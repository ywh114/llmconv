"""Story runner that manages scene transitions and persistent character state.

The :class:`Story` class is the "completed plot subsystem" missing from the
original proof-of-concept.  It loads scenes sequentially, carries character
scratchpad and memory state across scene boundaries, and finalises each scene
by running end-of-scene scratch updates for important characters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.tools import ToolRegistry, tool
from ara.memory.chroma import ChromaStore
from ara.models import GameRole, Importance
from ara.utils.debug import DebugConsole
from ara.utils.logger import get_logger
from ara.world.character import Character
from ara.world.engine import Engine
from ara.world.scene import Scene
from ara.world.summarizer import Summarizer

logger = get_logger(__name__)


def _merge_characters(prev_scene: Scene, new_scene: Scene) -> None:
    """Carry over memory state for characters present in both scenes.

    Scratchpads are intentionally NOT carried over — the summarizer produces
    fresh per-character orientations for the next scene instead.

    :param prev_scene: Scene that just ended.
    :param new_scene: Scene that is about to begin.
    """
    prev_by_name = {c.name: c for c in prev_scene.character_pool}
    carried = 0
    for char in new_scene.character_pool:
        if char.name in prev_by_name and char.importance >= Importance.IMPORTANT:
            old = prev_by_name[char.name]
            char.memory = old.memory
            carried += 1
    logger.debug(f"Carried over memory for {carried} characters into new scene")


def _finalize_scene(
    scene: Scene,
    engine: Engine,
    directives_log: dict[Character, str],
) -> None:
    """Run end-of-scene scratch updates for important characters.

    Only characters with :attr:`Importance.IMPORTANT` or higher participate.
    The player and narrator are skipped.

    :param scene: The scene that just ended.
    :param engine: Active engine instance (provides the LLM client).
    :param directives_log: Mapping from character to the last directive they
        received during the scene.
    """
    for char in scene.character_pool:
        if char == scene.player or char == scene.narrator:
            continue
        if char.importance < Importance.IMPORTANT:
            continue

        scratch_tool = tool(
            name="write_scratch",
            description="""Update your scratchpad based on the conversation.
Use this space to reason and come up with plans.
Make guesses on when you might meet the other character(s) again.
Clean up and only keep what will be useful to carry over into future conversations.
If the scratchpad does not need changing, do not provide the `contents` field.""",
            properties={
                "contents": {
                    "type": "string",
                    "description": "New scratch content. Replaces all previous content.",
                }
            },
            required=[],
            strict=True,
        )

        registry = ToolRegistry()

        def _make_scratch_hook(target: Character):
            def scratch_hook(args: str) -> str:
                data = json.loads(args)
                if "contents" in data:
                    target.scratch.text = data["contents"]
                    return "Updated scratch."
                return "No changes."
            return scratch_hook

        registry.register("write_scratch", _make_scratch_hook(char))

        prompt = (
            f"{char.name}, the current conversation has ended. "
            "Follow the instructions in your system prompt."
        )
        system = f"""IMPORTANT: Write scratch in {scene.language} only!
The current round of conversation has ended.
# Role
 - You are the ephemeral scratch-writing agent representing {char.name}.
 - Write how you think {char.name} would reply based on {char.name}'s previous messages.

## Instructions
 - Based on the previous rounds of conversation, update your scratchpad.
 - Use this space to reason and come up with plans.
 - Make guesses on when you might meet the other character(s) again.
 - Clean up and only keep what will be useful to carry over into future conversations.

## Additional directives given to {char.name}
    - {directives_log.get(char, "None")}
"""

        ctx = ConversationContext(char.name)
        ctx.concat_context(char.whoami)
        ctx.concat_context(char.scratch_context)
        ctx.user_message(prompt, name="System")

        result = engine.client.complete(
            role=GameRole.CHARACTER,
            system_prompt=system,
            messages=ctx.to_list(),
            tools=[scratch_tool],
            tool_choice="auto",
            stream=False,
        )

        for tc in result.tool_calls:
            registry.call(tc["function"]["name"], tc["function"]["arguments"])

        # Generate a structured memory summary via sub-agent
        summary = engine.client.complete_subagent(
            task=f"Summarise what {char.name} experienced and learned in this scene. "
                 f"Focus on facts, relationships, and open questions. "
                 f"Write from {char.name}'s perspective in {scene.language}.",
            context=char.scratch.text,
            max_tokens=256,
        )
        if summary.strip():
            char.memory.add_conversation([summary.strip()])
            logger.debug(f"Stored memory summary for {char.name}")


@dataclass
class StoryStep:
    """Result of a single :meth:`Story.step` call.

    :ivar event: What happened this tick.  One of ``scene_loaded``,
        ``turn``, ``needs_player_input``, ``scene_ended``, ``story_complete``.
    :ivar scene: The newly-loaded scene (only for ``scene_loaded``).
    :ivar suggestions: Orchestrator suggestions (only for ``needs_player_input``).
    :ivar next_scene: Identifier of the next scene (only for ``scene_ended``).
    :ivar speaker: Name of the character whose turn this was.
    :ivar enter: Names of characters that entered this turn.
    :ivar exit: Names of characters that exited this turn.
    :ivar sprite_changes: Mapping of character names → new sprite names.
    """

    event: str
    scene: Scene | None = None
    suggestions: list[str] | None = None
    next_scene: str | None = None
    speaker: str | None = None
    enter: list[str] = field(default_factory=list)
    exit: list[str] = field(default_factory=list)
    sprite_changes: dict[str, str] = field(default_factory=dict)


class Story:
    """Top-level controller that runs a sequence of scenes.

    A :class:`Story` instance is initialised with a starting scene file.
    Calling :meth:`start` followed by repeated :meth:`step` drives the story
    as a state machine:

    1. Load the next scene TOML.
    2. Merge character state from the previous scene.
    3. Run the :class:`Engine` one turn at a time.
    4. Finalise the scene (end-of-scene scratch updates).
    5. Resolve the chosen next scene and repeat.

    The old blocking :meth:`run` convenience wrapper is still available.

    :param config: Application settings.
    :param db: Shared ChromaDB store.
    :param client: LLM client instance.
    :param initial_scene_path: Path to the first scene ``.toml`` file.
    """

    def __init__(
        self,
        config: AraSettings,
        db: ChromaStore,
        client: LLMClient,
        initial_scene_path: Path,
    ) -> None:
        self.config = config
        self.db = db
        self.client = client
        self.engine = Engine(client, db=db)
        self.initial_scene_path = initial_scene_path
        self._scene_history: list[str] = []
        self._current_path: Path | None = None
        self._prev_scene: Scene | None = None
        self._current_scene: Scene | None = None
        self._state: str = "idle"
        self._skipped_scene: bool = False
        self._summarizer = Summarizer(client)
        self._next_scene_summaries: dict[str, str] = {}
        self._next_scene_location_desc: str = ""

    @property
    def finished(self) -> bool:
        """``True`` when the story has no more scenes to play."""
        return self._state == "complete"

    @property
    def current_scene(self) -> Scene | None:
        """The scene currently being played, or ``None``."""
        return self._current_scene

    @property
    def scene_history(self) -> list[str]:
        """List of scene identifiers visited so far."""
        return self._scene_history

    def start(self, scene_id: str | None = None) -> None:
        """Prepare the story for playback.

        Must be called before the first :meth:`step`.

        :param scene_id: Optional scene identifier to start at instead of the
            initial scene path.  The identifier is resolved relative to the
            initial scene's directory.
        """
        if scene_id:
            self._current_path = self._resolve_scene_path(scene_id)
        else:
            self._current_path = self.initial_scene_path
        self._prev_scene = None
        self._scene_history = []
        self._current_scene = None
        self._state = "loading"

    def step(self) -> StoryStep:
        """Advance the story by one tick.

        A tick is one of: load a scene, run one engine turn, request player
        input, finalise a scene, or complete the story.

        :return: Description of what happened this tick.
        :raises RuntimeError: If the story is not started, is already complete,
            or is waiting for player input.
        """
        if self._state == "idle":
            raise RuntimeError("Story not started. Call start() first.")
        if self._state == "loading":
            return self._load_scene()
        if self._state == "running":
            if self.engine.needs_player_input:
                raise RuntimeError(
                    "Story is waiting for player input. Call submit_player_input() first."
                )
            result = self.engine.step()
            if result.needs_player_input:
                return StoryStep(
                    event="needs_player_input", suggestions=result.suggestions,
                    speaker=result.speaker,
                    enter=result.enter,
                    exit=result.exit,
                    sprite_changes=result.sprite_changes,
                )
            if result.scene_ended:
                return self._finalize_and_transition()
            return StoryStep(
                event="turn", speaker=result.speaker,
                enter=result.enter,
                exit=result.exit,
                sprite_changes=result.sprite_changes,
            )
        if self._state == "complete":
            raise RuntimeError("Story is already complete")
        raise RuntimeError(f"Unknown story state: {self._state}")

    def submit_player_input(self, text: str) -> None:
        """Forward player input to the engine.

        Must only be called after :meth:`step` returned
        ``event="needs_player_input"``.
        """
        self.engine.submit_player_input(text)

    def generate_player_input(self, suggestion: str) -> str:
        """Generate natural player dialogue from a suggestion.

        :param suggestion: Suggestion text to expand.
        :return: Generated player dialogue.
        """
        return self.engine.generate_player_input(suggestion)

    def _load_scene(self) -> StoryStep:
        """Load the scene at :attr:`_current_path` and start the engine."""
        if self._current_path is None or not self._current_path.exists():
            self._state = "complete"
            return StoryStep(event="story_complete")

        prev_id = self._prev_scene.id if self._prev_scene else ""
        scene = Scene.load(self._current_path, self.db, self.config, prev_id=prev_id)
        self._current_scene = scene

        # Fin scenes immediately end the story without running turns.
        if scene.scene_type == "fin":
            self._state = "complete"
            return StoryStep(event="story_complete", scene=scene)

        if self._prev_scene is not None:
            _merge_characters(self._prev_scene, scene)

        self._scene_history.append(scene.id)
        self._state = "running"
        self.engine.start(scene)

        # Distribute per-character bridging summaries from the summarizer.
        if self._next_scene_summaries:
            for char in scene.character_pool:
                summary = self._next_scene_summaries.get(char.name, "")
                if summary:
                    char.prev_scene_summary = summary
            self._next_scene_summaries = {}

        # Apply finalized location description from the summarizer.
        if self._next_scene_location_desc:
            scene.starting_location.desc = self._next_scene_location_desc
            self._next_scene_location_desc = ""

        if self._skipped_scene and self.engine.ctx is not None:
            self.engine.ctx.user_message(
                "[SYSTEM NOTICE: A scene skip occurred. The narrative jumped directly to this scene. "
                "Characters may need to re-establish context.]",
                name="System",
            )
            self._skipped_scene = False
        return StoryStep(event="scene_loaded", scene=scene)

    def _finalize_and_transition(self) -> StoryStep:
        """Run end-of-scene finalisation and choose the next scene."""
        assert self._current_scene is not None
        _finalize_scene(self._current_scene, self.engine, self.engine.directives_log)
        self._prev_scene = self._current_scene

        # Store scene summary in story-wide history
        scene = self._current_scene
        summary = (
            f"Scene: {scene.id}\n"
            f"Characters: {[c.name for c in scene.starting_characters]}\n"
            f"Plot: {scene.plot_story[:500]}\n"
            f"Next scene: {self.engine.next_scene or '(end)'}"
        )
        self.db.upsert(
            "story_history",
            ids=[f"scene_{scene.id}"],
            documents=[summary],
            metadatas=[{"scene_id": scene.id}],
        )

        next_scene = self.engine.next_scene
        if next_scene and next_scene in self._current_scene.next_choices:
            next_choice_obj = self._current_scene.next_choices[next_scene]
            self._current_path = self._resolve_scene_path(next_choice_obj.id)

            # Run the transition summarizer to bridge context.
            self._run_summarizer(next_scene)

            self._state = "loading"
            return StoryStep(event="scene_ended", next_scene=next_scene)

        self._current_path = None
        self._state = "complete"
        return StoryStep(event="story_complete")

    def _run_summarizer(self, next_scene_id: str) -> None:
        """Generate a bridging summary and finalize location edits."""
        assert self._current_scene is not None
        assert self.engine.ctx is not None
        assert self.engine.loc is not None

        next_path = self._current_path
        if next_path is None or not next_path.exists():
            return

        try:
            import tomllib
            with next_path.open("rb") as f:
                next_data = tomllib.load(f)
        except Exception:
            return

        next_plot = next_data.get("plot", {}).get("scene", "")
        next_considerations = next_data.get("plot", {}).get("considerations", "")

        # Gather scratchpads from characters in the current scene.
        scratchpads = {
            c.name: c.scratch.text
            for c in self._current_scene.character_pool
            if c.scratch.text and c.scratch.text != "Nothing yet!"
        }

        # Peek at next scene's character pool from TOML.
        next_scene_chars: list[str] = []
        try:
            next_chars = next_data.get("character", {}).get("pool", [])
            if next_chars:
                next_scene_chars = list(next_chars)
        except Exception:
            pass

        bridging_summaries, finalized_loc = self._summarizer.summarize_transition(
            current_scene=self._current_scene,
            next_scene_plot=next_plot,
            next_scene_considerations=next_considerations,
            conversation_context=self.engine.ctx.to_list(),
            location_desc=self.engine.loc.desc,
            language=self._current_scene.language,
            scratchpads=scratchpads,
            next_scene_chars=next_scene_chars,
        )

        self._next_scene_summaries = bridging_summaries
        self._next_scene_location_desc = finalized_loc
        total_chars = len(bridging_summaries)
        logger.info(f"Summarizer produced {total_chars} character summaries for transition to {next_scene_id}")

    def run(
        self,
        get_user_input: Callable[[str, list[str]], str] | None = None,
        debug_console: DebugConsole | None = None,
    ) -> None:
        """Execute the full story from the initial scene onward.

        This is a convenience wrapper around :meth:`start` and :meth:`step`.
        For CLI-driven playback use the state-machine API directly.

        :param get_user_input: Callback that receives a prompt string and a
            list of suggestions, and returns the player's typed input.
            Defaults to a simple ``input()`` wrapper.
        :param debug_console: Optional debug console.
        """
        if get_user_input is None:
            get_user_input = self._get_user_input

        self.engine.set_debug_console(debug_console)
        self.start()

        while not self.finished:
            if (
                debug_console
                and debug_console.auto_pause
                and self._state == "running"
                and not self.engine.needs_player_input
            ):
                self._debug_pause(debug_console)

            result = self.step()

            if result.event == "scene_loaded" and result.scene is not None:
                print(f"\n=== Scene: {result.scene.id} ===")
                print(f"Location: {result.scene.starting_location.name}")
                print(f"Characters: {[c.name for c in result.scene.starting_characters]}\n")

            elif result.event == "needs_player_input":
                suggestions = result.suggestions or []
                while True:
                    text = get_user_input(
                        f"{self.current_scene.player.name}> ", suggestions
                    )
                    stripped = text.strip()
                    if stripped.startswith(("/", ":")):
                        if debug_console is not None:
                            self._debug_pause(debug_console, noshell=stripped[1:])
                        else:
                            print("Debug console not available. Start with --debug-console.")
                        continue
                    break
                self.submit_player_input(text)

        print("\n=== Story Complete ===")
        print(f"Scenes visited: {self._scene_history}")

    def _debug_pause(self, debug_console: DebugConsole, noshell: str = "") -> None:
        """Enter the debug console with the current engine state."""
        if self.current_scene is None:
            return
        debug_console.pause(
            scene=self.current_scene,
            ctx=self.engine.ctx,
            here_chars=self.engine.here_chars,
            away_chars=self.engine.away_chars,
            loc=self.engine.loc or self.current_scene.starting_location,
            decision=self.engine.last_decision,
            noshell=noshell,
        )

    def _get_user_input(self, prompt: str, suggestions: list[str]) -> str:
        """Display *prompt* and read a line from stdin.

        :param prompt: Text to display before the cursor.
        :param suggestions: Orchestrator-provided suggestions (unused by the
            default implementation but available for custom front-ends).
        :return: The player's input, or a default continuation string on EOF.
        """
        print(prompt, end="")
        try:
            return input()
        except EOFError:
            return "[OOC: continue]"

    def jump_to(self, scene_id: str) -> StoryStep:
        """Abandon the current scene and jump directly to *scene_id*.

        :param scene_id: Scene identifier (filename without extension).
        :return: The loaded scene step.
        :raises RuntimeError: If the scene file does not exist.
        """
        path = self._resolve_scene_path(scene_id)
        if not path.exists():
            raise RuntimeError(f"Scene '{scene_id}' not found at {path}")
        self._current_path = path
        self._state = "loading"
        self._skipped_scene = True
        return self._load_scene()

    def _resolve_scene_path(self, scene_id: str) -> Path:
        """Map a scene identifier to a ``.toml`` file path.

        Scene files are looked up in the same directory as the initial scene.

        :param scene_id: Scene identifier (filename without extension).
        :return: Resolved file path.
        """
        base = self.initial_scene_path.parent
        return base / f"{scene_id}.toml"
