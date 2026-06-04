# Visual Novel Engine

Ara is a multi-character AI roleplay / visual novel engine.  It drives
scene-based narrative via an LLM orchestrator that decides who speaks next,
while individual characters (and a narrator) generate dialogue with their own
personalities, memories, and tools.

## Bootstrap (after clone or pull)

```bash
# 1. Install Python dependencies (including webclient extras)
uv sync --extra web

# 2. Set your LLM API key
export DEEPSEEK_API_KEY="sk-..."

# 3. Verify everything works
uv run pytest tests/ -q
# Ignore errors, WIP.
```

## Quick start — Webclient

The web VN frontend is a thin polling GUI over the agent API.

**1. Start the agent server**:

```bash
uv run python -m ara.agent --scene data/assets/plot/0.toml 2>&1 | tee log
```

**2. In a second terminal, after a few seconds, start the web gateway**:

```bash
uv run python -m ara.webclient --port 8080 2>&1 | tee servlog
```

Then open `http://localhost:8080` in your browser. `0.toml` is the first demo
scene.

> TODO: later, wrap the two commands in a single helper script.

## Debugging via `aractl`

`examples/aractl.py` is a first-class CLI client for the agent API.  It can
start/step/reply through the story and also exposes the full debug console.

```bash
# Start or restart the story
uv run python examples/aractl.py start

# Advance one tick
uv run python examples/aractl.py step

# Auto-step until player input is required
uv run python examples/aractl.py next

# Submit player input
uv run python examples/aractl.py reply "Hello there"

# Any input starting with / or : is treated as a debug command
uv run python examples/aractl.py reply "/info"

# Run an explicit debug command
uv run python examples/aractl.py debug info
uv run python examples/aractl.py debug here
uv run python examples/aractl.py debug dump

# Show full state snapshot
uv run python examples/aractl.py state

# Kill the server daemon
uv run python examples/aractl.py --kill

# TODO: Make the command shorter. Wrap in shell script.
```

### Available debug commands

| Command | Description |
|---|---|
| `help`, `h` | Show debug help |
| `info`, `i` | Engine state summary (scene, location, who is here/away) |
| `here` | Characters present in the scene |
| `away` | Characters away from the scene |
| `loc` | Current location details |
| `scene` | Scene metadata |
| `decision`, `dec` | Last orchestrator decision |
| `scratch <name>` | Show a character's scratchpad |
| `summary <name>` | Show a character's prev-scene summary |
| `dump`, `d` | Pretty-print the full LLM conversation context |
| `exec`, `x <code>` | Execute arbitrary Python (DANGEROUS) |

## Architecture

- Ask an LLM.

## Testing with short scenes

Two minimal test scenes live under `data/assets/plot/test/`:

```bash
# Test the two-scene flow + summarizer bridge
uv run python -m ara.agent --scene data/assets/plot/test/test_a.toml 2>&1 | tee log
```

Then in another terminal:

```bash
uv run python examples/aractl.py start
uv run python examples/aractl.py next
# … continue through scene A → scene B → fin …
```

## Roadmap

### Parallel summarizer (future)

Currently the summarizer runs serially during scene finalisation.  The plan is
 to move it to a background worker model:

1. **Locking** — When a scene ends, lock the resources it modifies
   (character scratchpads, location descriptions).  Other operations that
   touch locked resources poll until unlocked.

2. **PID tracking** — Each summarizer worker gets a log and a PID.  When
   polling a locked resource, verify the worker that holds the lock is still
   alive.  If the worker died, spin up a replacement and retry.

3. **Continue without blocking** — The engine should not wait for the
   summarizer to finish before allowing the player to interact with the next
   scene.  The bridging summary is injected lazily when it becomes available.
