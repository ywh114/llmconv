# Ara - Visual Novel Engine

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
```

## Quick start - Webclient

The web VN frontend is a thin polling GUI over the agent API.

**Internal server mode** (single command, no UNIX socket):

```bash
uv run arawc --internal --scene demo --port 8081 2>&1 | tee servlog
```

**Two-process mode**:

```bash
# Terminal 1 - agent server
uv run aragent --scene demo 2>&1 | tee log

# Terminal 2 - web gateway
uv run arawc --port 8081 2>&1 | tee servlog
```

Then open `http://localhost:8081` in your browser.

## `aractl`

`aractl` is a CLI client for the agent API.  It can start/step/reply through the
story, manage saves, and exposes the full debug console.

```bash
# Start or restart the story
uv run aractl --scene <SCENE> start

# Advance one tick
uv run aractl step

# Auto-step until player input is required
uv run aractl next

# Submit player input
uv run aractl reply "Hello, world"

# Any input starting with / or : is treated as a debug command
uv run aractl reply "/info"

# Run an explicit debug command
uv run aractl debug info
uv run aractl debug here
uv run aractl debug dump

# Show full state snapshot
uv run aractl state

# Save / load
uv run aractl save 1
uv run aractl load 1
uv run aractl saves
uv run aractl delete-save 1

# Kill the server daemon
uv run aractl --kill
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
| `state` | Game state metadata |
| `decision`, `dec` | Last orchestrator decision |
| `scratch <name>` | Show a character's scratchpad |
| `summary <name>` | Show a character's prev-scene summary |
| `dump`, `d` | Pretty-print the full LLM conversation context |

## Asset layout

Assets are organised by type, then story:

```
data/assets/
  cc/<story>/<character>/card.toml   # character definitions
  lc/<story>/<location>/card.toml    # location definitions
  items/<story>/<item>.toml          # item definitions
  world/<story>.toml                 # world wiki (realms, factions, places)
  plot/<story>/<scene>.toml          # scene scripts
```

Directory/file names are canonical IDs; the `[names]` table is only for
display/localisation.
