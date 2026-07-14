"""Generate a procedural 'crowd' sprite for the arena story.

Scatters many mixed symbols on a transparent canvas to represent a dense,
abstract crowd. Run with:

    uv run python scripts/gen_crowd.py

Output: data/assets/cc/arena/Crowd/default_neutral.png
"""

from __future__ import annotations

import argparse
import collections
import math
import random
import subprocess
import tomllib
from pathlib import Path

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_OUT = 'data/assets/cc/arena/Crowd/default_neutral.png'
DEFAULT_FONT = '/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf'

DEFAULT_WIDTH = 1536
DEFAULT_HEIGHT = 1024
DEFAULT_SYMBOL_COUNT = 420

# Cluster shape: symbols are placed with a radial bias.
# Radius = random.betavariate(RADIUS_BETA_A, RADIUS_BETA_B) * RADIUS_SCALE.
# Smaller A / larger B pushes symbols toward the center; larger A spreads them out.
DEFAULT_RADIUS_BETA_A = 1.25
DEFAULT_RADIUS_BETA_B = 2.5
DEFAULT_RADIUS_SCALE = 0.96
DEFAULT_X_RADIUS = 615  # horizontal spread in px
DEFAULT_Y_RADIUS = 470  # vertical spread in px
DEFAULT_X_JITTER = 38  # gaussian jitter in px
DEFAULT_Y_JITTER = 44

DEFAULT_MIN_SIZE = 32
DEFAULT_MAX_SIZE = 96
DEFAULT_MIN_OPACITY = 55
DEFAULT_MAX_OPACITY = 100
DEFAULT_COLOR_JITTER = 18

DEFAULT_MIN_ROTATION = -18
DEFAULT_MAX_ROTATION = 18

DEFAULT_SEED = None

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

# Full ~/.Xresources Gruvbox palette.
COLORS = [
    (0x28, 0x28, 0x28),  # color0  black
    (0xCC, 0x24, 0x1D),  # color1  dark red
    # (0x98, 0x97, 0x1A),  # color2  dark green
    # (0xD7, 0x99, 0x21),  # color3  dark yellow
    (0x45, 0x85, 0x88),  # color4  dark blue
    (0xB1, 0x62, 0x86),  # color5  dark magenta
    # (0x68, 0x9D, 0x6A),  # color6  dark cyan
    (0xA8, 0x99, 0x84),  # color7  light grey
    (0x92, 0x83, 0x74),  # color8  dark grey
    (0xFB, 0x49, 0x34),  # color9  red
    # (0xB8, 0xBB, 0x26),  # color10 green
    (0xFA, 0xBD, 0x2F),  # color11 yellow
    (0x83, 0xA5, 0x98),  # color12 blue
    (0xD3, 0x86, 0x9B),  # color13 magenta
    # (0x8E, 0xC0, 0x7C),  # color14 cyan
    (0xEB, 0xDB, 0xB2),  # color15 white/cream
]


def load_symbols(toml_path: Path) -> tuple[list[str], dict[str, str]]:
    """Load symbol families from a TOML file.

    Returns a flat list of symbols and a mapping from each symbol to its
    family name.
    """
    with toml_path.open('rb') as f:
        data = tomllib.load(f)

    symbols: list[str] = []
    symbol_to_family: dict[str, str] = {}
    for family_name, family in data.items():
        if not isinstance(family, dict):
            continue
        family_symbols = family.get('symbols', [])
        if not isinstance(family_symbols, list):
            continue
        for raw in family_symbols:
            sym = str(raw)
            symbols.append(sym)
            symbol_to_family[sym] = family_name

    return symbols, symbol_to_family


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def build_command(
    args: argparse.Namespace,
    symbols_chosen: list[str],
) -> list[str]:
    cmd = [
        'magick',
        '-size',
        f'{args.width}x{args.height}',
        'xc:none',
    ]

    for sym in symbols_chosen:
        angle = random.uniform(0, 2 * math.pi)
        radius = (
            random.betavariate(args.beta_a, args.beta_b) * args.radius_scale
        )
        x = (
            args.width / 2
            + radius * args.x_radius * math.cos(angle)
            + random.gauss(0, args.x_jitter)
        )
        y = (
            args.height / 2
            + radius * args.y_radius * math.sin(angle)
            + random.gauss(0, args.y_jitter)
        )

        x = max(8, min(args.width - 8, x))
        y = max(8, min(args.height - 8, y))

        size = random.randint(args.min_size, args.max_size)
        opacity = random.randint(args.min_opacity, args.max_opacity)
        rotation = random.uniform(args.min_rotation, args.max_rotation)

        rgb = random.choice(COLORS)
        jitter = args.color_jitter
        r = max(0, min(255, rgb[0] + random.randint(-jitter, jitter)))
        g = max(0, min(255, rgb[1] + random.randint(-jitter // 2, jitter // 2)))
        b = max(0, min(255, rgb[2] + random.randint(-jitter // 2, jitter // 2)))
        color = f'rgba({r},{g},{b},{opacity / 100})'

        cmd.extend(
            [
                '-font',
                args.font,
                '-pointsize',
                str(size),
                '-fill',
                color,
                '-gravity',
                'none',
                '-draw',
                f"translate {int(x)},{int(y)} rotate {rotation:.1f} text 0,0 '{sym}'",
            ]
        )

    cmd.append(args.output)
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate a procedural crowd sprite.',
    )
    parser.add_argument(
        '-o', '--output', default=DEFAULT_OUT, help='Output PNG path'
    )
    parser.add_argument('--font', default=DEFAULT_FONT, help='Symbol font file')
    parser.add_argument(
        '-W', '--width', type=int, default=DEFAULT_WIDTH, help='Canvas width'
    )
    parser.add_argument(
        '-H', '--height', type=int, default=DEFAULT_HEIGHT, help='Canvas height'
    )
    parser.add_argument(
        '-n',
        '--count',
        type=int,
        default=DEFAULT_SYMBOL_COUNT,
        help='Number of symbols',
    )
    parser.add_argument(
        '--beta-a',
        type=float,
        default=DEFAULT_RADIUS_BETA_A,
        help='Beta dist alpha (center bias)',
    )
    parser.add_argument(
        '--beta-b',
        type=float,
        default=DEFAULT_RADIUS_BETA_B,
        help='Beta dist beta (tail falloff)',
    )
    parser.add_argument(
        '--radius-scale',
        type=float,
        default=DEFAULT_RADIUS_SCALE,
        help='Max normalized radius',
    )
    parser.add_argument(
        '--x-radius',
        type=float,
        default=DEFAULT_X_RADIUS,
        help='Horizontal spread in px',
    )
    parser.add_argument(
        '--y-radius',
        type=float,
        default=DEFAULT_Y_RADIUS,
        help='Vertical spread in px',
    )
    parser.add_argument(
        '--x-jitter',
        type=float,
        default=DEFAULT_X_JITTER,
        help='Horizontal gaussian jitter',
    )
    parser.add_argument(
        '--y-jitter',
        type=float,
        default=DEFAULT_Y_JITTER,
        help='Vertical gaussian jitter',
    )
    parser.add_argument(
        '--min-size', type=int, default=DEFAULT_MIN_SIZE, help='Min symbol size'
    )
    parser.add_argument(
        '--max-size', type=int, default=DEFAULT_MAX_SIZE, help='Max symbol size'
    )
    parser.add_argument(
        '--min-opacity',
        type=int,
        default=DEFAULT_MIN_OPACITY,
        help='Min opacity (0-100)',
    )
    parser.add_argument(
        '--max-opacity',
        type=int,
        default=DEFAULT_MAX_OPACITY,
        help='Max opacity (0-100)',
    )
    parser.add_argument(
        '--color-jitter',
        type=int,
        default=DEFAULT_COLOR_JITTER,
        help='RGB jitter amount',
    )
    parser.add_argument(
        '--min-rotation',
        type=float,
        default=DEFAULT_MIN_ROTATION,
        help='Min rotation angle in degrees',
    )
    parser.add_argument(
        '--max-rotation',
        type=float,
        default=DEFAULT_MAX_ROTATION,
        help='Max rotation angle in degrees',
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED, help='Random seed'
    )
    parser.add_argument(
        '--symbols',
        type=Path,
        default=Path(__file__).with_suffix('').parent
        / 'gen_crowd_symbols.toml',
        help='Path to TOML file containing symbol families',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols, symbol_to_family = load_symbols(args.symbols)
    if not symbols:
        raise SystemExit(f'No symbols found in {args.symbols}')

    random.seed(args.seed)
    symbols_chosen = [random.choice(symbols) for _ in range(args.count)]

    symbol_counts = collections.Counter(symbols_chosen)
    family_counts: dict[str, dict[str, int]] = collections.defaultdict(
        collections.Counter
    )
    for sym, count in symbol_counts.items():
        family = symbol_to_family.get(sym, 'unknown')
        family_counts[family][sym] = count

    cmd = build_command(args, symbols_chosen)
    subprocess.run(cmd, check=True)

    print(f'Generated {args.output}')
    print()
    for i, family in enumerate(sorted(family_counts)):
        if i:
            print()
        items = sorted(family_counts[family].items())
        family_total = sum(family_counts[family].values())
        print(f'[{family}: {family_total}]')
        print(', '.join(f'{sym}: {count}' for sym, count in items))
    print()
    print(f'[total] {sum(symbol_counts.values())}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
