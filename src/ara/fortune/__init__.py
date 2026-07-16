"""Fortune grammar package: title, ability, and human-name generation.

Self-contained submodule; the rest of the project should only use the public
API re-exported here instead of reaching into the submodules.
(``ara.world.fortune`` adds I-Ching, inspiration, and distribution sampling
on top of this package.)
"""

from ara.fortune import tokens
from ara.fortune.ability import (
    SLOTS as ABILITY_SLOTS,
    ability_dirs,
    categorized_ability_flavors,
    generate_ability,
    list_ability_flavors,
    load_ability_grammar,
    load_templates as load_ability_templates,
)
from ara.fortune.names import generate_name
from ara.fortune.title import (
    GENERIC_SLOTS as TITLE_SLOTS,
    LEVELS,
    apply_expose,
    build_grammar,
    categorized_title_flavors,
    cull_grammar,
    expand,
    expand_all,
    expand_all_traced,
    expand_traced,
    generate_title,
    list_title_flavors,
    load_templates as load_title_templates,
    load_title_grammar,
    load_toml,
    resolve_level,
    title_case,
    title_dirs,
)

__all__ = [
    "ABILITY_SLOTS",
    "LEVELS",
    "TITLE_SLOTS",
    "ability_dirs",
    "apply_expose",
    "build_grammar",
    "categorized_ability_flavors",
    "categorized_title_flavors",
    "cull_grammar",
    "expand",
    "expand_all",
    "expand_all_traced",
    "expand_traced",
    "generate_ability",
    "generate_name",
    "generate_title",
    "list_ability_flavors",
    "list_title_flavors",
    "load_ability_grammar",
    "load_ability_templates",
    "load_title_grammar",
    "load_title_templates",
    "load_toml",
    "resolve_level",
    "title_case",
    "title_dirs",
    "tokens",
]
