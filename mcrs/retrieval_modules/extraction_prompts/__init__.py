"""Extraction prompt registry — maps (category, specificity) to a prompt string."""

from . import default
from . import cat_a
from . import cat_b
from . import cat_c
from . import cat_d
from . import cat_e
from . import cat_f
from . import cat_g
from . import cat_h
from . import cat_i
from . import cat_j
from . import cat_k

_MODULES = {
    "A": cat_a,
    "B": cat_b,
    "C": cat_c,
    "D": cat_d,
    "E": cat_e,
    "F": cat_f,
    "G": cat_g,
    "H": cat_h,
    "I": cat_i,
    "J": cat_j,
    "K": cat_k,
}


def get_prompt(category: str | None, specificity: str | None) -> str:
    """Return the extraction prompt for the given category and specificity."""
    if category is None:
        return default.PROMPT
    mod = _MODULES.get(category)
    if mod is None:
        return default.PROMPT
    return mod.PROMPTS.get(specificity or "default", mod.PROMPTS.get("default", default.PROMPT))


def get_result_builder(category: str | None):
    """Return the build_result(data, specificity, fallback) function for the category.

    Returns None when the category has no custom builder (uses generic builder).
    """
    if category is None:
        return None
    mod = _MODULES.get(category)
    return getattr(mod, "build_result", None)
