"""Choice over more options than one call allows, by tournament.

Options are split into groups that fit a single Choice, the top few of each group advance, and
a final round picks among the finalists. Probabilities in the result come from the final round,
scaled by each finalist's first-round share so eliminated options keep a small, honest mass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kgqa.jev.questions import MAX_CHOICE_OPTIONS, Choice, ChoiceResult


def choose_large(
    ask,
    state: Any,
    criteria: Mapping[str, Any],
    instructions: Any = None,
    *,
    advance: int = 3,
    abstain_key: str | None = "other",
) -> ChoiceResult:
    """`ask(state, questions) -> JevResponse`; returns one ChoiceResult over all options."""
    options = dict(criteria)
    abstain = options.pop(abstain_key, None) if abstain_key else None
    group_size = MAX_CHOICE_OPTIONS - (1 if abstain_key else 0)
    if len(options) <= group_size:
        full = {**options, **({abstain_key: abstain} if abstain_key else {})}
        return ask(state, {"q": Choice(criteria=full, instructions=instructions)}).choice("q")

    keys = list(options)
    groups = [keys[i : i + group_size] for i in range(0, len(keys), group_size)]
    questions = {f"g{i}": Choice(criteria={k: options[k] for k in g}, instructions=instructions) for i, g in enumerate(groups)}
    first = ask(state, questions)
    finalists: dict[str, float] = {}
    for name in questions:
        for k, p in first.choice(name).ranked()[:advance]:
            finalists[k] = p
    final_criteria = {k: options[k] for k in finalists}
    if abstain_key:
        final_criteria[abstain_key] = abstain
    final = ask(state, {"final": Choice(criteria=final_criteria, instructions=instructions)}).choice("final")
    probs = dict(final.probabilities)
    for k in keys:
        probs.setdefault(k, 0.0)
    return ChoiceResult(final.choice, probs, final.confidence)
