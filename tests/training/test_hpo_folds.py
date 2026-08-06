"""Fold builders that do not require the heavy training runtime."""

from collections import Counter

from cases.alpha_d.hpo import balanced_parameter_folds
from training.hpo.splits import random_case_folds, validate_folds


def _grid_names() -> list[str]:
    return [
        f"Re_{re}__Dr_{str(dr).replace('.', 'p')}__Lr_{str(lr).replace('.', 'p')}"
        for re in (1000, 2000, 3000)
        for dr in (0.3, 0.5, 0.7)
        for lr in (0.1, 0.2, 0.3)
    ]


def test_random_case_folds_are_deterministic_and_exclude_outer_cases() -> None:
    names = [f"case-{index}" for index in range(12)]
    candidates = list(range(9))
    folds = random_case_folds(names, candidates, 3, 42)

    assert folds == random_case_folds(names, candidates, 3, 42)
    validate_folds(folds, candidates, 3)
    assert all(not ({9, 10, 11} & set(train + val)) for train, val in folds)


def test_alpha_d_folds_balance_each_parameter_margin() -> None:
    names = _grid_names()
    candidates = list(range(len(names)))
    folds = balanced_parameter_folds(names, candidates, 3, 42)
    validate_folds(folds, candidates, 3)

    for axis, token in (("Re", 0), ("Dr", 1), ("Lr", 2)):
        del axis
        per_level: dict[str, Counter] = {}
        for fold_index, (_train, validation) in enumerate(folds):
            for index in validation:
                level = names[index].split("__")[token]
                per_level.setdefault(level, Counter())[fold_index] += 1
        assert all(
            max(counts.values()) - min(counts.values()) <= 1 for counts in per_level.values()
        )
