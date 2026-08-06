"""Generic explicit case-fold builders for HPO."""

from __future__ import annotations

import random


def random_case_folds(
    sim_names: list[str],
    candidate_indices: list[int],
    n_folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Build deterministic, shuffled case folds from the candidate pool."""
    candidates = [int(index) for index in candidate_indices]
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_indices must not contain duplicates.")
    if any(index < 0 or index >= len(sim_names) for index in candidates):
        raise ValueError("candidate_indices contains an out-of-range case index.")
    if n_folds < 2 or n_folds > len(candidates):
        raise ValueError("n_folds must be between 2 and the number of candidate cases.")

    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    validation_groups = [sorted(shuffled[offset::n_folds]) for offset in range(n_folds)]
    candidate_set = set(candidates)
    return [
        (sorted(candidate_set - set(validation)), validation) for validation in validation_groups
    ]


def validate_folds(
    folds: list[tuple[list[int], list[int]]],
    candidate_indices: list[int],
    n_folds: int,
) -> None:
    """Validate the explicit fold-builder contract."""
    if not isinstance(folds, list) or len(folds) != n_folds:
        raise ValueError(f"Fold builder must return exactly {n_folds} folds.")
    candidates = set(int(index) for index in candidate_indices)
    seen_validation: list[int] = []
    for fold_index, fold in enumerate(folds):
        if not isinstance(fold, tuple) or len(fold) != 2:
            raise ValueError(f"Fold {fold_index} must be a (train_indices, val_indices) tuple.")
        train_indices = [int(index) for index in fold[0]]
        val_indices = [int(index) for index in fold[1]]
        train_set = set(train_indices)
        val_set = set(val_indices)
        if not train_indices or not val_indices:
            raise ValueError(f"Fold {fold_index} has an empty training or validation set.")
        if len(train_set) != len(train_indices) or len(val_set) != len(val_indices):
            raise ValueError(f"Fold {fold_index} contains duplicate indices.")
        if train_set & val_set:
            raise ValueError(f"Fold {fold_index} training and validation sets overlap.")
        if train_set | val_set != candidates:
            raise ValueError(f"Fold {fold_index} does not exactly cover candidate_indices.")
        seen_validation.extend(val_indices)
    if sorted(seen_validation) != sorted(candidate_indices):
        raise ValueError("Each candidate case must appear in validation exactly once.")
