"""Alpha-D fold construction for balanced HPO validation."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from cases.alpha_d.extrapolation import parse_case_params


def balanced_parameter_folds(
    sim_names: list[str],
    candidate_indices: list[int],
    n_folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Balance marginal Re, Dr, and Lr levels across validation folds."""
    candidates = [int(index) for index in candidate_indices]
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_indices must not contain duplicates.")
    if any(index < 0 or index >= len(sim_names) for index in candidates):
        raise ValueError("candidate_indices contains an out-of-range case index.")
    if n_folds < 2 or n_folds > len(candidates):
        raise ValueError("n_folds must be between 2 and the number of candidate cases.")

    parameters = {index: parse_case_params(sim_names[index]) for index in candidates}
    level_frequency = {
        axis: Counter(getattr(parameters[index], axis) for index in candidates)
        for axis in ("Re", "Dr", "Lr")
    }
    shuffled = list(candidates)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda index: sum(
            level_frequency[axis][getattr(parameters[index], axis)] for axis in ("Re", "Dr", "Lr")
        )
    )
    tie_order = list(range(n_folds))
    rng.shuffle(tie_order)
    tie_rank = {fold: rank for rank, fold in enumerate(tie_order)}

    validation_groups: list[list[int]] = [[] for _ in range(n_folds)]
    fold_level_counts = [defaultdict(Counter) for _ in range(n_folds)]
    for index in shuffled:
        params = parameters[index]

        def fold_cost(fold: int) -> tuple[int, int, int, int]:
            marginal_counts = [
                fold_level_counts[fold][axis][getattr(params, axis)] for axis in ("Re", "Dr", "Lr")
            ]
            return (
                sum(marginal_counts),
                max(marginal_counts),
                len(validation_groups[fold]),
                tie_rank[fold],
            )

        chosen_fold = min(range(n_folds), key=fold_cost)
        validation_groups[chosen_fold].append(index)
        for axis in ("Re", "Dr", "Lr"):
            fold_level_counts[chosen_fold][axis][getattr(params, axis)] += 1

    # Greedy placement can leave a removable one-count marginal imbalance.
    # Deterministically swap cases across folds while doing so reduces the
    # total squared deviation from each level's ideal per-fold count.
    while True:
        best_swap: tuple[float, int, int, int, int] | None = None
        for left_fold in range(n_folds):
            for right_fold in range(left_fold + 1, n_folds):
                for left_pos, left_index in enumerate(validation_groups[left_fold]):
                    left_params = parameters[left_index]
                    for right_pos, right_index in enumerate(validation_groups[right_fold]):
                        right_params = parameters[right_index]
                        delta = 0.0
                        for axis in ("Re", "Dr", "Lr"):
                            left_level = getattr(left_params, axis)
                            right_level = getattr(right_params, axis)
                            if left_level == right_level:
                                continue
                            target_left = level_frequency[axis][left_level] / n_folds
                            target_right = level_frequency[axis][right_level] / n_folds
                            before = (
                                (fold_level_counts[left_fold][axis][left_level] - target_left) ** 2
                                + (fold_level_counts[right_fold][axis][left_level] - target_left)
                                ** 2
                                + (fold_level_counts[left_fold][axis][right_level] - target_right)
                                ** 2
                                + (fold_level_counts[right_fold][axis][right_level] - target_right)
                                ** 2
                            )
                            after = (
                                (fold_level_counts[left_fold][axis][left_level] - 1 - target_left)
                                ** 2
                                + (
                                    fold_level_counts[right_fold][axis][left_level]
                                    + 1
                                    - target_left
                                )
                                ** 2
                                + (
                                    fold_level_counts[left_fold][axis][right_level]
                                    + 1
                                    - target_right
                                )
                                ** 2
                                + (
                                    fold_level_counts[right_fold][axis][right_level]
                                    - 1
                                    - target_right
                                )
                                ** 2
                            )
                            delta += after - before
                        candidate = (
                            delta,
                            left_fold,
                            right_fold,
                            left_pos,
                            right_pos,
                        )
                        if delta < -1.0e-12 and (best_swap is None or candidate < best_swap):
                            best_swap = candidate
        if best_swap is None:
            break
        _, left_fold, right_fold, left_pos, right_pos = best_swap
        left_index = validation_groups[left_fold][left_pos]
        right_index = validation_groups[right_fold][right_pos]
        for axis in ("Re", "Dr", "Lr"):
            left_level = getattr(parameters[left_index], axis)
            right_level = getattr(parameters[right_index], axis)
            if left_level != right_level:
                fold_level_counts[left_fold][axis][left_level] -= 1
                fold_level_counts[left_fold][axis][right_level] += 1
                fold_level_counts[right_fold][axis][left_level] += 1
                fold_level_counts[right_fold][axis][right_level] -= 1
        validation_groups[left_fold][left_pos] = right_index
        validation_groups[right_fold][right_pos] = left_index

    candidate_set = set(candidates)
    return [
        (sorted(candidate_set - set(validation)), sorted(validation))
        for validation in validation_groups
    ]
