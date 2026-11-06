"""Correct maximization-based Pareto ranking and a size-limited output archive.

All objectives are assumed to be scores for which *larger is better*.
The full Pareto front is maintained separately from the size-limited selection.
"""

from __future__ import annotations

import numpy as np


def _vector(scores, num_objectives):
    vec = np.asarray(scores, dtype=float)
    if vec.ndim != 1 or not (1 <= num_objectives <= vec.size):
        raise ValueError("Score must be a 1-D vector containing all active objectives")
    if not np.isfinite(vec).all():
        raise ValueError("Pareto scores must be finite (no NaN or infinity)")
    return vec


def dominates(a, b, num_objectives):
    """True if a is >= b in all active objectives and > in at least one."""
    a = _vector(a, num_objectives)[:num_objectives]
    b = _vector(b, num_objectives)[:num_objectives]
    if a.shape != b.shape:
        raise ValueError("Mismatched score vector lengths")
    return bool(np.all(a >= b) and np.any(a > b))


def add_nondominated(front, key, scores, num_objectives, payload=None, get_scores=None):
    """Insert key if not dominated, removing only entries dominated by it.

    Modifies `front` in place. Equal score vectors for different keys are retained.
    This assumes the input front was already Pareto-filtered for the same objectives.
    """
    scores = _vector(scores, num_objectives)
    if get_scores is None:
        get_scores = lambda value: value
    # Replacements must be rebuilt by callers; simply skip identical existing keys.
    if key in front:
        raise ValueError("An existing key must be removed and the front rebuilt")

    to_delete = []
    for old_key, old_payload in front.items():
        old_scores = get_scores(old_payload)
        if dominates(old_scores, scores, num_objectives):
            return False
        if dominates(scores, old_scores, num_objectives):
            to_delete.append(old_key)

    for old_key in to_delete:
        del front[old_key]
    front[key] = scores.copy() if payload is None else payload
    return True


def _crowding_subset(full_front, max_size, num_objectives):
    """Keep a diverse non-dominated subset with NSGA-II crowding distances."""
    if len(full_front) <= max_size:
        return dict(full_front)
    keys = list(full_front)
    values = np.stack([_vector(full_front[k]["scores"], num_objectives)
                       for k in keys])[:, :num_objectives]
    n = len(keys)
    crowding = np.zeros(n, dtype=float)
    for j in range(num_objectives):
        order = np.argsort(values[:, j], kind="stable")
        low, high = values[order[0], j], values[order[-1], j]
        crowding[order[0]] = np.inf
        crowding[order[-1]] = np.inf
        if high > low:
            for rank in range(1, n - 1):
                idx = order[rank]
                if np.isfinite(crowding[idx]):
                    crowding[idx] += (values[order[rank + 1], j] -
                                      values[order[rank - 1], j]) / (high - low)
    # Stable ordering breaks ties and makes experiments reproducible.
    chosen = set(sorted(range(n), key=lambda i: (-crowding[i], i))[:max_size])
    return {k: full_front[k] for i, k in enumerate(keys) if i in chosen}


class ParetoArchive:
    """All seen candidates, exact front, and at-most-max_size output subset.

    Rebuilds from all candidates when the number of active objectives changes,
    because a candidate discarded under earlier objectives can become Pareto
    optimal when additional objectives are introduced.
    """

    def __init__(self, max_size):
        self.max_size = int(max_size)
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        self.candidates = {}  # Every unique sequence and its predicted scores.
        self.full_front = {}  # Exact Pareto front across all seen sequences.
        self.selected = {}  # Bounded subset for saving as CSV.
        self.num_objectives = None
        self.score_length = None

    def add(self, sequence, scores, token_ids, num_objectives):
        scores = _vector(scores, num_objectives).copy()
        if self.score_length is not None and scores.size != self.score_length:
            raise ValueError("Score vector lengths must be consistent")
        self.score_length = scores.size
        if hasattr(token_ids, "detach"):
            token_ids = token_ids.detach().cpu()
        payload = {"scores": scores, "token_ids": token_ids}
        replaced = sequence in self.candidates
        self.candidates[sequence] = payload

        if replaced or self.num_objectives != num_objectives:
            # New objective dimensions can cause formerly discarded points to
            # become non-dominated. Rebuild using the complete candidate history.
            self.full_front = {}
            for seq, item in self.candidates.items():
                add_nondominated(self.full_front, seq, item["scores"],
                                 num_objectives, payload=item,
                                 get_scores=lambda v: v["scores"])
        else:
            add_nondominated(self.full_front, sequence, scores,
                             num_objectives, payload=payload,
                             get_scores=lambda v: v["scores"])
        self.num_objectives = num_objectives
        self.selected = _crowding_subset(self.full_front, self.max_size, num_objectives)
        return self.selected
