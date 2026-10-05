"""Immutable Train-only pilot membership for Dataset B TabMmini."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


PILOT_SAMPLE_FRACTION = 0.20
PILOT_STOP_FRACTION = 0.20
PILOT_RANDOM_STATE = 42


@dataclass(frozen=True)
class TabMMiniPilotSplit:
    """Positions and IDs for the locked Train-only pilot and inner split."""

    source_record_ids: tuple[str, ...]
    pilot_positions: tuple[int, ...]
    pilot_record_ids: tuple[str, ...]
    fit_positions: tuple[int, ...]
    fit_record_ids: tuple[str, ...]
    stop_positions: tuple[int, ...]
    stop_record_ids: tuple[str, ...]


def _validated_membership(
    record_ids: Sequence[object], labels: object
) -> tuple[tuple[str, ...], np.ndarray]:
    ids = tuple(str(value) for value in record_ids)
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("TabMmini pilot requires unique nonblank Train record IDs.")
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(ids):
        raise ValueError("TabMmini pilot labels must align one-to-one with Train IDs.")
    if not np.isin(y, (0, 1)).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("TabMmini Train population must contain both binary classes.")
    return ids, y.astype(np.int64, copy=False)


def make_tabmmini_pilot_split(
    train_record_ids: Sequence[object], labels: object
) -> TabMMiniPilotSplit:
    """Create the deterministic 20%-of-Train pilot and its stratified 80/20 split."""
    record_ids, y = _validated_membership(train_record_ids, labels)
    marker = np.zeros(len(record_ids), dtype=np.int8)
    try:
        _, pilot_positions = next(
            StratifiedShuffleSplit(
                n_splits=1,
                test_size=PILOT_SAMPLE_FRACTION,
                random_state=PILOT_RANDOM_STATE,
            ).split(marker, y)
        )
        pilot_positions = np.sort(np.asarray(pilot_positions, dtype=np.int64))
        pilot_y = y[pilot_positions]
        pilot_marker = np.zeros(len(pilot_positions), dtype=np.int8)
        fit_local, stop_local = next(
            StratifiedShuffleSplit(
                n_splits=1,
                test_size=PILOT_STOP_FRACTION,
                random_state=PILOT_RANDOM_STATE,
            ).split(pilot_marker, pilot_y)
        )
    except ValueError as exc:
        raise ValueError(
            "TabMmini Train membership cannot support the locked stratified pilot split."
        ) from exc
    fit_positions = np.sort(pilot_positions[np.asarray(fit_local, dtype=np.int64)])
    stop_positions = np.sort(pilot_positions[np.asarray(stop_local, dtype=np.int64)])
    if np.intersect1d(fit_positions, stop_positions).size:
        raise RuntimeError("TabMmini pilot-fit and pilot-stop positions overlap.")
    if not np.array_equal(
        np.sort(np.concatenate((fit_positions, stop_positions))), pilot_positions
    ):
        raise RuntimeError("TabMmini inner pilot split does not cover the pilot.")
    return TabMMiniPilotSplit(
        source_record_ids=record_ids,
        pilot_positions=tuple(int(value) for value in pilot_positions),
        pilot_record_ids=tuple(record_ids[position] for position in pilot_positions),
        fit_positions=tuple(int(value) for value in fit_positions),
        fit_record_ids=tuple(record_ids[position] for position in fit_positions),
        stop_positions=tuple(int(value) for value in stop_positions),
        stop_record_ids=tuple(record_ids[position] for position in stop_positions),
    )
