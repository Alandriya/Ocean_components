"""Chronological, target-disjoint splits for daily ocean forecasts.

All boundaries are *exclusive* end indices. Validation/test may use earlier
observations as context, but all target indices belong to their own split.
"""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ChronologicalSplit:
    train_end: int
    val_start: int
    test_start: int
    data_end: int

    def dataset_bounds(self, split, in_len, stride=1):
        if split == 'train':
            return 0, self.train_end
        if split == 'val':
            return self.val_start - in_len * stride, self.test_start
        if split == 'test':
            return self.test_start - in_len * stride, self.data_end
        raise KeyError(split)


def resolve_splits(reader_length, base_date, last_date, val_steps, test_steps,
                   in_len, out_len, stride=1):
    first = date.fromisoformat(str(base_date))
    last = date.fromisoformat(str(last_date))
    data_end = (last - first).days + 1
    if data_end <= 0:
        raise ValueError(f'End date {last_date} is before base date {base_date}.')
    if data_end > reader_length:
        raise ValueError(
            f'Data contain only {reader_length} frames starting {base_date}, '
            f'but requested end date {last_date} needs {data_end}. '
            'Correct the dates or supply a complete time series.'
        )
    val_steps, test_steps = int(val_steps), int(test_steps)
    if val_steps <= 0 or test_steps <= 0:
        raise ValueError('Validation and test intervals must be positive.')
    test_start = data_end - test_steps
    val_start = test_start - val_steps
    needed = (int(in_len) + int(out_len) - 1) * int(stride) + 1
    if val_start < needed or val_steps < (out_len - 1) * stride + 1 or test_steps < (out_len - 1) * stride + 1:
        raise ValueError('Insufficient data for the requested chronological splits and horizons.')
    split = ChronologicalSplit(val_start, val_start, test_start, data_end)
    if any(split.dataset_bounds(k, in_len, stride)[0] < 0 for k in ('val', 'test')):
        raise ValueError('Not enough historical context for validation/test.')
    return split
