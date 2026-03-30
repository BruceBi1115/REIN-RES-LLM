from __future__ import annotations


def _batch_time_seq_for_sample(batch_field, sample_idx: int) -> list[str]:
    out = []
    if not isinstance(batch_field, (list, tuple)):
        return out
    for step in batch_field:
        if isinstance(step, (list, tuple)):
            if sample_idx < len(step):
                out.append(str(step[sample_idx]))
        else:
            out.append(str(step))
    return out
