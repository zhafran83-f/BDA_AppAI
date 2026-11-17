"""Validate model probability outputs before exposing service results."""
import numpy as np


def probability_row(output: object, count: int) -> np.ndarray:
    values = np.asarray(output)
    if (
        values.shape != (1, count)
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.isclose(values.sum(), 1.0, rtol=1e-5, atol=1e-6)
    ):
        raise RuntimeError("Model returned an invalid probability distribution.")
    return values[0]
