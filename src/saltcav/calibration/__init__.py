"""Невязки с данными и байесовская калибровка (этап 8)."""

from saltcav.calibration.fit import CalibResult, calibrate
from saltcav.calibration.problem import SPECS, Problem, make_params, with_values
from saltcav.calibration.synthetic import read_truth, sansmic_dataset, synthesize, write_truth

__all__ = [
    "SPECS",
    "CalibResult",
    "Problem",
    "calibrate",
    "make_params",
    "read_truth",
    "sansmic_dataset",
    "synthesize",
    "with_values",
    "write_truth",
]
