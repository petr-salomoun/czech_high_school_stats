from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    cohort_lag_years: int = 8
    min_cohort_size: int = 10
    min_school_count: int = 10
    ridge_alpha: float = 1.0
    bootstrap_iterations: int = 500
    random_seed: int = 42
