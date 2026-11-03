"""Backward-compatible GRIB function imports without import-time execution."""

from Data_processing.grib import convert_grib, split_years

__all__ = ["convert_grib", "split_years"]
