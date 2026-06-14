"""Shared feature names used by the PG-TL SFCC model."""

SHARED_FEATURES = [
    "Sand",
    "Silt",
    "Clay",
    "SOM",
    "BD",
    "Salinity",
    "Porosity",
    "Saturation",
]
SPECIFIC_FEATURES = ["PL", "LL", "S_a"]
ALL_FEATURES = SHARED_FEATURES + SPECIFIC_FEATURES
