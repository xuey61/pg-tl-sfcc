"""PG-TL SFCC inference package."""

from .features import ALL_FEATURES, SHARED_FEATURES, SPECIFIC_FEATURES
from .physics import theta_u_from_vg

__all__ = [
    "ALL_FEATURES",
    "SHARED_FEATURES",
    "SPECIFIC_FEATURES",
    "PGTLSFCCModel",
    "predict_sfcc_curve",
    "predict_vg_parameters",
    "theta_u_from_vg",
]


def __getattr__(name: str):
    """Load PyTorch-backed inference objects only when requested."""

    if name in {"PGTLSFCCModel", "predict_sfcc_curve", "predict_vg_parameters"}:
        from .inference import PGTLSFCCModel, predict_sfcc_curve, predict_vg_parameters

        values = {
            "PGTLSFCCModel": PGTLSFCCModel,
            "predict_sfcc_curve": predict_sfcc_curve,
            "predict_vg_parameters": predict_vg_parameters,
        }
        return values[name]
    raise AttributeError(f"module 'pgtl_sfcc' has no attribute {name!r}")
