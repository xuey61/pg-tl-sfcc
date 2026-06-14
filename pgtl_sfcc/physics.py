"""Physical post-processing utilities for PG-TL SFCC predictions."""

from __future__ import annotations

import numpy as np


L_F = 3.34e5
R_GAS = 8.314
M_W = 0.018015
G = 9.81
RHO_W = 1000.0
T0_K = 273.15
SOFTPLUS_BETA = 0.9615321414563509


def _softplus(x: np.ndarray, beta: float = SOFTPLUS_BETA) -> np.ndarray:
    return np.logaddexp(0.0, beta * x) / beta


def theta_u_from_vg(
    temperature_c: np.ndarray | list[float],
    alpha: float,
    n: float,
    theta_r: float,
    theta_s: float,
    a_w: float,
) -> np.ndarray:
    """Compute unfrozen water content from GCE-VG parameters.

    Parameters
    ----------
    temperature_c:
        Temperature in degrees C.
    alpha:
        van Genuchten alpha parameter in cm^-1.
    n:
        van Genuchten n parameter. Must be greater than 1.
    theta_r:
        Residual water content in m3/m3.
    theta_s:
        Saturated or total water content upper bound in m3/m3.
    a_w:
        Water activity of the liquid phase.
    """

    if n <= 1.0:
        raise ValueError(f"VG parameter n must be > 1, got {n}.")

    temperature_c = np.asarray(temperature_c, dtype=np.float64)
    temperature_k = temperature_c + T0_K
    v_m = M_W / RHO_W
    psi_o_pa = -(R_GAS * temperature_k / v_m) * np.log(np.clip(a_w, 1e-6, 0.9999))
    delta_t = np.maximum(T0_K - temperature_k, 0.0)
    psi_cryo_pa = (L_F * RHO_W / T0_K) * delta_t
    psi_m_pa = _softplus(psi_cryo_pa - psi_o_pa)
    h_cm = (psi_m_pa / (RHO_W * G)) * 100.0

    m = 1.0 - 1.0 / n
    h_c = 2.0
    theta_m = theta_r + (theta_s - theta_r) * (1.0 + (alpha * h_c) ** n) ** (-m)
    ah = np.clip(alpha * h_cm, 0.0, 10000.0)
    theta_u = theta_r + (theta_m - theta_r) / (1.0 + ah**n) ** m
    theta_u = np.where(temperature_c >= 0.0, theta_s, np.clip(theta_u, theta_r, theta_s))
    return theta_u
