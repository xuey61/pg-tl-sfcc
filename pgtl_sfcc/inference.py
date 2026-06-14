"""Public inference API for the PG-TL SFCC model."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .model import (
    ALL_FEATURES,
    BEST_PARAMS,
    SHARED_FEATURES,
    SPECIFIC_FEATURES,
    ModelSFCCFinetune,
    with_sfcc_scaler_config,
)
from .physics import theta_u_from_vg
from .scaler import FrozenStandardScaler, load_scaler_stats


def default_model_path() -> Path:
    """Return the default model path for a source checkout."""

    return Path(__file__).resolve().parents[1] / "models" / "model.pth"


def _as_records(samples: Mapping[str, float] | Sequence[Mapping[str, float]]) -> list[Mapping[str, float]]:
    if isinstance(samples, Mapping):
        return [samples]
    return list(samples)


def _load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class PGTLSFCCModel:
    """Loaded PG-TL model for VG parameter and SFCC curve prediction."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        use_pitzer_iteration: bool = False,
        pitzer_iteration_steps: int = 32,
        pitzer_max_salinity_mol_l: float = 5.0,
    ):
        self.device = torch.device(device)
        self.model_path = default_model_path() if model_path is None else Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Cannot find model weights at {self.model_path}. "
                "Pass model_path=... if you moved models/model.pth."
            )

        self.scaler_stats = load_scaler_stats(scaler_path)
        self.scaler = FrozenStandardScaler(self.scaler_stats)
        config = with_sfcc_scaler_config(BEST_PARAMS, self.scaler_stats)
        config["sfcc_aw_full_iteration"] = bool(use_pitzer_iteration)
        config["sfcc_aw_iter_steps"] = max(int(pitzer_iteration_steps), 1)
        config["sfcc_aw_iter_max_salinity_mol_l"] = float(pitzer_max_salinity_mol_l)
        self.model = ModelSFCCFinetune(
            shared_dim=len(SHARED_FEATURES),
            spec_dim=len(SPECIFIC_FEATURES),
            config=config,
        ).to(self.device)
        self.model.load_state_dict(_load_state_dict(self.model_path, self.device))
        self.model.salinity_mean.fill_(float(self.scaler_stats["mean"][ALL_FEATURES.index("Salinity")]))
        self.model.salinity_scale.fill_(float(self.scaler_stats["scale"][ALL_FEATURES.index("Salinity")]))
        self.model.eval()

    @contextmanager
    def _temporary_pitzer_iteration(self, use_pitzer_iteration: bool | None):
        if use_pitzer_iteration is None:
            yield
            return

        old_value = self.model.config.get("sfcc_aw_full_iteration", False)
        self.model.config["sfcc_aw_full_iteration"] = bool(use_pitzer_iteration)
        try:
            yield
        finally:
            self.model.config["sfcc_aw_full_iteration"] = old_value

    def _tensor_batch(
        self,
        samples: Mapping[str, float] | Sequence[Mapping[str, float]],
        temperature_c: float | Sequence[float] | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Mapping[str, float]]]:
        records = _as_records(samples)
        scaled = self.scaler.transform_records(records).astype(np.float32)
        x_sh_raw = scaled[:, : len(SHARED_FEATURES)]
        x_sp_raw = scaled[:, len(SHARED_FEATURES) :]
        x_sh = np.concatenate([x_sh_raw, np.ones_like(x_sh_raw)], axis=1)
        x_sp = np.concatenate([x_sp_raw, np.ones_like(x_sp_raw)], axis=1)

        porosity = np.asarray([float(row["Porosity"]) for row in records], dtype=np.float32)[:, None]
        saturation = np.asarray([float(row["Saturation"]) for row in records], dtype=np.float32)[:, None]

        temp = np.asarray(temperature_c, dtype=np.float32)
        if temp.ndim == 0:
            temp = np.full((len(records), 1), float(temp), dtype=np.float32)
        elif temp.ndim == 1:
            temp = np.broadcast_to(temp[None, :], (len(records), temp.size)).astype(np.float32)
        elif temp.shape[0] != len(records):
            raise ValueError("2D temperature arrays must have shape (n_samples, n_temperatures).")

        return (
            torch.tensor(x_sh, dtype=torch.float32, device=self.device),
            torch.tensor(x_sp, dtype=torch.float32, device=self.device),
            torch.tensor(temp, dtype=torch.float32, device=self.device),
            torch.tensor(porosity, dtype=torch.float32, device=self.device),
            torch.tensor(saturation, dtype=torch.float32, device=self.device),
            records,
        )

    def predict_vg_parameters(
        self,
        samples: Mapping[str, float] | Sequence[Mapping[str, float]],
        temperature_c: float = 0.0,
        use_pitzer_iteration: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict VG/SFCC parameters for one or more soil descriptors.

        The `a_w` value is evaluated at `temperature_c`; the VG shape parameters
        are independent of this evaluation temperature.
        """

        x_sh, x_sp, temp, porosity, saturation, _records = self._tensor_batch(samples, temperature_c)
        with self._temporary_pitzer_iteration(use_pitzer_iteration):
            with torch.no_grad():
                _theta, params = self.model(x_sh, x_sp, temp, porosity, saturation)
                alpha, n, theta_r, a_w, theta_s, *_ = params
        return {
            "alpha": alpha.detach().cpu().numpy().reshape(-1),
            "n": n.detach().cpu().numpy().reshape(-1),
            "theta_r": theta_r.detach().cpu().numpy().reshape(-1),
            "theta_s": theta_s.detach().cpu().numpy().reshape(-1),
            "a_w": a_w[:, 0].detach().cpu().numpy().reshape(-1),
        }

    def predict_theta_u(
        self,
        samples: Mapping[str, float] | Sequence[Mapping[str, float]],
        temperature_c: Sequence[float] | np.ndarray,
        use_pitzer_iteration: bool | None = None,
    ) -> np.ndarray:
        """Predict unfrozen water content curves for one or more samples."""

        x_sh, x_sp, temp, porosity, saturation, _records = self._tensor_batch(samples, temperature_c)
        with self._temporary_pitzer_iteration(use_pitzer_iteration):
            with torch.no_grad():
                theta, _params = self.model(x_sh, x_sp, temp, porosity, saturation)
        return theta.detach().cpu().numpy()

    def predict_sfcc_curve(
        self,
        sample: Mapping[str, float],
        temperature_c: Sequence[float] | np.ndarray,
        use_pitzer_iteration: bool | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Predict a single SFCC curve and its VG parameters."""

        temps = np.asarray(temperature_c, dtype=np.float64)
        vg = self.predict_vg_parameters(
            sample,
            temperature_c=float(np.max(np.minimum(temps, 0.0))),
            use_pitzer_iteration=use_pitzer_iteration,
        )
        theta = self.predict_theta_u(sample, temps, use_pitzer_iteration=use_pitzer_iteration)[0]
        return {
            "temperature_c": temps,
            "theta_u": theta,
            "alpha": float(vg["alpha"][0]),
            "n": float(vg["n"][0]),
            "theta_r": float(vg["theta_r"][0]),
            "theta_s": float(vg["theta_s"][0]),
            "a_w": float(vg["a_w"][0]),
        }


def predict_vg_parameters(
    samples: Mapping[str, float] | Sequence[Mapping[str, float]],
    temperature_c: float = 0.0,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    use_pitzer_iteration: bool = False,
) -> dict[str, np.ndarray]:
    """Convenience wrapper for one-off VG parameter inference."""

    model = PGTLSFCCModel(
        model_path=model_path,
        scaler_path=scaler_path,
        use_pitzer_iteration=use_pitzer_iteration,
    )
    return model.predict_vg_parameters(samples, temperature_c=temperature_c)


def predict_sfcc_curve(
    sample: Mapping[str, float],
    temperature_c: Sequence[float] | np.ndarray,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    use_pitzer_iteration: bool = False,
) -> dict[str, np.ndarray | float]:
    """Convenience wrapper for one-off SFCC curve inference."""

    model = PGTLSFCCModel(
        model_path=model_path,
        scaler_path=scaler_path,
        use_pitzer_iteration=use_pitzer_iteration,
    )
    return model.predict_sfcc_curve(sample, temperature_c=temperature_c)
