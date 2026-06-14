"""PyTorch runtime for the PG-TL SFCC model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import ALL_FEATURES, SHARED_FEATURES, SPECIFIC_FEATURES

SFCC_PITZER_MODEL = "tempdep"

SALINITY_IDX = SHARED_FEATURES.index("Salinity")
ALL_SALINITY_IDX = ALL_FEATURES.index("Salinity")


BEST_PARAMS = {
    "swcc_hidden_dim": 512,
    "swcc_lr": 0.00028097636000388245,
    "swcc_dropout": 0.14771105377714538,
    "swcc_log_loss": 0.12622446207263935,
    "swcc_alpha_max": 0.15636709730138634,
    "swcc_alpha_min": 2.133731336372521e-06,
    "swcc_n_max": 10.634987456671611,
    "swcc_n_min": 1.0850182788370286,
    "swcc_ts_min_factor": 0.8099494679088017,
    "swcc_ts_range": 0.7304910240383485,
    "swcc_tr_max_frac": 0.6005580571037706,
    "swcc_max_ah": 14998.129494539007,
    "sfcc_adapter_dim": 512,
    "sfcc_adapter_layers": 3,
    "sfcc_batch_size": 16,
    "sfcc_lr_backbone": 4.981850640003099e-05,
    "sfcc_lr_adapter": 0.000266160423451742,
    "sfcc_lr_heads": 0.0001665338950669495,
    "sfcc_dropout": 0.00795891589270567,
    "sfcc_log_loss": 0.26058525351245404,
    "sfcc_tr_min": 1.0203790054275257e-05,
    "sfcc_max_d_log_alpha": 4.745614989006689,
    "sfcc_softplus_beta": 0.9615321414563509,
    "sfcc_aw_mode": "pitzer",
    "sfcc_pitzer_model": "tempdep",
    "sfcc_aw_full_iteration": False,
    "sfcc_aw_iter_steps": 32,
    "sfcc_aw_iter_theta_min": 1.0e-4,
    "sfcc_aw_iter_max_salinity_mol_l": 5.0,
}


def with_sfcc_scaler_config(config: dict, scaler_stats: dict) -> dict:
    """Attach raw-salinity inverse-scaling constants used by Pitzer a_w."""

    cfg = dict(config)
    salinity_mean = float(scaler_stats["mean"][ALL_SALINITY_IDX])
    salinity_scale = float(scaler_stats["scale"][ALL_SALINITY_IDX])
    if abs(salinity_scale) < 1e-12:
        salinity_scale = 1.0
    cfg["_sfcc_salinity_mean"] = salinity_mean
    cfg["_sfcc_salinity_scale"] = salinity_scale
    cfg["sfcc_salinity_mean"] = salinity_mean
    cfg["sfcc_salinity_scale"] = salinity_scale
    return cfg


class PitzerWaterActivity:
    """Differentiable NaCl-H2O Pitzer water activity used by the SFCC layer."""

    R = 8.314
    M_w = 0.018015
    M_NaCl = 0.05844
    rho_w_kg_l = 0.9998
    nu = 2.0
    b = 1.2
    alpha1 = 2.0
    t_ref_k = 298.15
    A_Debye = 1.175930

    # HMW/Cantera NaCl-H2O Pitzer coefficients. The HMW input reports A_Debye;
    # the osmotic-coefficient form below uses A_phi = A_Debye / 3.
    pitzer_25c = {
        "beta0": 0.0765,
        "beta1": 0.2664,
        "Cphi": 0.00127,
        "Aphi": A_Debye / 3.0,
    }
    can_tdep = {
        "beta0": [0.0765, 0.008946, -3.3158e-6, -777.03, -4.4706],
        "beta1": [0.2664, 6.1608e-5, 1.0715e-6, 0.0, 0.0],
        "Cphi": [0.00127, -4.655e-5, 0.0, 33.317, 0.09421],
    }

    @classmethod
    def molarity_to_molality(cls, s0_mol_l: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(cls.rho_w_kg_l - s0_mol_l * cls.M_NaCl, min=1e-8)
        return s0_mol_l / denominator

    @classmethod
    def _constant_like(cls, t_k: torch.Tensor, value: float) -> torch.Tensor:
        return torch.full_like(t_k, float(value))

    @classmethod
    def _complex_temp(cls, coeffs: list[float], t_k: torch.Tensor) -> torch.Tensor:
        q0, q1, q2, q3, q4 = [
            torch.as_tensor(v, dtype=t_k.dtype, device=t_k.device) for v in coeffs
        ]
        t_ref = torch.as_tensor(cls.t_ref_k, dtype=t_k.dtype, device=t_k.device)
        return (
            q0
            + q1 * (t_k - t_ref)
            + q2 * (t_k**2 - t_ref**2)
            + q3 * (1.0 / t_k - 1.0 / t_ref)
            + q4 * torch.log(t_k / t_ref)
        )

    @classmethod
    def parameters(cls, t_k: torch.Tensor, model: str = "tempdep") -> dict[str, torch.Tensor]:
        if model == "25c":
            return {key: cls._constant_like(t_k, value) for key, value in cls.pitzer_25c.items()}
        if model != "tempdep":
            raise ValueError(f"Unknown Pitzer model: {model}")

        return {
            "beta0": cls._complex_temp(cls.can_tdep["beta0"], t_k),
            "beta1": cls._complex_temp(cls.can_tdep["beta1"], t_k),
            "Cphi": cls._complex_temp(cls.can_tdep["Cphi"], t_k),
            "Aphi": cls._constant_like(t_k, cls.pitzer_25c["Aphi"]),
        }

    @classmethod
    def compute(cls, s0_mol_l: torch.Tensor, t_k: torch.Tensor, model: str = "tempdep") -> torch.Tensor:
        """Return a_w for salinity in mol/L and temperature in K."""

        s0 = torch.clamp(s0_mol_l.to(dtype=t_k.dtype, device=t_k.device), min=0.0, max=5.0)
        molality = cls.molarity_to_molality(s0).expand_as(t_k)

        params = cls.parameters(t_k, model=model)
        sqrt_i = torch.sqrt(torch.clamp(molality, min=0.0))
        f_phi = -params["Aphi"] * sqrt_i / (1.0 + cls.b * sqrt_i)
        b_phi = params["beta0"] + params["beta1"] * torch.exp(-cls.alpha1 * sqrt_i)
        phi = 1.0 + f_phi + molality * b_phi + molality**2 * params["Cphi"]

        ln_aw = -cls.nu * molality * cls.M_w * phi
        return torch.exp(torch.clamp(ln_aw, min=-5.0, max=0.0))


class UnifiedSWCCBackbone(nn.Module):
    """Pre-trained SWCC backbone used inside the transferred SFCC model."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )
        self.head_alpha = nn.Linear(hidden_dim + 2, 1)
        self.head_n = nn.Linear(hidden_dim + 3, 1)
        self.head_theta_r = nn.Linear(hidden_dim + 3, 1)
        self.head_theta_s_factor = nn.Linear(hidden_dim + 2, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.head_alpha.weight, 0.01)
        nn.init.constant_(self.head_alpha.bias, 0.0)
        nn.init.constant_(self.head_theta_s_factor.bias, 0.0)
        nn.init.constant_(self.head_theta_r.bias, -2.0)

    def forward(
        self,
        x_shared: torch.Tensor,
        porosity: torch.Tensor,
        saturation: torch.Tensor,
        config: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x_shared)
        feat_silt = x_shared[:, 1:2]
        feat_clay = x_shared[:, 2:3]
        feat_som = x_shared[:, 3:4]
        feat_poro = x_shared[:, 6:7]
        feat_sat = x_shared[:, 7:8]

        raw_alpha = self.head_alpha(torch.cat([features, feat_silt, feat_clay], dim=1))
        raw_n = self.head_n(torch.cat([features, feat_silt, feat_clay, feat_som], dim=1))
        raw_tr = self.head_theta_r(torch.cat([features, feat_silt, feat_clay, feat_som], dim=1))
        raw_ts = self.head_theta_s_factor(torch.cat([features, feat_poro, feat_sat], dim=1))

        theta_total = porosity * saturation
        alpha_max = config.get("swcc_alpha_max", 0.5)
        alpha_min = config.get("swcc_alpha_min", 0.0001)
        n_max = config.get("swcc_n_max", 9.05)
        n_min = config.get("swcc_n_min", 1.05)
        ts_min_factor = config.get("swcc_ts_min_factor", 0.7)
        ts_range = config.get("swcc_ts_range", 0.6)
        tr_max_frac = config.get("swcc_tr_max_frac", 0.5)

        alpha_base = alpha_min + (alpha_max - alpha_min) * torch.sigmoid(raw_alpha)
        n_base = n_min + (n_max - n_min) * torch.sigmoid(raw_n)
        ts_factor = ts_min_factor + ts_range * torch.sigmoid(raw_ts)
        ts_base = torch.clamp(torch.maximum(ts_factor * porosity, theta_total + 1e-4), max=0.99)
        tr_base = torch.sigmoid(raw_tr) * (theta_total * tr_max_frac)

        return alpha_base, n_base, tr_base, ts_base


class ModelSFCCFinetune(nn.Module):
    """Transferred SFCC model returning theta_u and physically interpretable parameters."""

    def __init__(self, shared_dim: int, spec_dim: int, config: dict):
        super().__init__()
        self.config = config
        self.swcc = UnifiedSWCCBackbone(shared_dim, config["swcc_hidden_dim"], config["sfcc_dropout"])

        self.register_buffer("L_f", torch.tensor(3.34e5))
        self.register_buffer("R", torch.tensor(8.314))
        self.register_buffer("M_w", torch.tensor(0.018015))
        self.register_buffer("g", torch.tensor(9.81))
        self.register_buffer("rho_w", torch.tensor(1000.0))
        self.register_buffer("T0_K", torch.tensor(273.15))

        input_dim = (shared_dim + spec_dim) * 2
        h_dim = config["sfcc_adapter_dim"]
        dropout_val = config["sfcc_dropout"]

        layers: list[nn.Module] = [
            nn.Linear(input_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_val),
        ]
        for _ in range(config["sfcc_adapter_layers"]):
            layers.extend(
                [
                    nn.Linear(h_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout_val),
                ]
            )
        layers.extend([nn.Linear(h_dim, h_dim // 2), nn.LeakyReLU(0.1), nn.Dropout(dropout_val)])
        self.adapter = nn.Sequential(*layers)

        self.head_d_alpha = nn.Linear(h_dim // 2, 1)
        self.head_d_n = nn.Linear(h_dim // 2, 1)
        self.head_d_tr = nn.Linear(h_dim // 2, 1)
        self.head_d_ts = nn.Linear(h_dim // 2, 1)
        self.head_aw = nn.Identity()
        self.pitzer_model = config.get("sfcc_pitzer_model", SFCC_PITZER_MODEL)
        self.register_buffer(
            "salinity_mean",
            torch.tensor(config.get("sfcc_salinity_mean", config.get("_sfcc_salinity_mean", 0.0)), dtype=torch.float32),
        )
        self.register_buffer(
            "salinity_scale",
            torch.tensor(config.get("sfcc_salinity_scale", config.get("_sfcc_salinity_scale", 1.0)), dtype=torch.float32),
        )

        for head in [self.head_d_alpha, self.head_d_n, self.head_d_tr, self.head_d_ts]:
            nn.init.constant_(head.weight, 0.0)
            nn.init.constant_(head.bias, 0.0)

    def _raw_salinity_from_input(self, x_sh: torch.Tensor) -> torch.Tensor:
        salinity_norm = x_sh[:, SALINITY_IDX : SALINITY_IDX + 1]
        salinity = salinity_norm * self.salinity_scale + self.salinity_mean
        return torch.clamp(salinity, min=0.0, max=5.0)

    def _theta_from_aw(
        self,
        a_w: torch.Tensor,
        t_k: torch.Tensor,
        alpha: torch.Tensor,
        n: torch.Tensor,
        tr: torch.Tensor,
        ts: torch.Tensor,
        theta_total: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        v_m = self.M_w / self.rho_w
        psi_o_pa = -(self.R * t_k / v_m) * torch.log(torch.clamp(a_w, min=1e-12, max=0.999999))
        delta_t = torch.clamp(self.T0_K - t_k, min=0.0)
        psi_cryo_pa = (self.L_f * self.rho_w / self.T0_K) * delta_t
        psi_m_pa = F.softplus(psi_cryo_pa - psi_o_pa, beta=self.config["sfcc_softplus_beta"])
        h_cm = (psi_m_pa / (self.rho_w * self.g)) * 100.0

        h_c = 2.0
        denom_c = torch.pow(1.0 + torch.pow(alpha * h_c, n), m)
        theta_m = tr + (ts - tr) * denom_c

        product_ah = torch.clamp(alpha * h_cm, max=10000.0)
        denom = torch.pow(1.0 + torch.pow(product_ah, n), m)
        theta_calc = tr + (theta_m - tr) / denom
        return torch.clamp(torch.minimum(theta_calc, theta_total), min=tr)

    def _salinity_from_theta(
        self,
        s0_curve: torch.Tensor,
        theta0: torch.Tensor,
        theta_l: torch.Tensor,
    ) -> torch.Tensor:
        theta_min = float(self.config.get("sfcc_aw_iter_theta_min", 1.0e-4))
        max_salinity = float(self.config.get("sfcc_aw_iter_max_salinity_mol_l", 5.0))

        salinity = s0_curve * theta0 / torch.clamp(theta_l, min=theta_min)
        salinity = torch.maximum(salinity, s0_curve)
        salinity = torch.nan_to_num(salinity, nan=max_salinity, posinf=max_salinity, neginf=0.0)
        return torch.clamp(salinity, min=0.0, max=max_salinity)

    def _theta_from_concentrated_theta(
        self,
        theta_l: torch.Tensor,
        s0_curve: torch.Tensor,
        theta0: torch.Tensor,
        t_k: torch.Tensor,
        alpha: torch.Tensor,
        n: torch.Tensor,
        tr: torch.Tensor,
        ts: torch.Tensor,
        theta_total: torch.Tensor,
        m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        salinity = self._salinity_from_theta(s0_curve, theta0, theta_l)
        a_w = PitzerWaterActivity.compute(salinity, t_k, model=self.pitzer_model)
        theta_next = self._theta_from_aw(a_w, t_k, alpha, n, tr, ts, theta_total, m)
        return theta_next, salinity, a_w

    def _implicit_aw_bisection(
        self,
        s0_mol_l: torch.Tensor,
        t_k: torch.Tensor,
        alpha: torch.Tensor,
        n: torch.Tensor,
        tr: torch.Tensor,
        ts: torch.Tensor,
        theta_total: torch.Tensor,
        m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        iter_steps = max(int(self.config.get("sfcc_aw_iter_steps", 32)), 1)
        s0_curve = s0_mol_l.expand_as(t_k)
        theta0 = theta_total.expand_as(t_k)

        lo = torch.clamp(
            torch.minimum(tr, theta_total),
            min=float(self.config.get("sfcc_aw_iter_theta_min", 1.0e-4)),
        )
        hi = torch.maximum(theta_total, lo + 1e-6)

        def residual(theta_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            theta_next, _salinity, a_w = self._theta_from_concentrated_theta(
                theta_l, s0_curve, theta0, t_k, alpha, n, tr, ts, theta_total, m
            )
            return theta_l - theta_next, theta_next, a_w

        g_lo, _theta_lo, _aw_lo = residual(lo)
        g_hi, _theta_hi, _aw_hi = residual(hi)
        bracketed = (g_lo * g_hi) <= 0.0

        for _ in range(iter_steps):
            mid = 0.5 * (lo + hi)
            g_mid, _theta_mid, _aw_mid = residual(mid)
            left_brackets_root = (g_lo * g_mid) <= 0.0
            update_hi = bracketed & left_brackets_root
            update_lo = bracketed & (~left_brackets_root)
            hi = torch.where(update_hi, mid, hi)
            g_hi = torch.where(update_hi, g_mid, g_hi)
            lo = torch.where(update_lo, mid, lo)
            g_lo = torch.where(update_lo, g_mid, g_lo)

        theta_root = 0.5 * (lo + hi)
        _g_root, _theta_root_next, a_w_root = residual(theta_root)
        g_lo_final, theta_lo, a_w_lo = residual(lo)
        g_hi_final, theta_hi, a_w_hi = residual(hi)

        lo_is_better = torch.abs(g_lo_final) <= torch.abs(g_hi_final)
        theta_endpoint = torch.where(lo_is_better, theta_lo, theta_hi)
        a_w_endpoint = torch.where(lo_is_better, a_w_lo, a_w_hi)
        theta_solution = torch.where(bracketed, theta_root, theta_endpoint)
        a_w_solution = torch.where(bracketed, a_w_root, a_w_endpoint)
        return a_w_solution, theta_solution

    def forward(
        self,
        x_sh: torch.Tensor,
        x_sp: torch.Tensor,
        temperature_c: torch.Tensor,
        porosity: torch.Tensor,
        saturation: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        alpha_b, n_b, tr_b, ts_b = self.swcc(x_sh, porosity, saturation, self.config)
        features = self.adapter(torch.cat([x_sh, x_sp], dim=1))

        d_alpha = torch.clamp(
            self.head_d_alpha(features),
            min=-self.config["sfcc_max_d_log_alpha"],
            max=self.config["sfcc_max_d_log_alpha"],
        )
        d_n = self.head_d_n(features)
        d_tr = self.head_d_tr(features)
        d_ts = self.head_d_ts(features)

        theta_total = porosity * saturation
        alpha_sfcc = alpha_b * torch.exp(d_alpha)
        n_min = self.config.get("swcc_n_min", 1.05)
        n_sfcc = n_min + (n_b - n_min) * torch.exp(d_n)
        tr_safe = torch.clamp(tr_b, min=self.config["sfcc_tr_min"])
        tr_sfcc = torch.minimum(tr_safe * torch.exp(d_tr), theta_total * 0.5)
        ts_sfcc = torch.minimum(torch.maximum(ts_b + d_ts, tr_sfcc + 1e-4), theta_total)

        seq_len = temperature_c.shape[1]
        a = alpha_sfcc.expand(-1, seq_len)
        n = n_sfcc.expand(-1, seq_len)
        tr = tr_sfcc.expand(-1, seq_len)
        ts = ts_sfcc.expand(-1, seq_len)
        tt = theta_total.expand(-1, seq_len)
        m = 1.0 - (1.0 / n)

        t_k = temperature_c + self.T0_K
        s0_mol_l = self._raw_salinity_from_input(x_sh)
        if bool(self.config.get("sfcc_aw_full_iteration", False)):
            a_w, theta_pred = self._implicit_aw_bisection(s0_mol_l, t_k, a, n, tr, ts, tt, m)
        else:
            a_w = PitzerWaterActivity.compute(s0_mol_l, t_k, model=self.pitzer_model)
            theta_pred = self._theta_from_aw(a_w, t_k, a, n, tr, ts, tt, m)
        return theta_pred, (alpha_sfcc, n_sfcc, tr_sfcc, a_w, ts_sfcc, d_alpha, d_n, d_tr, d_ts)
