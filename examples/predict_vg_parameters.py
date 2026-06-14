"""Predict VG/SFCC parameters for synthetic soil descriptors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pgtl_sfcc import PGTLSFCCModel


SAMPLES = [
    {
        "Sand": 10.0,
        "Silt": 60.0,
        "Clay": 30.0,
        "SOM": 0.12,
        "BD": 800.0,
        "Salinity": 0.05,
        "Porosity": 0.65,
        "Saturation": 0.95,
        "PL": 28.0,
        "LL": 52.0,
        "S_a": 40.0,
    },
    {
        "Sand": 20.0,
        "Silt": 65.0,
        "Clay": 15.0,
        "SOM": 0.03,
        "BD": 1300.0,
        "Salinity": 0.10,
        "Porosity": 0.50,
        "Saturation": 0.90,
        "PL": 18.0,
        "LL": 34.0,
        "S_a": 20.0,
    },
]


def main() -> None:
    model = PGTLSFCCModel()
    params = model.predict_vg_parameters(SAMPLES, temperature_c=0.0)
    for idx in range(len(SAMPLES)):
        print(
            f"sample {idx}: "
            f"alpha={params['alpha'][idx]:.6g}, "
            f"n={params['n'][idx]:.4f}, "
            f"theta_r={params['theta_r'][idx]:.4f}, "
            f"theta_s={params['theta_s'][idx]:.4f}, "
            f"a_w={params['a_w'][idx]:.5f}"
        )


if __name__ == "__main__":
    main()
