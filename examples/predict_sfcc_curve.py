"""Predict and save a synthetic SFCC curve."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pgtl_sfcc import PGTLSFCCModel


SAMPLE = {
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
}


def main() -> None:
    model = PGTLSFCCModel()
    temperatures_c = np.linspace(-20.0, 2.0, 100)
    result = model.predict_sfcc_curve(SAMPLE, temperatures_c)

    with open("sfcc_curve_example.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["temperature_c", "theta_u"])
        for temperature, theta_u in zip(result["temperature_c"], result["theta_u"]):
            writer.writerow([f"{temperature:.6f}", f"{theta_u:.8f}"])

    print("Saved sfcc_curve_example.csv")
    print(
        "Parameters: "
        f"alpha={result['alpha']:.6g}, "
        f"n={result['n']:.4f}, "
        f"theta_r={result['theta_r']:.4f}, "
        f"theta_s={result['theta_s']:.4f}, "
        f"a_w={result['a_w']:.5f}"
    )


if __name__ == "__main__":
    main()
