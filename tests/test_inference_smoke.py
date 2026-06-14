from __future__ import annotations

import numpy as np
from pathlib import Path

from pgtl_sfcc import PGTLSFCCModel, theta_u_from_vg
from pgtl_sfcc import gui
from pgtl_sfcc.gui import MAX_IMPORT_CURVES, curve_spec_from_mapping, load_curve_specs_csv


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


def test_predict_vg_parameters_have_physical_ranges():
    model = PGTLSFCCModel()
    params = model.predict_vg_parameters(SAMPLE)

    assert params["alpha"].shape == (1,)
    assert params["alpha"][0] > 0.0
    assert params["n"][0] > 1.0
    assert 0.0 <= params["theta_r"][0] < params["theta_s"][0] <= SAMPLE["Porosity"] * SAMPLE["Saturation"]
    assert 0.0 < params["a_w"][0] <= 1.0


def test_predict_sfcc_curve_is_bounded_and_thawed_value_matches_theta_s():
    model = PGTLSFCCModel()
    temperatures = np.array([-10.0, -5.0, -1.0, 2.0])
    curve = model.predict_sfcc_curve(SAMPLE, temperatures)
    theta_u = curve["theta_u"]

    assert theta_u.shape == temperatures.shape
    assert np.all(theta_u >= curve["theta_r"] - 1e-6)
    assert np.all(theta_u <= curve["theta_s"] + 1e-6)
    assert abs(theta_u[-1] - curve["theta_s"]) < 1e-6


def test_pitzer_full_iteration_can_be_enabled_for_prediction():
    model = PGTLSFCCModel()
    saline = dict(SAMPLE)
    saline["Salinity"] = 1.0
    temperatures = np.linspace(-20.0, 2.0, 30)

    base = model.predict_sfcc_curve(saline, temperatures, use_pitzer_iteration=False)
    iterative = model.predict_sfcc_curve(saline, temperatures, use_pitzer_iteration=True)

    assert iterative["theta_u"].shape == base["theta_u"].shape
    assert np.all(np.isfinite(iterative["theta_u"]))
    assert np.all(iterative["theta_u"] >= iterative["theta_r"] - 1e-6)
    assert np.all(iterative["theta_u"] <= iterative["theta_s"] + 1e-6)
    assert not np.allclose(iterative["theta_u"], base["theta_u"])
    assert iterative["a_w"] <= base["a_w"] + 1e-8


def test_model_constructor_sets_default_pitzer_iteration_config():
    model = PGTLSFCCModel(use_pitzer_iteration=True, pitzer_iteration_steps=12)

    assert model.model.config["sfcc_aw_full_iteration"] is True
    assert model.model.config["sfcc_aw_iter_steps"] == 12


def test_gui_exposes_pitzer_full_iteration_checkbox():
    source = Path(gui.__file__).read_text(encoding="utf-8")

    assert "Pitzer full iteration" in source
    assert "use_pitzer_iteration" in source


def test_numpy_theta_u_from_vg_matches_bounds():
    model = PGTLSFCCModel()
    params = model.predict_vg_parameters(SAMPLE)
    theta = theta_u_from_vg(
        np.array([-20.0, -1.0, 2.0]),
        float(params["alpha"][0]),
        float(params["n"][0]),
        float(params["theta_r"][0]),
        float(params["theta_s"][0]),
        float(params["a_w"][0]),
    )

    assert theta[0] <= theta[1] <= theta[2]
    assert abs(theta[2] - params["theta_s"][0]) < 1e-6


def test_curve_spec_from_mapping_accepts_optional_plot_fields():
    row = {
        "label": "synthetic organic",
        **{name: str(value) for name, value in SAMPLE.items()},
        "T_min_c": "-15",
        "T_max_c": "1",
        "n_points": "25",
    }

    spec = curve_spec_from_mapping(row)

    assert spec.label == "synthetic organic"
    assert spec.sample["Salinity"] == SAMPLE["Salinity"]
    assert spec.t_min_c == -15.0
    assert spec.t_max_c == 1.0
    assert spec.n_points == 25
    assert spec.temperatures().shape == (25,)


def test_load_curve_specs_csv_limits_import_to_five_rows(tmp_path):
    csv_path = tmp_path / "curves.csv"
    header = ["label", *SAMPLE.keys(), "T_min_c", "T_max_c", "n_points"]
    lines = [",".join(header)]
    for idx in range(MAX_IMPORT_CURVES + 2):
        row = [f"curve {idx}", *[str(SAMPLE[name]) for name in SAMPLE], "-20", "2", "20"]
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    specs = load_curve_specs_csv(csv_path)

    assert len(specs) == MAX_IMPORT_CURVES
    assert specs[0].label == "curve 0"
    assert specs[-1].label == f"curve {MAX_IMPORT_CURVES - 1}"
