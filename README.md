# PG-TL SFCC Model Release

This repository contains the public inference package for a physics-guided transfer-learning
(PG-TL) model for soil freezing characteristic curve (SFCC) parameterization. The released
artifact is intended for forward prediction and visualization of SFCC curves from user-supplied
soil descriptors.

The package provides:

- trained PG-TL model weights: `models/model.pth`
- frozen feature-scaling statistics: `models/scaler_stats.json`
- Python inference code for VG/SFCC parameters
- utilities for computing unfrozen water content curves from the GCE-VG relation
- a lightweight desktop GUI for plotting and comparing SFCC curves
- small examples using synthetic soil descriptors

This repository does not include the compiled SWCC/SFCC training databases, site-specific THC
simulation inputs, field forcing, borehole observations, model run outputs, or the coupled THC
simulator.

## Repository Contents

```text
models/
  model.pth              Trained PG-TL model weights
  scaler_stats.json      Frozen feature-scaling statistics
pgtl_sfcc/
  inference.py           Public model-loading and prediction API
  model.py               PyTorch model architecture
  physics.py             GCE-VG unfrozen-water calculation
  gui.py                 Tkinter/Matplotlib desktop GUI
examples/
  predict_vg_parameters.py
  predict_sfcc_curve.py
  run_gui.py
  sample_curves.csv      Synthetic GUI import example
tests/
  test_inference_smoke.py
```

## Installation

From the repository root:

```powershell
pip install -e .
```

The model requires Python 3.10 or later, NumPy, PyTorch, and Matplotlib.

## Quick Start

```python
import numpy as np
from pgtl_sfcc import PGTLSFCCModel

sample = {
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

model = PGTLSFCCModel()
params = model.predict_vg_parameters(sample, temperature_c=0.0)
curve = model.predict_sfcc_curve(sample, np.linspace(-20.0, 2.0, 100))
curve_iter = model.predict_sfcc_curve(
    sample,
    np.linspace(-20.0, 2.0, 100),
    use_pitzer_iteration=True,
)

print(params)
print(curve["theta_u"])
```

## Input Features

Each prediction uses a soil descriptor with the following fields:

| Name | Unit / meaning |
| --- | --- |
| `Sand` | percent |
| `Silt` | percent |
| `Clay` | percent |
| `SOM` | g/g |
| `BD` | kg/m3 |
| `Salinity` | mol/L of liquid water |
| `Porosity` | fraction |
| `Saturation` | fraction |
| `PL` | plastic limit, percent |
| `LL` | liquid limit, percent |
| `S_a` | specific surface area, m2/g |

## Command-Line Examples

Run the included examples:

```powershell
python examples/predict_vg_parameters.py
python examples/predict_sfcc_curve.py
```

## Desktop GUI

Launch the GUI from a source checkout:

```powershell
python examples/run_gui.py
```

After installation, the same interface can be launched with:

```powershell
pgtl-sfcc-gui
```

The GUI lets users:

- edit soil input parameters in text boxes
- add multiple SFCC curves to one plot
- select a curve and update only that curve's inputs
- remove selected curves or clear the plot
- import up to five curves from a CSV file
- optionally enable Pitzer full iteration with the `Pitzer full iteration` checkbox

An example CSV is provided at:

```text
examples/sample_curves.csv
```

CSV files must include the following columns:

```text
Sand,Silt,Clay,SOM,BD,Salinity,Porosity,Saturation,PL,LL,S_a
```

They may also include:

```text
label,T_min_c,T_max_c,n_points
```

The CSV importer reads at most five curves at a time. This limit keeps the GUI readable and
prevents accidental import of full datasets.

## Optional Pitzer Full Iteration

By default, the released model uses the initial NaCl-equivalent salinity to calculate Pitzer
water activity at each temperature. This reproduces the trained inference behavior and is the
fastest option.

For sensitivity checks, users can enable full Pitzer iteration:

```python
model = PGTLSFCCModel(use_pitzer_iteration=True)
curve = model.predict_sfcc_curve(sample, temperatures)

# Or enable it for one call only:
curve = model.predict_sfcc_curve(sample, temperatures, use_pitzer_iteration=True)
```

When enabled, the model solves an implicit salt-concentration problem at each temperature:

```text
theta_l = SFCC(a_w(S0 * theta0 / theta_l, T))
```

The solver uses bounded bisection with default `pitzer_iteration_steps=32`,
`theta_min=1e-4`, and a salinity cap of `5.0 mol/L`. It changes only the Pitzer
water-activity calculation during inference; it does not retrain the neural-network weights.
In the GUI, check `Pitzer full iteration` and the displayed curves will be recomputed.

### Windows Conda Note

Some Windows conda environments can report an Intel OpenMP duplicate-runtime warning when
Matplotlib and PyTorch are loaded in the same desktop session. Start from a clean, activated
environment first:

```powershell
conda activate soil_inn
python examples\run_gui.py
```

For local GUI inspection only, this workaround may be used if the environment still reports
`libiomp5md.dll` duplication:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python examples\run_gui.py
```

## Outputs

`predict_vg_parameters` returns:

- `alpha`: van Genuchten alpha parameter, cm^-1
- `n`: van Genuchten shape parameter
- `theta_r`: residual water content, m3/m3
- `theta_s`: upper-bound water content, m3/m3
- `a_w`: liquid-water activity at the requested evaluation temperature

`predict_sfcc_curve` returns a temperature array and the corresponding predicted unfrozen
water content `theta_u`.

## Scope and Data Availability

Only the trained model weights, frozen scaler statistics, inference code, GUI, tests, and
synthetic examples are released here. The underlying SWCC/SFCC databases and site-specific THC
simulation data are not included in this repository and are available from the corresponding
author upon reasonable request, subject to applicable data-use restrictions.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, inputs, outputs, and limitations.

## Testing

```powershell
python -m pytest tests -q
```

The smoke tests check model loading, physical output ranges, SFCC curve bounds, and CSV import
behavior.

## License

This inference package is released under the MIT License.
