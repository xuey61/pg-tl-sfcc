# PG-TL SFC parameterization

## Model Summary

This repository releases a trained physics-guided transfer-learning (PG-TL) model for soil
freezing characteristic (SFC) parameterization. The
model predicts physically interpretable van Genuchten-style SFCC parameters and can generate
unfrozen water content (UWC) curves over a specified temperature range.

## Released Artifacts

- `models/model.pth`: trained PyTorch model weights
- `models/scaler_stats.json`: frozen feature-scaling statistics required for inference
- `pgtl_sfcc/`: model architecture, inference API, GCE-VG physics utilities, and GUI
- `examples/sample_curves.csv`: synthetic input table for GUI testing

The raw SWCC/SFCC databases, site-specific forcing data, THC simulation inputs, and THC model
outputs are not included.

## Intended Use

The model is intended for:

- forward prediction of SFCC curves from soil property descriptors
- comparison of salinity, organic matter, texture, and saturation effects on unfrozen water
  content


## Inputs

Required input features are:

```text
Sand, Silt, Clay, SOM, BD, Salinity, Porosity, Saturation, PL, LL, S_a
```

Salinity is interpreted as liquid-phase salt concentration in mol/L. Organic matter is supplied
as `SOM` in g/g. Specific surface area is supplied as `S_a` in m2/g.

## Outputs

The model returns:

- `alpha`: van Genuchten alpha parameter, cm^-1
- `n`: van Genuchten shape parameter
- `theta_r`: residual water content, m3/m3
- `theta_s`: upper-bound water content, m3/m3
- `a_w`: liquid-water activity
- `theta_u(T)`: predicted unfrozen water content curve

## Limitations

- The model should be used within soil property ranges comparable to the training domain.
- Salinity is represented using a NaCl-equivalent liquid-phase concentration.

## Citation

If this model is used, please cite the associated manuscript. The repository citation metadata
can be updated once the manuscript DOI and final author metadata are available.
