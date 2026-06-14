from __future__ import annotations

import pytest
import torch

from pgtl_sfcc.model import PitzerWaterActivity


def test_pitzer_coefficients_match_hmw_nacl_reference_form():
    assert PitzerWaterActivity.A_Debye == pytest.approx(1.175930)
    assert PitzerWaterActivity.pitzer_25c["Aphi"] == pytest.approx(1.175930 / 3.0)

    assert PitzerWaterActivity.can_tdep["beta0"] == pytest.approx(
        [0.0765, 0.008946, -3.3158e-6, -777.03, -4.4706]
    )
    assert PitzerWaterActivity.can_tdep["beta1"] == pytest.approx(
        [0.2664, 6.1608e-5, 1.0715e-6, 0.0, 0.0]
    )
    assert PitzerWaterActivity.can_tdep["Cphi"] == pytest.approx(
        [0.00127, -4.655e-5, 0.0, 33.317, 0.09421]
    )

    params = PitzerWaterActivity.parameters(torch.tensor([298.15]), model="tempdep")
    assert float(params["beta0"][0]) == pytest.approx(0.0765)
    assert float(params["beta1"][0]) == pytest.approx(0.2664)
    assert float(params["Cphi"][0]) == pytest.approx(0.00127)
    assert float(params["Aphi"][0]) == pytest.approx(1.175930 / 3.0)
