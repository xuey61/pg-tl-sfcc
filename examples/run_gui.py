"""Launch the PG-TL SFCC desktop GUI from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pgtl_sfcc.gui import main


if __name__ == "__main__":
    main()
