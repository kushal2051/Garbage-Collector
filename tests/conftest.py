"""The stage modules sit at the repo root, not in an installed package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
