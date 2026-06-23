"""Run the Streamlit demo app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"


def main() -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(APP_PATH)],
            check=True,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
