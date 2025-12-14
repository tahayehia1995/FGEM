#!/usr/bin/env python
"""Run the FGEM Streamlit application."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    app_path = script_dir / "streamlit_app" / "app.py"
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path)
    ])

