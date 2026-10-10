"""Root launcher for the AI-Based Autonomous Navigation Streamlit Web Application.

Allows running either:
    streamlit run app.py
or
    python -m streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.app import main

if __name__ == "__main__":
    main()
