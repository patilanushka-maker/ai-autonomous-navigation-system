"""Root execution entry point for the AI-Based Autonomous Navigation System.

Delegates execution directly to `src.main.main()`.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path so src package is cleanly importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
