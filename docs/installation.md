# Installation & Environment Setup Guide

This guide walks you through setting up the environment for the **AI-Based Autonomous Navigation System**.

---

## 1. Prerequisites

- **Python Version**: Python 3.10 or higher.
- **Operating System**: Windows, macOS, or Linux.
- **Git** (optional for source tracking, but recommended).

---

## 2. Virtual Environment Setup (Recommended)

To isolate project dependencies, create and activate a Python virtual environment:

### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### On Linux / macOS (Bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

Install the minimal required dependencies from the project root:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `numpy`: Numerical operations, grid representations, kinematics matrix calculations.
- `matplotlib`: 2D trajectory rendering, map plotting, and output figures.
- `pytest`: Automated unit test framework.

---

## 4. Verify Installation

Run the root application entry point:

```bash
python main.py
```

Expected output:
```text
============================================================
AI-Based Autonomous Navigation System
============================================================
Project Foundation & Environment Setup: COMPLETE
Current Phase: Repository Scaffolding & Configuration
Next Phase   : Navigation Algorithm & Simulation Implementation
------------------------------------------------------------
Note: Navigation algorithms (A* planner, kinematics, controller)
      are scheduled for implementation in the next phase.
============================================================
```

---

## 5. Running Tests (Subsequent Phase)

Once unit tests are populated in `tests/`, execute:

```bash
pytest -v
```
