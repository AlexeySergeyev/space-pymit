from pathlib import Path


MODULE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = MODULE_DIR.parent.parent

DAMIT_DIR = PROJECT_ROOT / "damit"
CONVEXINV_EXEC = DAMIT_DIR / "convexinv" / "convexinv"
PERIOD_SCAN_EXEC = DAMIT_DIR / "convexinv" / "period_scan"
MINKOWSKI_EXEC = DAMIT_DIR / "fortran" / "minkowski"
