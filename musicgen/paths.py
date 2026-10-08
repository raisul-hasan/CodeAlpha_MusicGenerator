from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
GENERATED = ROOT / "outputs" / "generated"
EXAMPLES = ROOT / "outputs" / "examples"
