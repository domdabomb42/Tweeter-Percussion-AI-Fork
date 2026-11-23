"""Independent, full-track percussion experiment for one pinned SmolLM2 variant."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT.parent
OUTPUT = MODEL_ROOT / "MostRecentOutput"
V3_ROOT = MODEL_ROOT / "V3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

# V3 data installs the model-local CustomTraining and shared extractor paths.
from percussion_generation_v3 import data as _v3_data  # noqa: E402
from custom_training import MODEL_ID, MODEL_REVISION  # noqa: E402

VERSION = "v4ea-synth-full-track-v1"
TARGET_MARKER = "\nPERCUSSION_TRACK_TARGET="
END_MARKER = "[END_TRACK]"
