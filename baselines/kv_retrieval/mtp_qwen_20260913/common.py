from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
CAMPAIGN = BASE / 'decode_comparison_20260912'
sys.path[:0] = [str(BASE), str(CAMPAIGN), str(BASE / 'infinigen/InfiniGen')]
MODEL = ROOT / 'models/Qwen3.8-27B-BF16'
WEIGHTS = ROOT / 'models/Qwen3.8-27B-Q8_0.gguf'
