import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demo_dataset_creates_three_classes(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "src/make_demo_dataset.py"), "--output-dir", str(tmp_path),
                    "--per-class", "3", "--size", "64"], check=True, capture_output=True)
    folders = sorted(p.name for p in tmp_path.iterdir())
    assert folders == ["circles", "squares", "triangles"]
    assert all(len(list(p.glob("*.png"))) == 3 for p in tmp_path.iterdir())
