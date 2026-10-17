"""The shipped model files match models/SHA256SUMS, and every shipped model is listed there and has a model card."""

import hashlib
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
DOCS = MODELS.parent / "docs"


def manifest() -> dict[str, str]:
    rows = [ln.split() for ln in (MODELS / "SHA256SUMS").read_text().splitlines() if ln.strip()]
    return {name: digest for digest, name in rows}


def test_every_shipped_model_matches_its_checksum():
    sums = manifest()
    shipped = sorted(p.name for p in MODELS.glob("*.pt"))
    assert shipped == sorted(sums), "models/ and models/SHA256SUMS list different files"
    for name in shipped:
        assert hashlib.sha256((MODELS / name).read_bytes()).hexdigest() == sums[name], name


def test_every_shipped_model_has_a_card():
    for name in manifest():
        assert (DOCS / f"MODEL_CARD_{Path(name).stem}.md").exists(), name
