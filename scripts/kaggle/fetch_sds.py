"""Download the SeaDronesSee subset used in this repo straight from the dataset's public Nextcloud share.

For a machine with normal internet (Kaggle, a rented GPU). Images come from the ODv2 "Compressed Version" share,
one WebDAV request per image, are resized to 1920 px (as in every experiment here) and paired with the YOLO
labels shipped in data/kaggle/sds_labels.tar.gz (SeaDronesSee is CC0 1.0).

  python scripts/kaggle/fetch_sds.py --out /kaggle/tmp/sds --roles train val
  SDS_SHARE=https://<host>/index.php/s/<token> python scripts/kaggle/fetch_sds.py ...   # if the default moved

The share link is on https://seadronessee.cs.uni-tuebingen.de/dataset ("Object Detection v2").
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SHARE = "https://cloud.cs.uni-tuebingen.de/index.php/s/ZZxX65FGnQ8zjBP"


def webdav(share: str) -> tuple[str, str]:
    u = urllib.parse.urlparse(share)
    token = u.path.rstrip("/").split("/")[-1]
    return f"{u.scheme}://{u.netloc}/public.php/webdav/", "Basic " + base64.b64encode(f"{token}:".encode()).decode()


def get(url: str, auth: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Authorization": auth, "User-Agent": "esa-dataset-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roles", nargs="+", default=["train", "val"], choices=["train", "val", "test", "seen"])
    ap.add_argument("--share", default=os.environ.get("SDS_SHARE", DEFAULT_SHARE))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1920)
    a = ap.parse_args()
    out = Path(a.out)
    base, auth = webdav(a.share)
    sel = json.loads((REPO / "data/seadronessee_selection.json").read_text())

    # 0. check the share answers before starting 2,900 requests
    first_fn, first_split = sel[a.roles[0]][0]
    try:
        get(base + urllib.parse.quote(f"Compressed Version/images/{first_split}/{first_fn}"), auth, tries=2)
    except Exception as e:
        sys.exit(
            f"SeaDronesSee share not reachable at {base} ({e}).\n"
            "Open https://seadronessee.cs.uni-tuebingen.de/dataset, copy the 'Object Detection v2' link and rerun with "
            "SDS_SHARE=<that link>."
        )

    with tarfile.open(REPO / "data/kaggle/sds_labels.tar.gz") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.split("/")[1] in a.roles:
                dst = out / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(t.extractfile(m).read())

    jobs = [(role, fn, split) for role in a.roles for fn, split in sel[role]]

    def one(job):
        role, fn, split = job
        dst = out / "images" / role / fn
        if dst.exists():
            return 0
        raw = get(base + urllib.parse.quote(f"Compressed Version/images/{split}/{fn}"), auth)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        s = a.max_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=92)
        return len(raw)

    t0, done, nbytes = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for b in ex.map(one, jobs):
            done += 1
            nbytes += b
            if done % 250 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} images, {nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)

    names = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
    cfg = {"path": str(out.resolve()), **{r: f"images/{r}" for r in a.roles}, "names": dict(enumerate(names))}
    cfg.setdefault("val", cfg.get("train"))
    y = REPO / "configs/datasets/seadronessee.yaml"
    y.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    y.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("dataset yaml ->", y)


if __name__ == "__main__":
    main()
