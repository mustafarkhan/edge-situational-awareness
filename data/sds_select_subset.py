"""Choose a video-disjoint SeaDronesSee subset small enough to train on a CPU.

Finding that motivates this: every video in the official SeaDronesSee ODv2 *val* split also appears
in *train* (frames are split, videos are not). Scoring on official val therefore partly scores the
model on scenes it trained on. Here, whole source videos are assigned to train / val / test, and a
fourth set ("seen") keeps official-val frames from TRAIN videos, so the leakage effect can be measured.

  python data/sds_select_subset.py --train-json instances_train.json --val-json instances_val.json \
      --out selection.json --n-train 2500 --n-val 400 --n-test 1200 --n-seen 400

Output: {"train": [file names], "val": [...], "test": [...], "seen": [...], "videos": {...}}
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

RARE = {"jetski", "life_saving_appliances", "buoy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1200)
    ap.add_argument("--n-seen", type=int, default=400)
    ap.add_argument("--test-share", type=float, default=0.15, help="share of images (by video) for test")
    ap.add_argument("--val-share", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    images, labels = {}, defaultdict(Counter)
    for split, path in (("train", args.train_json), ("val", args.val_json)):
        d = json.loads(Path(path).read_text())
        cats = {c["id"]: c["name"] for c in d["categories"]}
        for im in d["images"]:
            images[im["file_name"]] = {"split": split, "video": im["source"].get("folder_name") or "?"}
        id2fn = {im["id"]: im["file_name"] for im in d["images"]}
        for a in d["annotations"]:
            if cats[a["category_id"]] != "ignored":
                labels[id2fn[a["image_id"]]][cats[a["category_id"]]] += 1

    by_video = defaultdict(list)
    for fn, m in images.items():
        by_video[m["video"]].append(fn)
    total = len(images)

    # assign whole videos to test, then val, greedily from a shuffled list, until each share is
    # reached; reshuffle (up to 200 tries) until test contains every class
    vids = sorted(by_video)
    classes = {c for cnt in labels.values() for c in cnt}
    for _ in range(200):
        rng.shuffle(vids)
        test_v, val_v, n_t, n_v = [], [], 0, 0
        for v in vids:
            n = len(by_video[v])
            if n > 0.5 * args.test_share * total:  # no single giant collection in test
                continue
            if n_t < args.test_share * total:
                test_v.append(v)
                n_t += n
            elif n_v < args.val_share * total:
                val_v.append(v)
                n_v += n
        test_classes = {c for v in test_v for fn in by_video[v] for c in labels[fn]}
        val_classes = {c for v in val_v for fn in by_video[v] for c in labels[fn]}
        if test_classes == classes and len(val_classes) >= len(classes) - 1:
            break
    train_v = [v for v in vids if v not in test_v and v not in val_v]

    def sample(vlist, n, only_split=None, prefer_rare=False):
        pool = [fn for v in vlist for fn in by_video[v] if only_split is None or images[fn]["split"] == only_split]
        pool = [fn for fn in pool if labels[fn]]  # images with at least one object
        rng.shuffle(pool)
        if prefer_rare:  # keep rare classes represented
            pool.sort(key=lambda fn: -sum(labels[fn][c] > 0 for c in RARE))
            rare = [fn for fn in pool if any(labels[fn][c] for c in RARE)][: n // 3]
            rest = [fn for fn in pool if fn not in set(rare)]
            rng.shuffle(rest)
            return sorted(rare + rest[: n - len(rare)])
        return sorted(pool[:n])

    sel = {
        "train": sample(train_v, args.n_train, only_split="train", prefer_rare=True),
        "val": sample(val_v, args.n_val),
        "test": sample(test_v, args.n_test),
        "seen": sample(train_v, args.n_seen, only_split="val"),  # official-val frames of TRAIN videos
        "videos": {"train": sorted(train_v), "val": sorted(val_v), "test": sorted(test_v)},
    }
    for k in ("train", "val", "test", "seen"):
        cnt = Counter()
        for fn in sel[k]:
            cnt.update(labels[fn])
        src = Counter(images[fn]["split"] for fn in sel[k])
        print(f"{k:5s}: {len(sel[k]):5d} images from official {dict(src)}; boxes {dict(cnt)}")
    print("videos:", {k: len(v) for k, v in sel["videos"].items()})
    Path(args.out).write_text(json.dumps(sel))


if __name__ == "__main__":
    main()
