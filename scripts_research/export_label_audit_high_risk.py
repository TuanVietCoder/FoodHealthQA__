"""Export every image that has a scored UNSAFE / CRITICAL_ALERT record and was not in the first audit sample.

Output: data/label_audit_high_risk.csv (same reviewer columns as label_audit_sample.csv, ordered by number of dependent scored records).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_label_audit_sample as base  # noqa: E402

ROOT = base.ROOT
OUT = ROOT / "data" / "label_audit_high_risk.csv"
DONE = ROOT / "data" / "label_audit_sample.csv"
TOP = {"UNSAFE", "CRITICAL_ALERT"}


def main():
    rows = [json.loads(l) for l in base.ACTIVE.read_text(encoding="utf-8").splitlines() if l.strip()]
    scored = [r for r in rows if r["eligible_for_scoring"] == "yes"]
    by_image = defaultdict(list)
    for r in scored:
        by_image[r["image_ref"]].append(r)
    done = {r["image_file"] for r in csv.DictReader(DONE.open(encoding="utf-8-sig"))}
    imgs = [i for i, rs in by_image.items() if any(r["label_v5"] in TOP for r in rs)
            and ("assets/" + i).replace("\\", "/") not in done]
    imgs.sort(key=lambda i: (-len(by_image[i]), i))
    out = []
    for k, img in enumerate(imgs, 1):
        recs = by_image[img]
        r0 = recs[0]
        label = json.loads((ROOT / r0["label_transcription_path"]).read_text(encoding="utf-8"))
        p = ROOT / "assets" / img
        out.append({
            "sample_no": k, "image_file": str(p.relative_to(ROOT)).replace("\\", "/"),
            "image_exists": "yes" if p.exists() else "NO", "label_json": r0["label_transcription_path"],
            "product_name": label.get("product_name", ""), "n_scored_records_depending": len(recs),
            "labels_depending": ", ".join(f"{a}x{b}" for a, b in sorted(Counter(r["label_v5"] for r in recs).items())),
            "record_ids": " ".join(r["id"] for r in recs), "what_to_check": base.what_to_check(recs),
            "transcribed_ingredient": base.join(label.get("ingredient")),
            "transcribed_additive": base.join(label.get("additive")),
            "transcribed_warning": base.join(label.get("warning")),
            "transcribed_nutrition": "",
            "ingredient_ok (Y/N)": "", "additive_codes_ok (Y/N)": "", "warning_ok (Y/N)": "", "product_name_ok (Y/N)": "",
            "what_is_wrong": "", "corrected_text": "",
        })
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    n_rec = sum(r["n_scored_records_depending"] for r in out)
    print(f"{len(out)} images (not yet audited) with scored UNSAFE/CRITICAL_ALERT; they carry {n_rec} of {len(scored)} scored records")
    print("missing images:", sum(r["image_exists"] == "NO" for r in out))
    top = Counter(l.split("x")[0] for r in out for l in r["labels_depending"].split(", "))
    print(top.most_common(8))


if __name__ == "__main__":
    main()
