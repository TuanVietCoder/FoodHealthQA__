"""Build the additive identity registry from the official Vietnamese annex.

The registry is deliberately identity-only.  It answers "which INS code does
this declared name denote?" and must not be used by itself to assert food-
category permission, ML compliance, ADI compliance, or product safety.

The default source is the 2024 consolidated text of Circular 24/2019/TT-BYT
(including the 2023 amendment and the 2024 partial repeal).  PyMuPDF is only a
build-time dependency; the application reads the generated JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "knowledge_base" / "sources" / "raw" / "vn" / "09_VBHN_BYT_2024.pdf"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "knowledge_base" / "additive_registry.json"
SAFETY_KB = PROJECT_ROOT / "knowledge_base" / "additives_standard.json"

SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/09-vbhn-byt.pdf"


EXTRA_ALIASES = {
    "150a": [
        "caramen nhóm i",
        "caramen nhóm 1",
        "caramen nguyên chất",
        "caramel class i",
        "plain caramel",
    ],
    "150d": [
        "caramen nhóm iv",
        "caramen nhóm 4",
        "caramen amoni sulfit",
        "caramel class iv",
        "sulfite ammonia caramel",
        "sulphite ammonia caramel",
    ],
    "307c": [
        "dl-alpha tocopherol",
        "dl alpha tocopherol",
        "tocopherol dl-alpha",
        "tocopherol dl alpha",
    ],
    "322(i)": [
        "lecithin đậu nành",
        "lecithin đậu tương",
        "soy lecithin",
        "soya lecithin",
        "phosphatides",
        "phospholipids",
    ],
}

# E-number mappings are stored explicitly where the INS suffix does not map
# mechanically to the EU display code.  Do not derive these values blindly.
E_NUMBER_OVERRIDES = {
    "150a": ["E150a"],
    "150d": ["E150d"],
    "307c": ["E307"],
    "322(i)": ["E322"],
}


def _clean_cell(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def _code_key(value: object) -> str:
    text = _clean_cell(value).casefold()
    text = re.sub(r"^(?:ins|e)\s*", "", text)
    text = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"(\d{3,4})(?:\(?([a-z]+|[ivx]+)\)?)?", text)
    if not match:
        return text
    base, suffix = match.groups()
    if not suffix:
        return base
    if re.fullmatch(r"[ivx]+", suffix):
        return f"{base}({suffix})"
    return f"{base}{suffix}"


def _load_safety_aliases() -> dict[str, list[str]]:
    if not SAFETY_KB.exists():
        return {}
    records = json.loads(SAFETY_KB.read_text(encoding="utf-8"))
    aliases: dict[str, list[str]] = {}
    for record in records:
        key = _code_key(record.get("ins_code"))
        if not key:
            continue
        values = [
            record.get("canonical_name"),
            record.get("canonical_name_vi"),
            record.get("canonical_name_en"),
            *(record.get("synonyms") or []),
            *(record.get("common_names") or []),
        ]
        aliases.setdefault(key, []).extend(_clean_cell(v) for v in values if v)
    return aliases


def _find_annex_pages(document) -> range:
    annex_1 = None
    annex_2a = None
    for index in range(document.page_count):
        text = document[index].get_text()
        if annex_1 is None and "PHỤ LỤC 1" in text:
            annex_1 = index
        if annex_1 is not None and index > annex_1 and "PHỤ LỤC 2A" in text:
            annex_2a = index
            break
    if annex_1 is None or annex_2a is None:
        raise RuntimeError("Không xác định được phạm vi Phụ lục 1 trong PDF nguồn")
    return range(annex_1, annex_2a)


def build_registry(source_path: Path) -> list[dict]:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - build-time guard
        raise RuntimeError("Cần cài PyMuPDF để dựng additive_registry.json") from exc

    document = pymupdf.open(source_path)
    safety_aliases = _load_safety_aliases()
    records: list[dict] = []

    for page_index in _find_annex_pages(document):
        tables = document[page_index].find_tables().tables
        if not tables:
            continue
        for row in tables[0].extract():
            if len(row) < 5 or not re.fullmatch(r"\d{1,3}", _clean_cell(row[0])):
                continue
            ordinal = int(_clean_cell(row[0]))
            raw_code = _clean_cell(row[1])
            key = _code_key(raw_code)
            name_vi = _clean_cell(row[2])
            name_en = _clean_cell(row[3])
            function = _clean_cell(row[4])
            aliases = [name_vi, name_en]
            aliases.extend(safety_aliases.get(key, []))
            aliases.extend(EXTRA_ALIASES.get(key, []))
            aliases = sorted({a for a in aliases if a}, key=lambda value: value.casefold())

            records.append(
                {
                    "registry_id": f"INS_{key.upper()}",
                    "ins_code": f"INS {key}",
                    "ins_key": key,
                    "e_numbers": E_NUMBER_OVERRIDES.get(key, []),
                    "canonical_name": name_vi or name_en,
                    "canonical_name_vi": name_vi,
                    "canonical_name_en": name_en,
                    "aliases": aliases,
                    "common_names": aliases,
                    "function": function,
                    "identity_only": True,
                    "regulatory_scope": "identity_only_not_compliance",
                    "source": {
                        "id": "VN_TT24_2019_CONSOLIDATED_2024_ANNEX_1",
                        "url": SOURCE_URL,
                        "locator": f"Phụ lục 1, STT {ordinal}, INS {raw_code}",
                    },
                }
            )

    records.sort(key=lambda item: int(re.match(r"\d+", item["ins_key"]).group()))
    ordinals = [item["source"]["locator"] for item in records]
    if len(records) != 400 or len(set(ordinals)) != len(ordinals):
        raise RuntimeError(
            f"Registry không đạt kiểm tra toàn vẹn: expected=400, actual={len(records)}"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    registry = build_registry(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(registry)} additive identities -> {args.output}")


if __name__ == "__main__":
    main()
