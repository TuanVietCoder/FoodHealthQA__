"""Detect additives written by NAME only (no INS code) for the vulnerable-group authority flags.

The v5 flags (EFSA 2019 phosphates, EFSA 2008 McCann colours + sodium benzoate, EFSA 2017 nitrite, EFSA 2021 E171,
21 CFR 172.804 aspartame) were keyed on INS numbers. Labels often name the substance instead (e.g. "Dinatri phosphat").
This module maps names to the INS base number of the group, with two precautions:
  * negated mentions ("Không chứa Natri benzoat", "Không phẩm màu Sunset yellow") are ignored;
  * substances that are NOT in the EFSA group are excluded on purpose (iron pyrophosphate, distarch phosphate = INS 1412/1442,
    thiamine mononitrate, trisodium citrate).
"""

from __future__ import annotations

import re
import unicodedata

_CATION = r"(?:natri|sodium|kali|potassium|canxi|calci|calcium|magnesi|magnesium)"
_PHOS_WORD = r"(?:pyro|poly|hexameta|meta)?(?:phosphat\w*|photphat\w*|phosphate\w*)"

# (group key, INS base reported, regex on accent-stripped lower-case text)
_RULES = [
    ("phosphate", "341", re.compile(
        r"(?<!distarch )(?<!distarch\s)\b(?:mono|di|tri|tetra|penta|hexa|poly)?" + _CATION + r"\s*(?:di|tri)?(?:hydro\w*\s*)?(?:di\s*|tri\s*)?" + _PHOS_WORD)),
    ("phosphate", "452", re.compile(r"\b(?:amoni|ammonium)\s*polyphosphat\w*")),
    ("phosphate", "338", re.compile(r"\bacid\s*(?:ortho)?ph(?:o)?s?phoric|\bphosphoric\s*acid")),
    ("mccann", "129", re.compile(r"\ballura\s*red")),
    ("mccann", "102", re.compile(r"\btartrazin\w*")),
    ("mccann", "110", re.compile(r"\bsunset\s*yellow")),
    ("mccann", "124", re.compile(r"\bponceau\s*4\s*r|\bcochineal\s*red\s*a")),
    ("mccann", "122", re.compile(r"\bcarmoisin\w*|\bazorubin\w*")),
    ("mccann", "104", re.compile(r"\bquinolin\w*\s*yellow|\bvang\s*quinolin\w*")),
    ("mccann", "211", re.compile(r"\b(?:natri|sodium)\s*benzoat\w*|\bbenzoat\w*\s*natri")),
    ("nitrite", "250", re.compile(r"\b(?:natri|sodium)\s*nitrit\w*")),
    ("nitrite", "249", re.compile(r"\b(?:kali|potassium)\s*nitrit\w*")),
    ("tio2", "171", re.compile(r"\btitan(?:ium)?\s*dioxy?d\w*")),
    ("aspartame", "951", re.compile(r"\baspartame?\b(?!\s*-?\s*acesulfam)")),
]
_NEGATION = re.compile(r"\b(?:khong|no added|free from|without)\b")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _label_text(label: dict) -> str:
    parts = []
    for k in ("ingredient", "additive", "warning"):
        v = label.get(k)
        parts.append(" ; ".join(map(str, v)) if isinstance(v, list) else str(v or ""))
    return _norm(" ; ".join(parts))


def name_hits(label: dict) -> dict[str, list[tuple[str, str]]]:
    """{group: [(ins_base, matched_name), ...]} for substances named on the label and not negated."""
    text = _label_text(label)
    out: dict[str, list[tuple[str, str]]] = {}
    for group, base, rx in _RULES:
        for m in rx.finditer(text):
            clause_start = max(text.rfind(";", 0, m.start()), text.rfind(".", 0, m.start()))
            window = text[max(clause_start + 1, m.start() - 60):m.start()]
            if _NEGATION.search(window):
                continue
            hit = (base, re.sub(r"\s+", " ", m.group(0)).strip())
            if hit not in out.setdefault(group, []):
                out[group].append(hit)
    return out


def name_codes(label: dict) -> list[str]:
    """Pseudo-codes such as 'INS341[theo tên: dicalci phosphat]' (code_base() still reads the leading number)."""
    codes = []
    for hits in name_hits(label).values():
        for base, name in hits:
            codes.append(f"INS{base}[theo tên: {name}]")
    return codes
