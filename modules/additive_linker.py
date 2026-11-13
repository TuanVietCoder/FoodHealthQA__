"""Deterministic additive-name to INS entity linking.

This module performs identity resolution only.  A successful link means that
the declared name denotes an INS identity; it does not establish permission
for a food category, maximum-level compliance, exposure, or safety.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "knowledge_base" / "additive_registry.json"

_EXPLICIT_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<prefix>INS|E)\s*[.:-]?\s*"
    r"(?P<code>\d{3,4}(?:\s*\(\s*[ivx]+\s*\)|[a-z]|[ivx]{1,4})?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_BARE_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<code>\d{3,4}(?:\s*\(\s*[ivx]+\s*\)|[a-z]|[ivx]{1,4})?)"
    r"(?![A-Za-z0-9%])",
    re.IGNORECASE,
)
_ADDITIVE_CONTEXT_RE = re.compile(
    r"\b(?:chất|phẩm\s*màu|màu|acid|axit|emulsifier|preservative|sweetener|"
    r"antioxidant|stabili[sz]er|colour|color)\b",
    re.IGNORECASE,
)

# These surface forms denote a family, not a unique child identity.
_AMBIGUOUS_GENERIC_ALIASES = {
    "caramel",
    "caramen",
    "tocopherol",
    "tocopherols",
    "vitamin e",
}


def normalize_text(value: object) -> str:
    """Normalize Vietnamese/English label text for deterministic matching."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_ins_key(value: object) -> str:
    """Return a stable key such as ``150d`` or ``322(i)``."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"^(?:ins|e)\s*[.:-]?\s*", "", text)
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


def benchmark_ins_code(value: object) -> str:
    """Format an INS identity using the existing benchmark convention."""
    key = normalize_ins_key(value)
    return f"INS{key.upper()}"


def _identity_entity(record: dict) -> dict:
    entity = dict(record)
    entity.setdefault("canonical_name", record.get("canonical_name_vi", ""))
    entity.setdefault("common_names", record.get("aliases", []))
    entity.setdefault("jecfa_adi", "")
    entity.setdefault("contraindications", [])
    entity.setdefault("health_considerations", [])
    entity.setdefault("foodon_curie", "FOODON:00000000")
    entity["identity_only"] = True
    return entity


@lru_cache(maxsize=4)
def _load_registry_cached(path_text: str) -> tuple[dict, ...]:
    path = Path(path_text)
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(_identity_entity(item) for item in raw)


def load_additive_registry(path: str | Path | None = None) -> list[dict]:
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    return [dict(item) for item in _load_registry_cached(str(registry_path.resolve()))]


@lru_cache(maxsize=4)
def _build_indexes(path_text: str):
    records = _load_registry_cached(path_text)
    code_index: dict[str, list[dict]] = {}
    e_index: dict[str, list[dict]] = {}
    alias_index: dict[str, list[dict]] = {}

    for record in records:
        code_index.setdefault(normalize_ins_key(record.get("ins_code")), []).append(record)
        for e_number in record.get("e_numbers", []):
            e_index.setdefault(normalize_ins_key(e_number), []).append(record)
        aliases: Iterable[object] = (
            record.get("canonical_name"),
            record.get("canonical_name_vi"),
            record.get("canonical_name_en"),
            *(record.get("aliases") or []),
        )
        for alias in aliases:
            normalized = normalize_text(alias)
            if not normalized or normalized in _AMBIGUOUS_GENERIC_ALIASES:
                continue
            alias_index.setdefault(normalized, []).append(record)

    # De-duplicate identical records introduced by canonical name + aliases.
    for index in (code_index, e_index, alias_index):
        for key, values in list(index.items()):
            by_code = {normalize_ins_key(v.get("ins_code")): v for v in values}
            index[key] = list(by_code.values())

    aliases_longest_first = tuple(
        sorted(alias_index, key=lambda alias: (len(alias.split()), len(alias)), reverse=True)
    )
    return code_index, e_index, alias_index, aliases_longest_first


def _result(record: dict, mention: str, method: str, confidence: float, alias: str = "") -> dict:
    return {
        "mention": mention,
        "ins_code": record.get("ins_code", ""),
        "matched_entity": dict(record),
        "match_method": method,
        "confidence": confidence,
        "code_origin": (
            "explicit_on_label" if method == "explicit_code" else "inferred_from_name"
        ),
        "matched_alias": alias,
        "identity_only": bool(record.get("identity_only", True)),
        "source": record.get("source", {}),
    }


def link_additive_mentions(text: object, registry_path: str | Path | None = None) -> list[dict]:
    """Return every high-confidence additive identity found in ``text``.

    Explicit codes take precedence.  Exact canonical/alias phrases are then
    matched longest-first.  Fuzzy similarity is intentionally excluded from
    automatic regulatory linking.
    """
    mention = str(text or "").strip()
    if not mention:
        return []

    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    resolved_path = str(path.resolve())
    code_index, e_index, alias_index, aliases = _build_indexes(resolved_path)
    matches: dict[str, dict] = {}

    for code_match in _EXPLICIT_CODE_RE.finditer(mention):
        key = normalize_ins_key(code_match.group("code"))
        prefix = code_match.group("prefix").upper()
        candidates = (e_index.get(key, []) if prefix == "E" else code_index.get(key, []))
        if not candidates:
            candidates = code_index.get(key, [])
        if not candidates and "(" not in key:
            children = [
                record
                for candidate_key, values in code_index.items()
                if candidate_key.startswith(f"{key}(")
                for record in values
            ]
            if len(children) == 1:
                candidates = children
        if len(candidates) == 1:
            record = candidates[0]
            record_key = normalize_ins_key(record.get("ins_code"))
            matches[record_key] = _result(
                record, mention, "explicit_code", 1.0, code_match.group(0)
            )

    # Vietnamese labels commonly omit the literal "INS" inside a declared
    # functional class, e.g. "chất ổn định [170(i)]".  Only accept a bare
    # number when additive context is present, so percentages and quantities
    # are not mistaken for additive codes.
    if _ADDITIVE_CONTEXT_RE.search(mention):
        for code_match in _BARE_CODE_RE.finditer(mention):
            key = normalize_ins_key(code_match.group("code"))
            candidates = code_index.get(key, [])
            if len(candidates) == 1:
                record = candidates[0]
                record_key = normalize_ins_key(record.get("ins_code"))
                matches.setdefault(
                    record_key,
                    _result(record, mention, "explicit_code", 1.0, code_match.group(0)),
                )

    normalized_mention = normalize_text(mention)
    padded_mention = f" {normalized_mention} "
    for alias in aliases:
        if f" {alias} " not in padded_mention:
            continue
        candidates = alias_index.get(alias, [])
        if len(candidates) != 1:
            continue
        record = candidates[0]
        record_key = normalize_ins_key(record.get("ins_code"))
        if record_key in matches:
            continue
        matches[record_key] = _result(
            record, mention, "exact_alias_in_context", 0.99, alias
        )

    return list(matches.values())


def link_additive_name(text: object, registry_path: str | Path | None = None) -> dict | None:
    """Return the most specific deterministic match for a single mention."""
    matches = link_additive_mentions(text, registry_path=registry_path)
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            item["match_method"] == "explicit_code",
            len(item.get("matched_alias", "")),
        ),
        reverse=True,
    )
    return matches[0]


def clear_registry_cache() -> None:
    _load_registry_cached.cache_clear()
    _build_indexes.cache_clear()
