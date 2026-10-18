"""Loading, indexed lookup, and conservative CURIE assignment for FoodOn."""

import bisect
import json
import os
import re
import unicodedata
from functools import lru_cache

from modules.file_resolver import find_data_file


def _normalize_label(value):
    value = unicodedata.normalize("NFKD", str(value).casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", value)


def _label_tokens(value):
    value = unicodedata.normalize("NFKD", str(value).casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return frozenset(re.findall(r"[a-z0-9]+", value))


class FoodOnCatalog:
    """Immutable ontology terms with indexes for exact, prefix, and token searches."""

    def __init__(self, entities):
        self.entities = tuple(entities)
        self.by_curie = {}
        labels_by_normalized = {}
        self._label_prefixes = []
        self._label_tokens = {}

        for entity in self.entities:
            curie = str(entity["curie"]).strip().upper()
            label = str(entity["label"]).strip()
            self.by_curie[curie] = entity
            if label.casefold().startswith("obsolete:"):
                continue

            normalized = _normalize_label(label)
            if normalized:
                labels_by_normalized.setdefault(normalized, []).append(entity)
                self._label_prefixes.append((label.casefold(), entity))
                for token in _label_tokens(label):
                    self._label_tokens.setdefault(token, []).append(entity)

        self.by_normalized_label = {
            label: matches[0]
            for label, matches in labels_by_normalized.items()
            if len(matches) == 1
        }
        self._label_prefixes.sort(key=lambda item: item[0])
        self._label_prefix_keys = [item[0] for item in self._label_prefixes]
        self._curie_prefixes = sorted(self.by_curie)

    def __len__(self):
        return len(self.entities)


@lru_cache(maxsize=4)
def _load_foodon_catalog(file_path, modified_ns):
    with open(file_path, "r", encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, list):
        raise ValueError(f"FoodOn catalogue must contain a JSON list: {file_path}")

    entities = [
        item for item in data
        if isinstance(item, dict)
        and item.get("curie")
        and item.get("label")
    ]
    return FoodOnCatalog(entities)


def load_foodon_entities(data_dir):
    """Load and index the complete FoodOn catalogue, cached until the file changes."""
    file_path = find_data_file(data_dir, ["foodon_entities", "foodon"])
    if not file_path:
        return FoodOnCatalog(())
    return _load_foodon_catalog(
        os.path.abspath(file_path),
        os.stat(file_path).st_mtime_ns,
    )


def _as_catalog(entities):
    if isinstance(entities, FoodOnCatalog):
        return entities
    return FoodOnCatalog(entities)


def find_foodon_exact_label(query, entities):
    """Return a unique, non-obsolete term with an exact normalized label match."""
    normalized_query = _normalize_label(query)
    if not normalized_query:
        return None
    return _as_catalog(entities).by_normalized_label.get(normalized_query)


def search_foodon_entities(query, entities, limit=100):
    """Search by CURIE or label using prefix and token indexes."""
    query_lower = str(query).strip().casefold()
    normalized_query = _normalize_label(query)
    if not query_lower or limit <= 0:
        return []

    catalog = _as_catalog(entities)
    exact = catalog.by_normalized_label.get(normalized_query)
    matches = {}
    if exact:
        matches[exact["curie"]] = (0, str(exact["label"]).casefold(), exact)

    curie_prefix_index = bisect.bisect_left(catalog._curie_prefixes, query_lower.upper())
    while curie_prefix_index < len(catalog._curie_prefixes):
        curie = catalog._curie_prefixes[curie_prefix_index]
        if not curie.startswith(query_lower.upper()):
            break
        entity = catalog.by_curie[curie]
        matches[curie] = (1, str(entity["label"]).casefold(), entity)
        curie_prefix_index += 1

    label_prefix_index = bisect.bisect_left(catalog._label_prefix_keys, query_lower)
    while label_prefix_index < len(catalog._label_prefixes):
        label, entity = catalog._label_prefixes[label_prefix_index]
        if not label.startswith(query_lower):
            break
        rank = 0 if _normalize_label(label) == normalized_query else 1
        matches[entity["curie"]] = (rank, label, entity)
        label_prefix_index += 1

    query_tokens = _label_tokens(query)
    if query_tokens:
        postings = [catalog._label_tokens.get(token, ()) for token in query_tokens]
        if all(postings):
            postings.sort(key=len)
            candidates = {entity["curie"]: entity for entity in postings[0]}
            for posting in postings[1:]:
                posting_curies = {entity["curie"] for entity in posting}
                candidates = {
                    curie: entity for curie, entity in candidates.items()
                    if curie in posting_curies
                }
            for curie, entity in candidates.items():
                label = str(entity["label"]).casefold()
                rank = 0 if _normalize_label(label) == normalized_query else 2
                matches.setdefault(curie, (rank, label, entity))

    ordered = sorted(matches.values(), key=lambda item: (item[0], item[1]))
    return [entity for _, _, entity in ordered[:limit]]


def enrich_additive_foodon_curie(additives, entities):
    """Assign only unique exact-label CURIEs and clear invalid or ambiguous mappings."""
    catalog = _as_catalog(entities)

    for additive in additives:
        raw_synonyms = additive.get("common_names") or additive.get("synonyms") or []
        if isinstance(raw_synonyms, str):
            raw_synonyms = re.split(r"[,;]", raw_synonyms)
        labels = (
            additive.get("canonical_name"),
            additive.get("canonical_name_vi"),
            additive.get("canonical_name_en"),
            *raw_synonyms,
        )
        normalized_labels = {
            _normalize_label(label) for label in labels if label
        }

        matched_terms = {
            catalog.by_normalized_label[label]["curie"]
            for label in normalized_labels
            if label in catalog.by_normalized_label
        }
        additive["foodon_curie"] = (
            next(iter(matched_terms)) if len(matched_terms) == 1 else "FOODON:00000000"
        )

    return additives
