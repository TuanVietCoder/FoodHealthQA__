"""
Passage-level retrieval over the curated knowledge base (RAG retrieval layer).

Why this exists: modules/hybrid_retriever.py only links *additive entities*.
This module indexes *evidence passages* (regulation rows, JECFA ADI records,
allergen scope notes, external claim excerpts) so the answer generator can be
given ranked, citable evidence instead of rule-assembled context.

Each passage carries `citation_id` (a code recognised by modules.benchmark) and
`locator` (claim locator, same format as benchmark `source_claim_locators`),
so citations can be verified against what was actually retrieved.

Retrieval modes (for ablation): "bm25", "tfidf", "hybrid" (RRF fusion of the
two) and "hybrid_exact" (hybrid + exact INS-code lookup boost). No heavy
dependencies: numpy only. A dense retriever can be plugged in via
`PassageRetriever(dense_encoder=callable)` (vd OllamaEmbedder("bge-m3")).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from modules.hybrid_retriever import BM25Index, tokenize_vi

_ROOT = Path(__file__).resolve().parent.parent
_KB = _ROOT / "knowledge_base"
_DERIVED = _KB / "sources" / "derived"

# external_claim_verification.json key -> (citation_id, claim locator), aligned with
# knowledge_base/sources/catalog/source_citation_catalog.json and benchmark gold locators.
_CLAIM_KEY_LOCATOR = {
    "FALCPA_321qq_NINE_ALLERGENS": ("US_FDA_FALCPA", "US_CODE_21_321#qq_major_food_allergen"),
    "CFR_172_804_PKU_STATEMENT": ("US_FDA_21CFR172_804", "US_FDA_21CFR172.804#b"),
    "JEMRA1_PRIORITY_LIST": ("WHO_JEMRA_PRIORITY_ALLERGENS", "FAO_WHO_JEMRA_2021_PART1#PriorityAllergens"),
    "JEMRA1_SOY_AND_OATS_NOT_GLOBAL": ("WHO_JEMRA_PRIORITY_ALLERGENS", "FAO_WHO_JEMRA_2021_PART1#PriorityAllergens"),
    "JEMRA3_PAL_REPORT": ("FAO_WHO_JEMRA_PART_3", "FAO_WHO_JEMRA_2021_PART3#PAL_Guidance"),
    "EFSA_2021_TIO2_NOT_SAFE": ("EFSA_2021_TITANIUM_DIOXIDE_E171", "EFSA_ANS_2021_6585#E171_NotSafe"),
    "EFSA_2021_TIO2_FULLTEXT": ("EFSA_2021_TITANIUM_DIOXIDE_E171", "EFSA_ANS_2021_6585#E171_NotSafe"),
    "CFR_101_61_SODIUM_PER_RACC": ("US_FDA_21CFR101_61", "US_CFR_21_101_61#sodium_claims"),
    "CFR_101_54_HIGH_20PCT_DV": ("US_FDA_21CFR101_54", "US_CFR_21_101_54#high_good_source"),
    "FDA_BENZENE_BEVERAGES_ONLY": ("US_FDA_BENZENE_QA", "FDA_BENZENE_QA#beverages"),
    "EFSA_OPINION_2019_PHOSPHATES": ("EFSA_OPINIONS", "EFSA_ANS_2019_5674#Phosphates_ADI40"),
    "EFSA_OPINION_2008_MCCANN": ("EFSA_OPINIONS", "EFSA_ANS_2008_852#AzoDyes_Hyperactivity_Children"),
    "EFSA_OPINION_2017_NITRITE": ("EFSA_OPINIONS", "EFSA_ANS_2017_4786#Nitrite_Methemoglobin"),
    "TT24_PHULUC1_LISTS_INS_AND_NAME": ("VN_TT24_2019", "VN_TT24_2019#PhuLuc1_Bang1"),
    "TT24_ML_IN_MG_PER_KG": ("VN_TT24_2019", "VN_TT24_2019#PhuLuc2A"),
    "CODEX192_TABLE1_TABLE3_PARSED": ("CODEX_STAN_192_1995", "CODEX_STAN_192_1995#TableOne_TableThree"),
    "CODEX_CXG36_INS_AND_CLASS_NAMES": ("CODEX_CXG_36_1989", "CODEX_CXG_36_1989#ClassNames_INS"),
    "CXS1_4214_MANDATORY_ALLERGEN_LIST": ("CODEX_CXS_1_1985", "CODEX_CXS_1_1985#Sec4.2.1.4"),
    "CXS1_4215_OPTIONAL_LIST_SOY_OATS": ("CODEX_CXS_1_1985", "CODEX_CXS_1_1985#Sec4.2.1.5"),
    "CXS1_PAL_ANNEX_DEFINITION": ("CODEX_CXS_1_1985", "CODEX_CXS_1_1985#PAL_Annex"),
    "CXS1_4217_SULPHITE_DECLARATION": ("CODEX_CXS_1_1985", "CODEX_CXS_1_1985#Sec4.2.1.7"),
    "CODEX_GL23_SODIUM_PER_100G": ("CODEX_CAC_GL_23_1997", "CODEX_CAC_GL_23_1997#Table_Sodium"),
    "CODEX_GL23_SUGARS_FREE_PER_100G": ("CODEX_CAC_GL_23_1997", "CODEX_CAC_GL_23_1997#Table_Sugars"),
    "FDA_QA5_REFINED_OIL_EXEMPT": ("US_FDA_ALLERGEN_QA_ED5", "US_FDA_ALLERGEN_QA_ED5#D9_RefinedOils"),
    "EU_ANNEXV_COLOURS_WARNING": ("EU_REG_1333_2008_ANNEX_V", "EU_REG_1333_2008#AnnexV_ColourWarning"),
    "FDA_RED3_REVOKED_FOOD": ("US_FDA_RED3_REVOCATION", "US_FR_2025_00830#Red3_Revocation"),
    "IARC_V134_ASPARTAME_2B": ("IARC_MONOGRAPHS_V134", "IARC_MONOGRAPHS_V134#Aspartame_Group2B"),
    "WHO_NSS_2023_PREGNANCY": ("WHO_NSS_GUIDELINE_2023", "WHO_NSS_GUIDELINE_2023#Pregnancy"),
    "WHO_NSS_2023_CHILDREN": ("WHO_NSS_GUIDELINE_2023", "WHO_NSS_GUIDELINE_2023#Children"),
    "EFSA_CAFFEINE_2015_CHILDREN": ("EFSA_CAFFEINE_2015", "EFSA_CAFFEINE_2015#Children"),
}


# Preference among passages of the same INS when slots are limited: identity/ADI first,
# then permitted-list status, then JECFA record, Codex membership, and raw usage-limit table chunks.
_KIND_PRIORITY = {
    "additive_identity": 0, "tt24_positive_list": 1, "vn_regulation": 2, "jecfa_adi": 3,
    "codex": 4, "codex_membership": 5, "tt24_usage_limits": 6,
}


def _jecfa_locators(kb):
    """chemical_id -> catalog locator (WHO_FAO_JECFA#Chemical<id>_<Name>)."""
    out = {}
    for loc in (_load_json(kb / "sources" / "catalog" / "source_citation_catalog.json") or {}):
        m = re.match(r"WHO_FAO_JECFA(?:_\d+)?#Chemical(\d+)_", loc)
        if m:
            out.setdefault(int(m.group(1)), loc)
    return out



def normalize_ins(code) -> str:
    """'INS 331(III)' / 'E331iii' / '331 (iii)' -> '331iii'."""
    s = re.sub(r"(?i)\b(?:ins|e)(?=[\s.\-]*\d)[\s.\-]*", "", str(code or "")).lower()
    return re.sub(r"[\s().\-]", "", s)


def extract_ins_codes(text: str) -> list[str]:
    """Pull INS/E codes from free text (question or OCR line)."""
    out = []
    for m in re.finditer(r"(?i)\b(?:ins|e)[\s.\-]*(\d{3,4}\s*(?:\(\s*[ivx]+\s*\)|[a-z](?![a-z]))?)", text or ""):
        code = normalize_ins(m.group(1))
        if code and code not in out:
            out.append(code)
    return out


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _windows(text: str, size: int = 110, overlap: int = 20):
    words = text.split()
    if len(words) <= size:
        yield text
        return
    step = size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + size])
        if i + size >= len(words):
            break


def build_corpus(kb_dir: Path | None = None) -> list[dict]:
    """Build passage dicts: {pid, text, citation_id, locator, ins_codes, kind}."""
    kb = Path(kb_dir) if kb_dir else _KB
    derived = kb / "sources" / "derived"
    passages: list[dict] = []
    jecfa_loc = _jecfa_locators(kb)

    def add(kind, text, citation_id, locator, ins=()):
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) < 20:
            return
        passages.append({
            "pid": f"P{len(passages):05d}",
            "kind": kind,
            "text": text,
            "citation_id": citation_id,
            "locator": locator,
            "ins_codes": sorted({normalize_ins(c) for c in ins if c}),
        })

    # 1) curated additive entries: identity + function + ADI, regulatory status
    for a in _load_json(kb / "additives_standard.json") or []:
        ins = a.get("ins_code", "")
        names = " ".join([a.get("canonical_name_vi", ""), a.get("canonical_name_en", ""),
                          " ".join(a.get("synonyms", []))])
        head = f"{ins} {a.get('e_number', '')} {names}"
        add("additive_identity",
            f"{head}. Chức năng: {a.get('function', '')}. ADI: {a.get('adi_value', '')}.",
            "JECFA_EVALUATIONS", f"JECFA_EVALUATIONS#{normalize_ins(ins)}", [ins])
        reg = a.get("vn_regulation") or ""
        if reg:
            add("vn_regulation", f"{head}. Quy định Việt Nam: {reg}",
                "VN_TT24_2019", "VN_TT24_2019#PhuLuc1_Bang1", [ins])
        codex = a.get("codex_standard") or ""
        if codex and "Pending" not in codex:
            add("codex", f"{head}. Codex: {codex}", "CODEX_STAN_192_1995",
                "CODEX_STAN_192_1995#TableOne_TableThree", [ins])

    # 2) TT24 extracted evidence: Phu luc 1 line + chunked Phu luc 2A/2B blocks
    tt24 = _load_json(derived / "tt24_extracted_evidence_for_kb_additives.json") or {}
    for ins, ev in tt24.items():
        p1 = ev.get("phu_luc_1")
        if p1:
            add("tt24_positive_list", f"Phụ lục 1 TT24/2019: {p1}",
                "VN_TT24_2019", "VN_TT24_2019#PhuLuc1_Bang1", [ins])
        for tag, loc in (("phu_luc_2a_blocks", "PhuLuc2A"), ("phu_luc_2b_blocks", "PhuLuc2B")):
            for block in ev.get(tag, []) or []:
                for w in _windows(block):
                    add("tt24_usage_limits", f"INS {ins} Phụ lục {loc[-2:]} TT24/2019: {w}",
                        "VN_TT24_2019", f"VN_TT24_2019#{loc}", [ins])

    # 3) JECFA ADI verification records (primary ADI evidence)
    for fname in ("jecfa_adi_verification.json", "jecfa_adi_verification_more.json"):
        data = _load_json(derived / fname)
        items = data.items() if isinstance(data, dict) else (data or [])
        for ins, rec in items:
            if not isinstance(rec, dict):
                continue
            add("jecfa_adi",
                f"JECFA {rec.get('jecfa_name', '')} (INS {ins}): lần đánh giá gần nhất "
                f"{rec.get('latest_evaluation_year', '?')}, ADI {rec.get('latest_adi', '?')}.",
                "JECFA_EVALUATIONS",
                jecfa_loc.get(rec.get("chemical_id"), f"JECFA_EVALUATIONS#{normalize_ins(ins)}"), [ins])

    # 4) Codex 192 membership (INS present in Table 1 / Table 3)
    codex = _load_json(derived / "codex_192_ins_codes.json") or {}
    for key, loc in (("table_one", "TableOne"), ("table_three", "TableThree")):
        for ins in codex.get(key, []) or []:
            add("codex_membership",
                f"INS {ins} có trong Codex CXS 192-1995 {'Bảng 1' if key == 'table_one' else 'Bảng 3 (GMP)'}.",
                "CODEX_STAN_192_1995", "CODEX_STAN_192_1995#TableOne_TableThree", [ins])

    # 5) external claim excerpts (FALCPA, JEMRA, EFSA, FDA, IARC...)
    for key, rec in (_load_json(derived / "external_claim_verification.json") or {}).items():
        if not isinstance(rec, dict) or not rec.get("passed", True):
            continue
        cid_loc = _CLAIM_KEY_LOCATOR.get(key)
        if not cid_loc:
            continue
        ev = rec.get("evidence")
        ev_text = " ".join(map(str, ev.values())) if isinstance(ev, dict) else str(ev or "")
        add("external_claim", f"{rec.get('description', '')} {ev_text}", cid_loc[0], cid_loc[1])

    # 6) allergen groups
    ag = _load_json(kb / "allergen_groups.json")
    for g in (ag if isinstance(ag, list) else (ag or {}).get("groups", [])):
        kw = ", ".join(g.get("keywords", [])[:20])
        text = (f"Nhóm dị nguyên {g.get('name_vi', '')} ({g.get('name_en', '')}). "
                f"Từ khóa: {kw}. {g.get('note', '')} Giới hạn: {g.get('screening_limit', '')}")
        add("allergen_group", text, "CODEX_CXS_1_1985", "CODEX_CXS_1_1985#Sec4.2.1.4")
    return passages


class PassageRetriever:
    """Passage retriever: BM25 + TF-IDF (+ dense embeddings) with RRF fusion and exact-INS slots.

    Modes: bm25 | tfidf | dense | hybrid (bm25+tfidf) | hybrid_dense (bm25+tfidf+dense), and
    `_exact` variants that reserve part of the top-k for passages mentioning a linked INS code.
    Dense modes need `dense_encoder` (see modules.ollama_embedder).
    """

    MODES = ("bm25", "tfidf", "dense", "hybrid", "hybrid_exact", "hybrid_dense", "hybrid_dense_exact")
    DENSE_MODES = ("dense", "hybrid_dense", "hybrid_dense_exact")

    def __init__(self, passages: list[dict] | None = None, dense_encoder=None, rrf_k: int = 60,
                 exact_frac: float = 0.5):
        self.passages = passages if passages is not None else build_corpus()
        self.rrf_k = rrf_k
        self.exact_frac = exact_frac  # share of top-k slots reserved for exact INS matches
        self._bm25_cache = (None, None)
        self.bm25 = BM25Index()
        self.bm25.index([p["text"] for p in self.passages])
        self._build_tfidf()
        self._by_ins: dict[str, list[int]] = {}
        for i, p in enumerate(self.passages):
            for c in p["ins_codes"]:
                self._by_ins.setdefault(c, []).append(i)
        # optional dense encoder: callable(list[str]) -> np.ndarray (L2-normalised)
        self._dense = None
        self._encode = dense_encoder
        if dense_encoder is not None:
            self._dense = np.asarray(dense_encoder([p["text"] for p in self.passages]), dtype=np.float32)

    # ---- TF-IDF (word uni+bigrams) --------------------------------------
    def _terms(self, text):
        toks = tokenize_vi(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def _build_tfidf(self):
        docs = [self._terms(p["text"]) for p in self.passages]
        df = Counter(t for d in docs for t in set(d))
        self._vocab = {t: i for i, t in enumerate(df)}
        n = len(docs)
        self._idf = np.zeros(len(self._vocab), dtype=np.float32)
        for t, i in self._vocab.items():
            self._idf[i] = math.log((n + 1) / (df[t] + 1)) + 1.0
        m = np.zeros((n, len(self._vocab)), dtype=np.float32)
        for r, d in enumerate(docs):
            for t, c in Counter(d).items():
                m[r, self._vocab[t]] = (1 + math.log(c)) * self._idf[self._vocab[t]]
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._tfidf = m / norms

    def _tfidf_scores(self, query):
        q = np.zeros(len(self._vocab), dtype=np.float32)
        for t, c in Counter(self._terms(query)).items():
            if t in self._vocab:
                q[self._vocab[t]] = (1 + math.log(c)) * self._idf[self._vocab[t]]
        nq = np.linalg.norm(q)
        return self._tfidf @ (q / nq) if nq else np.zeros(len(self.passages), dtype=np.float32)

    def _bm25_scores(self, query):
        if self._bm25_cache[0] != query:  # same query is re-scored for several top_k values
            s = np.zeros(len(self.passages), dtype=np.float32)
            for i, sc in self.bm25.search(query, top_k=len(self.passages)):
                s[i] = sc
            self._bm25_cache = (query, s)
        return self._bm25_cache[1]

    @property
    def has_dense(self) -> bool:
        return self._dense is not None

    def _dense_scores(self, query):
        q = np.asarray(self._encode([query]), dtype=np.float32)[0]
        return self._dense @ q

    @staticmethod
    def _rank(scores, positive_only=True):
        order = np.argsort(-scores, kind="stable")
        return {int(i): r for r, i in enumerate(order) if not positive_only or scores[i] > 0}

    # ---- public API -----------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid_exact",
                 ins_codes: list[str] | None = None) -> list[dict]:
        """Return top_k passages as dicts with `score` and `rank` added."""
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if mode in self.DENSE_MODES and not self.has_dense:
            raise ValueError(f"mode '{mode}' cần dense_encoder (vd modules.ollama_embedder.OllamaEmbedder)")
        if not self.passages:
            return []
        fused: dict[int, float] = {}
        if mode == "bm25":
            fused = {i: float(s) for i, s in enumerate(self._bm25_scores(query)) if s > 0}
        elif mode == "tfidf":
            fused = {i: float(s) for i, s in enumerate(self._tfidf_scores(query)) if s > 0}
        elif mode == "dense":
            fused = {i: float(s) for i, s in enumerate(self._dense_scores(query))}
        else:
            rankings = [self._rank(self._bm25_scores(query)), self._rank(self._tfidf_scores(query))]
            if mode.startswith("hybrid_dense"):
                rankings.append(self._rank(self._dense_scores(query), positive_only=False))
            for ranks in rankings:
                for i, r in ranks.items():
                    fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:top_k]
        if mode.endswith("_exact"):
            # Reserve a bounded share of slots for passages that mention a linked INS code
            # (round-robin across codes, best fused score first within a code); the remaining
            # slots keep the semantic ranking so topical evidence (allergen scope, JEMRA, WHO...)
            # is not crowded out by the many per-INS passages.
            codes = [normalize_ins(c) for c in (ins_codes or [])] or extract_ins_codes(query)
            quota = min(top_k, max(1, round(top_k * self.exact_frac))) if codes and top_k >= 2 else 0
            per_code = []
            for c in dict.fromkeys(codes):
                cand = sorted(
                    self._by_ins.get(c, []),
                    key=lambda i: (_KIND_PRIORITY.get(self.passages[i]["kind"], 9), -fused.get(i, 0.0)),
                )
                per_code.append(cand)
            picked: list[int] = []
            depth = 0
            while len(picked) < quota and any(depth < len(c) for c in per_code):
                for cand in per_code:
                    if depth < len(cand) and cand[depth] not in picked and len(picked) < quota:
                        picked.append(cand[depth])
                depth += 1
            rest = [i for i, _ in ranked if i not in picked][: top_k - len(picked)]
            top = [(i, fused.get(i, 0.0) + 1.0) for i in picked] + [(i, fused[i]) for i in rest]
        out = []
        for rank, (i, s) in enumerate(top, 1):
            p = dict(self.passages[i])
            p["score"], p["rank"] = round(s, 6), rank
            out.append(p)
        return out


_SINGLETON: PassageRetriever | None = None


def get_passage_retriever() -> PassageRetriever:
    """Singleton. Dense nhánh bật khi đặt PASSAGE_EMBEDDING=bge-m3 (hoặc tên model Ollama khác) và Ollama sẵn sàng."""
    global _SINGLETON
    if _SINGLETON is None:
        encoder = None
        model = os.environ.get("PASSAGE_EMBEDDING", "").strip()
        if model:
            try:
                from modules.ollama_embedder import OllamaEmbedder
                encoder = OllamaEmbedder(model)
                encoder(["kiểm tra kết nối"])
            except Exception as exc:  # không có Ollama/model → vẫn chạy với truy xuất sparse
                import logging
                logging.getLogger(__name__).warning("Dense embedding '%s' không dùng được: %s", model, exc)
                encoder = None
        _SINGLETON = PassageRetriever(dense_encoder=encoder)
    return _SINGLETON


def reset_passage_retriever() -> None:
    global _SINGLETON
    _SINGLETON = None


def render_passages(passages: list[dict], max_chars: int = 600) -> str:
    """Format retrieved passages as citable evidence for the LLM prompt."""
    if not passages:
        return ""
    lines = ["### Đoạn bằng chứng truy xuất (xếp hạng theo độ liên quan):"]
    for p in passages:
        text = p["text"] if len(p["text"]) <= max_chars else p["text"][:max_chars] + "…"
        lines.append(f"- [{p['citation_id']}] ({p['locator']}) {text}")
    return "\n".join(lines)
