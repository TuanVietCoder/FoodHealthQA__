"""
Module Hybrid Retriever cho hệ thống FoodHealthQA
Triển khai kiến trúc 4-Agent của FoodOntoRAG:
  Agent 1: Hybrid Retriever (BM25 Lexical + TF-IDF Semantic + Fusion)
  Agent 2: Selector Agent (Exact-match preference + Specificity rule)
  Agent 3: LLM Scorer (Confidence Calibration τ)
  Agent 4: Synonym Generator & Feedback Loop

Hỗ trợ:
- Chế độ lightweight (numpy): BM25 thuần + TF-IDF cosine (không cần cài thêm thư viện)
- Chế độ nâng cao (tùy chọn): Whoosh index + FAISS + SentenceTransformer all-MiniLM-L6-v2
"""

import re
import math
import numpy as np
from collections import Counter

# Thử nạp thư viện nâng cao, fallback nếu không có
_HAS_WHOOSH = False
_HAS_FAISS = False
_HAS_SBERT = False

try:
    from whoosh.index import create_in, open_dir
    from whoosh.fields import Schema, TEXT, ID, KEYWORD
    from whoosh.qparser import MultifieldParser
    from whoosh import scoring
    _HAS_WHOOSH = True
except ImportError:
    pass

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    pass


# =====================================================================
# TOKENIZER TIẾNG VIỆT ĐƠN GIẢN
# =====================================================================

_VIET_STOPWORDS = frozenset([
    "và", "của", "các", "cho", "từ", "trong", "với", "là", "có", "được",
    "theo", "để", "hoặc", "hay", "không", "một", "những", "này", "đó",
    "khi", "về", "tại", "trên", "bởi", "cũng", "như", "đã", "sẽ", "đang",
])


def tokenize_vi(text):
    """Tách token tiếng Việt: lowercase, loại stopwords, giữ mã số."""
    text = text.lower().strip()
    text = re.sub(r'[()[\]{}<>\"\'`~!@#$%^&*+=|\\/:;]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t and t not in _VIET_STOPWORDS and len(t) > 1]


# =====================================================================
# AGENT 1: BM25 LEXICAL RETRIEVER (Triển khai thuần numpy)
# =====================================================================

class BM25Index:
    """
    Thuật toán BM25 (Okapi BM25) triển khai thuần numpy.
    Công thức: score(D, Q) = Σ IDF(qi) * (tf(qi,D) * (k1+1)) / (tf(qi,D) + k1*(1-b+b*|D|/avgdl))
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = []
        self.doc_lengths = []
        self.avgdl = 0.0
        self.n_docs = 0
        self.df = Counter()
        self.idf = {}

    def index(self, documents):
        """Xây dựng chỉ mục BM25 từ danh sách văn bản."""
        self.corpus_tokens = [tokenize_vi(doc) for doc in documents]
        self.n_docs = len(self.corpus_tokens)
        self.doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.n_docs, 1)

        self.df = Counter()
        for tokens in self.corpus_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1

        self.idf = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query, top_k=5):
        """Tìm kiếm BM25, trả về danh sách (doc_index, score) giảm dần."""
        query_tokens = tokenize_vi(query)
        if not query_tokens:
            return []

        scores = []
        for idx, doc_tokens in enumerate(self.corpus_tokens):
            score = 0.0
            tf_counter = Counter(doc_tokens)
            dl = self.doc_lengths[idx]
            for qt in query_tokens:
                if qt not in self.idf:
                    continue
                tf = tf_counter.get(qt, 0)
                idf = self.idf[qt]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += idf * numerator / denominator
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =====================================================================
# AGENT 1: TF-IDF SEMANTIC RETRIEVER (Cosine similarity trên vector TF-IDF)
# =====================================================================

class TFIDFSemanticIndex:
    """
    Tìm kiếm ngữ nghĩa bằng TF-IDF vector + Cosine Similarity.
    Lightweight thay thế cho SentenceTransformer + FAISS khi chưa cài thư viện.
    Nếu sentence_transformers + faiss có sẵn, tự động nâng cấp lên dense embedding.
    """

    def __init__(self):
        self.vocabulary = {}
        self.idf = None
        self.tfidf_matrix = None
        self.n_docs = 0
        self._sbert_model = None
        self._dense_matrix = None
        self._use_dense = False

    def index(self, documents):
        """Xây dựng chỉ mục TF-IDF (hoặc dense embedding nếu có SBERT)."""
        self.n_docs = len(documents)

        if _HAS_SBERT:
            try:
                self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = self._sbert_model.encode(documents, normalize_embeddings=True)
                self._dense_matrix = np.array(embeddings, dtype=np.float32)
                self._use_dense = True
                return
            except Exception:
                pass

        all_tokens = [tokenize_vi(doc) for doc in documents]

        vocab_set = set()
        for tokens in all_tokens:
            vocab_set.update(tokens)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocabulary)

        tf_matrix = np.zeros((self.n_docs, vocab_size), dtype=np.float64)
        for doc_idx, tokens in enumerate(all_tokens):
            counter = Counter(tokens)
            doc_len = max(len(tokens), 1)
            for token, count in counter.items():
                if token in self.vocabulary:
                    tf_matrix[doc_idx, self.vocabulary[token]] = count / doc_len

        df = np.sum(tf_matrix > 0, axis=0).astype(np.float64)
        self.idf = np.log((self.n_docs + 1) / (df + 1)) + 1.0

        self.tfidf_matrix = tf_matrix * self.idf
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_matrix = self.tfidf_matrix / norms

    def search(self, query, top_k=5):
        """Tìm kiếm cosine similarity, trả về danh sách (doc_index, score)."""
        if self._use_dense and self._sbert_model is not None:
            q_emb = self._sbert_model.encode([query], normalize_embeddings=True)
            q_vec = np.array(q_emb, dtype=np.float32)
            if _HAS_FAISS:
                index = faiss.IndexFlatIP(self._dense_matrix.shape[1])
                index.add(self._dense_matrix)
                scores, indices = index.search(q_vec, min(top_k, self.n_docs))
                return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0])) if indices[0][i] >= 0]
            else:
                similarities = np.dot(self._dense_matrix, q_vec.T).flatten()
                top_indices = np.argsort(similarities)[::-1][:top_k]
                return [(int(idx), float(similarities[idx])) for idx in top_indices]

        query_tokens = tokenize_vi(query)
        if not query_tokens or self.tfidf_matrix is None:
            return []

        q_vec = np.zeros(len(self.vocabulary), dtype=np.float64)
        counter = Counter(query_tokens)
        q_len = max(len(query_tokens), 1)
        for token, count in counter.items():
            if token in self.vocabulary:
                q_vec[self.vocabulary[token]] = count / q_len
        q_vec = q_vec * self.idf
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        similarities = np.dot(self.tfidf_matrix, q_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


# =====================================================================
# AGENT 1: HYBRID FUSION
# =====================================================================

class HybridRetriever:
    """
    Kết hợp BM25 (lexical) + TF-IDF/SBERT (semantic) bằng thuật toán Fusion.
    Ưu tiên exact-match lên đầu danh sách (Surface-form exact match priority).
    """

    def __init__(self, alpha=0.5):
        """alpha: trọng số BM25 (1-alpha: trọng số semantic)"""
        self.alpha = alpha
        self.bm25 = BM25Index()
        self.semantic = TFIDFSemanticIndex()
        self.documents = []
        self.additives = []

    def build_index(self, kb_additives):
        """Xây dựng chỉ mục từ danh sách phụ gia."""
        self.additives = kb_additives
        self.documents = []
        for add in kb_additives:
            doc_parts = [
                add.get("ins_code", ""),
                add.get("e_number", ""),
                add.get("canonical_name", ""),
                add.get("canonical_name_en", ""),
                add.get("function", ""),
                " ".join(add.get("common_names", [])),
            ]
            self.documents.append(" ".join(doc_parts))

        self.bm25.index(self.documents)
        self.semantic.index(self.documents)

    def retrieve(self, query, top_k=5):
        """
        Truy xuất lai: BM25 + Semantic + Fusion.
        Trả về danh sách (additive, fused_score, retrieval_method).
        """
        if not self.additives:
            return []

        bm25_results = self.bm25.search(query, top_k=top_k)
        sem_results = self.semantic.search(query, top_k=top_k)

        bm25_max = max((s for _, s in bm25_results), default=1.0) or 1.0
        sem_max = max((s for _, s in sem_results), default=1.0) or 1.0

        score_map = {}
        for idx, score in bm25_results:
            norm_score = score / bm25_max
            score_map[idx] = {"bm25": norm_score, "semantic": 0.0}
        for idx, score in sem_results:
            norm_score = score / sem_max
            if idx not in score_map:
                score_map[idx] = {"bm25": 0.0, "semantic": 0.0}
            score_map[idx]["semantic"] = norm_score

        fused = []
        for idx, scores in score_map.items():
            fused_score = self.alpha * scores["bm25"] + (1 - self.alpha) * scores["semantic"]
            method_parts = []
            if scores["bm25"] > 0:
                method_parts.append(f"BM25={scores['bm25']:.2f}")
            if scores["semantic"] > 0:
                method_parts.append(f"Semantic={scores['semantic']:.2f}")
            fused.append((idx, fused_score, " + ".join(method_parts)))

        fused.sort(key=lambda x: x[1], reverse=True)
        return [(self.additives[idx], score, method) for idx, score, method in fused[:top_k]]

    def get_engine_info(self):
        """Trả về thông tin engine đang sử dụng."""
        if self._has_advanced():
            return "Whoosh BM25 + SentenceTransformer all-MiniLM-L6-v2 + FAISS"
        return "BM25 (numpy) + TF-IDF Cosine Similarity (numpy)"

    def _has_advanced(self):
        return _HAS_SBERT and self.semantic._use_dense


# =====================================================================
# AGENT 2: SELECTOR AGENT (Exact-match preference + Specificity Rule)
# =====================================================================

# Bảng phân cấp ontology cho specificity rule
_ONTOLOGY_DEPTH = {
    "Chất nhũ hóa": 1,
    "Chất tạo xốp": 1,
    "Chất bảo quản": 1,
    "Chất điều vị": 1,
    "Phẩm màu tổng hợp": 1,
    "Chất tạo ngọt tổng hợp": 1,
    "Chất điều chỉnh độ acid": 1,
    "Chất làm dày": 1,
}


def compute_specificity_score(additive):
    """
    Quy tắc Độ đặc hiệu (Specificity Rule):
    Ưu tiên thực thể cụ thể hơn (tầng lá) so với nhóm chung.
    Tiêu chí:
      - Có mã INS cụ thể (3-4 chữ số) -> cụ thể hơn
      - Có suffix (i, ii, iii) -> cụ thể hơn (biến thể con)
      - Tên dài hơn -> thường cụ thể hơn
      - FoodOn CURIE khác 00000000 -> đã được ánh xạ ontology
    """
    score = 0.0
    ins = additive.get("ins_code", "")

    if re.match(r'INS\s*\d{3,4}', ins):
        score += 3.0
    if re.search(r'[ivx]+$', ins, re.IGNORECASE):
        score += 1.0

    curie = additive.get("foodon_curie", "FOODON:00000000")
    if curie and curie != "FOODON:00000000":
        score += 2.0

    name = additive.get("canonical_name", "")
    score += min(len(name) / 20.0, 1.5)

    func = additive.get("function", "")
    if func not in _ONTOLOGY_DEPTH:
        score += 0.5

    return score


def selector_agent(candidates, query):
    """
    Selector Agent: Chọn ứng viên tốt nhất từ danh sách candidates.
    Quy tắc 1: Exact-match preference (ưu tiên khớp chính xác bề mặt)
    Quy tắc 2: Specificity rule (ưu tiên thực thể tầng lá cụ thể nhất)
    """
    if not candidates:
        return None, 0.0, "Không tìm thấy ứng viên."

    query_lower = query.lower()
    numbers = re.findall(r'\d{3,4}', query_lower)

    scored = []
    for additive, retrieval_score, method in candidates:
        ins_code = additive.get("ins_code", "").lower()
        e_number = additive.get("e_number", "").lower()
        name = additive.get("canonical_name", "").lower()
        synonyms = [s.lower() for s in additive.get("common_names", []) if s]

        exact_bonus = 0.0

        for num in numbers:
            if num in ins_code:
                exact_bonus = 3.0
                break
            if e_number and num in e_number:
                exact_bonus = 2.5
                break

        if name and name in query_lower:
            exact_bonus = max(exact_bonus, 2.0)
        for syn in synonyms:
            if syn and syn in query_lower:
                exact_bonus = max(exact_bonus, 1.5)
                break

        specificity = compute_specificity_score(additive)

        final_score = retrieval_score * 2.0 + exact_bonus * 3.0 + specificity * 0.5
        scored.append((additive, final_score, method, exact_bonus, specificity))

    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0]
    additive, final_score, method, exact_bonus, specificity = best

    if exact_bonus >= 2.5:
        rationale_prefix = "Exact-match INS/E-number"
    elif exact_bonus >= 1.5:
        rationale_prefix = "Exact-match tên/từ đồng nghĩa"
    else:
        rationale_prefix = "Hybrid Retrieval (BM25 + Semantic)"

    specificity_note = f" | Specificity={specificity:.1f}"
    rationale = f"{rationale_prefix} [{method}]{specificity_note}"

    return additive, final_score, rationale


# =====================================================================
# AGENT 3: LLM SCORER (Confidence Calibration τ)
# =====================================================================

def llm_scorer(additive, mention_text, exact_bonus, retrieval_score):
    """
    Tính điểm tin cậy τ ∈ [0, 1] dựa trên mức độ tương đồng.
    Kết hợp: exact match weight + retrieval score + specificity + penalty factors.
    """
    if additive is None:
        return 0.50

    tau = 0.50

    ins_code = additive.get("ins_code", "")
    mention_lower = mention_text.lower()
    numbers = re.findall(r'\d{3,4}', mention_lower)

    if numbers and any(n in ins_code for n in numbers):
        tau = 0.97
    elif additive.get("canonical_name", "").lower() in mention_lower:
        tau = 0.94
    else:
        synonyms = [s.lower() for s in additive.get("common_names", []) if s]
        if any(syn in mention_lower for syn in synonyms):
            tau = 0.91
        else:
            tau = 0.60 + min(retrieval_score * 0.3, 0.25)

    curie = additive.get("foodon_curie", "FOODON:00000000")
    if curie and curie != "FOODON:00000000":
        tau = min(tau + 0.02, 1.0)

    if len(mention_lower.split()) <= 1 and not numbers:
        tau = max(tau - 0.05, 0.50)

    return round(tau, 3)


# =====================================================================
# AGENT 4: SYNONYM GENERATOR & FEEDBACK LOOP
# =====================================================================

_SYNONYM_EXPANSIONS = {
    "bột ngọt": ["msg", "mì chính", "monosodium glutamate", "glutamate", "mononatri glutamat"],
    "lecithin": ["phosphatidylcholine", "chất nhũ hóa 322", "soy lecithin", "lecithin đậu nành"],
    "aspartame": ["chất tạo ngọt 951", "aspartam", "sweetener 951"],
    "tartrazine": ["màu vàng 102", "FD&C Yellow No. 5", "tartrazin", "phẩm màu 102"],
    "benzoat": ["natri benzoat", "sodium benzoate", "chất bảo quản 211"],
    "baking soda": ["natri bicacbonat", "bột nở", "sodium bicarbonate", "natri hydro carbonat"],
    "bột khai": ["ammonium bicarbonate", "amoni hydro carbonat"],
    "acid citric": ["axit xitric", "citric acid", "chất điều chỉnh độ chua 330"],
    "tinh bột biến tính": ["modified starch", "acetylated starch", "chất làm dày 1420"],
    "monoglyceride": ["mono- và diglycerid", "chất nhũ hóa 471", "diglyceride"],
}


def synonym_generator(mention_text, failed_query):
    """
    Agent 4: Sinh từ đồng nghĩa mới khi τ < 0.70.
    Trả về danh sách query mở rộng để Hybrid Retriever tìm kiếm lần 2.
    """
    mention_lower = mention_text.lower()
    expanded_queries = []

    for key, synonyms in _SYNONYM_EXPANSIONS.items():
        if key in mention_lower:
            for syn in synonyms:
                expanded_queries.append(syn)
            break
        for syn in synonyms:
            if syn in mention_lower:
                expanded_queries.append(key)
                for other_syn in synonyms:
                    if other_syn != syn:
                        expanded_queries.append(other_syn)
                break
        if expanded_queries:
            break

    numbers = re.findall(r'\d{3,4}', mention_lower)
    if numbers and not expanded_queries:
        for num in numbers:
            expanded_queries.append(f"INS {num}")
            expanded_queries.append(f"E{num}")

    return expanded_queries


# =====================================================================
# ENTRY POINT: 4-AGENT ENTITY LINKING PIPELINE
# =====================================================================

_GLOBAL_RETRIEVER = None


def get_hybrid_retriever(kb_additives):
    """Lấy hoặc khởi tạo Hybrid Retriever singleton."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None or len(_GLOBAL_RETRIEVER.additives) != len(kb_additives):
        _GLOBAL_RETRIEVER = HybridRetriever(alpha=0.5)
        _GLOBAL_RETRIEVER.build_index(kb_additives)
    return _GLOBAL_RETRIEVER


def foodontorag_4agent_linking(mention, kb_additives):
    """
    Pipeline 4-Agent FoodOntoRAG Entity Linking đầy đủ:
      Agent 1: Hybrid Retriever (BM25 + TF-IDF/SBERT) -> candidates
      Agent 2: Selector Agent (Exact-match + Specificity) -> best candidate
      Agent 3: LLM Scorer -> confidence τ
      Agent 4: Synonym Generator & Feedback Loop (nếu τ < 0.70)
    """
    raw_text = mention["raw_mention"]
    retriever = get_hybrid_retriever(kb_additives)

    # === Agent 1: Hybrid Retrieval ===
    candidates = retriever.retrieve(raw_text, top_k=5)

    # === Agent 2: Selector Agent ===
    best_additive, selector_score, selector_rationale = selector_agent(candidates, raw_text)

    # === Agent 2b: Kiểm tra có exact match thực sự không ===
    has_exact_match = False
    if best_additive:
        query_lower = raw_text.lower()
        numbers = re.findall(r'\d{3,4}', query_lower)
        ins = best_additive.get("ins_code", "").lower()
        e_num = best_additive.get("e_number", "").lower()
        canonical = best_additive.get("canonical_name", "").lower()
        synonyms = [s.lower() for s in best_additive.get("common_names", []) if s]

        if numbers and any(n in ins or (e_num and n in e_num) for n in numbers):
            has_exact_match = True
        elif canonical and canonical in query_lower:
            has_exact_match = True
        elif any(syn and syn in query_lower for syn in synonyms):
            has_exact_match = True

    # === Agent 3: LLM Scorer ===
    exact_bonus = 3.0 if has_exact_match else 0.0
    tau = llm_scorer(best_additive, raw_text, exact_bonus, selector_score)

    # === Agent 4: Synonym Generator & Feedback Loop ===
    feedback_note = ""
    if tau < 0.70:
        expanded_queries = synonym_generator(raw_text, raw_text)
        for eq in expanded_queries:
            retry_candidates = retriever.retrieve(eq, top_k=3)
            if retry_candidates:
                retry_best, retry_score, retry_rationale = selector_agent(retry_candidates, eq)
                retry_tau = llm_scorer(retry_best, raw_text, 0, retry_score)
                if retry_tau > tau:
                    best_additive = retry_best
                    tau = retry_tau
                    selector_rationale = f"Synonym Feedback Loop [{eq}] -> {retry_rationale}"
                    feedback_note = f" (Feedback loop: mở rộng '{eq}')"
                    break

    # Nếu không có exact match (INS/tên/synonym), loại bỏ false positive từ retriever
    if best_additive is not None and not has_exact_match and tau < 0.90:
        best_additive = None

    if best_additive is None and mention.get("category") == "ingredient":
        tau = 0.85
        rationale = "Thành phần nguyên liệu tự nhiên (không thuộc nhóm phụ gia bắt buộc kiểm soát ADI)."
    elif best_additive is None and mention.get("category") == "allergen":
        tau = 0.90
        rationale = "Cụm cảnh báo dị nguyên (Allergen advice) cần thẩm định lâm sàng."
    elif best_additive is None:
        tau = 0.80
        rationale = "Không tìm thấy thực thể phụ gia phù hợp trong cơ sở tri thức."
    else:
        curie = best_additive.get("foodon_curie", "") if best_additive else ""
        curie_info = f" [{curie}]" if curie and curie != "FOODON:00000000" else ""
        rationale = f"{selector_rationale}{curie_info}{feedback_note}"

    engine_info = retriever.get_engine_info()

    return {
        "raw_mention": raw_text,
        "category": mention.get("category", "ingredient"),
        "matched_entity": best_additive,
        "confidence_tau": tau,
        "rationale": rationale,
        "retrieval_engine": engine_info,
    }
