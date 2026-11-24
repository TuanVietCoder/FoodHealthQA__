"""Lớp bằng chứng cho giao diện hỏi đáp: đánh số trích dẫn, danh sách nguồn, nhãn lĩnh vực, lý do thiếu bằng chứng.

Không gọi LLM và không đọc ảnh; chỉ biến kết quả đã có của pipeline (câu trả lời có thẻ [MÃ_NGUỒN],
các đoạn đã truy xuất, kết quả guardrail) thành dữ liệu để hiển thị:

    numbered = build_cited_answer(answer, passages, ins_codes)
    numbered["text"]     # câu trả lời, thẻ [MÃ_NGUỒN] đã đổi thành [1], [2]...
    numbered["sources"]  # [{n, citation_id, title, scope, locator, quote, url, tier, tier_label, verified}]
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from modules.benchmark import RECOGNIZED_CITATION_SOURCES, canonicalize_citation, is_recognized_citation

_ROOT = Path(__file__).resolve().parent.parent
_CATALOG_PATH = _ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json"

_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_PREFIX = re.compile(r"^(?:nguồn|nguon|source|ref)\s*:\s*", re.I)
QUOTE_MAX_CHARS = 420
ABSTAIN_THRESHOLD = 0.65  # mặc định; luôn đọc lại từ pipeline qua _abstain_threshold()


def _abstain_threshold() -> float:
    """Dùng chung đúng ngưỡng của tầng guardrail (modules.pipeline.CONFIDENCE_ABSTAIN_THRESHOLD)."""
    try:
        from modules.pipeline import CONFIDENCE_ABSTAIN_THRESHOLD
        return float(CONFIDENCE_ABSTAIN_THRESHOLD)
    except Exception:
        return ABSTAIN_THRESHOLD

# Mức tin cậy của nguồn theo bản chất tài liệu (không phải độ chắc chắn của từng nhận định).
_TIER_RULES = (
    (("VN_TT24", "CODEX_", "US_FDA", "US_CFR", "US_CODE", "EU_REG"), 1,
     "Cao: văn bản pháp quy / tiêu chuẩn chính thức"),
    (("JECFA", "WHO_FAO", "EFSA", "FAO_WHO", "WHO_JEMRA", "IARC", "WHO_"), 2,
     "Cao: đánh giá khoa học của cơ quan chuyên môn quốc tế"),
    (("USDA", "OFF_", "OPEN_FOOD"), 3,
     "Trung bình: cơ sở dữ liệu tham khảo (giá trị ước tính)"),
)


def source_tier(citation_id: str) -> tuple[int, str]:
    cid = str(citation_id or "").upper()
    for prefixes, tier, label in _TIER_RULES:
        if cid.startswith(prefixes):
            return tier, label
    return 4, "Chưa phân loại"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    try:
        return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_RAW = _ROOT / "knowledge_base" / "sources" / "raw"

# Bản lưu cục bộ của tài liệu gốc (knowledge_base/sources/raw). Khớp theo URL trước (chính xác nhất:
# mỗi bài EFSA / trang JECFA có file riêng), rồi theo mã nguồn. Chỉ trả về file thực sự tồn tại.
_LOCAL_BY_URL = (
    ("aspartame-and-other-sweeteners-food", "external/FDA_SWEETENERS_IN_FOOD.html"),
    ("efsa.2019.5674", "external/efsa_2019_5674.pdf"),
    ("efsa.2008.660", "external/efsa_2008_660.pdf"),
    ("efsajournal/pub/4786", "external/efsa_2017_4786.pdf"),
    ("efsajournal/pub/6585", "external/efsa_2021_6585.pdf"),
    ("efsa.2016.4438", "external/efsa_2016_4438.pdf"),
    ("efsa.2017.4786","external/efsa_2017_4786.pdf"),
    ("efsa.2021.6585", "external/efsa_2021_6585.pdf"),
    ("efsa.2015.4102","external/EFSA_Journal_2015_Scientific_Opinion_safety_caffeine.pdf"),
    ("j.ajkd.2020.05.006", "external/KDOQI_Nutrition_2020.pdf"),
    ("jecfa87_4-13-june-2019", "external/jecfa87_2019_summary.pdf"),
    ("CELEX:02011R1169", "external/CELEX_02011R1169_2025.pdf"),
    ("efsajournal/pub/7594", "external/efsa_2022_7594.pdf"),
    ("efsa.2022.7594", "external/efsa_2022_7594.pdf"),
    ("efsajournal/pub/1331", "external/efsa_2009_1331.pdf"),
    ("efsajournal/pub/3765", "external/efsa_2014_3765.pdf"),
    ("efsajournal/pub/1327", "external/efsa_2009_1327.pdf"),
    ("efsajournal/pub/1332", "external/efsa_2009_1332.pdf"),
    ("efsajournal/pub/1328", "external/efsa_2009_1328.pdf"),
    ("efsajournal/pub/1329", "external/efsa_2009_1329.pdf"),
    ("efsajournal/pub/1853", "external/efsa_2010_1853.pdf"),
    ("efsajournal/pub/4787", "external/efsa_2017_4787.pdf"),
    ("efsa.2017.4787", "external/efsa_2017_4787.pdf"),
    ("efsajournal/pub/2392", "external/efsa_2011_2392.pdf"),
    ("efsa.2011.2392", "external/efsa_2011_2392.pdf"),
    ("efsajournal/pub/2004", "external/efsa_2011_2004.pdf"),
    ("efsajournal/pub/1537", "external/efsa_2010_1537.pdf"),
    ("efsajournal/pub/9480", "external/efsa_2025_9480.pdf"),
    ("efsajournal/pub/9317", "external/efsa_2025_9317.pdf"),
    ("efsa.2025.9317", "external/efsa_2025_9317.pdf"),
    ("PMC7009848", "external/efsa_2017_4910.pdf"),  # bản PMC của EFSA Journal 2017;15(7):4910 (glutamat)
    ("PMC7009865", "external/efsa_2017_4911.pdf"),  # bản PMC của EFSA Journal 2017;15(7):4911 (tinh bột biến tính)
    ("titanium-dioxide-e171-no-longer", "external/EFSA_2021_TIO2_NEWS.html"),
    ("9789240072510", "external/JEMRA_PART3.pdf"),
    ("1st-allergen-summary-report", "external/JEMRA_2021_PART1_SUMMARY.pdf"),
)
_LOCAL_BY_CITATION = {
    "VN_TT24_2019": "vn/09_VBHN_BYT_2024.pdf",
    "VN_QCVN_11_1_2012": "external/QCVN_11_1_2012_BYT.pdf",
    "VN_QCVN_11_4_2012": "external/QCVN_11_4_2012_BYT.pdf",
    "CODEX_STAN_192_1995": "external/cxs_192e.pdf",
    "CODEX_CXS_1_1985": "external/cxs_001e.pdf",
    "CODEX_CXG_36_1989": "external/cxg_036e.pdf",
    "CODEX_CAC_GL_23_1997": "external/cxg_023e.pdf",
    "CODEX_CXS_72_1981": "external/Codex_CXS_72_1981.pdf",
    "CODEX_CXS_74_1981": "external/CXS_74_1981.pdf",
    "WHO_COMPLEMENTARY_FEEDING_2023": "external/complementary_feeding_infants_young_children_6_23_months.pdf",
    "WHO_JEMRA_PRIORITY_ALLERGENS": "external/JEMRA_2021_PART1_SUMMARY.pdf",
    "WHO_NSS_GUIDELINE_2023": "external/Bookshelf_NBK592258.pdf",  # trang bìa: "Use of non-sugar sweeteners WHO guideline" (90 trang)
    "FAO_WHO_JEMRA_PART_3": "external/JEMRA_PART3.pdf",
    "EFSA_2021_TITANIUM_DIOXIDE_E171": "external/efsa_2021_6585.pdf",
    "EFSA_CAFFEINE_2015": "external/EFSA_Journal_2015_Scientific_Opinion_safety_caffeine.pdf",
    "EU_REG_1333_2008_ANNEX_V": "external/EU_REG_1333_2008_ANNEX_V.pdf",
    "IARC_MONOGRAPHS_V134": "external/IARC_V134_ASPARTAME.pdf",
    "US_FDA_ALLERGEN_QA_ED5": "external/FDA_ALLERGEN_QA_ED5.pdf",
    "US_FDA_RED3_REVOCATION": "external/FR_RED3_REVOCATION.pdf",
    "US_FDA_BENZENE_QA": "external/FDA_BENZENE_QA.html",
    "US_FDA_ASPARTAME_SWEETENERS": "external/FDA_SWEETENERS_IN_FOOD.html",
    "US_FDA_21CFR101_54": "external/US_CFR_21_101_54.html",
    "US_FDA_21CFR101_61": "external/US_CFR_21_101_61.html",
    "US_FDA_21CFR172_804": "external/US_CFR_21_172_804.html",
    "US_FDA_FALCPA": "external/US_CODE_21_321.html",
}
_JECFA_CHEMICAL = re.compile(r"/Home/Chemical/(\d+)")


def local_source_file(citation_id: str, url: str = "") -> str:
    """Đường dẫn (tương đối với thư mục gốc dự án) tới bản lưu cục bộ của nguồn, hoặc "" nếu không có."""
    url = str(url or "")
    candidates = []
    m = _JECFA_CHEMICAL.search(url)
    if m:
        candidates.append(f"external/jecfa/Chemical_{m.group(1)}.html")
    candidates += [rel for key, rel in _LOCAL_BY_URL if key in url]
    if citation_id in _LOCAL_BY_CITATION and not m:
        candidates.append(_LOCAL_BY_CITATION[citation_id])
    for rel in candidates:
        path = _RAW / rel
        if path.is_file():
            return path.relative_to(_ROOT).as_posix()
    return ""


@lru_cache(maxsize=64)
def local_text(local_file: str) -> str:
    """Nội dung chữ của bản lưu cục bộ để xem ngay trong app: HTML → chữ thuần (bỏ script/style);
    .doc nhị phân → bản chữ đã chuyển sẵn ở knowledge_base/sources/derived/doc_text/<tên>.txt. PDF → "" (xem theo trang)."""
    path = _ROOT / local_file
    suffix = path.suffix.lower()
    try:
        if suffix in (".html", ".htm"):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return re.sub(r"\n\s*\n+", "\n", soup.get_text("\n")).strip()
        if suffix == ".doc":
            return (_ROOT / "knowledge_base" / "sources" / "derived" / "doc_text" / f"{path.stem}.txt").read_text(
                encoding="utf-8", errors="replace")
    except (OSError, ImportError):
        return ""
    return ""


@lru_cache(maxsize=1)
def _additive_names() -> dict:
    """Mã INS chuẩn hoá → các từ khoá tên (vi + en) dùng để nhận ra đúng dòng của phụ gia trong văn bản gốc."""
    try:
        rows = json.loads((_ROOT / "knowledge_base" / "additives_standard.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for r in rows:
        names = []
        for key in ("canonical_name_vi", "canonical_name_en"):
            text = re.sub(r"\([^)]*\)", " ", str(r.get(key) or ""))
            names += [t.lower()[:7] for t in re.findall(r"[^\W\d_]{5,}", text)]
        out[_norm_code(r.get("ins_code", ""))] = tuple(dict.fromkeys(names))
    return out


def _norm_code(code) -> str:
    return re.sub(r"[()\s]", "", re.sub(r"^(ins|e)\s*", "", str(code or "").strip(), flags=re.I)).lower()


def _row_bounds(words, code_word):
    mid = (code_word[1] + code_word[3]) / 2
    return [w for w in words if abs((w[1] + w[3]) / 2 - mid) < 5]


def _code_matches(word: str, code: str, loose: bool) -> bool:
    w = _norm_code(word.strip(".,;:"))
    if w == code:
        return True
    # Codex viết nhóm phụ gia là 100(i), 420(ii), 160a(i … : khi dò lỏng chấp nhận mã nhóm là tiền tố + phần chữ/ngoặc
    return loose and w.startswith(code) and re.fullmatch(r"[a-z]+", w[len(code):] or "0") is not None


def _match_row(words, code: str, names, loose: bool):
    """Dòng của phụ gia trên trang. Chặt: mã INS đứng đầu dòng (hoặc sau số thứ tự) và dòng có tên phụ gia.
    Lỏng (khi không có tên trong sổ đăng ký hoặc tên viết khác): mã đứng đầu dòng, chấp nhận 100(i) cho mã 100."""
    for cw in words:
        if not _code_matches(cw[4], code, loose):
            continue
        row = sorted(_row_bounds(words, cw), key=lambda w: w[0])
        if row.index(cw) > 1:
            continue  # mã nằm giữa câu (ví dụ "(INS 110)" trong chú thích), không phải dòng của phụ gia
        text = " ".join(w[4] for w in row).lower()
        if loose or (names and any(n in text for n in names)):
            return row
    return None


@lru_cache(maxsize=8)
def _section_ranges(local_file: str) -> dict:
    """Khoảng trang của từng Phụ lục trong TT24 (dò theo tiêu đề 'PHỤ LỤC 2A'… ở đầu dòng)."""
    try:
        import fitz
        with fitz.open(_ROOT / local_file) as doc:
            starts = {}
            for i, page in enumerate(doc):
                for m in re.finditer(r"(?m)^\s*PHỤ LỤC\s+(\d+[AB]?)\s*$", page.get_text()):
                    starts.setdefault(m.group(1), i + 1)
            order = sorted(starts.items(), key=lambda kv: kv[1])
            return {k: (s, (order[j + 1][1] - 1) if j + 1 < len(order) else len(doc))
                    for j, (k, s) in enumerate(order)}
    except Exception:
        return {}


_LOCATOR_SECTION = re.compile(r"#PhuLuc(\d+[AB]?)", re.I)


@lru_cache(maxsize=256)
def locate_pages(local_file: str, ins_code: str, max_pages: int = 4, locator: str = "") -> tuple:
    """Trang của file PDF cục bộ có dòng chứa đúng mã INS (dò theo văn bản, không phải số trang đã kiểm chứng).
    `locator` dạng …#PhuLuc2A giới hạn trong đúng Phụ lục đó. Trả về ((số_trang, đoạn_dòng), ...); rỗng nếu không dò được."""
    code = _norm_code(ins_code)
    names = _additive_names().get(code, ())
    path = _ROOT / local_file
    if not code or path.suffix.lower() != ".pdf" or not path.is_file():
        return ()
    lo, hi = 1, 10 ** 6
    m = _LOCATOR_SECTION.search(locator or "")
    if m:
        lo, hi = _section_ranges(local_file).get(m.group(1).upper(), (1, 10 ** 6))
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
    except Exception:
        return ()
    with doc:
        for loose in ((False, True) if names else (True,)):
            hits = []
            for i in range(lo - 1, min(len(doc), hi)):
                words = doc[i].get_text("words")
                row = _match_row(words, code, names, loose)
                if row:
                    hits.append((i + 1, " ".join(w[4] for w in row)[:110]))
                    if len(hits) >= max_pages:
                        break
            if hits:
                return tuple(hits)
    return ()


def pdf_page_count(local_file: str) -> int:
    try:
        import fitz
        with fitz.open(_ROOT / local_file) as doc:
            return len(doc)
    except Exception:
        return 0


def render_pdf_page(local_file: str, page_no: int, dpi: int = 100) -> bytes:
    """Ảnh PNG nguyên một trang PDF của bản lưu cục bộ (để đọc tài liệu ngay trong app). b"" nếu lỗi."""
    try:
        import fitz
        with fitz.open(_ROOT / local_file) as doc:
            return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")
    except Exception:
        return b""


def render_pdf_row(local_file: str, page_no: int, ins_code: str, band_pt: float = 150.0, dpi: int = 130) -> bytes:
    """Ảnh PNG vùng quanh dòng của phụ gia trên trang PDF, có tô sáng dòng đó. b"" nếu không dựng được."""
    code = _norm_code(ins_code)
    names = _additive_names().get(code, ())
    try:
        import fitz
        with fitz.open(_ROOT / local_file) as doc:
            page = doc[page_no - 1]
            words = page.get_text("words")
            target = _match_row(words, code, names, False) or _match_row(words, code, names, True)
            clip = page.rect
            if target:
                y0 = min(w[1] for w in target); y1 = max(w[3] for w in target)
                clip = fitz.Rect(page.rect.x0, max(page.rect.y0, y0 - band_pt), page.rect.x1,
                                 min(page.rect.y1, y1 + band_pt))
                for w in target:
                    page.add_highlight_annot(fitz.Rect(w[:4]))
            return page.get_pixmap(dpi=dpi, clip=clip, annots=True).tobytes("png")
    except Exception:
        return b""


_TT24_SCOPES = {
    "1": "Phụ lục 1, Bảng 1: Danh mục phụ gia thực phẩm được phép sử dụng tại Việt Nam (Positive List).",
    "2A": "Phụ lục 2A: Mức tối đa (ML) của phụ gia theo nhóm thực phẩm (hài hòa theo Codex GSFA).",
    "2B": "Phụ lục 2B: Mức tối đa (ML) theo nhóm thực phẩm riêng của Việt Nam.",
    "3": "Phụ lục 3: Phụ gia được dùng theo thực hành sản xuất tốt (GMP).",
}


@lru_cache(maxsize=1)
def _jecfa_records() -> dict:
    """Mã INS chuẩn hoá → bản ghi kiểm chứng JECFA (chemical_id, url, file đã lưu trong raw/external/jecfa)."""
    out = {}
    for fname in ("jecfa_adi_verification.json", "jecfa_adi_verification_more.json"):
        try:
            data = json.loads((_ROOT / "knowledge_base" / "sources" / "derived" / fname).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for ins, rec in (data.items() if isinstance(data, dict) else []):
            if isinstance(rec, dict):
                out[_norm_code(ins)] = rec
    return out


def resolve_source(citation_id: str, locator: str | None = None, ins_codes=None) -> dict:
    """Tên tài liệu, mục/điều khoản, URL cho một mã nguồn; ưu tiên khớp đúng locator của đoạn trích.

    Passage JECFA có locator dạng JECFA_EVALUATIONS#<INS> không nằm trong catalog: tra theo bản ghi kiểm chứng của đúng
    mã INS (trang JECFA của đúng phụ gia), KHÔNG lấy mục JECFA đầu tiên (từng khiến mọi phụ gia trỏ về trang Aspartame)."""
    catalog = load_catalog()
    entry = catalog.get(locator or "") if locator else None
    meta = RECOGNIZED_CITATION_SOURCES.get(citation_id, {})
    if not entry and citation_id == "JECFA_EVALUATIONS":
        rec = next((_jecfa_records().get(_norm_code(c)) for c in (ins_codes or []) if _jecfa_records().get(_norm_code(c))), None)
        code = next(iter(ins_codes or []), "")
        if rec and not rec.get("passed"):
            # Chưa tìm được trang JECFA cho phụ gia này → không có tên/năm/ADI để dẫn; nói rõ thay vì in dấu "?"
            return {"title": f"WHO/FAO JECFA (INS {code})",
                    "scope": "Chưa có trang đánh giá JECFA đã kiểm chứng cho phụ gia này; không có giá trị ADI để dẫn.",
                    "url": "", "verified": False, "local_file": ""}
        if rec:
            local = _RAW / "external" / "jecfa" / str(rec.get("file") or "")
            name = str(rec.get("jecfa_name") or "").title()
            year, adi = rec.get("latest_evaluation_year"), rec.get("latest_adi")
            parts = ([f"Đánh giá JECFA gần nhất {year}"] if year else []) + ([f"ADI: {adi}"] if adi else [])
            return {
                "title": "WHO/FAO JECFA" + (f" — {name}" if name else "") + f" (INS {code})",
                "scope": ("; ".join(parts) + ".") if parts else "Trang JECFA không ghi năm đánh giá hay ADI.",
                "url": rec.get("url", ""), "verified": True,
                "local_file": local.relative_to(_ROOT).as_posix() if local.is_file() else "",
            }
        return {"title": meta.get("title") or "JECFA — cơ sở dữ liệu đánh giá phụ gia (chưa có trang riêng cho phụ gia này)",
                "scope": "", "url": "", "verified": False, "local_file": ""}
    scope_override = ""
    if not entry:
        entry = next((v for v in catalog.values() if v.get("citation_id") == citation_id), None)
        if citation_id == "VN_TT24_2019" and locator:
            sec = _LOCATOR_SECTION.search(locator)
            scope_override = _TT24_SCOPES.get(sec.group(1).upper(), "") if sec else ""
    return {
        "title": (entry or {}).get("title") or meta.get("title") or citation_id,
        "scope": scope_override or (entry or {}).get("scope", ""),
        "url": (entry or {}).get("url", ""),
        "verified": bool((entry or {}).get("verified_against")),
        "local_file": local_source_file(citation_id, (entry or {}).get("url", "")),
    }


def _clip(text: str, limit: int = QUOTE_MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _fallback_passage(citation_id: str, ins_codes) -> dict | None:
    """Khi không có đoạn nào được truy xuất cho mã nguồn này, lấy một đoạn trong kho khớp mã nguồn và mã INS."""
    try:
        from modules.passage_retriever import get_passage_retriever, normalize_ins
        retriever = get_passage_retriever()
    except Exception:
        return None
    wanted = {normalize_ins(c) for c in (ins_codes or []) if c}
    candidates = [p for p in retriever.passages if p.get("citation_id") == citation_id]
    for p in candidates:
        if wanted & set(p.get("ins_codes", [])):
            return p
    return candidates[0] if candidates else None


def build_cited_answer(answer: str | None, passages=None, ins_codes=None) -> dict:
    """Đổi thẻ [MÃ_NGUỒN] trong câu trả lời thành số [n] và dựng danh sách nguồn tương ứng.

    Thẻ không thuộc danh mục nguồn được công nhận được giữ nguyên (không tự gán số cho nguồn bịa).
    """
    passages = list(passages or [])
    order: list[str] = []

    def number_for(cid: str) -> int:
        if cid not in order:
            order.append(cid)
        return order.index(cid) + 1

    def repl(match: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"[,;&]", _PREFIX.sub("", match.group(1).strip())) if p.strip()]
        if not parts or not all(is_recognized_citation(p) for p in parts):
            return match.group(0)
        nums = sorted({number_for(canonicalize_citation(p)) for p in parts})
        return "".join(f"[{n}]" for n in nums)

    text = _BRACKET.sub(repl, str(answer or ""))

    sources = []
    for cid in order:
        passage = next((p for p in passages if canonicalize_citation(p.get("citation_id", "")) == cid), None)
        if passage is None:
            passage = _fallback_passage(cid, ins_codes)
        locator = (passage or {}).get("locator")
        info = resolve_source(cid, locator, (passage or {}).get("ins_codes") or ins_codes)
        tier, tier_label = source_tier(cid)
        quote_text = (passage or {}).get("text", "")
        if not quote_text and info.get("scope"):
            quote_text = f"Quy chuẩn / Phạm vi thẩm định: {info['scope']}"
        sources.append({
            "n": order.index(cid) + 1,
            "citation_id": cid,
            "title": info["title"],
            "scope": info["scope"],
            "locator": locator or "",
            "quote": _clip(quote_text),
            "url": info["url"],
            "local_file": info["local_file"],
            "ins_codes": list((passage or {}).get("ins_codes") or [])[:8] or [str(c) for c in (ins_codes or []) if c][:8],
            "tier": tier,
            "tier_label": tier_label,
            "verified": info["verified"],
        })
    return {"text": text, "sources": sources}


# ── Khung "máy đọc được gì trên nhãn": dựng bằng code từ kết quả liên kết, không do LLM viết ────────────

_ORIGIN_VI = {
    "explicit_on_label": "mã INS/E ghi trên nhãn",
    "inferred_from_name": "suy ra từ tên chất, nhãn không ghi mã",
}


def _source_view(citation_id: str, locator: str | None, code: str, *, quote: str = "") -> dict:
    """Một nguồn ở dạng app.render_source_view dùng được (cùng khoá với build_cited_answer)."""
    info = resolve_source(citation_id, locator, [code])
    tier, tier_label = source_tier(citation_id)
    return {"citation_id": citation_id, "title": info["title"], "scope": info["scope"], "locator": locator or "",
            "quote": _clip(quote), "url": info["url"], "local_file": info["local_file"], "ins_codes": [code],
            "tier": tier, "tier_label": tier_label, "verified": info["verified"]}


def _adi_source_view(adi: dict, code: str, adi_text: str) -> dict | None:
    """Nguồn của ĐÚNG con số ADI đang hiển thị (không lấy nguồn khác: Tartrazine KB dùng EFSA 0–7,5, còn trang JECFA ghi
    0–10). Ưu tiên bản lưu của nguồn mà KB ghi; nếu không có thì dùng trang JECFA của đúng mã INS khi ADI là của JECFA."""
    source_id, url = str(adi.get("source_id") or ""), str(adi.get("source_url") or "")
    local = local_source_file(source_id, url)
    if local:
        entry = next((v for v in load_catalog().values() if v.get("url") and v["url"] == url), None)
        tier, tier_label = source_tier(source_id)
        locator = str(adi.get("source_locator") or "")
        title = (entry or {}).get("title") or re.split(r"\.\s", locator, maxsplit=1)[0][:100] or source_id
        return {"role": "ADI", "citation_id": source_id, "title": title,
                "scope": str(adi.get("authority") or ""), "locator": str(adi.get("source_locator") or ""),
                "quote": _clip(adi_text), "url": url, "local_file": local, "ins_codes": [code], "tier": tier,
                "tier_label": tier_label, "verified": adi.get("verification_status") == "source_checked"}
    record = _jecfa_records().get(_norm_code(code))
    if record and record.get("passed") and record.get("latest_adi") \
            and str(adi.get("authority") or "").upper().lstrip().startswith(("JECFA", "WHO")):
        return {**_source_view("JECFA_EVALUATIONS", None, code, quote=f"ADI: {record.get('latest_adi', '')}"), "role": "ADI"}
    return None


def build_recognition_summary(linked_results) -> list[dict]:
    """Mỗi phụ gia trên nhãn một dòng: chữ máy đọc được → chất được nhận diện (mã INS, tên) → ADI → nguồn.

    Gồm cả cụm được coi là phụ gia nhưng KHÔNG nhận diện được trong KB (status="unlinked") để người kiểm duyệt thấy
    chỗ máy chưa chắc chắn. Chỉ lấy dữ liệu đã có trong kết quả liên kết và KB; không tạo thông tin mới."""
    rows: list[dict] = []
    seen: set[tuple] = set()
    for item in linked_results or []:
        entity = item.get("matched_entity")
        raw = str(item.get("raw_mention") or "").strip()
        if not entity:
            if item.get("category") == "additive" and raw and (raw, "") not in seen:
                seen.add((raw, ""))
                rows.append({"status": "unlinked", "raw": raw, "ins_code": "", "name": "", "how": "",
                             "adi_text": "", "sources": []})
            continue
        ins = str(entity.get("ins_code") or "").strip()
        if not ins or (raw, ins) in seen:
            continue
        seen.add((raw, ins))
        code = re.sub(r"^(?:INS|E)\s*", "", ins, flags=re.I).strip()
        name = entity.get("canonical_name") or entity.get("canonical_name_vi") or entity.get("canonical_name_en") or ""
        adi = entity.get("adi_assessment") or {}
        adi_text = adi.get("display_text") or entity.get("adi_value") or ""
        sources: list[dict] = []
        adi_src = _adi_source_view(adi, code, adi_text) if adi_text else None
        if adi_src:
            sources.append(adi_src)
        claim = entity.get("regulatory_claim") or {}
        if claim.get("verification_status", "").startswith("source_checked"):
            sources.append({**_source_view("VN_TT24_2019", "VN_TT24_2019#PhuLuc1_Bang1", code,
                                           quote=str(claim.get("claim") or "")), "role": "Có trong danh mục TT 24/2019"})
        rows.append({
            "status": "linked", "raw": raw, "ins_code": ins, "name": str(name),
            "how": _ORIGIN_VI.get(item.get("code_origin"), ""),
            "adi_text": str(adi_text), "sources": sources,
        })
    return rows


def condition_gap_groups(evaluation) -> list[str]:
    """Tên các bệnh nền người dùng khai mà KB không có dữ liệu phụ gia nào (từ abstention.condition_notes).
    Dùng để KHÔNG hiển thị ô "an toàn/không thấy điểm cần lưu ý" như thể bệnh nền đã được xét."""
    return [n.get("group", "") for n in (evaluation or {}).get("population_notes") or []
            if n.get("code") == "CONDITION_NO_ADDITIVE_EVIDENCE"]


def friendly_refusal_reason(reason: str) -> str:
    """Lý do kỹ thuật khi cổng nguồn từ chối câu trả lời của mô hình → một mệnh đề dễ hiểu cho người dùng."""
    r = str(reason or "")
    if "khác nhãn của bộ luật" in r:
        return "phần giải thích do mô hình viết chưa khớp với kết luận ở trên"
    if "ngoài danh sách" in r:
        return "phần giải thích dẫn tới tài liệu không nằm trong số tài liệu hệ thống đã tra cứu cho câu hỏi này"
    if "không dẫn nguồn" in r or "không truy xuất được nguồn" in r:
        return "phần giải thích không dẫn được tài liệu nào để kiểm chứng"
    return "hệ thống chưa chắc phần giải thích này đáng tin"


def _join_names(items, limit=3) -> str:
    items = list(dict.fromkeys(str(i).strip() for i in items if i and str(i).strip()))
    shown, rest = items[:limit], len(items) - limit
    text = ", ".join(shown[:-1]) + " và " + shown[-1] if len(shown) > 1 else (shown[0] if shown else "")
    return text + (f" (và {rest} chất khác)" if rest > 0 else "")


def build_bottom_line(evaluation, rec_rows, profile_text="", declared=False, allergen_names=None) -> str:
    """Câu "Tóm lại" cuối câu trả lời: VỚI AI · THẤY GÌ · NÊN LÀM GÌ, bằng lời thường, dựng từ kết quả bộ luật
    (không do mô hình ngôn ngữ viết, nên luôn khớp với kết luận). `allergen_names(alerts)` → tên dị nguyên gọn ("sữa, tôm")."""
    ev = evaluation or {}
    status = str(ev.get("status") or "").upper()
    alerts = ev.get("alerts") or []
    rows = list(rec_rows or [])
    who = f"Với hồ sơ của bạn ({profile_text})" if declared and profile_text else "Với thông tin hiện có"
    names_of = lambda group: _join_names([a.get("additive", "") for a in group])  # noqa: E731
    allergens = lambda group: (allergen_names(group) if allergen_names else "") or names_of(group)  # noqa: E731
    doctor = "Nếu đã dùng và thấy khó chịu, hãy gặp bác sĩ."

    if status == "CRITICAL_ALERT":
        return (f"{who}: bạn **không nên dùng** sản phẩm này vì có {names_of(alerts) or 'một chất'} — chất mà quy chuẩn "
                f"chống chỉ định với bệnh bạn đã khai. {doctor}")
    if status == "WARNING":
        age = [a for a in alerts if a.get("label_age_rule")]
        rest = [a for a in alerts if not a.get("label_age_rule")]
        pal = [a for a in rest if a.get("regulation") == "FAO_WHO_JEMRA_PART_3" or "PAL" in str(a.get("message", ""))]
        direct = [a for a in rest if a not in pal]
        if age:
            return f"{who}: **không nên cho bé dùng** sản phẩm này vì nhà sản xuất ghi sản phẩm chưa dành cho độ tuổi của bé."
        if direct:
            return (f"{who}: bạn **không nên dùng** sản phẩm này vì trên nhãn có {allergens(direct)}, trùng với dị ứng bạn đã khai. "
                    f"{doctor}")
        return (f"{who}: nhãn ghi sản phẩm \"có thể chứa\" {allergens(pal)} do dây chuyền sản xuất. Nếu dị ứng nặng, "
                "**tốt nhất không dùng**; nếu chỉ dị ứng nhẹ, hãy hỏi bác sĩ trước khi dùng.")
    if status == "CAUTION":
        infant = [a for a in alerts if a.get("infant_rule")]
        if infant and len(infant) == len(alerts):
            return (f"{who}: **chưa nên cho bé dưới 12 tháng dùng** vì có {names_of(infant)}, không nằm trong danh mục phụ gia "
                    "được phép cho trẻ ở tuổi này. Hãy hỏi bác sĩ nhi khoa.")
        if alerts:
            return (f"{who}: có thể dùng nhưng **nên hạn chế**, vì có {names_of(alerts)} cần dè chừng với bạn. "
                    "Không nên dùng thường xuyên hoặc với lượng lớn.")
        return (f"{who}: **nên dùng ít** vì số liệu dinh dưỡng trên nhãn vượt mức khuyến nghị cho hồ sơ của bạn. "
                "Hãy xem lại khẩu phần trong ngày.")
    if status == "ABSTAIN":
        needs = [g.get("need", "") for g in (ev.get("data_gaps") or []) if g.get("need")]
        if ev.get("scope_issues"):
            return ("**Chưa thể trả lời** vì câu hỏi cần ý kiến y khoa; hệ thống chỉ sàng lọc nhãn thực phẩm. "
                    "Bạn hãy hỏi bác sĩ, dược sĩ hoặc chuyên gia dinh dưỡng.")
        todo = "; ".join(dict.fromkeys(needs)) if needs else "chụp lại nhãn rõ hơn, đủ phần thành phần và cảnh báo dị nguyên"
        return f"**Chưa thể kết luận** vì thông tin trên nhãn chưa đủ. Bạn có thể {todo}."
    gap = condition_gap_groups(ev)
    if gap:
        return (f"**Chưa thể nói sản phẩm có hợp với {_join_names(gap)} hay không** vì hệ thống chưa có dữ liệu về bệnh này. "
                "Bạn nên hỏi bác sĩ hoặc chuyên gia dinh dưỡng trước khi dùng.")
    linked = [r for r in rows if r.get("status") == "linked"]
    unlinked = [r for r in rows if r.get("status") != "linked"]
    seen = _join_names([str(r.get("name", "")).split(" (")[0] for r in linked]) or ""
    if not linked:
        return (f"{who}: máy chưa nhận diện được phụ gia nào trên nhãn nên **chưa thấy gì cần tránh**, nhưng kết quả này "
                "chưa chắc chắn. Hãy tự đọc lại phần thành phần trên bao bì.")
    tail = (f" Có {len(unlinked)} cụm chữ máy chưa nhận diện được, bạn nên tự đối chiếu với nhãn." if unlinked else
            " Máy có thể đọc nhầm chữ, nên hãy so lại với ảnh nhãn nếu bạn dị ứng nặng.")
    return (f"{who}: trong {len(linked)} phụ gia máy đọc được ({seen}), **chưa thấy chất nào cần tránh**. "
            f"Đây là kết quả sàng lọc, chưa phải bảo đảm an toàn.{tail}")


NOT_CLEAR_ENOUGH = "Không đủ dữ liệu rõ nét"


def recognition_headline(mentions, rows, product_name: str = "", max_keywords: int = 6) -> str:
    """Mục [NHẬN DIỆN TỪ ẢNH]: một dòng ngắn tên sản phẩm / mã chất / từ khóa máy đọc được.
    Không đọc được cụm chữ nào → đúng câu quy định, không điền thêm gì."""
    rows = list(rows or [])
    additive_text = {r["raw"].lower() for r in rows}
    keywords = []
    for m in mentions or []:
        raw = str((m or {}).get("raw_mention") or "").strip()
        if raw and raw.lower() not in additive_text and (m or {}).get("category") != "additive" and raw not in keywords:
            keywords.append(raw)
    if not rows and not keywords:
        return NOT_CLEAR_ENOUGH
    parts = []
    if product_name:
        parts.append(f"Sản phẩm: {product_name}")
    linked = [f"{r['ins_code']} ({r['name'].split(' (')[0]})" if r["name"] else r["ins_code"] for r in rows if r["status"] == "linked"]
    if linked:
        parts.append("Mã/phụ gia: " + ", ".join(dict.fromkeys(linked)))
    unlinked = [f"“{r['raw']}”" for r in rows if r["status"] != "linked"]
    if unlinked:
        parts.append("Chưa nhận diện được: " + ", ".join(unlinked))
    if keywords:
        parts.append("Từ khóa khác: " + ", ".join(keywords[:max_keywords]) + ("…" if len(keywords) > max_keywords else ""))
    return "; ".join(parts)


def reference_lines(cited_sources, rows) -> list[dict]:
    """Mục [NGUỒN ĐỐI CHIẾU]: tên tài liệu/cơ sở dữ liệu + mục/bảng/vị trí. Chỉ lấy nguồn đã được dùng thật (nguồn câu trả
    lời trích dẫn, hoặc nguồn của ADI / danh mục mà luật đã tra cho từng phụ gia nhận diện); không có thì trả danh sách rỗng."""
    out, seen = [], set()
    for s in list(cited_sources or []):
        where = s.get("scope") or s.get("locator") or ""
        key = (s["title"], where)
        if key not in seen:
            seen.add(key)
            out.append({"title": s["title"], "where": where, "n": s.get("n"), "from": "câu trả lời"})
    for r in rows or []:
        for s in r.get("sources", []):
            where = s.get("scope") or s.get("locator") or ""
            key = (s["title"], where)
            if key not in seen:
                seen.add(key)
                out.append({"title": s["title"], "where": (f"{s['role']} của {r['ins_code']}" + (f" — {where}" if where else "")),
                            "n": None, "from": "tra cứu theo phụ gia"})
    return out


# ── Nhãn lĩnh vực: cùng bộ nhãn với dataset (domain / domain_vi) ──────────────────────────────

DOMAIN_LABELS = {
    "additive_regulation": "Đánh giá phụ gia",
    "allergen_detection": "Phát hiện dị nguyên",
    "vulnerable_groups": "Nhóm dễ tổn thương",
    "insufficient_data": "Thiếu dữ liệu",
}


def classify_domain(qa: dict | None, evaluation: dict | None, persona: dict | None) -> tuple[str, str]:
    """Gán một trong 4 nhãn lĩnh vực của dataset cho câu hỏi hiện tại.

    Thứ tự ưu tiên theo cách dataset được dựng: thiếu dữ liệu > dị nguyên > nhóm dễ tổn thương > phụ gia.
    """
    qa = qa or {}
    evaluation = evaluation or {}
    persona = persona or {}
    ents = qa.get("entities", {})
    intents = " ".join(qa.get("intents", [])).lower()

    if evaluation.get("status") == "ABSTAIN" or qa.get("knowledge_gaps"):
        code = "insufficient_data"
    elif ents.get("allergens") or "dị nguyên" in intents or persona.get("allergies"):
        code = "allergen_detection"
    elif ents.get("populations") or (
        {str(c).lower() for c in persona.get("conditions") or []} - {"healthy", "healthy_adult"}
    ):
        code = "vulnerable_groups"
    else:
        code = "additive_regulation"
    return code, DOMAIN_LABELS[code]


# ── Lý do "không đủ bằng chứng" ────────────────────────────────────────────────────────────────

def insufficiency_reasons(evaluation: dict, ocr_result: dict | None, linked_results, passages, qa: dict | None) -> list[str]:
    """Các lý do cụ thể khiến câu trả lời chưa đủ căn cứ (rỗng nếu đủ)."""
    threshold = _abstain_threshold()
    reasons: list[str] = []
    ocr_result = ocr_result or {}
    s_perception = float(evaluation.get("s_perception", 1.0) or 0.0)
    s_total = float(evaluation.get("s_total", 1.0) or 0.0)

    if not str(ocr_result.get("full_text", "")).strip():
        reasons.append("Không đọc được chữ nào trên ảnh: ảnh có thể quá mờ, tối hoặc không chứa bảng thành phần.")
    elif s_perception < threshold:
        reasons.append(f"Ảnh khó đọc: độ tin cậy OCR {s_perception:.2f} thấp hơn ngưỡng {threshold:.2f}.")

    if not any(item.get("matched_entity") for item in (linked_results or [])):
        reasons.append("Không nhận diện được phụ gia hoặc dị nguyên nào trong thành phần đọc được.")

    if not passages:
        reasons.append("Không truy xuất được đoạn bằng chứng nào từ cơ sở tri thức cho câu hỏi này.")

    for reason in evaluation.get("abstention_reasons", []) or []:
        if reason not in reasons:
            reasons.append(str(reason))
    for gap in (qa or {}).get("knowledge_gaps", []) or []:
        text = gap.get("reason") if isinstance(gap, dict) else str(gap)
        if text and text not in reasons:
            reasons.append(text)

    if s_total < threshold:
        reasons.append(f"Điểm tin cậy tổng thể {s_total:.2f} thấp hơn ngưỡng {threshold:.2f}.")
    return reasons
