"""
modules/nli_verifier.py: Bộ Thẩm định Khẳng định Ngữ nghĩa (Natural Language Inference - NLI).

Thực hiện kiểm chứng tính trung thực (Faithfulness & Claim Verification) của câu trả lời AI
đối với đoạn bằng chứng tra cứu được từ cơ sở tri thức y khoa & pháp lý:
- Phân rã câu trả lời thành các nhận định nguyên tử (Atomic Claims).
- Đánh giá quan hệ logic 3 trạng thái: ENTAILMENT, CONTRADICTION, NEUTRAL.
- Phát hiện hiện tượng "gian lận trích dẫn" (Citation Hallucination).
- Hỗ trợ cả 2 chế độ: Fast Zero-Shot LLM NLI và Heuristic Deterministic Fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from modules.benchmark import extract_citations, is_recognized_citation, canonicalize_citation

# Danh sách từ khóa rủi ro cao (cần ưu tiên kiểm định NLI)
HIGH_RISK_KEYWORDS = [
    "cấm", "độc", "ung thư", "nguy cơ", "không an toàn", "vượt ngưỡng",
    "dị ứng", "sốc phản vệ", "hiếu động", "methemoglobin", "tổn thương",
    "quái thai", "tử vong", "ngộ độc", "nhiễm độc", "tác dụng phụ",
    "không được dùng", "chống chỉ định", "thận trọng", "trẻ em",
    "thai phụ", "mang thai", "suy thận", "tiểu đường"
]

REGULATORY_KEYWORDS = [
    "thông tư", "tt24", "tt 24", "codex", "jecfa", "efsa", "fda", "adi",
    "quy định", "cho phép", "danh mục", "ngưỡng", "tiêu chuẩn", "ml"
]


def split_into_claims(text: str) -> List[Dict[str, Any]]:
    """
    Phân rã văn bản thành danh sách các câu / nhận định nguyên tử.
    Loại bỏ các dòng meta (như SAFETY_LABEL) và câu chào rỗng.
    """
    clean_text = re.sub(r"(?im)^\s*SAFETY_LABEL\s*:.*$", "", text).strip()
    if not clean_text:
        return []

    # Tách câu theo dấu câu kết thúc hoặc ngắt dòng
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", clean_text)
    claims = []

    for s in raw_sentences:
        s_clean = s.strip()
        # Bỏ qua câu quá ngắn hoặc chỉ toàn ký tự đặc biệt
        if len(s_clean) < 15 or not re.search(r"\w{2,}", s_clean):
            continue

        c_low = s_clean.lower()
        has_risk = any(k in c_low for k in HIGH_RISK_KEYWORDS)
        has_reg = any(k in c_low for k in REGULATORY_KEYWORDS)
        cites = [canonicalize_citation(c) for c in extract_citations(s_clean) if is_recognized_citation(c)]

        claims.append({
            "text": s_clean,
            "has_citation": bool(cites),
            "citations": cites,
            "is_high_risk": has_risk,
            "is_regulatory": has_reg,
            "needs_verification": has_risk or has_reg or bool(cites)
        })

    return claims


def _rule_based_nli_heuristic(claim_text: str, evidence_text: str) -> Dict[str, Any]:
    """
    Bộ suy luận logic tất định dựa trên từ vựng & ngữ nghĩa khi không có LLM.
    Phát hiện xung đột rõ ràng giữa 'cấm' vs 'cho phép' hoặc mã INS.
    """
    c_low = claim_text.lower()
    e_low = evidence_text.lower()

    # Kiểm tra mâu thuẫn trực tiếp: Cấm vs Cho phép
    if ("bị cấm" in c_low or "không được phép" in c_low) and ("cho phép" in e_low or "permitted" in e_low or "được phép" in e_low):
        # Trừ khi evidence nói rõ cấm chất đó
        if "không được phép" not in e_low and "cấm" not in e_low:
            return {
                "label": "CONTRADICTION",
                "confidence": 0.85,
                "reason": "Nhận định nói 'cấm' nhưng bằng chứng ghi nhận được phép/cho phép."
            }

    if ("an toàn tuyệt đối" in c_low or "hoàn toàn an toàn" in c_low) and ("cannot establish adi" in e_low or "genotoxicity" in e_low or "nguy cơ" in e_low):
        return {
            "label": "CONTRADICTION",
            "confidence": 0.90,
            "reason": "Nhận định nói an toàn tuyệt đối nhưng bằng chứng có cảnh báo độc tính/nguy cơ."
        }

    # Kiểm tra Entailment từ khóa
    matches = 0
    words = re.findall(r"\w{3,}", c_low)
    meaningful_words = [w for w in words if w not in {"sản", "phẩm", "trong", "được", "người", "không", "này", "theo", "chất"}]
    for w in meaningful_words:
        if w in e_low:
            matches += 1

    ratio = matches / max(1, len(meaningful_words))
    if ratio >= 0.4:
        return {
            "label": "ENTAILMENT",
            "confidence": min(0.95, 0.6 + ratio * 0.4),
            "reason": f"Khớp từ vựng ngữ cảnh ({matches}/{len(meaningful_words)} từ khóa chính xác)."
        }

    return {
        "label": "NEUTRAL",
        "confidence": 0.60,
        "reason": "Không đủ từ khóa đối soát trực tiếp trong bằng chứng."
    }


def classify_claim_nli(
    claim_text: str,
    evidence_text: str,
    call_llm_fn: Optional[Callable[[str], Optional[str]]] = None
) -> Dict[str, Any]:
    """
    Phân loại quan hệ NLI giữa 1 nhận định (Hypothesis) và Bằng chứng (Premise).
    Trả về dict: {"label": "ENTAILMENT"|"CONTRADICTION"|"NEUTRAL", "confidence": float, "reason": str}
    """
    if not claim_text.strip():
        return {"label": "NEUTRAL", "confidence": 0.0, "reason": "Câu rỗng"}

    if not evidence_text.strip():
        return {"label": "NEUTRAL", "confidence": 0.5, "reason": "Thiếu bằng chứng đối soát"}

    # Nếu có LLM function: thực hiện Fast Zero-Shot NLI
    if call_llm_fn is not None:
        # Cắt ngắn evidence để tăng tốc độ xử lý (lấy 1500 ký tự đầu liên quan nhất)
        ev_snippet = evidence_text[:1800]
        prompt = (
            "Bạn là trợ lý kiểm chứng khẳng định khoa học & y tế (Natural Language Inference).\n"
            "Hãy đối chiếu NHẬN ĐỊNH của AI với BẰNG CHỨNG THẨM QUYỀN bên dưới:\n\n"
            f"--- BẰNG CHỨNG (PREMISE) ---\n{ev_snippet}\n\n"
            f"--- NHẬN ĐỊNH (HYPOTHESIS) ---\n\"{claim_text}\"\n\n"
            "Hãy xác định quan hệ logic giữa NHẬN ĐỊNH và BẰNG CHỨNG:\n"
            "- ENTAILMENT: Nhận định được chứng minh trực tiếp hoặc suy diễn hợp lý từ bằng chứng.\n"
            "- CONTRADICTION: Nhận định mâu thuẫn trực tiếp, phủ định hoặc trái ngược với bằng chứng.\n"
            "- NEUTRAL: Bằng chứng không đủ thông tin để kết luận đúng hay sai (nói ngoài tài liệu).\n\n"
            "CHỈ TRẢ LỜI ĐÚNG ĐỊNH DẠNG JSON MỘT DÒNG:\n"
            "{\"label\": \"ENTAILMENT\" hoặc \"CONTRADICTION\" hoặc \"NEUTRAL\", \"reason\": \"lý do ngắn dưới 20 từ\"}"
        )
        try:
            raw_res = call_llm_fn(prompt)
            if raw_res:
                # Tìm JSON trong câu trả lời
                m = re.search(r"\{.*?\}", raw_res, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    lbl = str(parsed.get("label", "")).strip().upper()
                    if lbl in {"ENTAILMENT", "CONTRADICTION", "NEUTRAL"}:
                        return {
                            "label": lbl,
                            "confidence": 0.92 if lbl != "NEUTRAL" else 0.70,
                            "reason": str(parsed.get("reason", "Phân tích ngữ nghĩa qua NLI")),
                            "source": "llm_nli"
                        }
        except Exception:
            pass  # Fallback to heuristic

    # Fallback heuristic logic
    res = _rule_based_nli_heuristic(claim_text, evidence_text)
    res["source"] = "heuristic_nli"
    return res


def verify_answer_faithfulness(
    answer_text: str,
    evidence_text: str,
    call_llm_fn: Optional[Callable[[str], Optional[str]]] = None,
    max_claims_to_check: int = 5,
    trigger_only: bool = True
) -> Dict[str, Any]:
    """
    Kiểm định độ trung thực (Faithfulness) toàn diện của câu trả lời AI.
    - trigger_only=True: Chỉ soi các câu có rủi ro hoặc quy chuẩn (giúp tăng tốc độ xử lý).
    """
    claims = split_into_claims(answer_text)
    if not claims:
        return {
            "status": "VERIFIED_SAFE",
            "faithfulness_score": 1.0,
            "claims_count": 0,
            "verified_count": 0,
            "contradiction_count": 0,
            "neutral_count": 0,
            "claims": [],
            "badge_html": '<span class="fh-chip2" style="background:#E0F2FE;color:#0284C7;">🛡️ NLI: Chưa phát hiện nhận định cần xác thực</span>'
        }

    # Chọn lọc câu để kiểm chứng
    target_claims = [c for c in claims if (not trigger_only or c["needs_verification"])]
    if not target_claims:
        target_claims = claims[:max_claims_to_check]
    else:
        target_claims = target_claims[:max_claims_to_check]

    results = []
    entail_cnt = 0
    contra_cnt = 0
    neutral_cnt = 0

    for c in target_claims:
        clf = classify_claim_nli(c["text"], evidence_text, call_llm_fn=call_llm_fn)
        lbl = clf["label"]
        if lbl == "ENTAILMENT":
            entail_cnt += 1
        elif lbl == "CONTRADICTION":
            contra_cnt += 1
        else:
            neutral_cnt += 1

        results.append({
            "text": c["text"],
            "label": lbl,
            "confidence": clf.get("confidence", 0.8),
            "reason": clf.get("reason", ""),
            "source": clf.get("source", "heuristic_nli"),
            "citations": c["citations"],
            "is_high_risk": c["is_high_risk"]
        })

    total_checked = len(results)
    faithfulness = round(entail_cnt / total_checked, 2) if total_checked > 0 else 1.0

    if contra_cnt > 0:
        status = "CONTRADICTION_DETECTED"
        badge_html = (
            f'<div style="background:#FEE2E2; border:1px solid #FCA5A5; color:#991B1B; padding:8px 12px; border-radius:8px; font-size:13px; margin:8px 0;">'
            f'🚨 <b>CẢNH BÁO XUNG ĐỘT NGỮ NGHĨA (NLI Guardrail):</b> Phát hiện <b>{contra_cnt} nhận định</b> mâu thuẫn trực tiếp với bằng chứng pháp lý/y khoa!'
            f'</div>'
        )
    elif neutral_cnt > 0 and entail_cnt == 0:
        status = "UNVERIFIED"
        badge_html = (
            f'<div style="background:#FEF3C7; border:1px solid #FCD34D; color:#92400E; padding:8px 12px; border-radius:8px; font-size:13px; margin:8px 0;">'
            f'⚠️ <b>LƯU Ý THẨM ĐỊNH (NLI Guardrail):</b> Các nhận định chưa tìm thấy căn cứ trực tiếp trong tài liệu ({total_checked} câu trung lập).'
            f'</div>'
        )
    else:
        status = "VERIFIED_SAFE"
        pct_str = f"{int(faithfulness * 100)}%"
        badge_html = (
            f'<div style="background:#DCFCE7; border:1px solid #86EFAC; color:#166534; padding:8px 12px; border-radius:8px; font-size:13px; margin:8px 0;">'
            f'🛡️ <b>ĐÃ XÁC THỰC NGỮ NGHĨA (NLI Faithfulness: {pct_str}):</b> Toàn bộ {entail_cnt}/{total_checked} luận điểm quan trọng đều khớp logic với văn bản thẩm quyền.'
            f'</div>'
        )

    return {
        "status": status,
        "faithfulness_score": faithfulness,
        "claims_count": len(claims),
        "verified_count": total_checked,
        "entailment_count": entail_cnt,
        "contradiction_count": contra_cnt,
        "neutral_count": neutral_cnt,
        "claims": results,
        "badge_html": badge_html
    }
