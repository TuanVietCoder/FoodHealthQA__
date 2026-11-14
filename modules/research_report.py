"""Mục "Báo cáo nghiên cứu": so sánh 3 cấu hình A/B/C từ data/evaluation_results.json.

Chỉ hiển thị số liệu có thật trong tệp kết quả (sinh bởi scripts_research/run_evaluation.py); không điền số mặc định.
Nếu kết quả chưa đầy đủ (thiếu prediction) hoặc bộ tham chiếu chưa phải gold, trang nói rõ điều đó trước bảng số.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

CONFIG_LABELS = {
    "A_LLM_Only": "[A] LLM không truy xuất (No-RAG baseline)",
    "B_GT_RAG": "[B] RAG từ văn bản nhãn tham chiếu",
    "C_OCR_RAG": "[C] OCR + RAG (full pipeline)",
}

METRICS = (
    ("count", "Số câu chấm", "n"),
    ("exact_match", "Exact Match", "%"),
    ("token_f1", "Token F1", "%"),
    ("citation_precision", "Citation Precision", "%"),
    ("citation_recall", "Citation Recall", "%"),
    ("citation_compliance", "Citation Compliance", "%"),
    ("groundedness", "Groundedness", "f"),
    ("safety_accuracy", "Safety Accuracy", "%"),
    ("unsafe_recall", "Unsafe Recall", "%"),
    ("abstention_rate_predicted", "Tỉ lệ từ chối (dự đoán)", "%"),
    ("abstention_rate_gold", "Tỉ lệ từ chối (tham chiếu)", "%"),
)


def _fmt(value, kind):
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "n":
        return f"{int(v):,}".replace(",", ".")
    if kind == "%":
        return f"{v * 100:.1f}%"
    return f"{v:.3f}"


def load_results(data_dir):
    path = os.path.join(data_dir, "evaluation_results.json")
    if not os.path.isfile(path):
        return None, f"Chưa có tệp kết quả `{path}`. Hãy chạy `scripts_research/run_evaluation.py`."
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, ValueError) as exc:
        return None, f"Không đọc được `evaluation_results.json`: {exc}"


def summary_rows(data):
    rows = []
    for key, label in CONFIG_LABELS.items():
        ov = (data.get("configs") or {}).get(key, {}).get("overall")
        if ov is None:
            continue
        row = {"Cấu hình": label}
        for mkey, mlabel, kind in METRICS:
            row[mlabel] = _fmt(ov.get(mkey), kind)
        rows.append(row)
    return rows


def render_research_report(data_dir):
    st.markdown("## 🔬 Báo cáo nghiên cứu — so sánh 3 cấu hình")
    st.caption("A: LLM trả lời trực tiếp · B: RAG trên văn bản nhãn tham chiếu · C: OCR ảnh thật + RAG. "
               "B và C dùng chung toàn bộ tầng xử lý phía sau, chỉ khác nguồn văn bản nhãn.")

    data, err = load_results(data_dir)
    if err:
        st.warning(err)
        return

    ref_count = int(data.get("reference_count") or 0)
    configs = data.get("configs") or {}
    scored = {k: int((configs.get(k, {}).get("overall") or {}).get("count") or 0) for k in CONFIG_LABELS}
    complete = bool(data.get("complete"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Câu hỏi trong bộ tham chiếu", _fmt(ref_count, "n"))
    c2.metric("Câu đã chấm (ít nhất / cấu hình)", _fmt(min(scored.values()) if scored else 0, "n"))
    c3.metric("Trạng thái kết quả", "Đầy đủ" if complete else "Chưa đầy đủ")

    if not complete:
        st.error(
            f"Kết quả **chưa đầy đủ**: mỗi cấu hình mới có {', '.join(f'{k[0]} = {v}' for k, v in scored.items())} "
            f"câu được chấm trên tổng {ref_count} câu. Các con số dưới đây **không đại diện** cho hiệu năng hệ thống; "
            "cần sinh đủ prediction A/B/C (`scripts_research/generate_predictions.py`) rồi chạy lại "
            "`scripts_research/run_evaluation.py`."
        )
    if not data.get("gold_validated", False):
        st.warning("Bộ tham chiếu chưa được chuyên gia thẩm định: chỉ số là *exploratory weak-label metrics*, "
                   "không phải hiệu năng trên gold standard. "
                   + (f"Ghi chú: {data['score_interpretation']}" if data.get("score_interpretation") else ""))

    rows = summary_rows(data)
    if not rows:
        st.info("Tệp kết quả không có số liệu cấu hình nào.")
        return
    st.markdown("#### 📋 Bảng đối chiếu tổng quan")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    tabs = st.tabs([CONFIG_LABELS[k] for k in CONFIG_LABELS if k in configs])
    for tab, key in zip(tabs, [k for k in CONFIG_LABELS if k in configs]):
        cfg = configs[key]
        with tab:
            per_domain = cfg.get("per_domain") or {}
            if per_domain:
                st.markdown("##### Theo lĩnh vực")
                st.dataframe(pd.DataFrame([
                    {"Lĩnh vực": dom, **{lbl: _fmt(m.get(mk), kd) for mk, lbl, kd in METRICS}}
                    for dom, m in per_domain.items()
                ]), width="stretch", hide_index=True)
            cm = cfg.get("confusion_matrix") or {}
            if cm:
                st.markdown("##### Ma trận nhầm lẫn nhãn an toàn (hàng: tham chiếu · cột: dự đoán)")
                st.dataframe(pd.DataFrame(cm).T, width="stretch")
            missing = cfg.get("missing_predictions") or []
            extra = cfg.get("extra_predictions") or []
            if missing or extra:
                st.caption(f"Thiếu prediction: {len(missing)} câu · Prediction ngoài bộ tham chiếu: {len(extra)} câu.")
