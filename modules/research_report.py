"""Tab "🔬 Báo cáo nghiên cứu": hiển thị KẾT QUẢ ĐÁNH GIÁ BỘ QA (không phải công cụ phân tích từng câu).

Mọi con số đọc trực tiếp từ các file kết quả do script đánh giá sinh ra; phần nào chưa chạy thì ghi rõ "chưa có dữ liệu"
kèm lệnh cần chạy — không điền số mặc định.

  Bộ QA                  data/foodhealthqa_eval_set.jsonl, data/benchmark_manifest.json
  Kết quả 3 cấu hình     data/evaluation_results.json                (scripts_research/run_evaluation.py)
  Hiệu chỉnh độ tin cậy  data/calibration/calibration_report.json    (scripts_research/calibrate_confidence.py)
  Phân tích lỗi & ca     data/analysis/normalization_errors.json     (analyze_normalization_errors.py)
                         data/analysis/faithfulness_summary.json     (judge_faithfulness.py)
                         data/analysis/case_analysis.json            (analyze_cases.py)
"""

from __future__ import annotations

import html
import json
import os
from collections import Counter

import pandas as pd
import streamlit as st

CONFIG_LABELS = {
    "A_LLM_Only": "[A] LLM không truy xuất",
    "B_GT_RAG": "[B] RAG + văn bản nhãn chuẩn",
    "C_OCR_RAG": "[C] OCR ảnh thật + RAG",
}

METRICS = (
    ("count", "Số câu chấm", "n"),
    ("exact_match", "Exact Match", "%"),
    ("token_f1", "Token F1", "%"),
    ("citation_precision", "Citation Precision", "%"),
    ("citation_recall", "Citation Recall", "%"),
    ("citation_compliance", "Có trích nguồn hợp lệ", "%"),
    ("unrecognized_citation_rate", "Có thẻ nguồn tự bịa", "%"),
    ("groundedness", "Groundedness (trùng từ)", "f"),
    ("safety_accuracy", "Safety Accuracy", "%"),
    ("unsafe_recall", "Unsafe Recall", "%"),
    ("false_safe_rate", "False-safe (nguy hiểm → SAFE)", "%"),
    ("safe_from_screened_no_alert", "SAFE do 'chưa thấy cảnh báo'", "n"),
    ("invalid_label_count", "Nhãn không hợp lệ", "n"),
    ("abstention_rate_predicted", "Tỉ lệ từ chối (dự đoán)", "%"),
    ("abstention_rate_gold", "Tỉ lệ từ chối (nhãn chuẩn)", "%"),
)

HERO_CSS = """
<style>
[data-testid="stMainBlockContainer"],.main .block-container{max-width:100%!important;padding-bottom:2.5rem!important}
.rr-hero{border-radius:16px;padding:22px 26px;margin:0 0 14px 0;color:#fff;
  background:linear-gradient(135deg,#09476b 0%,#0d6e73 50%,#087f8c 100%);box-shadow:0 10px 25px -5px rgba(15,23,42,.10)}
.rr-hero h1{margin:0!important;padding:0!important;font-size:1.45rem!important;font-weight:800!important;color:#fff!important}
.rr-hero p{margin:8px 0 0;font-size:.86rem;line-height:1.6;color:rgba(240,253,250,.92)!important}
.rr-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.rr-chip{padding:4px 11px;border-radius:8px;font-size:.74rem;font-weight:600;color:#fff;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18)}
.rr-chip.ok{background:rgba(16,185,129,.28)} .rr-chip.no{background:rgba(245,158,11,.30)}
.rr-empty{padding:14px 16px;border-radius:12px;background:#f8fafc;border:1px dashed #cbd5e1;font-size:.82rem;color:#475569;line-height:1.6}
.rr-empty code{font-size:.76rem}
</style>
"""


# ---------------------------------------------------------------------------
# tiện ích
# ---------------------------------------------------------------------------
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


def _load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _empty(msg, command):
    st.markdown(f'<div class="rr-empty">⏳ {html.escape(msg)}<br>Chạy: <code>{html.escape(command)}</code></div>',
                unsafe_allow_html=True)


def _counts_table(counter, label, total):
    return pd.DataFrame([{label: k or "(trống)", "Số câu": v, "Tỉ lệ": f"{v / total * 100:.1f}%"}
                         for k, v in counter.most_common()])


# ---------------------------------------------------------------------------
# các phần
# ---------------------------------------------------------------------------
def _hero(data_dir, records, eval_data, calib, analyses):
    def chip(name, ok):
        return f'<span class="rr-chip {"ok" if ok else "no"}">{"✓" if ok else "…"} {name}</span>'
    complete = bool(eval_data and eval_data.get("complete"))
    st.markdown(
        '<div class="rr-hero"><h1>🔬 Báo cáo đánh giá bộ QA FoodHealthQA</h1>'
        f'<p>Bộ đánh giá {_fmt(len(records), "n")} câu hỏi–đáp trên {_fmt(len({r.get("image_ref") for r in records}), "n")} ảnh nhãn; '
        'so sánh 3 cấu hình A (LLM không truy xuất) · B (RAG trên văn bản nhãn chuẩn) · C (OCR ảnh thật + RAG). '
        'Bộ QA là silver/pre-gold, chưa được chuyên gia thẩm định: các chỉ số mang tính tham khảo (exploratory).</p>'
        '<div class="rr-chips">'
        + chip("Bộ QA", bool(records))
        + chip("Hiệu chỉnh độ tin cậy", bool(calib))
        + chip("Kết quả A/B/C" + ("" if complete else " (chưa đầy đủ)"), complete)
        + chip("Lỗi chuẩn hóa", bool(analyses.get("norm")))
        + chip("Faithfulness", bool(analyses.get("faith")))
        + chip("Phân tích ca", bool(analyses.get("cases")))
        + "</div></div>",
        unsafe_allow_html=True,
    )


def _tab_dataset(records, manifest):
    if not records:
        _empty("Chưa tìm thấy bộ đánh giá data/foodhealthqa_eval_set.jsonl.",
               "python scripts_research/build_foodhealthqa_subset.py")
        return
    n = len(records)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Câu hỏi", _fmt(n, "n"))
    c2.metric("Ảnh nhãn", _fmt(len({r.get("image_ref") for r in records}), "n"))
    c3.metric("Tầng core / silver", f"{sum(r.get('tier') == 'core' for r in records)} / {sum(r.get('tier') == 'silver' for r in records)}")
    c4.metric("Đã kiểm nguồn nguyên văn", _fmt(sum(r.get('source_verification_level') == 'label_and_primary_source_text_verified' for r in records), "n"))
    if manifest is not None and not manifest.get("expert_validated", False):
        st.caption("Bộ QA kiểm chứng bằng quy tắc, không có chuyên gia (expert_validated = false).")
    left, right = st.columns(2)
    with left:
        st.markdown("##### Nhãn an toàn chuẩn")
        st.dataframe(_counts_table(Counter(r.get("gold_safety_label") for r in records), "Nhãn", n), width="stretch", hide_index=True)
        st.markdown("##### Lĩnh vực")
        st.dataframe(_counts_table(Counter(r.get("domain_vi") or r.get("domain") for r in records), "Lĩnh vực", n), width="stretch", hide_index=True)
    with right:
        st.markdown("##### Hồ sơ người dùng (persona)")
        st.dataframe(_counts_table(Counter(r.get("persona_name") for r in records), "Persona", n), width="stretch", hide_index=True)
        st.markdown("##### Nguồn được trích trong đáp án chuẩn")
        src = Counter(s.strip() for r in records for s in str(r.get("source_citations") or "").split(",") if s.strip())
        st.dataframe(pd.DataFrame([{"Nguồn": k, "Số câu": v} for k, v in src.most_common()]), width="stretch", hide_index=True)
    reasons = Counter(r.get("abstain_reason") for r in records if r.get("abstain_reason"))
    if reasons:
        st.caption("Lý do từ chối trong nhãn chuẩn: " + " · ".join(f"{k}: {v}" for k, v in reasons.items()))


def _tab_results(eval_data):
    if not eval_data:
        _empty("Chưa có kết quả chấm 3 cấu hình.",
               "python scripts_research/generate_predictions.py --eval-750 --output-dir data --resume  →  python scripts_research/run_evaluation.py")
        return
    configs = eval_data.get("configs") or {}
    ref_count = int(eval_data.get("reference_count") or 0)
    scored = {k: int(((configs.get(k) or {}).get("overall") or {}).get("count") or 0) for k in CONFIG_LABELS}
    complete = bool(eval_data.get("complete"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Câu hỏi trong bộ tham chiếu", _fmt(ref_count, "n"))
    c2.metric("Câu đã chấm (ít nhất / cấu hình)", _fmt(min(scored.values()) if scored else 0, "n"))
    c3.metric("Trạng thái", "Đầy đủ" if complete else "Chưa đầy đủ")
    if not complete:
        st.error(f"Kết quả **chưa đầy đủ**: {', '.join(f'{k[0]} = {v}' for k, v in scored.items())} câu được chấm trên "
                 f"{ref_count} câu. Các số dưới đây **không đại diện** cho hiệu năng hệ thống.")
    if not eval_data.get("gold_validated", False):
        st.warning("Nhãn tham chiếu chưa được chuyên gia thẩm định: chỉ số là exploratory weak-label metrics.")
    rows = []
    for key, label in CONFIG_LABELS.items():
        ov = (configs.get(key) or {}).get("overall")
        if ov is not None:
            rows.append({"Cấu hình": label, **{lbl: _fmt(ov.get(mk), kd) for mk, lbl, kd in METRICS}})
    if rows:
        st.markdown("##### Bảng đối chiếu tổng quan")
        st.dataframe(pd.DataFrame(rows).set_index("Cấu hình").T, width="stretch")
        st.caption("'—' = không áp dụng (mẫu số bằng 0 hoặc không đo được). Groundedness là trùng từ với bằng chứng thật mà "
                   "mô hình đã đọc; faithfulness chính thức ở tab Phân tích lỗi & ca.")
    perc = []
    for key, label in CONFIG_LABELS.items():
        pc = (configs.get(key) or {}).get("perception")
        if pc:
            perc.append({"Cấu hình": label, "S_perception TB": _fmt(pc.get("mean"), "f"), "Nhỏ nhất": _fmt(pc.get("min"), "f"),
                         "P10": _fmt(pc.get("p10"), "f"), f"S_total < {pc.get('threshold', 0.65)}": _fmt(pc.get("below_threshold_total"), "n"),
                         "Còn hằng số 0,85 cũ": _fmt(pc.get("legacy_constant_count"), "n"),
                         "Phiên bản OCR": ", ".join(pc.get("ocr_pipeline_versions") or [])})
    if perc:
        st.markdown("##### Độ tin cậy OCR (S_perception)")
        st.dataframe(pd.DataFrame(perc), width="stretch", hide_index=True)
    present = [k for k in CONFIG_LABELS if k in configs]
    if present:
        st.markdown("##### Chi tiết từng cấu hình")
        for tab, key in zip(st.tabs([CONFIG_LABELS[k] for k in present]), present):
            cfg = configs[key]
            with tab:
                per_domain = cfg.get("per_domain") or {}
                if per_domain:
                    st.dataframe(pd.DataFrame([{"Lĩnh vực": d, **{lbl: _fmt(m.get(mk), kd) for mk, lbl, kd in METRICS}}
                                               for d, m in per_domain.items()]), width="stretch", hide_index=True)
                cm = cfg.get("confusion_matrix") or {}
                if cm:
                    st.caption("Ma trận nhầm lẫn nhãn an toàn (hàng: nhãn chuẩn · cột: dự đoán)")
                    st.dataframe(pd.DataFrame(cm).T, width="stretch")
                missing, extra = cfg.get("missing_predictions") or [], cfg.get("extra_predictions") or []
                if missing or extra:
                    st.caption(f"Thiếu dự đoán: {len(missing)} câu · dự đoán ngoài bộ tham chiếu: {len(extra)} câu.")


def _tab_calibration(calib):
    if not calib:
        _empty("Chưa có kết quả hiệu chỉnh độ tin cậy / chọn ngưỡng từ chối (bước này OCR mọi ảnh, không gọi LLM).",
               "python scripts_research/calibrate_confidence.py --stage all")
        return
    st.caption(f"Ngưỡng hiện tại: {calib.get('current_threshold')} · tiêu chí chọn: {calib.get('selection_rule')} · "
               f"chia theo ảnh, {int(calib.get('dev_fraction', 0.5) * 100)}% ảnh để chọn ngưỡng, phần còn lại để kiểm tra.")
    for config, c in (calib.get("configs") or {}).items():
        st.markdown(f"##### Cấu hình {config} — {c.get('records')} câu (phát triển {c.get('n_dev')} / kiểm tra {c.get('n_test')})")
        a, b, d = st.columns(3)
        a.metric("S_retrieval không áp dụng", _fmt(c.get("s_retrieval_not_applicable"), "n"))
        b.metric("S_total < 0,65", _fmt(c.get("s_total_below_current"), "n"))
        d.metric("Ngưỡng chọn được", _fmt(c.get("chosen_threshold"), "f"))
        lk = c.get("linking_vs_gold_ins") or {}
        st.caption(f"Liên kết mã INS so với mã được hỏi: recall {_fmt(lk.get('recall'), '%')} (precision không áp dụng)")
        _v = lk.get("vs_reference_label")
        if _v:
            st.caption(f"So với mã liên kết từ nhãn tham chiếu (B): precision {_fmt(_v.get('precision'), '%')} · "
                       f"recall {_fmt(_v.get('recall'), '%')} · F1 {_fmt(_v.get('f1'), '%')}")
        oc = c.get("ocr_confidence_vs_cer")
        if oc:
            rho = oc.get("spearman_sperception_vs") or {}
            st.markdown(f"**Độ tin cậy OCR có phản ánh lỗi OCR thật không?** ({oc.get('images')} ảnh) — Spearman: thiếu từ "
                        f"{_fmt(rho.get('token_miss'), 'f')} · thiếu thành phần {_fmt(rho.get('item_miss'), 'f')} · "
                        f"CER theo thứ tự đọc {_fmt(rho.get('cer_reading_order'), 'f')}")
            st.dataframe(pd.DataFrame([{"S_perception": b_["s_perception"], "Số ảnh": b_["images"],
                                        "Thiếu từ TB": _fmt(b_.get("mean_token_miss"), "%"),
                                        "Thiếu thành phần TB": _fmt(b_.get("mean_item_miss"), "%"),
                                        "CER (thứ tự đọc)": _fmt(b_.get("mean_cer_reading_order"), "%")} for b_ in oc.get("bins") or []]),
                         width="stretch", hide_index=True)
            st.caption(oc.get("interpretation", ""))
        test = c.get("test") or {}
        if test:
            st.markdown("**Trên phần kiểm tra:**")
            names = {"no_threshold": "Không dùng ngưỡng", "current_0.65": "Ngưỡng hiện tại 0,65", "chosen": "Ngưỡng chọn được"}
            st.dataframe(pd.DataFrame([{"": names.get(k, k), "Ngưỡng": _fmt(m.get("threshold"), "f"), "Accuracy": _fmt(m.get("accuracy"), "%"),
                                        "Tỉ lệ trả lời": _fmt(m.get("coverage"), "%"), "Đúng khi trả lời": _fmt(m.get("selective_accuracy"), "%"),
                                        "False-safe": _fmt(m.get("false_safe_rate"), "%"), "Từ chối đúng (recall)": _fmt(m.get("abstain_recall"), "%"),
                                        "Từ chối đúng (precision)": _fmt(m.get("abstain_precision"), "%")} for k, m in test.items() if m]),
                         width="stretch", hide_index=True)
        curve = c.get("dev_curve") or []
        if curve:
            with st.expander("Đường cong trên phần phát triển (accuracy · tỉ lệ trả lời theo ngưỡng)"):
                st.line_chart(pd.DataFrame([{"Ngưỡng": m["threshold"], "Accuracy": m["accuracy"], "Tỉ lệ trả lời": m["coverage"]}
                                            for m in curve]).set_index("Ngưỡng"))


def _tab_analysis(analyses):
    norm, faith, cases = analyses.get("norm"), analyses.get("faith"), analyses.get("cases")
    st.markdown("##### Lỗi chuẩn hóa thành phần (OCR so với liên kết)")
    if norm:
        rows = [{"Cấu hình": c, "Precision": _fmt(m.get("precision"), "%"), "Recall": _fmt(m.get("recall"), "%"), "F1": _fmt(m.get("f1"), "%"),
                 "Đúng": m.get("tp"), "Thừa/sai": m.get("fp"), "Bỏ sót": m.get("fn")}
                for c, m in (norm.get("linking_vs_label_codes") or {}).items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        cause = norm.get("missed_codes_in_C_by_cause") or {}
        st.caption(f"Mã C bỏ sót: do OCR {cause.get('ocr', 0)} · do liên kết {cause.get('linking', 0)} · "
                   f"mã OCR bị đánh dấu không đọc chắc: {(norm.get('uncertain_code_flags_in_C') or {}).get('flags', 0)}")
    else:
        _empty("Chưa phân tích lỗi chuẩn hóa (cần dự đoán B và C).", "python scripts_research/analyze_normalization_errors.py")

    st.markdown("##### Faithfulness (LLM làm giám khảo, kiểm chứng bằng người chấm)")
    if faith:
        ag = faith.get("human_agreement")
        if ag:
            st.caption(f"Đồng thuận người – giám khảo: {ag.get('pairs')} nhận định · tỉ lệ trùng {_fmt(ag.get('agreement'), '%')} · "
                       f"Cohen's kappa {_fmt(ag.get('cohen_kappa'), 'f')}")
        else:
            st.warning("Chưa có kết quả chấm tay để kiểm chứng giám khảo — chưa nên báo cáo các số dưới đây như kết quả chính thức.")
        st.dataframe(pd.DataFrame([{"Cấu hình": c, "Câu trả lời": m.get("answers"), "Nhận định": m.get("claims"),
                                    "Faithfulness TB": _fmt(m.get("faithfulness_mean"), "%"), "Mâu thuẫn": _fmt(m.get("contradicted_rate"), "%"),
                                    "Thiếu căn cứ": _fmt(m.get("not_enough_info_rate"), "%"), "Giám khảo": ", ".join(m.get("judge_models") or [])}
                                   for c, m in (faith.get("configs") or {}).items()]), width="stretch", hide_index=True)
    else:
        _empty("Chưa chấm faithfulness.", "python scripts_research/judge_faithfulness.py judge  →  export-annotation  →  agreement  →  summarize")

    st.markdown("##### Hallucination · thiếu căn cứ · cần chuyên gia y tế")
    if cases:
        labels = {"fake_citation": "Thẻ nguồn tự bịa", "ins_not_in_evidence": "Nêu mã INS không có căn cứ",
                  "judge_contradicted": "Nhận định trái bằng chứng", "judge_not_enough_info": "Nhận định thiếu căn cứ",
                  "low_lexical_grounding": "Trùng từ thấp (sàng lọc)", "system_referred": "Đã chuyển chuyên gia",
                  "question_needs_expert_not_referred": "Cần chuyên gia nhưng không chuyển", "false_safe": "False-safe",
                  "critical_missed": "Bỏ sót CRITICAL_ALERT"}
        counts = cases.get("counts") or {}
        groups = sorted({g for c in counts.values() for g in c})
        st.dataframe(pd.DataFrame([{"Nhóm": labels.get(g, g), **{f"{c} (câu)": counts.get(c, {}).get(g, 0) for c in sorted(counts)}}
                                   for g in groups]), width="stretch", hide_index=True)
        st.caption(cases.get("note", "") + " Danh sách ca cụ thể: data/analysis/cases_<nhóm>.csv")
    else:
        _empty("Chưa phân tích ca.", "python scripts_research/analyze_cases.py")


def render_research_report(data_dir):
    st.markdown(HERO_CSS, unsafe_allow_html=True)
    records = _load_jsonl(os.path.join(data_dir, "foodhealthqa_eval_set.jsonl"))
    manifest_all = _load_json(os.path.join(data_dir, "benchmark_manifest.json")) or {}
    manifest = (manifest_all.get("benchmarks") or {}).get(manifest_all.get("active_version", ""), None)
    eval_data = _load_json(os.path.join(data_dir, "evaluation_results.json"))
    calib = _load_json(os.path.join(data_dir, "calibration", "calibration_report.json"))
    analyses = {
        "norm": _load_json(os.path.join(data_dir, "analysis", "normalization_errors.json")),
        "faith": _load_json(os.path.join(data_dir, "analysis", "faithfulness_summary.json")),
        "cases": _load_json(os.path.join(data_dir, "analysis", "case_analysis.json")),
    }
    _hero(data_dir, records, eval_data, calib, analyses)
    t1, t2, t3, t4 = st.tabs(["📊 Bộ QA", "🧪 Kết quả 3 cấu hình A/B/C", "📷 Hiệu chỉnh độ tin cậy", "🔍 Phân tích lỗi & ca"])
    with t1:
        _tab_dataset(records, manifest)
    with t2:
        _tab_results(eval_data)
    with t3:
        _tab_calibration(calib)
    with t4:
        _tab_analysis(analyses)
