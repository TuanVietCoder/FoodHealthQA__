"""Hiển thị bảng so sánh A/B/C (và Gemini với Qwen2.5) trong Streamlit. Logic nằm ở modules/compare_runner.py."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from modules import compare_runner as cr

LABEL_ICONS = {"SAFE": "🟢", "CAUTION": "🟡", "UNSAFE": "🟠", "CRITICAL_ALERT": "🔴", "ABSTAIN": "⚪"}


def _label(value) -> str:
    return f"{LABEL_ICONS.get(value, '❔')} {value}" if value else "—"


def _result(by, provider, config):
    return by.get((provider, config))


def _render_cell(result, config, disabled_reason=None):
    st.markdown(f"**{cr.CONFIG_TITLES[config]}**")
    if disabled_reason:
        st.caption(f"⛔ {disabled_reason}")
        return
    if result is None:
        st.caption("⏳ Đang chờ chạy…")
        return
    if result.get("error"):
        st.error(result["error"])
        return
    st.markdown(f"### {_label(result['safety_label'])}")
    n_cit = len(result["citations"])
    n_pas = len(result["passages"]) if config != "A" else "—"
    sec = f"{result['seconds']}s"
    st.markdown(
        f'''<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 0 12px 0;">
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:8px 4px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                <div style="font-size:0.72rem;font-weight:600;color:#64748b;line-height:1.2;">Trích dẫn</div>
                <div style="font-size:1.15rem;font-weight:800;color:#0f172a;margin-top:2px;font-family:ui-monospace,Consolas,monospace;">{n_cit}</div>
            </div>
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:8px 4px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                <div style="font-size:0.72rem;font-weight:600;color:#64748b;line-height:1.2;">Truy xuất</div>
                <div style="font-size:1.15rem;font-weight:800;color:#0f172a;margin-top:2px;font-family:ui-monospace,Consolas,monospace;">{n_pas}</div>
            </div>
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:8px 4px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                <div style="font-size:0.72rem;font-weight:600;color:#64748b;line-height:1.2;">Thời gian</div>
                <div style="font-size:1.15rem;font-weight:800;color:#0f172a;margin-top:2px;font-family:ui-monospace,Consolas,monospace;">{sec}</div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )
    if result.get("citations"):
        st.caption("Nguồn: " + ", ".join(result["citations"]))
    elif config == "A":
        st.caption("Không tra cứu nên không có nguồn trích dẫn.")
    else:
        st.warning("Câu trả lời không có trích dẫn nguồn hợp lệ — kém tin cậy.")
    if result.get("unrecognized_citations"):
        st.warning("Thẻ nguồn lạ (mô hình tự bịa, không tính là trích dẫn): " + ", ".join(result["unrecognized_citations"]))
    with st.expander("Câu trả lời", expanded=True):
        st.markdown(result["answer"])


def render_board(data: dict, running_note: str | None = None) -> None:
    """Vẽ toàn bộ kết quả: lưới provider × cấu hình, bảng so sánh nhanh, nhận xét, đối chiếu OCR."""
    results = data["results"]
    providers, configs = data["providers"], data["configs"]
    by = {(r["provider"], r["config"]): r for r in results}
    ocr = data.get("ocr") or {}
    c_reason = None
    if "C" in configs and ocr.get("unavailable"):
        c_reason = ocr.get("error") or "OCR không khả dụng"

    if running_note:
        st.caption(running_note)

    for provider in providers:
        if len(providers) > 1:
            st.markdown(f"#### 🤖 {cr.PROVIDER_TITLES.get(provider, provider)}")
        cols = st.columns(len(configs))
        for col, config in zip(cols, configs):
            with col:
                _render_cell(_result(by, provider, config), config,
                             disabled_reason=c_reason if config == "C" else None)

    done = [r for r in results if not r.get("error")]
    if done:
        st.markdown("#### So sánh nhanh")
        rows = {
            "Nhãn an toàn": lambda r: _label(r["safety_label"]),
            "Nhãn do LLM tự gán": lambda r: _label(r.get("llm_label")),
            "Số nguồn trích dẫn": lambda r: len(r["citations"]),
            "Thẻ nguồn lạ": lambda r: len(r.get("unrecognized_citations") or []),
            "Số đoạn truy xuất": lambda r: len(r["passages"]) if r["config"] != "A" else "—",
            "Thời gian (giây)": lambda r: r["seconds"],
            "Model trả lời": lambda r: r.get("model") or "—",
        }
        table = {}
        for provider in providers:
            for config in configs:
                r = by.get((provider, config))
                name = (f"{cr.PROVIDER_TITLES.get(provider, provider)} · " if len(providers) > 1 else "") + config
                table[name] = {k: (fn(r) if r and not r.get("error") else "—") for k, fn in rows.items()}
        df = pd.DataFrame(table)
        df = df.astype(str)
        st.dataframe(df, use_container_width=True)

        notes = cr.summarize(results, providers, ocr)
        if notes:
            st.markdown("\n\n".join(f"- {n}" for n in notes))

    if ocr.get("requested") and ocr.get("text") is not None and ocr.get("reference"):
        with st.expander("🔍 Đối chiếu văn bản chuẩn và kết quả OCR (cấu hình C)", expanded=False):
            ratio = ocr.get("match_ratio")
            if ratio is not None:
                st.caption(f"OCR đọc đúng **{ratio:.1%}** ký tự của mục thành phần · thời gian OCR {ocr.get('seconds')}s "
                           "· chỉ đối chiếu đoạn từ \"Thành phần\" (chữ khác trên bao bì được bỏ qua)")
            left, right = st.columns(2)
            with left:
                st.markdown("**Văn bản chuẩn (annotation)**")
                st.markdown(  # không dùng widget có key: bảng được vẽ lại nhiều lần trong một lần chạy
                    f'<div style="font-size:0.9rem;line-height:1.5">{html.escape(ocr["reference"]).replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown("**OCR đọc được** (đỏ: sai chính tả/ngoài nhãn, ▯ vàng: thiếu; thành phần đảo vị trí vẫn nhận diện đúng)")
                st.markdown(
                    f'<div style="font-size:0.9rem;line-height:1.5">{cr.diff_html(ocr["reference"], ocr.get("section") or ocr["text"])}</div>',
                    unsafe_allow_html=True,
                )
