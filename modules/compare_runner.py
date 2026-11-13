"""
Chạy cùng một đầu vào qua cả ba cấu hình A/B/C (và một hoặc hai nhà cung cấp LLM) để so sánh cạnh nhau.

Không phụ thuộc Streamlit nên kiểm thử được độc lập. Giao diện nằm ở modules/compare_ui.py.

Quy ước đầu vào (giống scripts_research/generate_predictions.py):
  A: văn bản thành phần -> LLM không truy xuất
  B: văn bản thành phần -> chuẩn hóa/RAG/guardrail -> LLM
  C: ảnh -> OCR (chạy một lần, dùng chung cho mọi nhà cung cấp) -> cùng downstream với B
Văn bản thành phần dùng cho A/B là annotation của nhãn mẫu hoặc văn bản người dùng nhập; C cần ảnh.
"""

from __future__ import annotations

import difflib
import html
import re
import time

from modules import pipeline as P
from modules.benchmark import analyze_answer_citations
from modules.ocr_extractor import parse_ingredients_from_ocr

CONFIGS = ("A", "B", "C")
CONFIG_TITLES = {
    "A": "A · LLM thuần (không RAG)",
    "B": "B · Văn bản + RAG",
    "C": "C · Ảnh + OCR + RAG",
}
PROVIDER_TITLES = {"gemini": "Gemini", "ollama": "Qwen2.5 (Ollama)"}


def with_ingredient_prefix(text: str) -> str:
    """Thêm 'Thành phần:' nếu chưa có (đồng bộ với lúc chạy đánh giá hàng loạt)."""
    text = str(text or "").strip()
    if not text:
        return ""
    lowered = text.casefold()
    if lowered.startswith("thành phần") or lowered.startswith("ingredients"):
        return text
    return f"Thành phần: {text}"


def provider_ready(provider: str) -> tuple[bool, str]:
    """Nhà cung cấp LLM có thể được gọi không (có khóa Gemini / Ollama đang chạy)."""
    if provider == "ollama":
        return P.ollama_status()
    if provider == "gemini":
        from modules import llm_keys
        if llm_keys.get_pool().keys:
            return True, "Gemini sẵn sàng"
        return False, "Chưa có khóa Gemini (đặt GEMINI_API_KEYS hoặc GEMINI_API_KEY trong .env)"
    return False, f"Nhà cung cấp không hợp lệ: {provider}"


def ingredient_section(ocr_text: str, reference: str) -> str:
    """
    OCR đọc cả mặt bao bì và "Thành phần" có thể xuất hiện nhiều lần (ví dụ "Thành phần dinh dưỡng").
    Thử từng vị trí "Thành phần", cắt đoạn dài tối đa ~1,3 lần văn bản chuẩn và chọn đoạn khớp nhất.
    """
    ocr_text = ocr_text or ""
    if not reference:
        return ocr_text
    limit = int(len(reference) * 1.3) + 20
    lowered = ocr_text.casefold()
    best, best_score, start = None, -1.0, lowered.find("thành phần")
    while start >= 0:
        window = ocr_text[start:start + limit]
        score = match_ratio(reference, window)
        if score > best_score:
            best, best_score = window, score
        start = lowered.find("thành phần", start + 1)
    return best if best is not None else ocr_text[:limit]


def match_ratio(reference: str, observed: str) -> float:
    """Tỷ lệ ký tự của văn bản chuẩn được OCR đọc đúng (cho phép hoán đổi vị trí các khối thành phần); 1,0 = đọc đúng hoàn toàn."""
    if not reference:
        return 0.0
    ref_work, obs_work = reference, observed or ""
    matched_chars = 0
    while True:
        s = difflib.SequenceMatcher(None, ref_work, obs_work, autojunk=False)
        m = s.find_longest_match(0, len(ref_work), 0, len(obs_work))
        if m.size < 3:
            if m.size == 2 and any(ch.isalnum() for ch in obs_work[m.b:m.b + 2]):
                pass
            else:
                break
        matched_chars += m.size
        ref_work = ref_work[:m.a] + ("\x00" * m.size) + ref_work[m.a + m.size:]
        obs_work = obs_work[:m.b] + ("\x01" * m.size) + obs_work[m.b + m.size:]
    return min(1.0, matched_chars / len(reference))


def run_ocr_once(image_path, sample_key: str = "custom") -> dict:
    """OCR thật một lần cho cấu hình C; lỗi được trả về dạng dict thay vì ném ngoại lệ."""
    try:
        return P.run_perception_pipeline(image_path, sample_key=sample_key, force_real_ocr=True)
    except Exception as exc:  # pragma: no cover - phụ thuộc PaddleOCR
        return {"is_ocr_unavailable": True, "ocr_error": f"{type(exc).__name__}: {exc}"}


def _citations(answer: str) -> tuple[list[str], list[str]]:
    """(thẻ nguồn được công nhận, thẻ lạ) trong câu trả lời."""
    analysis = analyze_answer_citations(answer or "")
    return analysis["recognized"], analysis["unrecognized"]


def _empty_result(config: str, provider: str) -> dict:
    return {
        "config": config, "provider": provider, "model": None, "seconds": 0.0,
        "answer": "", "safety_label": "ABSTAIN", "llm_label": None, "guardrail_status": None,
        "citations": [], "unrecognized_citations": [], "passages": [], "mentions": 0, "error": None,
    }


def run_config(config, provider, *, persona, question, label_text, product_name, ocr_result,
               resources, reference_text=None) -> dict:
    """Chạy một cấu hình với một nhà cung cấp; luôn trả dict kết quả (lỗi nằm ở trường `error`)."""
    res = _empty_result(config, provider)
    ready, message = provider_ready(provider)
    if not ready:
        res["error"] = message
        return res

    previous = P.get_llm_provider()
    P.set_llm_provider(provider)
    started = time.perf_counter()
    try:
        label_text = with_ingredient_prefix(label_text)
        if config == "A":
            if not label_text:
                res["error"] = "Thiếu văn bản thành phần cho cấu hình A"
                return res
            answer = P.generate_llm_only_answer(label_text, persona, question)
            res["answer"] = answer or ""
            res["safety_label"] = P.extract_safety_label(answer or "")
            res["llm_label"] = res["safety_label"]
        else:
            if config == "B":
                if not label_text:
                    res["error"] = "Thiếu văn bản thành phần cho cấu hình B"
                    return res
                mentions = parse_ingredients_from_ocr(label_text, include_allergen_statements=True)
                confidence = 1.0
            else:
                if not ocr_result or ocr_result.get("is_ocr_unavailable"):
                    res["error"] = (ocr_result or {}).get("ocr_error") or "Cấu hình C cần ảnh và OCR khả dụng"
                    return res
                ocr_text = ocr_result.get("full_text", "")
                mentions = parse_ingredients_from_ocr(ocr_text, include_allergen_statements=True)
                confidence = float(ocr_result.get("avg_confidence", 0.0) or 0.0)
            grounded = P.run_grounded_pipeline(
                mentions, confidence, product_name, persona, question,
                resources["kb_additives"], resources["hazards"], resources["fndds"], resources["off_products"],
            )
            answer = grounded.get("llm_answer")
            res["answer"] = answer or ""
            res["safety_label"] = grounded.get("safety_label", "ABSTAIN")
            res["llm_label"] = P.extract_safety_label(answer or "") if answer else None
            res["guardrail_status"] = (grounded.get("evaluation") or {}).get("status")
            res["mentions"] = len(mentions)
            res["passages"] = [
                {"citation_id": p.get("citation_id"), "locator": p.get("locator"), "kind": p.get("kind")}
                for p in grounded.get("retrieved_passages") or []
            ]
        res["citations"], res["unrecognized_citations"] = _citations(res["answer"])
        if not res["answer"]:
            res["error"] = "Mô hình ngôn ngữ không trả về câu trả lời (kiểm tra khóa/hạn mức Gemini hoặc Ollama)"
        else:
            res["model"] = P.get_last_llm_model()
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        res["seconds"] = round(time.perf_counter() - started, 1)
        P.set_llm_provider(previous)
    return res


def run_comparison(providers, configs, *, persona, question, label_text, product_name,
                   image_path, sample_key, resources, progress_cb=None) -> dict:
    """
    Chạy tuần tự: với mỗi nhà cung cấp lần lượt các cấu hình. OCR chạy một lần và dùng chung.
    `progress_cb(result, done, total, ocr_info)` được gọi sau mỗi kết quả để giao diện cập nhật dần.
    """
    label_text = with_ingredient_prefix(label_text)  # cùng dạng với văn bản OCR để CER không bị lệch
    ocr_info = {"requested": "C" in configs, "unavailable": False, "error": None,
                "text": None, "section": None, "reference": label_text or None, "match_ratio": None, "seconds": 0.0}
    ocr_result = None
    if "C" in configs:
        if image_path is None:
            ocr_info.update(unavailable=True, error="Không có ảnh: cấu hình C bị bỏ qua")
        else:
            started = time.perf_counter()
            ocr_result = run_ocr_once(image_path, sample_key)
            ocr_info["seconds"] = round(time.perf_counter() - started, 1)
            if ocr_result.get("is_ocr_unavailable"):
                ocr_info.update(unavailable=True, error=ocr_result.get("ocr_error") or "OCR không khả dụng")
            else:
                ocr_info["text"] = ocr_result.get("full_text", "")
                if label_text:
                    ocr_info["section"] = ingredient_section(ocr_info["text"], label_text)
                    ocr_info["match_ratio"] = round(match_ratio(label_text, ocr_info["section"]), 4)

    total = len(providers) * len(configs)
    results = []
    for provider in providers:
        for config in configs:
            result = run_config(
                config, provider, persona=persona, question=question, label_text=label_text,
                product_name=product_name, ocr_result=ocr_result, resources=resources,
            )
            results.append(result)
            if progress_cb:
                progress_cb(result, len(results), total, ocr_info)
    return {"results": results, "ocr": ocr_info, "providers": list(providers), "configs": list(configs)}


def _ok(result):
    return result is not None and not result.get("error")


def _by_key(results):
    return {(r["provider"], r["config"]): r for r in results}


def summarize(results, providers, ocr_info=None) -> list[str]:
    """Nhận xét ngắn gắn kết quả với câu hỏi nghiên cứu (A→B: giá trị của RAG, B→C: ảnh hưởng OCR)."""
    by = _by_key(results)
    lines = []
    for provider in providers:
        name = PROVIDER_TITLES.get(provider, provider)
        a, b, c = by.get((provider, "A")), by.get((provider, "B")), by.get((provider, "C"))
        if _ok(a) and _ok(b):
            change = (f"RAG đổi nhãn {a['safety_label']} → {b['safety_label']}"
                      if a["safety_label"] != b["safety_label"] else f"RAG không đổi nhãn ({a['safety_label']})")
            lines.append(f"**{name} · A→B:** {change}; trích dẫn {len(a['citations'])} → {len(b['citations'])}, "
                         f"đoạn truy xuất 0 → {len(b['passages'])}.")
        if _ok(b) and _ok(c):
            change = (f"nhãn đổi {b['safety_label']} → {c['safety_label']}"
                      if b["safety_label"] != c["safety_label"] else f"nhãn không đổi ({b['safety_label']})")
            ratio = ocr_info.get("match_ratio") if ocr_info else None
            ocr_note = f"; OCR đọc đúng {ratio:.1%} ký tự của mục thành phần" if ratio is not None else ""
            lines.append(f"**{name} · B→C:** đọc ảnh làm {change}{ocr_note}.")
    if len(providers) == 2:
        names = [PROVIDER_TITLES.get(p, p) for p in providers]
        for config in sorted({r["config"] for r in results}):
            x, y = by.get((providers[0], config)), by.get((providers[1], config))
            if _ok(x) and _ok(y):
                same = "giống nhau" if x["safety_label"] == y["safety_label"] else "KHÁC nhau"
                lines.append(f"**Cấu hình {config} · {names[0]} so với {names[1]}:** nhãn {x['safety_label']} / "
                             f"{y['safety_label']} ({same}); thời gian {x['seconds']}s / {y['seconds']}s.")
    errors = [f"{PROVIDER_TITLES.get(r['provider'], r['provider'])}/{r['config']}: {r['error']}"
              for r in results if r.get("error")]
    if errors:
        lines.append("⚠️ Có ô không chạy được: " + "; ".join(errors))
    return lines


def _split_items_with_delimiters(text: str) -> list[tuple[str, str]]:
    """Tách văn bản thành các thành phần kèm dấu phân cách (bảo toàn số thập phân như 34,242%)."""
    pattern = r'((?<!\d),(?!\d)|;|\.\s+|\n+)'
    tokens = re.split(pattern, text)
    items = []
    for i in range(0, len(tokens), 2):
        chunk = tokens[i]
        sep = tokens[i + 1] if i + 1 < len(tokens) else ""
        if chunk or sep:
            items.append((chunk, sep))
    return items


def _diff_single_item(ref_item: str, obs_item: str) -> str:
    """Tô màu sai khác cục bộ bên trong một thành phần."""
    if not ref_item:
        return f'<span style="background:#fecaca">{html.escape(obs_item)}</span>'
    if ref_item == obs_item:
        return html.escape(obs_item)
    matcher = difflib.SequenceMatcher(None, ref_item, obs_item, autojunk=False)
    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(html.escape(obs_item[j1:j2]))
        else:
            if j2 > j1:
                parts.append(f'<span style="background:#fecaca">{html.escape(obs_item[j1:j2])}</span>')
            if tag in ("delete", "replace") and i2 > i1 and j2 == j1:
                parts.append('<span style="background:#fde68a" title="ký tự bị thiếu">▯</span>')
    return "".join(parts)


def diff_html(reference: str, observed: str, limit: int = 4000) -> str:
    """HTML tô sai khác giữa văn bản chuẩn và văn bản OCR:
    - Ký tự/từ sai chính tả quang học: màu đỏ (#fecaca).
    - Các thành phần có trong nhãn chuẩn nhưng bị lệch vị trí dòng/thứ tự: giữ nguyên màu bình thường.
    - Ký tự thiếu: ▯ màu vàng (#fde68a).
    - Đoạn thông tin thừa ngoài vùng thành phần: màu đỏ (#fecaca).
    """
    reference, observed = (reference or "")[:limit], (observed or "")[:limit]
    if not reference or not observed:
        if observed:
            return f'<span style="background:#fecaca">{html.escape(observed)}</span>'
        return ""
    if reference == observed:
        return html.escape(observed).replace("\n", "<br>")

    # Tách nhãn chuẩn thành danh sách các thành phần độc lập
    ref_pattern = r'(?<!\d),(?!\d)|;|\.\s+|\n+'
    ref_items = [it.strip() for it in re.split(ref_pattern, reference) if it.strip()]

    # Tách văn bản OCR đọc được thành các đoạn kèm dấu phân cách
    obs_tokens = _split_items_with_delimiters(observed)

    used_ref_indices = set()
    out_parts = []

    for chunk, sep in obs_tokens:
        clean_chunk = chunk.strip()
        if not clean_chunk:
            out_parts.append(html.escape(chunk + sep))
            continue

        best_idx = None
        best_score = -1.0
        clean_lower = clean_chunk.casefold()

        # 1. Thử khớp chính xác hoặc quan hệ tập con
        for idx, r_item in enumerate(ref_items):
            r_lower = r_item.casefold()
            if clean_lower == r_lower or clean_lower in r_lower or r_lower in clean_lower:
                score = len(clean_lower) / max(len(r_lower), 1)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        # 2. So khớp mờ
        if best_score < 0.6:
            for idx, r_item in enumerate(ref_items):
                score = difflib.SequenceMatcher(None, r_item.casefold(), clean_lower).ratio()
                if score > best_score:
                    best_score = score
                    best_idx = idx

        # Ngưỡng khớp thành phần (>= 0.45):
        if best_score >= 0.45 and best_idx is not None:
            matched_ref = ref_items[best_idx]
            used_ref_indices.add(best_idx)
            l_ws = chunk[:len(chunk) - len(chunk.lstrip())]
            r_ws = chunk[len(chunk.rstrip()):]
            diffed = _diff_single_item(matched_ref, clean_chunk)
            out_parts.append(l_ws + diffed + r_ws + html.escape(sep))
        else:
            # Đoạn văn bản hoàn toàn không thuộc thành phần chuẩn (thông tin thừa ngoài nhãn)
            out_parts.append(f'<span style="background:#fecaca">{html.escape(chunk)}</span>' + html.escape(sep))

    return "".join(out_parts).replace("\n", "<br>")
