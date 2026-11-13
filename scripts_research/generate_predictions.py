"""Generate real predictions for the canonical three-configuration experiment.

A: Ground-truth label text + exact persona + question -> LLM only.
B: Ground-truth text -> normalization/retrieval/guardrail -> grounded LLM.
C: Image -> PaddleOCR + VietOCR -> the exact same downstream path as B.

Gold answers, gold labels and gold citations are never passed to prediction code.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from modules.ocr_extractor import (  # noqa: E402
    extract_with_paddle_ocr,
    is_paddleocr_available,
    parse_ingredients_from_ocr,
)
from modules.pipeline import (  # noqa: E402
    build_persona_from_benchmark_record,
    extract_safety_label,
    generate_llm_only_answer,
    load_fndds_nutrition,
    load_food_hazards,
    load_knowledge_base,
    load_openfoodfacts_vietnam,
    load_personas,
    run_grounded_pipeline,
)
from modules.benchmark_manifest import (  # noqa: E402
    get_active_benchmark_path,
    get_default_eval_benchmark_path,
    is_evaluation_eligible,
)

DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
PIPELINE_VERSION = "canonical_abc_v1"


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def _load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_existing(path):
    if not path.is_file():
        return {}
    return {record["id"]: record for record in _load_jsonl(path)}


def _ground_truth_text(record):
    text = str(record.get("product_ingredients") or record.get("ingredients") or "").strip()
    if not text:
        return ""
    lowered = text.casefold()
    if lowered.startswith("thành phần") or lowered.startswith("ingredients"):
        return text
    return f"Thành phần: {text}"


def _resolve_image(record):
    raw = str(record.get("image_ref") or "").strip().replace("\\", "/")
    if not raw:
        return None
    candidates = [
        PROJECT_ROOT / raw,
        ASSETS_DIR / raw,
        ASSETS_DIR / "images" / Path(raw).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _serializable_ocr_result(result):
    return {
        "full_text": result.get("full_text", ""),
        "raw_text_before_norm": result.get("raw_text_before_norm", ""),
        "text_after_ocr_dict": result.get("text_after_ocr_dict", ""),
        "avg_confidence": float(result.get("avg_confidence", 0.0) or 0.0),
        "mentions": result.get("mentions", []),
        "engine_name": result.get("engine_name", ""),
        "is_ocr_unavailable": bool(result.get("is_ocr_unavailable", False)),
        "ocr_error": result.get("ocr_error", ""),
    }


def _load_ocr_cache(path):
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("pipeline_version") != PIPELINE_VERSION:
            return {}
        return payload.get("images", {})
    except (OSError, ValueError, TypeError):
        return {}


def _save_ocr_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pipeline_version": PIPELINE_VERSION, "images": cache}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False)
    os.replace(temporary, path)


def _ocr_input(record, cache, cache_path):
    image_path = _resolve_image(record)
    if image_path is None:
        return None, "missing_image"
    cache_key = str(image_path).casefold()
    if cache_key not in cache:
        result = extract_with_paddle_ocr(str(image_path), sample_key=image_path.stem)
        cache[cache_key] = _serializable_ocr_result(result)
        _save_ocr_cache(cache_path, cache)
    result = cache[cache_key]
    if result.get("is_ocr_unavailable"):
        return result, result.get("ocr_error") or "ocr_unavailable"
    return result, None


def _base_prediction(record, config_name):
    return {
        "id": record["id"],
        "config": config_name,
        "evaluation_type": "real",
        "pipeline_version": PIPELINE_VERSION,
        "input_source": {
            "A": "ground_truth_text",
            "B": "ground_truth_text",
            "C": "image_ocr",
        }[config_name],
    }


def _predict_one(config_name, record, resources, ocr_cache, ocr_cache_path):
    prediction = _base_prediction(record, config_name)
    persona = build_persona_from_benchmark_record(record, resources["personas"])
    question = str(record.get("question") or "")
    product_name = str(record.get("product_name") or "")

    if config_name == "A":
        label_text = _ground_truth_text(record)
        if not label_text:
            prediction.update(answer="", safety_label="ABSTAIN", error="missing_ground_truth_text")
            return prediction
        answer = generate_llm_only_answer(label_text, persona, question)
        prediction.update(
            answer=answer or "",
            safety_label=extract_safety_label(answer),
            llm_safety_label=extract_safety_label(answer),
            error=None if answer else "llm_generation_failed",
        )
        return prediction

    if config_name == "B":
        label_text = _ground_truth_text(record)
        if not label_text:
            prediction.update(answer="", safety_label="ABSTAIN", error="missing_ground_truth_text")
            return prediction
        mentions = parse_ingredients_from_ocr(label_text, include_allergen_statements=True)
        perception_confidence = 1.0
        prediction["ground_truth_text"] = label_text
    else:
        ocr_result, ocr_error = _ocr_input(record, ocr_cache, ocr_cache_path)
        if ocr_error:
            prediction.update(answer="", safety_label="ABSTAIN", error=ocr_error)
            if ocr_result:
                prediction["ocr"] = ocr_result
            return prediction
        mentions = parse_ingredients_from_ocr(
            ocr_result.get("full_text", ""),
            include_allergen_statements=True,
        )
        perception_confidence = float(ocr_result.get("avg_confidence", 0.0) or 0.0)
        prediction["ocr"] = {
            "full_text": ocr_result.get("full_text", ""),
            "raw_text_before_norm": ocr_result.get("raw_text_before_norm", ""),
            "text_after_ocr_dict": ocr_result.get("text_after_ocr_dict", ""),
            "avg_confidence": perception_confidence,
            "engine_name": ocr_result.get("engine_name", ""),
        }

    grounded = run_grounded_pipeline(
        mentions=mentions,
        avg_ocr_conf=perception_confidence,
        product_name=product_name,
        persona=persona,
        question=question,
        kb_additives=resources["kb_additives"],
        hazards_list=resources["hazards"],
        fndds_list=resources["fndds"],
        off_products=resources["off_products"],
    )
    answer = grounded.get("llm_answer")
    evaluation = grounded.get("evaluation", {})
    prediction.update(
        answer=answer or "",
        safety_label=grounded.get("safety_label", "ABSTAIN"),
        llm_safety_label=extract_safety_label(answer),
        guardrail_status=evaluation.get("status", "ABSTAIN"),
        perception_confidence=perception_confidence,
        retrieval_confidence=evaluation.get("s_retrieval", 0.0),
        total_confidence=evaluation.get("s_total", 0.0),
        citations=evaluation.get("citations", []),
        citation_validation=grounded.get("citation_validation"),
        normalized_mentions=mentions,
        error=None if answer else "llm_generation_failed",
    )
    return prediction


def _load_resources():
    return {
        "kb_additives": load_knowledge_base(str(DATA_DIR)),
        "personas": load_personas(str(DATA_DIR)),
        "hazards": load_food_hazards(str(DATA_DIR)),
        "fndds": load_fndds_nutrition(str(DATA_DIR)),
        "off_products": load_openfoodfacts_vietnam(str(DATA_DIR)),
    }


def _is_good(prediction):
    """Bản ghi đạt yêu cầu: không lỗi và có câu trả lời. Bản ghi lỗi sẽ được chạy lại khi resume."""
    return bool(prediction) and not prediction.get("error") and bool(str(prediction.get("answer", "")).strip())


def _atomic_write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_good_predictions(path):
    """Đọc file dự đoán, giữ bản ghi tốt (bản ghi cuối thắng nếu trùng id), loại bản ghi lỗi để chạy lại."""
    good = {}
    dropped = 0
    for record in (_load_jsonl(path) if path.is_file() else []):
        if _is_good(record):
            good[record["id"]] = record
        else:
            dropped += 1
    return good, dropped


def _print_status(output_dir, references, configs):
    ids = {record["id"] for record in references}
    print(f"Thư mục: {output_dir} | tổng câu cần chạy: {len(ids)}")
    for config_name in configs:
        path = output_dir / f"predictions_config_{config_name.lower()}.jsonl"
        good, dropped = _load_good_predictions(path)
        done = len(ids & set(good))
        print(f"  [{config_name}] hoàn thành {done}/{len(ids)} ({100 * done / max(len(ids), 1):.1f}%), "
              f"lỗi cần chạy lại: {dropped}")


def main():
    from modules import llm_keys
    from modules.pipeline import (
        get_last_llm_model,
        get_llm_provider,
        get_ollama_model,
        set_llm_provider,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=get_default_eval_benchmark_path(DATA_DIR, "jsonl"))
    parser.add_argument(
        "--eval-750",
        action="store_true",
        help="Sử dụng bộ đánh giá phân tầng chuẩn 750 câu (data/foodhealthqa_eval_set.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Mặc định: data/runs/<llm-provider>/ để kết quả Gemini và Ollama không đè lên nhau",
    )
    parser.add_argument("--configs", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--limit", type=int, default=0, help="Chỉ lấy N câu đầu của benchmark (thử nghiệm)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Chạy tiếp: giữ bản ghi tốt, chạy lại bản ghi lỗi/rỗng, không ghi đè kết quả đã có",
    )
    parser.add_argument(
        "--max-new", type=int, default=0,
        help="Dừng sau khi sinh N dự đoán mới trong lần chạy này (0 = chạy đến hết hoặc hết hạn mức)",
    )
    parser.add_argument(
        "--order", choices=["record", "config"], default="record",
        help="record: mỗi câu chạy lần lượt A,B,C (các cấu hình tiến cùng nhau, hợp lý khi bị giới hạn hạn mức); "
             "config: chạy hết A rồi tới B, C",
    )
    parser.add_argument("--status", action="store_true", help="Chỉ in tiến độ rồi thoát")
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Debug only: include records still waiting for expert adjudication",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between LLM calls in seconds")
    parser.add_argument(
        "--ocr-cache",
        default=str(DATA_DIR / "ocr_cache_config_c.json"),
        help="Persistent OCR cache; one OCR run per unique image",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "ollama"],
        default=None,
        help="LLM sinh câu trả lời: gemini (mặc định) hoặc ollama (Qwen2.5 cục bộ; đổi model bằng OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--keys-file", default=None,
        help="File danh sách khóa Gemini (mỗi dòng một khóa). Mặc định: GEMINI_API_KEYS trong .env hoặc gemini_keys.txt",
    )
    args = parser.parse_args()
    if args.llm_provider:
        set_llm_provider(args.llm_provider)
    provider = get_llm_provider()

    _load_dotenv()
    if args.keys_file:
        llm_keys.KEYS_FILE = Path(args.keys_file)
        llm_keys._POOL = llm_keys.KeyPool(llm_keys.load_keys(Path(args.keys_file)))
    if args.eval_750:
        args.benchmark = str(DATA_DIR / "foodhealthqa_eval_set.jsonl")
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_file():
        parser.error(f"Benchmark not found: {benchmark_path}")
    if provider == "gemini" and not (llm_keys.get_pool().keys):
        parser.error(
            "Chưa có khóa Gemini: đặt GEMINI_API_KEYS (nhiều khóa, ngăn bằng dấu phẩy) hoặc GEMINI_API_KEY trong .env, "
            "hoặc tạo gemini_keys.txt (xem gemini_keys.example.txt)"
        )
    if provider == "ollama" and not args.status:
        from modules.pipeline import ollama_status
        ok, message = ollama_status()
        if not ok:
            parser.error(message)
    if "C" in args.configs and not args.status and not is_paddleocr_available():
        parser.error("Config C requires the working PaddleOCR + VietOCR perception pipeline")

    references = _load_jsonl(benchmark_path)
    if not args.include_pending:
        references = [record for record in references if is_evaluation_eligible(record)]
    if args.limit > 0:
        references = references[: args.limit]
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / "runs" / provider
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.status:
        _print_status(output_dir, references, args.configs)
        return

    resources = _load_resources()
    ocr_cache_path = Path(args.ocr_cache)
    ocr_cache = _load_ocr_cache(ocr_cache_path)
    model_label = get_ollama_model() if provider == "ollama" else "gemini (chuỗi fallback)"
    print(f"LLM provider: {provider} | model: {model_label} | output: {output_dir}")
    llm_keys.set_strict(True)  # hết hạn mức / mất kết nối → dừng sạch, không ghi bản ghi lỗi hàng loạt

    paths = {c: output_dir / f"predictions_config_{c.lower()}.jsonl" for c in args.configs}
    existing = {}
    for config_name, path in paths.items():
        if args.resume:
            good, dropped = _load_good_predictions(path)
            _atomic_write_jsonl(path, list(good.values()))  # bỏ bản ghi lỗi/trùng, giữ bản ghi tốt
            if dropped:
                print(f"[{config_name}] bỏ {dropped} bản ghi lỗi/rỗng để chạy lại")
        else:
            if path.is_file() and path.stat().st_size > 0:
                parser.error(
                    f"{path} đã có dữ liệu. Dùng --resume để chạy tiếp, hoặc xóa file/đổi --output-dir nếu muốn chạy lại từ đầu."
                )
            good = {}
        existing[config_name] = good

    counts = {c: {"generated": 0, "failed": 0, "skipped": 0} for c in args.configs}
    stop_reason = "completed"
    new_total = 0
    started = time.time()

    if args.order == "record":
        work = [(index, record, c) for index, record in enumerate(references, 1) for c in args.configs]
    else:
        work = [(index, record, c) for c in args.configs for index, record in enumerate(references, 1)]

    handles = {c: open(paths[c], "a", encoding="utf-8") for c in args.configs}
    try:
        for index, record, config_name in work:
            if record["id"] in existing[config_name]:
                counts[config_name]["skipped"] += 1
                continue
            if args.max_new and new_total >= args.max_new:
                stop_reason = f"max_new={args.max_new}"
                break
            try:
                prediction = _predict_one(config_name, record, resources, ocr_cache, ocr_cache_path)
            except (llm_keys.LLMQuotaExhausted, llm_keys.LLMUnavailable) as stop:
                stop_reason = f"{type(stop).__name__}: {stop}"
                print(f"\n⛔ Dừng: {stop_reason}\n   Chạy lại cùng lệnh với --resume (khi có hạn mức/Ollama) để tiếp tục.")
                break
            except Exception as error:
                prediction = _base_prediction(record, config_name)
                prediction.update(
                    answer="",
                    safety_label="ABSTAIN",
                    error=f"{type(error).__name__}: {error}",
                )
            prediction["llm_provider"] = provider
            prediction["llm_model"] = get_last_llm_model() if prediction.get("answer") else None
            handles[config_name].write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handles[config_name].flush()
            counts[config_name]["generated"] += 1
            counts[config_name]["failed"] += int(bool(prediction.get("error")))
            new_total += 1
            print(
                f"[{config_name}] {index}/{len(references)} {record['id']} "
                f"label={prediction.get('safety_label')} error={prediction.get('error') or '-'}"
            )
            if args.delay > 0:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt"
        print("\nĐã dừng bằng Ctrl+C — chạy lại với --resume để tiếp tục.")
    finally:
        for handle in handles.values():
            handle.close()

    ids = {record["id"] for record in references}
    for config_name in args.configs:
        good, _ = _load_good_predictions(paths[config_name])
        c = counts[config_name]
        print(
            f"Config {config_name}: mới={c['generated']}, lỗi={c['failed']}, bỏ qua(đã có)={c['skipped']}, "
            f"hoàn thành {len(ids & set(good))}/{len(ids)} -> {paths[config_name]}"
        )
    if provider == "gemini":
        for row in llm_keys.get_pool().status():
            print(f"  khóa {row['key']}: đã dùng hôm nay={row['used_today']}, hết hạn mức lần chạy này={row['exhausted_this_run']}")
    log_entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "configs": args.configs,
        "new_predictions": new_total,
        "counts": counts,
        "stop_reason": stop_reason,
        "seconds": round(time.time() - started, 1),
    }
    with open(output_dir / "run_log.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print(f"Kết thúc: {stop_reason}")


if __name__ == "__main__":
    main()
