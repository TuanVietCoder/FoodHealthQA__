"""
Sinh prediction files thật cho 3 cấu hình A/B/C bằng Gemini API.

Chạy 462 câu benchmark × 3 configs = 1386 API calls.
Hỗ trợ --resume (tiếp tục từ chỗ dừng) và --limit (giới hạn số câu).

Cấu hình:
  A (LLM-only):  Gemini trả lời từ kiến thức, KHÔNG có evidence
  B (GT+RAG):    Gemini + evidence_excerpt từ gold benchmark (upper bound)
  C (OCR+RAG):   Gemini + evidence tra cứu bằng HybridRetriever từ KB

Yêu cầu: GEMINI_API_KEY hoặc GOOGLE_API_KEY (env hoặc .env file)

Chạy:
  python scripts/generate_predictions.py
  python scripts/generate_predictions.py --limit 20
  python scripts/generate_predictions.py --resume --configs A B
  python scripts/generate_predictions.py --configs C --limit 50
"""

import argparse
import json
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_NAME = "gemini-3.8-flash"

_SAFETY_LABEL_PATTERN = re.compile(
    r"\b(SAFE|CAUTION|WARNING|CRITICAL_ALERT|ABSTAIN)\b", re.IGNORECASE
)

_PROMPT_CONFIG_A = """Bạn là chuyên gia tư vấn an toàn thực phẩm Việt Nam.
Trả lời câu hỏi dưới đây DỰA TRÊN KIẾN THỨC CỦA BẠN (không có cơ sở dữ liệu tra cứu).

## Hồ sơ người dùng:
{persona_info}

## Câu hỏi:
{question}

## Yêu cầu:
- Trả lời bằng tiếng Việt, ngắn gọn, chính xác.
- Nếu câu hỏi ngoài phạm vi hoặc bạn không chắc, nói rõ và khuyên tham vấn chuyên gia.
- Cuối câu trả lời, ghi nhãn an toàn trên 1 dòng riêng: SAFETY_LABEL: <SAFE|CAUTION|WARNING|CRITICAL_ALERT|ABSTAIN>

## Trả lời:
"""

_PROMPT_CONFIG_BC = """Bạn là chuyên gia tư vấn an toàn thực phẩm Việt Nam.
Dựa trên bằng chứng khoa học bên dưới, hãy trả lời câu hỏi của người dùng.

## Hồ sơ người dùng:
{persona_info}

## Câu hỏi:
{question}

## Bằng chứng:
{evidence}

## Nguồn cho phép trích dẫn:
{sources}

## Yêu cầu:
- Trả lời bằng tiếng Việt, ngắn gọn, chính xác.
- Mỗi câu phải có trích dẫn nguồn trong ngoặc vuông [Nguồn].
- Chỉ dùng nguồn trong danh sách cho phép, không bịa nguồn.
- Nếu bằng chứng không đủ, nói rõ và khuyên tham vấn chuyên gia.
- Cuối câu trả lời, ghi nhãn an toàn trên 1 dòng riêng: SAFETY_LABEL: <SAFE|CAUTION|WARNING|CRITICAL_ALERT|ABSTAIN>

## Trả lời:
"""


def _get_api_key():
    from dotenv import dotenv_values
    env_path = os.path.join(PROJECT_ROOT, ".env")
    file_values = dotenv_values(env_path) if os.path.isfile(env_path) else {}
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or file_values.get("GEMINI_API_KEY")
        or file_values.get("GOOGLE_API_KEY")
    )


def _load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_existing_predictions(path):
    """Load predictions đã có (cho --resume)."""
    if not os.path.isfile(path):
        return {}
    existing = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                existing[rec["id"]] = rec
    return existing


def _extract_safety_label(answer_text):
    """Trích xuất safety label từ câu trả lời LLM."""
    lines = answer_text.strip().split("\n")
    for line in reversed(lines):
        if "SAFETY_LABEL" in line.upper():
            match = _SAFETY_LABEL_PATTERN.search(line)
            if match:
                return match.group(1).upper()

    match = _SAFETY_LABEL_PATTERN.search(answer_text)
    if match:
        return match.group(1).upper()
    return "SAFE"


def _get_persona_for_question(question_record, personas_map):
    """Xác định persona phù hợp cho câu hỏi dựa trên domain/sub_type/text."""
    domain = question_record.get("domain", "")
    question_text = question_record.get("question", "").lower()

    if domain == "condition_safety" or domain == "pregnancy_safety" or domain == "children_safety":
        for pid, persona in personas_map.items():
            name_lower = persona.get("name", "").lower()
            if name_lower and name_lower in question_text:
                return persona

    if "mang thai" in question_text or "thai kỳ" in question_text:
        return personas_map.get("pregnant_woman", personas_map.get("healthy_adult"))
    if "trẻ" in question_text or "trẻ nhỏ" in question_text:
        return personas_map.get("child_5yo", personas_map.get("healthy_adult"))
    if "tăng huyết áp" in question_text or "huyết áp" in question_text:
        return personas_map.get("hypertension_patient", personas_map.get("healthy_adult"))
    if "đái tháo đường" in question_text or "tiểu đường" in question_text:
        return personas_map.get("diabetes_type2", personas_map.get("healthy_adult"))
    if "suy thận" in question_text or "thận" in question_text:
        return personas_map.get("ckd_patient", personas_map.get("healthy_adult"))
    if "hen suyễn" in question_text or "hen" in question_text:
        return personas_map.get("asthma_patient", personas_map.get("healthy_adult"))
    if "cao tuổi" in question_text or "75 tuổi" in question_text:
        return personas_map.get("elderly_patient", personas_map.get("healthy_adult"))
    if "dị ứng" in question_text:
        return personas_map.get("allergy_patient", personas_map.get("healthy_adult"))
    if "pku" in question_text or "phenylketon" in question_text:
        return personas_map.get("pku_allergy_patient", personas_map.get("healthy_adult"))

    return personas_map.get("healthy_adult", list(personas_map.values())[0])


def _build_retriever_evidence(question_text, retriever, additives):
    """Dùng HybridRetriever tra cứu evidence từ KB cho Config C."""
    evidence_parts = []
    source_ids = set()

    candidates = retriever.retrieve(question_text, top_k=5)
    for entity, score, method in candidates:
        if score <= 0:
            continue
        ins = entity.get("ins_code", "")
        name = entity.get("canonical_name", entity.get("canonical_name_vi", ""))
        func = entity.get("function", "")
        adi = entity.get("jecfa_adi", entity.get("adi_value", ""))

        evidence_parts.append(
            f"- {name} ({ins}): Chức năng: {func}. ADI: {adi}."
        )

        for contra in entity.get("contraindications", []):
            evidence_parts.append(
                f"  Lưu ý sức khỏe: {contra.get('condition', '')} — "
                f"{contra.get('message', '')} (mức: {contra.get('severity', '')})"
            )

        source_id = f"KB:{ins}" if ins else f"KB:{name}"
        source_ids.add(source_id)

    if not evidence_parts:
        evidence_parts.append("Không tìm thấy bằng chứng liên quan trong Knowledge Base.")

    return "\n".join(evidence_parts), sorted(source_ids)


def _call_gemini(client, prompt, retries=3, delay=2.0):
    """Gọi Gemini API với retry logic."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 1024},
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as e:
            if attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"    [RETRY {attempt+1}] {e.__class__.__name__}: {str(e)[:80]}. Chờ {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    [ERROR] Hết retry: {e}")
                return None
    return None


def generate_for_config(
    config_name, references, personas_map, client, retriever, additives,
    output_path, existing, delay_seconds=2.0
):
    """Sinh predictions cho 1 cấu hình."""
    total = len(references)
    skipped = 0
    generated = 0
    failed = 0

    with open(output_path, "a", encoding="utf-8") as out:
        for i, ref in enumerate(references):
            qid = ref["id"]

            if qid in existing:
                skipped += 1
                continue

            persona = _get_persona_for_question(ref, personas_map)
            conditions = ", ".join(persona.get("conditions", [])) or "Không có"
            allergies = ", ".join(persona.get("allergies", [])) or "Không có"
            persona_info = (
                f"Tên: {persona.get('name', 'N/A')}. "
                f"Bệnh lý: {conditions}. Dị ứng: {allergies}."
            )
            question = ref["question"]

            if config_name == "A":
                prompt = _PROMPT_CONFIG_A.format(
                    persona_info=persona_info,
                    question=question,
                )
            elif config_name == "B":
                evidence = ref.get("evidence_excerpt", "")
                sources = ref.get("source_ids", [])
                if not evidence:
                    evidence = ref.get("reference_answer", "")
                prompt = _PROMPT_CONFIG_BC.format(
                    persona_info=persona_info,
                    question=question,
                    evidence=evidence,
                    sources=", ".join(sources) if sources else "Không có nguồn cụ thể",
                )
            elif config_name == "C":
                evidence_text, source_ids = _build_retriever_evidence(
                    question, retriever, additives
                )
                prompt = _PROMPT_CONFIG_BC.format(
                    persona_info=persona_info,
                    question=question,
                    evidence=evidence_text,
                    sources=", ".join(source_ids) if source_ids else "KB (Knowledge Base)",
                )
            else:
                continue

            answer = _call_gemini(client, prompt)

            if answer:
                safety_label = _extract_safety_label(answer)
                answer_clean = re.sub(
                    r"\n*SAFETY_LABEL\s*:\s*\w+\s*$", "", answer, flags=re.IGNORECASE
                ).strip()

                pred = {
                    "id": qid,
                    "answer": answer_clean,
                    "safety_label": safety_label,
                    "config": config_name,
                    "model": MODEL_NAME,
                }
                out.write(json.dumps(pred, ensure_ascii=False) + "\n")
                out.flush()
                existing[qid] = pred
                generated += 1
            else:
                pred = {
                    "id": qid,
                    "answer": "",
                    "safety_label": "ABSTAIN",
                    "config": config_name,
                    "model": MODEL_NAME,
                    "error": True,
                }
                out.write(json.dumps(pred, ensure_ascii=False) + "\n")
                out.flush()
                existing[qid] = pred
                failed += 1

            done = skipped + generated + failed
            if (done % 20 == 0) or (done == total):
                print(f"  [{config_name}] {done}/{total} — generated={generated} skipped={skipped} failed={failed}")

            if generated > 0 and i < total - 1:
                time.sleep(delay_seconds)

    return {"generated": generated, "skipped": skipped, "failed": failed, "total": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--benchmark",
        default=os.path.join(DATA_DIR, "qa_benchmark_merged_draft.jsonl"),
        help="Benchmark JSONL (mặc định: qa_benchmark_merged_draft.jsonl)",
    )
    parser.add_argument(
        "--configs", nargs="+", default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Cấu hình cần chạy (mặc định: A B C)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số câu (mặc định: tất cả)")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ predictions đã có")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay giữa API calls (giây)")
    parser.add_argument("--output-dir", default=DATA_DIR, help="Thư mục lưu predictions")
    args = parser.parse_args()

    api_key = _get_api_key()
    if not api_key:
        print("❌ Chưa cấu hình API key!")
        print("   Đặt biến môi trường GEMINI_API_KEY hoặc GOOGLE_API_KEY,")
        print("   hoặc tạo file .env trong thư mục project.")
        sys.exit(1)

    print("=" * 70)
    print("  FoodHealthQA — Generate Real Predictions (Gemini API)")
    print("=" * 70)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Configs: {', '.join(args.configs)}")
    print(f"  Delay: {args.delay}s giữa mỗi API call")

    references = _load_jsonl(args.benchmark)
    if args.limit:
        references = references[:args.limit]
    print(f"\n  Benchmark: {len(references)} câu hỏi")

    print("\n  Đang nạp Knowledge Base...")
    from modules.pipeline import load_knowledge_base, load_personas
    additives = load_knowledge_base(os.path.join(PROJECT_ROOT, "data"))
    print(f"  KB: {len(additives)} phụ gia")

    personas_list = load_personas(os.path.join(PROJECT_ROOT, "data"))
    personas_map = {p["id"]: p for p in personas_list}
    print(f"  Personas: {len(personas_map)} hồ sơ")

    retriever = None
    if "C" in args.configs:
        print("\n  Đang xây dựng HybridRetriever index...")
        from modules.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(alpha=0.5)
        retriever.build_index(additives)
        print("  HybridRetriever sẵn sàng")

    from google import genai
    client = genai.Client(api_key=api_key)

    est_calls = len(references) * len(args.configs)
    est_minutes = est_calls * args.delay / 60
    print(f"\n  Ước tính: {est_calls} API calls ≈ {est_minutes:.0f} phút")
    print(f"  (Dùng --resume nếu bị gián đoạn)")

    for config_name in args.configs:
        output_path = os.path.join(args.output_dir, f"predictions_config_{config_name.lower()}.jsonl")

        existing = {}
        if args.resume:
            existing = _load_existing_predictions(output_path)
            if existing:
                print(f"\n  [RESUME] Config {config_name}: đã có {len(existing)} predictions")
        elif os.path.isfile(output_path):
            try:
                os.rename(output_path, output_path + ".bak")
            except OSError:
                pass

        print(f"\n{'─' * 50}")
        print(f"  Config {config_name} → {output_path}")
        print(f"{'─' * 50}")

        stats = generate_for_config(
            config_name=config_name,
            references=references,
            personas_map=personas_map,
            client=client,
            retriever=retriever,
            additives=additives,
            output_path=output_path,
            existing=existing,
            delay_seconds=args.delay,
        )

        print(f"\n  ✅ Config {config_name}: generated={stats['generated']} "
              f"skipped={stats['skipped']} failed={stats['failed']}")

    print(f"\n{'=' * 70}")
    print("  HOÀN TẤT — Prediction files:")
    for config_name in args.configs:
        path = os.path.join(args.output_dir, f"predictions_config_{config_name.lower()}.jsonl")
        count = len(_load_existing_predictions(path)) if os.path.isfile(path) else 0
        print(f"    {config_name}: {path} ({count} records)")

    print(f"\n  Tiếp theo, chạy evaluation:")
    pred_args = " ".join(
        f"--pred-{c.lower()} data/predictions_config_{c.lower()}.jsonl"
        for c in args.configs
    )
    print(f"    python scripts/run_evaluation.py {pred_args}")


if __name__ == "__main__":
    main()
