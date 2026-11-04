"""Run the real RAG pipeline on evaluation/golden.json and report quality.

Run from the repo root:  python -m evaluation.run_eval
Quick test:              python -m evaluation.run_eval --limit 3

No mocks: real retrieval, real answer generation, real judge model.
Uses a temporary ChromaDB folder, so backend/chroma_db is never touched.
"""
import argparse
import csv
import json
import shutil
import statistics
import sys
import tempfile
import time

from backend.app.services.answer_generator import AnswerGenerator
from backend.app.services.chroma_service import ChromaService
from backend.app.services.embeddings import EmbeddingsService
from backend.app.services.hybrid_retriever import HybridRetriever
from evaluation.utils import (
    EVAL_DIR, GOLDEN_PATH, check_ollama, gen_model, ingest_documents,
    judge_model, list_docs, normalize, ollama_json,
)
from backend.app.config import Config

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer (taken from the source): {expected}
Source passage the question came from:
\"\"\"
{passage}
\"\"\"
System answer: {answer}

Decide whether the system answer correctly answers the question.
- CORRECT: it gives the same fact as the reference answer (different wording is fine), or another answer that the source passage clearly supports.
- PARTIAL: it is on the right track but incomplete, or mixes the right fact with wrong or unsupported details.
- WRONG: it gives a different fact, contradicts the source, or says it lacks the information.
Reply ONLY with JSON: {{"verdict": "CORRECT" or "PARTIAL" or "WRONG", "reason": "one short sentence"}}"""

REFUSAL_PHRASES = (
    "don't have enough", "do not have enough", "not enough information",
    "not enough context", "cannot find", "can't find", "no information",
    "does not contain", "do not contain", "not mentioned",
)


def judge(question, expected, passage, answer):
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, passage=passage[:1500], answer=answer[:1500]
    )
    try:
        out = ollama_json(prompt, judge_model())
    except Exception as e:
        return "JUDGE_ERROR", str(e)[:150]
    verdict = str(out.get("verdict", "")).strip().upper()
    if verdict not in ("CORRECT", "PARTIAL", "WRONG"):
        return "JUDGE_ERROR", f"unexpected verdict: {verdict!r}"
    return verdict, str(out.get("reason", ""))[:200]


def retrieval_rank(entry, retrieved):
    """1-based rank of the source passage among the retrieved results, or None."""
    target = normalize(entry["source_chunk_text"])
    for i, (doc, meta) in enumerate(zip(retrieved["documents"], retrieved["metadatas"]), 1):
        same_meta = (
            meta.get("file_name") == entry["source_file"]
            and entry.get("source_chunk_index") is not None
            and meta.get("chunk_index") == entry["source_chunk_index"]
        )
        if same_meta or normalize(doc) == target:
            return i
    return None


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only run the first N answerable questions (plus 2 unanswerable)")
    args = ap.parse_args()

    if not GOLDEN_PATH.exists():
        sys.exit("golden.json not found. Run:  python -m evaluation.build_golden")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.limit:
        ans = [g for g in golden if g["answerable"]][: args.limit]
        golden = ans + [g for g in golden if not g["answerable"]][:2]

    problems = check_ollama([Config.LLM_MODEL, Config.EMBEDDING_MODEL, judge_model()])
    if problems:
        sys.exit("\n".join(problems))
    print(f"Answer model: {Config.LLM_MODEL} | Judge model: {judge_model()} | Embeddings: {Config.EMBEDDING_MODEL}")

    tmp = tempfile.mkdtemp(prefix="rag_eval_")
    try:
        chroma = ChromaService(db_path=tmp, embeddings_service=EmbeddingsService())
        print("Ingesting documents (this takes a few minutes)...")
        for name, n in ingest_documents(chroma, list_docs()).items():
            print(f"  {name}: {n} chunks")
        retriever = HybridRetriever(chroma_service=chroma)
        generator = AnswerGenerator()

        results, judge_ok, judge_fail = [], 0, 0
        jsonl = (EVAL_DIR / "results.jsonl").open("w", encoding="utf-8")
        print(f"\nRunning {len(golden)} questions...\n")
        for i, g in enumerate(golden, 1):
            t0 = time.time()
            app_error = ""
            try:
                retrieved = retriever.retrieve(g["question"], n_results=5)
                out = generator.generate(g["question"], retrieved, max_sources=5)
                answer = out.get("answer", "")
            except Exception as e:
                retrieved, answer, app_error = {"documents": [], "metadatas": []}, "", str(e)[:150]
            latency = time.time() - t0

            r = {
                "id": g["id"], "source_file": g["source_file"], "kind": g["kind"],
                "answerable": g["answerable"], "question": g["question"],
                "expected_answer": g["expected_answer"], "app_answer": answer,
                "latency_s": round(latency, 1), "retrieval_rank": None,
                "exact_match": None, "verdict": "", "reason": app_error,
            }
            if g["answerable"]:
                r["retrieval_rank"] = retrieval_rank(g, retrieved)
                r["exact_match"] = normalize(g["expected_answer"]) in normalize(answer)
                if app_error:
                    r["verdict"] = "WRONG"
                    r["reason"] = f"app error: {app_error}"
                else:
                    r["verdict"], r["reason"] = judge(
                        g["question"], g["expected_answer"], g["source_chunk_text"], answer
                    )
                    if r["verdict"] == "JUDGE_ERROR":
                        judge_fail += 1
                    else:
                        judge_ok += 1
                    if judge_fail >= 3 and judge_ok == 0:
                        sys.exit("The judge model failed 3 times in a row, so grading would be meaningless. "
                                 f"Last error: {r['reason']}")
            else:
                refused = any(p in answer.lower() for p in REFUSAL_PHRASES)
                r["verdict"] = "PASS" if refused else "FAIL"
                if app_error:
                    r["verdict"], r["reason"] = "FAIL", f"app error: {app_error}"

            results.append(r)
            jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
            jsonl.flush()
            extra = f"rank={r['retrieval_rank']}" if g["answerable"] else g["kind"]
            print(f"[{i}/{len(golden)}] {r['id']} {r['verdict']:<11} {extra:<10} {r['latency_s']}s  {g['question'][:60]}")
        jsonl.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---------------- summary ----------------
    ans = [r for r in results if r["answerable"]]
    una = [r for r in results if not r["answerable"]]
    judged = [r for r in ans if r["verdict"] in ("CORRECT", "PARTIAL", "WRONG")]
    correct = sum(r["verdict"] == "CORRECT" for r in judged)
    partial = sum(r["verdict"] == "PARTIAL" for r in judged)
    hits = [r for r in ans if r["retrieval_rank"]]
    recall5 = len(hits) / len(ans) if ans else 0
    mrr = sum(1 / r["retrieval_rank"] for r in hits) / len(ans) if ans else 0
    hit_judged = [r for r in judged if r["retrieval_rank"]]
    acc_if_hit = sum(r["verdict"] == "CORRECT" for r in hit_judged) / len(hit_judged) if hit_judged else 0
    refusal = sum(r["verdict"] == "PASS" for r in una)
    lat = [r["latency_s"] for r in results]
    p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] if lat else 0
    exact = sum(bool(r["exact_match"]) for r in ans)

    per_doc = {}
    for r in ans:
        d = per_doc.setdefault(r["source_file"], {"n": 0, "correct": 0, "hit": 0})
        d["n"] += 1
        d["correct"] += r["verdict"] == "CORRECT"
        d["hit"] += bool(r["retrieval_rank"])

    summary = {
        "questions_total": len(results), "answerable": len(ans), "unanswerable": len(una),
        "recall_at_5": round(recall5, 3), "mrr": round(mrr, 3),
        "answer_accuracy_correct": round(correct / len(judged), 3) if judged else None,
        "answer_accuracy_partial_half": round((correct + 0.5 * partial) / len(judged), 3) if judged else None,
        "accuracy_when_source_retrieved": round(acc_if_hit, 3),
        "exact_match_rate": round(exact / len(ans), 3) if ans else None,
        "refusal_accuracy": round(refusal / len(una), 3) if una else None,
        "judge_errors": judge_fail,
        "latency_avg_s": round(statistics.mean(lat), 1) if lat else 0,
        "latency_p95_s": p95, "latency_max_s": max(lat) if lat else 0,
        "models": {"answer": Config.LLM_MODEL, "judge": judge_model(), "embeddings": Config.EMBEDDING_MODEL},
        "per_document": per_doc,
    }

    lines = [
        "# RAG Evaluation Report", "",
        f"- Questions: {len(results)} ({len(ans)} answerable, {len(una)} unanswerable)",
        f"- Answer model: {Config.LLM_MODEL} | Judge: {judge_model()}", "",
        "| Metric | Result |", "|---|---|",
        f"| Recall@5 (source passage in top 5) | {pct(len(hits), len(ans))} ({len(hits)}/{len(ans)}) |",
        f"| MRR | {mrr:.2f} |",
        f"| Answer accuracy (CORRECT) | {pct(correct, len(judged))} ({correct}/{len(judged)}) |",
        f"| Accuracy, PARTIAL counted as half | {pct(correct + 0.5 * partial, len(judged))} |",
        f"| Accuracy when the source was retrieved | {pct(sum(r['verdict'] == 'CORRECT' for r in hit_judged), len(hit_judged))} |",
        f"| Exact-text match (no judge) | {pct(exact, len(ans))} |",
        f"| Correct refusals (unanswerable) | {pct(refusal, len(una))} ({refusal}/{len(una)}) |",
        f"| Latency avg / p95 / max | {summary['latency_avg_s']}s / {p95}s / {summary['latency_max_s']}s |",
        f"| Judge errors | {judge_fail} |", "", "## Per document", "",
        "| Document | Questions | Correct | Source retrieved |", "|---|---|---|---|",
    ]
    for name, d in sorted(per_doc.items()):
        lines.append(f"| {name} | {d['n']} | {pct(d['correct'], d['n'])} | {pct(d['hit'], d['n'])} |")
    bad = [r for r in results if r["verdict"] in ("PARTIAL", "WRONG", "FAIL", "JUDGE_ERROR")]
    lines += ["", f"## Items to review ({len(bad)})", ""]
    for r in bad:
        lines += [f"**{r['id']}** [{r['verdict']}] {r['question']}",
                  f"- Expected: {r['expected_answer']}",
                  f"- App said: {r['app_answer'][:300]}",
                  f"- Judge: {r['reason']}", ""]
    report_md = "\n".join(lines)
    (EVAL_DIR / "report.md").write_text(report_md, encoding="utf-8")
    (EVAL_DIR / "report.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EVAL_DIR / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols = ["id", "source_file", "kind", "question", "expected_answer", "app_answer",
                "verdict", "reason", "retrieval_rank", "exact_match", "latency_s"]
        w = csv.DictWriter(f, fieldnames=cols + ["manual_verdict"], extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print("\n" + report_md)
    print("\nSaved: evaluation/report.md, report.json, results.csv, results.jsonl")
    if judge_fail:
        print(f"WARNING: {judge_fail} answers could not be graded (judge errors); they are excluded from accuracy.")


if __name__ == "__main__":
    main()
