"""Build evaluation/golden.json from the documents in evaluation/docs/.

Run from the repo root:  python -m evaluation.build_golden
Optional:                --per-doc 8 --candidates 16

A small LLM writes one question per passage. Weak questions (circular, refer to
"the figure", answer not in the passage, ...) are rejected by code, not by the LLM.
"""
import argparse
import json
import re
import sys
from collections import Counter

from backend.app.services.document_processor import process_document
from evaluation.utils import (
    GOLDEN_PATH, check_ollama, gen_model, list_docs, normalize, ollama_json,
)

PROMPT = """You write exam questions for a document question-answering test.
Read the passage and write ONE question about a specific fact it states (a name, number, date, defined term, condition, penalty, duty, step, ...).
Rules:
- The question must make sense to someone who has NOT seen the passage: name the subject explicitly. Never write "the passage", "the text", "this section", "the figure", "the equation", "above" or "below".
- It must have ONE clear answer, so avoid wording like "one of the" or "the only".
- The answer must be short (1 to 12 words) and copied EXACTLY, word for word, from the passage.
- The passage may start or end in the middle of a sentence. Ignore the broken edges and use any complete fact in the middle.
- Reply {{"skip": true}} ONLY if the passage has no complete fact at all (for example only equations, code or a figure caption).
Example: passage "The warranty period for the X200 router is 24 months from the date of purchase."
-> {{"question": "How long is the warranty period for the X200 router?", "answer": "24 months"}}
{extra}Reply ONLY with JSON: {{"question": "...", "answer": "..."}}

PASSAGE:
\"\"\"
{chunk}
\"\"\""""

TABLE_HINT = (
    "The passage is rows of a table. Ask for one value of one specific row, "
    "identifying the row by two of its other values.\n"
)

BAD_REFS = re.compile(
    r"\b(passage|the text|this text|the document|this document|this section|following|above|below|"
    r"the figure|this figure|figure\s*\d|fig\.|equation|table\s*\d|the algorithm|the example)\b",
    re.I,
)
CLAUSE_REF = re.compile(r"\b(clause|sub-?section|paragraph|item)\s*\(", re.I)
VAGUE = re.compile(r"\b(one of the|the only|some of the|a few of the)\b", re.I)
GENERIC_ANSWERS = {"yes", "no", "true", "false", "none", "n/a"}

# Questions the documents cannot answer: the app should refuse, not guess.
UNANSWERABLE = [
    ("What is the boiling point of tungsten?", "trivia"),
    ("Who won the 2010 FIFA World Cup?", "trivia"),
    ("What is the capital of Australia?", "trivia"),
    ("How many moons does Jupiter have?", "trivia"),
    ("Who wrote the novel Pride and Prejudice?", "trivia"),
    ("What is the phone number of the customer support helpline?", "plausible"),
    ("What was the total budget allocated to the Mars colony project?", "plausible"),
    ("Who is the current chief executive officer of the organization that published this?", "plausible"),
]


def validate(question, answer, chunk_text, seen):
    """Return None if the pair is good, otherwise a short rejection reason."""
    if not question or not answer:
        return "empty"
    if not question.endswith("?") or not (20 <= len(question) <= 220):
        return "bad_question_shape"
    n_words = len(answer.split())
    if n_words > 15 or len(answer) < 2:
        return "bad_answer_length"
    if normalize(answer) in GENERIC_ANSWERS:
        return "generic_answer"
    if normalize(answer) not in normalize(chunk_text):
        return "answer_not_in_passage"
    if normalize(answer) in normalize(question):
        return "circular"
    if BAD_REFS.search(question) or CLAUSE_REF.search(question):
        return "refers_to_context"
    if VAGUE.search(question):
        return "vague_wording"
    q_words = {w for w in re.findall(r"[a-z]{4,}", normalize(question))}
    c_words = set(re.findall(r"[a-z]{4,}", normalize(chunk_text)))
    if q_words and not (q_words & c_words):
        return "unrelated_to_passage"
    if normalize(question) in seen:
        return "duplicate"
    return None


def candidate_chunks(chunks, tabular, limit, used_idx=frozenset()):
    good = []
    for c in chunks:
        t = c["text"]
        if c.get("chunk_index") in used_idx:
            continue  # already has a question in golden.json
        if len(t) < 250:
            continue
        if not tabular and sum(ch.isalpha() for ch in t) / len(t) < 0.70:
            continue  # equations, code, tables of numbers
        good.append(c)
    if len(good) <= limit:
        return good
    step = len(good) / limit
    return [good[int(i * step)] for i in range(limit)]


def next_id(golden):
    nums = [int(g["id"][1:]) for g in golden if str(g.get("id", "")).startswith("Q") and g["id"][1:].isdigit()]
    return f"Q{(max(nums) + 1 if nums else 1):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=8, help="questions to keep per document")
    ap.add_argument("--candidates", type=int, default=40, help="passages to try per document")
    ap.add_argument("--append", action="store_true", help="keep the existing golden.json and add new questions to it")
    ap.add_argument("--only", nargs="+", help="only use these file names from evaluation/docs")
    args = ap.parse_args()

    docs = list_docs()
    if not docs:
        sys.exit("No supported files in evaluation/docs/. Add 2-4 documents and run again.")
    if args.only:
        docs = [d for d in docs if d.name in set(args.only)]
        if not docs:
            sys.exit(f"None of these files are in evaluation/docs/: {args.only}")
    problems = check_ollama([gen_model()])
    if problems:
        sys.exit("\n".join(problems))
    print(f"Question-writer model: {gen_model()}\n")

    golden, seen, used = [], set(), set()
    if args.append and GOLDEN_PATH.exists():
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        seen = {normalize(g["question"]) for g in golden}
        used = {(g["source_file"], g["source_chunk_index"]) for g in golden if g["answerable"]}
        print(f"Keeping {sum(g['answerable'] for g in golden)} existing answerable questions\n")
    for path in docs:
        tabular = path.suffix.lower() in (".csv", ".xlsx")
        try:
            chunks = process_document(str(path), original_filename=path.name)
        except Exception as e:
            print(f"[{path.name}] could not be read: {e}")
            continue
        used_idx = {idx for (f, idx) in used if f == path.name}
        cands = candidate_chunks(chunks, tabular, args.candidates, used_idx)
        print(f"[{path.name}] {len(chunks)} chunks, trying up to {len(cands)} passages")
        kept, rejected = 0, Counter()
        for c in cands:
            if kept >= args.per_doc:
                break
            prompt = PROMPT.format(extra=TABLE_HINT if tabular else "", chunk=c["text"][:1800])
            try:
                out = ollama_json(prompt, gen_model())
            except Exception as e:
                rejected["llm_error"] += 1
                print(f"   LLM error: {str(e)[:100]}")
                continue
            if out.get("skip"):
                rejected["model_skipped"] += 1
                continue
            q, a = str(out.get("question", "")).strip(), str(out.get("answer", "")).strip()
            reason = validate(q, a, c["text"], seen)
            if reason:
                rejected[reason] += 1
                continue
            seen.add(normalize(q))
            kept += 1
            golden.append({
                "id": next_id(golden),
                "question": q,
                "expected_answer": a,
                "source_file": path.name,
                "source_page": c.get("page"),
                "source_chunk_index": c.get("chunk_index"),
                "source_chunk_text": c["text"],
                "answerable": True,
                "kind": "document",
            })
            print(f"   + {golden[-1]['id']}: {q}  ->  {a}")
        print(f"   kept {kept}; rejected {dict(rejected)}\n")

    if not golden:
        sys.exit("No valid questions were generated. Check that Ollama is running and the model works.")

    for q, kind in ([] if any(not g["answerable"] for g in golden) else UNANSWERABLE):
        golden.append({
            "id": next_id(golden), "question": q, "expected_answer": "UNANSWERABLE",
            "source_file": "none", "source_page": None, "source_chunk_index": None,
            "source_chunk_text": "", "answerable": False, "kind": kind,
        })

    GOLDEN_PATH.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ans = sum(g["answerable"] for g in golden)
    print(f"Saved {len(golden)} questions ({n_ans} answerable, {len(golden) - n_ans} unanswerable) to {GOLDEN_PATH}")
    print("Skim the questions above. Delete any bad entry from golden.json before running run_eval.")


if __name__ == "__main__":
    main()