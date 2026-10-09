"""Grade one run file: deterministic checks first, the LLM judge only for the cases they cannot settle."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import llm
from rag import load_docs

ROOT = Path(__file__).parent
SETS = {"eval": "questions.json", "dev": "dev_questions.json"}
RUBRIC = (ROOT / "eval" / "rubric.md").read_text()

ABSTAIN = [
    r"documents? (?:do not|don't|does not|doesn't) (?:say|mention|specify|state|include|contain|provide|cover)",
    r"(?:context|sources?|information|text) (?:provided )?(?:do not|don't|does not|doesn't) (?:say|mention|specify|state|include|contain|provide|cover)",
    r"(?:is|are) not (?:mentioned|specified|stated|provided|covered|included|available) in",
    r"(?:isn't|aren't) (?:mentioned|specified|stated|provided|covered)",
    r"no (?:information|mention|details?) (?:\w+ )?(?:about|on|of|regarding)",
    r"there is no (?:information|mention)",
    r"(?:cannot|can't|unable to) (?:find|answer|determine|tell)",
    r"not enough information",
    r"there is no mention",
]


def norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("–", "-").replace("—", "-").replace("**", "")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def found(variant: str, text: str) -> bool:
    v = norm(variant)
    pattern = re.escape(v)
    if v[0].isdigit():
        pattern = r"(?<![\d.])" + pattern
    if v[-1].isdigit():
        pattern += r"(?![\d])"
    return re.search(pattern, text) is not None


def abstains(text: str) -> bool:
    return any(re.search(p, text) for p in ABSTAIN)


ERRORS = ["none", "outdated", "wrong_value", "invented", "incomplete", "false_dont_know"]
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong", "abstained"]},
        "hallucination": {"type": "boolean"},
        "error_type": {"type": "string", "enum": ERRORS},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "hallucination", "error_type", "reason"],
    "additionalProperties": False,
}
ALL_DOCS = "\n\n".join(
    f"=== [{d['doc_id']}] {d['meta']['title']} | effective {d['meta']['effective_date']} | status: {d['meta']['status']} ===\n{d['body']}"
    for d in load_docs())


def judge(q: dict, answer: str) -> dict:
    gold = ", ".join(q.get("gold_sources", [])) or "none (the documents do not contain this answer)"
    prompt = (
        f"You grade answers from a customer-support chatbot. Follow this rubric exactly.\n\n{RUBRIC}\n\n"
        f"The complete knowledge base the chatbot could use:\n\n{ALL_DOCS}\n\n"
        f"Question: {q['question']}\n"
        f"Answerable from the documents: {'yes' if q.get('answerable', True) else 'no'}\n"
        f"Gold answer: {q['gold_answer']}\nGold source documents: {gold}\n\n"
        f"Answer to grade:\n<<<\n{answer}\n>>>\n\n"
        "Return the verdict, the hallucination flag, the main error type and a one-sentence reason."
    )
    out = llm.complete_json([{"role": "user", "content": prompt}], JUDGE_SCHEMA, purpose="judge")
    return {**out["json"], "judge_model": out["model"], "judge_cost_usd": out["cost_usd"]}


def rule_grade(q: dict, answer: str) -> dict | None:
    """Settle only the clear cases; everything mixed goes to the judge."""
    if q.get("judge_only"):
        return None
    text = norm(answer)
    abst = abstains(text)
    if not q.get("answerable", True):
        return {"verdict": "correct", "hallucination": False, "error_type": "none",
                "reason": "says the documents do not contain the answer"} if abst else None
    groups = [any(found(v, text) for v in g) for g in q.get("include", [])]
    trap = [t for t in q.get("traps", []) if found(t, text)]
    if groups and all(groups) and not trap and not abst:
        return {"verdict": "correct", "hallucination": False, "error_type": "none", "reason": "all key facts present, no trap value"}
    if abst and not any(groups) and not trap:
        return {"verdict": "abstained", "hallucination": False, "error_type": "false_dont_know",
                "reason": "said the documents don't say, but they do"}
    return None


def grade_run(path: Path) -> Path:
    run = json.loads(path.read_text())
    questions = {q["id"]: q for q in json.loads((ROOT / "eval" / SETS[run["set"]]).read_text())}
    for item in run["items"]:
        q = questions[item["id"]]
        g = rule_grade(q, item["answer"])
        item["grader"] = "rule" if g else "judge"
        item.update(g or judge(q, item["answer"]))
        gold = set(q.get("gold_sources", []))
        item["gold_sources_retrieved"] = bool(gold) and gold <= {h["doc_id"] for h in item["retrieved"]}
    out = ROOT / "outputs" / "graded" / path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=1, ensure_ascii=False))
    correct = sum(i["verdict"] == "correct" for i in run["items"])
    print(f"{path.name}: {correct}/{len(run['items'])} correct, judge used {sum(i['grader'] == 'judge' for i in run['items'])} times")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        grade_run(Path(p))
