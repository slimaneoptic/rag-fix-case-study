"""Build outputs/report.json and outputs/report.md from the graded runs, the spend log and the manual review."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from rag import AFTER_SETTINGS, BEFORE_SETTINGS, load_docs

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
RUNS = ["before_run1", "before_run2", "after_run1", "after_run2"]
CATEGORIES = ["table lookup", "current vs outdated policy", "multi-hop (two documents)", "exact numbers", "unanswerable"]
EXAMPLES = [  # chosen to show one failure type each; answers are copied from run 1 of both pipelines
    ("T1", "The 500-character split cut the pricing table inside its first row (Starter), so the Team row never reached the model. Keeping tables whole put the full table in the context."),
    ("C6", "The old 2024 refund policy and the current one were both retrieved, with no dates, so the bot offered both rules. With document dates and status, the outdated policy ranks below the current one and is labelled as superseded."),
    ("M3", "The answer needs two documents: the price table and the SLA credit table. Before, the price table was cut off; after, both whole tables were retrieved and the bot showed the calculation."),
]
# Measured add-on for each example, from the ablation runs (checked against the graded files in example_notes()).
EXAMPLE_NOTES = {
    "T1": ("abl-before-top6", "correct", "Measured: the old bot with 6 chunks instead of 2 also got this one right, in both runs, so more chunks alone fix it too."),
    "C6": ("abl-after-no-metadata", "mixed", "Measured: without the dates and status, the fixed bot gave the old $25 fee as a second valid rule in 1 of 2 runs."),
    "M3": ("abl-after-fixed-chunks", "not correct", "Measured: with the old 500-character chunks, the fixed bot still missed it in both runs."),
}
ABLATIONS = [  # each is the old or the fixed pipeline with ONE change (ablation.py, committed before these runs)
    ("abl-before-top6", "Old bot, 6 chunks instead of 2"),
    ("abl-after-fixed-chunks", "Fixed bot, but the old 500-character chunks"),
    ("abl-after-no-metadata", "Fixed bot, but no document dates or status"),
    ("abl-after-no-rerank", "Fixed bot, but no reranker"),
    ("abl-after-old-prompt", "Fixed bot, but the old prompt"),
]
ABLATION_FOR_CAUSE = {  # the one-change run closest to each fix, and what exactly it removes
    "Tables and sections cut in the middle": ("abl-after-fixed-chunks", "Fixed bot with the old 500-character chunks"),
    "Outdated policy retrieved next to the current one, with no dates": ("abl-after-no-metadata", "Fixed bot without dates and status"),
    "The right document was not in the top 2 results": ("abl-after-no-rerank", "Fixed bot without the reranker only (hybrid search and 6 chunks kept)")}
ABLATION_START = "2026-10-02T00:57:24"  # commit time of ablation.py; spend rows from then on are the ablation runs
CAUSES = [  # found by reading the chunks each before answer received (same chunks in both runs: retrieval is deterministic)
    {"problem": "Tables and sections cut in the middle", "questions": ["T1", "M1", "M3", "M5"],
     "fix": "Heading- and table-aware chunking: one chunk per section, tables never split, each chunk tagged with its document and section title"},
    {"problem": "Outdated policy retrieved next to the current one, with no dates", "questions": ["C3", "C4", "C6"],
     "fix": "Document metadata (effective date, current or superseded): outdated documents rank below current ones and are labelled in the prompt"},
    {"problem": "The right document was not in the top 2 results", "questions": ["T7", "E6", "M4"],
     "fix": "Hybrid search (keywords + embeddings), a reranker, and 6 chunks instead of 2"},
]


def load(name: str) -> dict:
    return json.loads((OUT / "graded" / f"eval_{name}.json").read_text())


def cites(answer: str) -> bool:
    return re.search(r"\[[a-z0-9-]+(?:, ?[a-z0-9-]+)*\]", answer) is not None


def summarize(run: dict) -> dict:
    items = run["items"]
    answerable = [i for i in items if i["category"] != "unanswerable"]
    unanswerable = [i for i in items if i["category"] == "unanswerable"]
    per_cat = {c: {"correct": sum(i["verdict"] == "correct" for i in items if i["category"] == c),
                   "total": sum(i["category"] == c for i in items)} for c in CATEGORIES}
    return {
        "correct": sum(i["verdict"] == "correct" for i in items), "total": len(items),
        "per_category": per_cat,
        "verdicts": {v: sum(i["verdict"] == v for i in items) for v in ["correct", "partial", "wrong", "abstained"]},
        "hallucinations": sum(bool(i["hallucination"]) for i in items),
        "hallucination_ids": [i["id"] for i in items if i["hallucination"]],
        "unanswerable_correctly_declined": sum(i["verdict"] == "correct" for i in unanswerable),
        "unanswerable_total": len(unanswerable),
        "false_dont_know": sum(i.get("error_type") == "false_dont_know" for i in answerable),
        "false_dont_know_ids": [i["id"] for i in answerable if i.get("error_type") == "false_dont_know"],
        "gold_documents_retrieved": sum(bool(i["gold_sources_retrieved"]) for i in answerable),
        "answerable_total": len(answerable),
        "answers_with_citation": sum(cites(i["answer"]) for i in items),
        "median_latency_s": round(statistics.median(i["latency_s"] for i in items), 2),
        "median_retrieval_s": round(statistics.median(i["retrieval_s"] for i in items), 3),
        "mean_cost_per_question_usd": sum(i["cost_usd"] for i in items) / len(items),
        "median_cost_per_question_usd": statistics.median(i["cost_usd"] for i in items),
        "models_used": sorted({i["model"] for i in items}),
        "graded_by_rule": sum(i["grader"] == "rule" for i in items),
        "graded_by_judge": sum(i["grader"] == "judge" for i in items),
        "missed_ids": [i["id"] for i in items if i["verdict"] != "correct"],
    }


def spend() -> dict:
    rows = [json.loads(line) for line in (OUT / "spend.jsonl").read_text().splitlines() if line.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["purpose"], [0, 0.0])
        by[r["purpose"]][0] += 1
        by[r["purpose"]][1] += r["cost_usd"] or 0.0
    abl = [r for r in rows if r["time"] >= ABLATION_START]
    return {"calls": len(rows), "total_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4),
            "by_purpose": {k: {"calls": n, "usd": round(c, 4)} for k, (n, c) in by.items()},
            "before_ablations": {"calls": len(rows) - len(abl), "usd": round(sum(r["cost_usd"] or 0 for r in rows if r["time"] < ABLATION_START), 4)},
            "ablation_runs": {"calls": len(abl), "usd": round(sum(r["cost_usd"] or 0 for r in abl), 4)}}


def manual_review(graded: dict, file: str = "manual_review.json") -> dict:
    review = json.loads((OUT / file).read_text())
    for m in review["items"]:
        actual = {i["id"]: i for i in graded[m["run"].replace("eval_", "")]["items"]}[m["id"]]
        assert m["grade"].split(" ")[0] == actual["verdict"], (m["run"], m["id"], actual["verdict"])
        m["answer"] = actual["answer"]
        m["grader"] = actual["grader"]
    items = review["items"]
    return {"items": items, "read": len(items), "score_agree": sum(m["agree"] for m in items),
            "label_agree": sum(m["label_agree"] for m in items), "judge_graded_read": sum(m["grader"] == "judge" for m in items)}


def verdicts(graded: dict, qid: str, runs) -> list[str]:
    return [{i["id"]: i for i in graded[r]["items"]}[qid]["verdict"] for r in runs]


def example_note(graded: dict, qid: str) -> str:
    variant, expect, text = EXAMPLE_NOTES[qid]
    v = verdicts(graded, qid, (f"{variant}_run1", f"{variant}_run2"))
    ok = {"correct": v == ["correct", "correct"], "not correct": "correct" not in v,
          "mixed": sorted(x == "correct" for x in v) == [False, True]}[expect]
    assert ok, (qid, variant, v)
    return text


def m2_story(graded: dict) -> dict:
    """Every fact used in the 'one question got worse' text, checked against the saved runs."""
    get = lambda r: {i["id"]: i for i in graded[r]["items"]}["M2"]
    chunks = lambda r: [(h["doc_id"], h.get("section")) for h in get(r)["retrieved"]]
    assert verdicts(graded, "M2", ("before_run1", "before_run2")) == ["correct", "correct"]
    assert "correct" not in verdicts(graded, "M2", ("after_run1", "after_run2"))
    assert verdicts(graded, "M2", ("abl-after-old-prompt_run1", "abl-after-old-prompt_run2")) == ["correct", "correct"]
    assert chunks("after_run1") == chunks("after_run2") == chunks("abl-after-old-prompt_run1") == chunks("abl-after-old-prompt_run2")
    assert {d for d, _ in chunks("before_run1")} == {d for d, _ in chunks("after_run1") if d != "release-notes-2026"} == {"sla", "api-rate-limits"}
    assert not any("$" in h["text"] for r in ("before_run1", "after_run1") for h in get(r)["retrieved"]), "a price reached the model"
    a1, a2 = get("after_run1")["answer"], get("after_run2")["answer"]
    assert "1,000" not in a1 and "1000" not in a1 and "1,000" in a2 and "5,000" in a2
    assert "price" in a1.lower(), "run 1 gives missing prices as its reason"
    # T7 (cheapest plan with Salesforce): same strict prompt answers when the price table is in its chunks
    t7 = lambda r: {i["id"]: i for i in graded[r]["items"]}["T7"]
    has_table = lambda r: any("Custom quote" in h["text"] and "$39" in h["text"] for h in t7(r)["retrieved"])
    assert verdicts(graded, "T7", ("after_run1", "after_run2")) == ["correct", "correct"]
    assert has_table("after_run1") and has_table("after_run2")
    assert "business" in t7("after_run1")["answer"].lower() and "business" in t7("after_run2")["answer"].lower()
    # without the reranker the price table was not in its T7 chunks, and it refused T7 the same way
    nr = ("abl-after-no-rerank_run1", "abl-after-no-rerank_run2")
    assert verdicts(graded, "T7", nr) == ["abstained", "abstained"]
    assert not any("$" in h["text"] for r in nr for h in t7(r)["retrieved"])
    # no document says Business costs less than Enterprise: Enterprise is a custom quote
    kb = {f.stem: f.read_text() for f in (ROOT / "kb").glob("*.md")}
    assert "| Enterprise | Custom quote | Custom quote |" in kb["pricing-plans"]
    assert not any(w in t.lower() for t in kb.values() for w in ("cheaper than enterprise", "less than enterprise", "less than the enterprise"))
    gold = {q["id"]: q for q in json.loads((ROOT / "eval" / "questions.json").read_text())}["M2"]["gold_sources"]
    assert gold == ["sla", "api-rate-limits"]
    return {"question": get("before_run1")["question"], "before_verdicts": verdicts(graded, "M2", ("before_run1", "before_run2")),
            "after_verdicts": verdicts(graded, "M2", ("after_run1", "after_run2")),
            "old_prompt_on_same_chunks_verdicts": verdicts(graded, "M2", ("abl-after-old-prompt_run1", "abl-after-old-prompt_run2")),
            "before_answer_run1": get("before_run1")["answer"], "after_answer_run1": a1, "after_answer_run2": a2,
            "price_in_chunks": False, "gold_sources": gold,
            "t7_after_verdicts": verdicts(graded, "T7", ("after_run1", "after_run2")), "t7_after_price_table_in_chunks": True,
            "t7_no_rerank_verdicts": verdicts(graded, "T7", nr), "t7_no_rerank_price_in_chunks": False,
            "kb_says_business_cheaper_than_enterprise": False}


def main() -> None:
    graded = {name: load(name) for name in RUNS}
    s = {name: summarize(run) for name, run in graded.items()}
    abl_runs = [f"{v}_run{n}" for v, _ in ABLATIONS for n in (1, 2)]
    graded.update({name: load(name) for name in abl_runs})
    first = {p: {i["id"]: i for i in graded[f"{p}_run1"]["items"]} for p in ("before", "after")}
    questions = {q["id"]: q for q in json.loads((ROOT / "eval" / "questions.json").read_text())}
    examples = [{"id": qid, "category": questions[qid]["category"], "question": questions[qid]["question"],
                 "gold_answer": questions[qid]["gold_answer"],
                 "before_answer": first["before"][qid]["answer"], "before_verdict": first["before"][qid]["verdict"],
                 "after_answer": first["after"][qid]["answer"], "after_verdict": first["after"][qid]["verdict"],
                 "after_sources": sorted({h["doc_id"] for h in first["after"][qid]["retrieved"]}),
                 "why": why, "measured": example_note(graded, qid)} for qid, why in EXAMPLES]
    causes = []
    for c in CAUSES:
        fixed = [q for q in c["questions"] if all(first_run["verdict"] == "correct" for first_run in
                 ({i["id"]: i for i in graded[r]["items"]}[q] for r in ("after_run1", "after_run2")))]
        failed_before = [q for q in c["questions"] if all({i["id"]: i for i in graded[r]["items"]}[q]["verdict"] != "correct"
                         for r in ("before_run1", "before_run2"))]
        abl, abl_label = ABLATION_FOR_CAUSE[c["problem"]]
        causes.append({**c, "wrong_before_in_both_runs": failed_before, "correct_after_in_both_runs": fixed,
                       "ablation": abl, "ablation_label": abl_label,
                       "ablation_correct": [summarize(graded[f"{abl}_run{n}"])["correct"] for n in (1, 2)]})
    docs = load_docs()
    review = manual_review(graded)
    review_abl = manual_review(graded, "manual_review_ablation.json")
    v1 = {n: sum(i["verdict"] == "correct" for i in json.loads((OUT / "graded_v1" / f"eval_{n}.json").read_text())["items"]) for n in RUNS}
    assert s["after_run1"]["missed_ids"] == s["after_run2"]["missed_ids"] == ["M2"], "update the 'One question got worse' text"
    # what changed, question by question (the same in both runs of each pipeline)
    ids = [i["id"] for i in graded["before_run1"]["items"]]
    ok = lambda r: {i["id"] for i in graded[r]["items"] if i["verdict"] == "correct"}
    fixed_ids = [q for q in ids if q not in ok("before_run1") | ok("before_run2") and q in ok("after_run1") & ok("after_run2")]
    worse_ids = [q for q in ids if q in ok("before_run1") & ok("before_run2") and q not in ok("after_run1") | ok("after_run2")]
    assert len(fixed_ids) - len(worse_ids) == s["after_run1"]["correct"] - s["before_run1"]["correct"] == s["after_run2"]["correct"] - s["before_run2"]["correct"]
    assert worse_ids == ["M2"], "update the 'One question got worse' text"
    for n in (1, 2):  # claim in the ablation text: M1 and M3 missed because the Business price row never reached the model
        fc_items = {i["id"]: i for i in graded[f"abl-after-fixed-chunks_run{n}"]["items"]}
        assert all(fc_items[q]["verdict"] != "correct" and not any("$39" in h["text"] or "$33" in h["text"] for h in fc_items[q]["retrieved"]) for q in ("M1", "M3"))
        assert set(fc_items) - {i["id"] for i in graded[f"abl-after-fixed-chunks_run{n}"]["items"] if i["verdict"] == "correct"} == {"M1", "M2", "M3"}
    ablations = []
    for v, label in ABLATIONS:
        runs = [summarize(graded[f"{v}_run{n}"]) for n in (1, 2)]
        ablations.append({"variant": v, "label": label, "settings_changed": graded[f"{v}_run1"]["settings"]["ablation"],
                          **{k: [x[k] for x in runs] for k in ("correct", "hallucinations", "answers_with_citation", "missed_ids",
                                                               "hallucination_ids", "median_latency_s", "mean_cost_per_question_usd",
                                                               "unanswerable_correctly_declined", "models_used")}})
    report = {
        "knowledge_base": {"documents": len(docs), "characters": sum(len(d["body"]) for d in docs),
                           "superseded_documents": [d["doc_id"] for d in docs if d["meta"]["status"] == "superseded"]},
        "eval_set": {"questions": 30, "by_category": {c: s["before_run1"]["per_category"][c]["total"] for c in CATEGORIES},
                     "dev_questions_used_while_building": 8},
        "settings": {"before": BEFORE_SETTINGS, "after": AFTER_SETTINGS},
        "runs": s, "examples": examples, "causes": causes,
        "changes": {"fixed_ids": fixed_ids, "worse_ids": worse_ids,
                    "note": "fixed = wrong in both before runs and right in both after runs; worse = right in both before runs and wrong in both after runs"},
        "worse_question": m2_story(graded),
        "ablations": ablations,
        "manual_review": {k: v for k, v in review.items() if k != "items"},
        "manual_review_ablation": {k: v for k, v in review_abl.items() if k != "items"},
        "first_grading_pass_correct": v1,
        "spend": spend(),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    write_markdown(report, review)
    print(json.dumps({n: (v["correct"], v["hallucinations"], v["median_latency_s"]) for n, v in s.items()}))
    print(json.dumps({a["variant"]: (a["correct"], a["hallucinations"], a["answers_with_citation"]) for a in ablations}))
    print("fixed", fixed_ids, "worse", worse_ids, "spend", report["spend"]["total_usd"], report["spend"]["ablation_runs"])


def money(x: float) -> str:
    return f"${x:.4f}"


def write_markdown(r: dict, review: dict) -> None:
    s = r["runs"]
    b1, b2, a1, a2 = (s[n] for n in RUNS)
    L = ["# RAG chatbot fix: before and after", "",
         f"Same {r['knowledge_base']['documents']} documents, same 30 questions, same embedding model and same LLM. Only the pipeline changed.", "",
         "Note: this public copy includes this report, but not the raw runs, grades, second-review files or spend log it names under `outputs/`.", "",
         "## Result", "",
         "| | Before, run 1 | Before, run 2 | After, run 1 | After, run 2 |", "|---|---|---|---|---|",
         f"| Correct answers | {b1['correct']}/30 | {b2['correct']}/30 | {a1['correct']}/30 | {a2['correct']}/30 |"]
    for c in CATEGORIES:
        L.append(f"| {c} | " + " | ".join(f"{x['per_category'][c]['correct']}/{x['per_category'][c]['total']}" for x in (b1, b2, a1, a2)) + " |")
    L += [f"| Hallucinations (wrong or outdated fact given as valid) | {b1['hallucinations']} | {b2['hallucinations']} | {a1['hallucinations']} | {a2['hallucinations']} |",
          f"| Unanswerable questions correctly declined | {b1['unanswerable_correctly_declined']}/5 | {b2['unanswerable_correctly_declined']}/5 | {a1['unanswerable_correctly_declined']}/5 | {a2['unanswerable_correctly_declined']}/5 |",
          f"| False \"don't know\" on answerable questions | {b1['false_dont_know']} | {b2['false_dont_know']} | {a1['false_dont_know']} | {a2['false_dont_know']} |",
          f"| Right document(s) retrieved (25 answerable) | {b1['gold_documents_retrieved']}/25 | {b2['gold_documents_retrieved']}/25 | {a1['gold_documents_retrieved']}/25 | {a2['gold_documents_retrieved']}/25 |",
          f"| Answers citing a source | {b1['answers_with_citation']}/30 | {b2['answers_with_citation']}/30 | {a1['answers_with_citation']}/30 | {a2['answers_with_citation']}/30 |",
          f"| Median latency per question | {b1['median_latency_s']} s | {b2['median_latency_s']} s | {a1['median_latency_s']} s | {a2['median_latency_s']} s |",
          f"| Mean LLM cost per question | {money(b1['mean_cost_per_question_usd'])} | {money(b2['mean_cost_per_question_usd'])} | {money(a1['mean_cost_per_question_usd'])} | {money(a2['mean_cost_per_question_usd'])} |",
          f"| Graded by rules / by LLM judge | {b1['graded_by_rule']} / {b1['graded_by_judge']} | {b2['graded_by_rule']} / {b2['graded_by_judge']} | {a1['graded_by_rule']} / {a1['graded_by_judge']} | {a2['graded_by_rule']} / {a2['graded_by_judge']} |",
          "", f"Missed questions: before run 1 {', '.join(b1['missed_ids'])}; before run 2 {', '.join(b2['missed_ids'])}; after run 1 {', '.join(a1['missed_ids'])}; after run 2 {', '.join(a2['missed_ids'])}.",
          f"Hallucinations: before run 1 {', '.join(b1['hallucination_ids']) or 'none'}; before run 2 {', '.join(b2['hallucination_ids']) or 'none'}; after: {', '.join(a1['hallucination_ids'] + a2['hallucination_ids']) or 'none'}.",
          f"Latency is retrieval + LLM call, measured after the models were loaded. Median retrieval time: before {b1['median_retrieval_s']} s, after {a1['median_retrieval_s']} s (BM25 + embeddings + reranker on CPU). Models used: {', '.join(sorted(set(b1['models_used'] + a1['models_used'] + b2['models_used'] + a2['models_used'])))}.",
          "", f"Net change, question by question: {len(r['changes']['fixed_ids'])} questions wrong in both before runs became right in both after runs "
          f"({', '.join(r['changes']['fixed_ids'])}); {len(r['changes']['worse_ids'])} question right in both before runs became a miss in both after runs "
          f"({', '.join(r['changes']['worse_ids'])}, see below). {b1['correct']} + {len(r['changes']['fixed_ids'])} - {len(r['changes']['worse_ids'])} = {a1['correct']}.",
          "", "## What went wrong before, and the fix", "",
          "The cause of each miss was found by reading the chunks each before answer received (retrieval is deterministic, so both before runs saw the same chunks). "
          "All fixes were applied together; the counts after each fix are for the full fix. The next section measures each fix on its own.", ""]
    for c in r["causes"]:
        L.append(f"- **{c['problem']}** ({', '.join(c['questions'])}). Fix: {c['fix']}. Wrong before in both runs: {len(c['wrong_before_in_both_runs'])}/{len(c['questions'])}; "
                 f"correct after the full fix in both runs: {len(c['correct_after_in_both_runs'])}/{len(c['questions'])}. "
                 f"{c['ablation_label']}: {' and '.join(str(x) for x in c['ablation_correct'])} of 30 (`{c['ablation']}`).")
    L.append("- **Where the dates and status come from:** each of the 12 synthetic documents has its effective date and status "
             "(current or superseded) in its header, written when the documents were made; the old pipeline does not use these headers. On real documents this comes from file dates, version names, "
             "or a one-time review in the first step. Here the current refund policy's own text also says which policy it replaces.")
    op = {a["variant"]: a for a in r["ablations"]}
    L += ["- **Prompt** (cite a source for every fact, say \"The documents don't say.\" when the answer is missing, never guess). "
          f"After: {a1['answers_with_citation']}/30 and {a2['answers_with_citation']}/30 answers cite a source, against {b1['answers_with_citation']}/30 before. "
          f"Honest note: the before bot already declined all 5 unanswerable questions ({b1['unanswerable_correctly_declined']}/5 in both runs), so this fix did not change that score, "
          f"and the fixed bot with the old prompt scored {' and '.join(map(str, op['abl-after-old-prompt']['correct']))} of 30. The strict prompt adds citations, not accuracy, on this test set.",
          "", "## Which fix mattered (ablation)", "",
          "Each row changes ONE thing in the old or the fixed pipeline (`ablation.py`, committed before these runs; `rag.py` was not changed). "
          "Same 30 questions, same grader, two runs each. These runs were made after the main results, to answer \"which fix mattered, or was it just more text?\".", "",
          "| Version | Correct (run 1 / run 2) | Outdated or wrong facts given as valid | Answers citing a source | Missed |", "|---|---|---|---|---|",
          f"| Old bot (top 2 chunks) | {b1['correct']} / {b2['correct']} | {b1['hallucinations']} / {b2['hallucinations']} | {b1['answers_with_citation']} / {b2['answers_with_citation']} | {', '.join(b1['missed_ids'])} |",
          f"| Fixed bot | {a1['correct']} / {a2['correct']} | {a1['hallucinations']} / {a2['hallucinations']} | {a1['answers_with_citation']} / {a2['answers_with_citation']} | {', '.join(a1['missed_ids'])} |"]
    for a in r["ablations"]:
        missed = " / ".join(", ".join(m) or "none" for m in a["missed_ids"]) if a["missed_ids"][0] != a["missed_ids"][1] else (", ".join(a["missed_ids"][0]) or "none")
        L.append(f"| {a['label']} | {a['correct'][0]} / {a['correct'][1]} | {a['hallucinations'][0]} / {a['hallucinations'][1]} | "
                 f"{a['answers_with_citation'][0]} / {a['answers_with_citation'][1]} | {missed} |")
    t6, fc, nm, nr = op["abl-before-top6"], op["abl-after-fixed-chunks"], op["abl-after-no-metadata"], op["abl-after-no-rerank"]
    L += ["", "What this shows:", "",
          f"- More text alone helps a lot: 6 chunks instead of 2 took the old bot from {b1['correct']} to {t6['correct'][0]} and {t6['correct'][1]}. "
          f"It still gave an outdated refund rule as valid ({', '.join(t6['hallucination_ids'][0])} in run 1; {', '.join(t6['hallucination_ids'][1])} in run 2) and missed {', '.join(t6['missed_ids'][0])}.",
          f"- Whole tables: without them the fixed bot scored {fc['correct'][0]} and {fc['correct'][1]}. It missed M1 and M3 in both runs because the chunk with the Business price row was not among its 6 chunks (it also missed M2, like the full fix).",
          f"- Dates and status: without them the fixed bot scored {nm['correct'][0]} and {nm['correct'][1]}; in run 2 it gave the old 2024 refund rule as a second valid rule ({', '.join(nm['hallucination_ids'][1])}). "
          "The current policy's own text says it replaces the 2024 one, but without the tags that was not enough in run 2.",
          f"- Reranker: without it, {nr['correct'][0]} and {nr['correct'][1]} (missed {', '.join(nr['missed_ids'][0])}).",
          f"- Prompt: with the old prompt on the same 6 chunks, {op['abl-after-old-prompt']['correct'][0]} and {op['abl-after-old-prompt']['correct'][1]}, but no answer cites a source. "
          "On the same chunks, the strict prompt is what turned M2 into a miss (next section). The main pipelines were NOT changed after these runs: that would be tuning on the test answers.",
          "", f"Second check of the ablation grades: {r['manual_review_ablation']['read']} answers checked again one by one against the documents in a second AI review, separate from the LLM judge: all judge-graded answers behind the findings above, plus the outdated-policy and M2 answers. "
          f"Agree on correct / not correct: {r['manual_review_ablation']['score_agree']}/{r['manual_review_ablation']['read']}; on the exact label: {r['manual_review_ablation']['label_agree']}/{r['manual_review_ablation']['read']} "
          "(two M3 answers graded \"abstained\" are \"partial\" in that check; not correct either way). File: `outputs/manual_review_ablation.json`.",
          "", "## One question got worse (M2)", "",
          f"M2 (\"{r['worse_question']['question']}\"): the old bot answered it right in both runs (Business and Enterprise; 1,000 requests per minute on Business). "
          "The fixed bot missed it in both runs. Both bots had the same two tables (SLA uptime and API limits) and no prices in their chunks. "
          "The fixed bot had no price table in its 6 chunks, and its rule is never guess, so it would not say which plan is cheaper. "
          "Run 1 answered \"The documents don't say.\" because the sources \"do not disclose the prices of the plans\"; run 2 named both plans and both limits but not the cheaper plan. "
          "When the price table is in its chunks (T7, the cheapest plan with the Salesforce integration), the same prompt does answer: Business, right in both runs. "
          "Without the reranker, the price table was not in its T7 chunks and it refused T7 the same way, in both runs. "
          "The old prompt guessed that Business is the cheaper plan and happened to be right; with the old prompt on the same 6 chunks, the fixed bot also answered M2 right in both runs. "
          f"M2's gold sources are only {' and '.join('`' + g + '`' for g in r['worse_question']['gold_sources'])}, and no document says Business costs less than Enterprise "
          "(Enterprise is \"Custom quote\" in `pricing-plans`), so the refusal is defensible. It is still counted as a miss. "
          f"Net result: {len(r['changes']['fixed_ids'])} old misses fixed, {len(r['changes']['worse_ids'])} old pass lost, {b1['correct']} to {a1['correct']}.",
          "", "## Three worked examples (run 1)", ""]
    for e in r["examples"]:
        L += [f"### {e['id']} ({e['category']}): {e['question']}", "",
              f"**Before ({e['before_verdict']}):** {e['before_answer']}", "",
              f"**After ({e['after_verdict']}):** {e['after_answer']}", "",
              f"**Why it changed:** {e['why']} {e['measured']}", ""]
    L += ["## Exact settings", "", "| | Before | After |", "|---|---|---|"]
    for k in BEFORE_SETTINGS:
        L.append(f"| {k} | {r['settings']['before'][k]} | {r['settings']['after'][k]} |")
    L += ["", "## How it was graded", "",
          "Full rubric: `eval/rubric.md`. Deterministic checks first (key numbers and phrases, known outdated values, \"don't know\" phrases); "
          "the LLM judge (gemini-3.8-flash, low reasoning, with the whole knowledge base and the rubric) only for mixed cases, yes/no answers and two-part answers.",
          "",
          f"Second check: {review['read']} graded answers checked again one by one against the documents in a second AI review, separately from the LLM judge ({review['judge_graded_read']} of them judge-graded). "
          f"Agree on correct / not correct: {review['score_agree']}/{review['read']}. Agree on the exact label: {review['label_agree']}/{review['read']}. Details: `outputs/manual_review.md`.",
          "",
          "Grader change, stated openly: the first grading pass (kept in `outputs/graded_v1/`) had two grader problems found while reading the answers: "
          "the judge only saw the gold document (so it flagged a true fact from the release notes as a hallucination), and one rule could mark a two-part answer "
          "as wrong and hallucinated from a single outdated keyword. The judge now sees the whole knowledge base, rules can only mark an answer correct or a false \"don't know\", "
          "everything else (and every question marked judge_only) goes to the judge, and all four runs were re-graded. "
          f"Correct answers, first pass: {', '.join(str(r['first_grading_pass_correct'][n]) for n in RUNS)}; final: {', '.join(str(s[n]['correct']) for n in RUNS)} (before 1, before 2, after 1, after 2). "
          "No answer and no pipeline setting was changed after the eval runs.",
          "", "The 30 eval questions and both pipelines were committed to git before the first eval run. 8 separate dev questions (`eval/dev_questions.json`) were used while building.",
          "", "## Cost", "",
          f"All LLM calls of this demo, including building, the 4 main eval runs, the 10 ablation runs and grading: {r['spend']['calls']} calls, ${r['spend']['total_usd']:.4f} in total (from OpenRouter usage data, `outputs/spend.jsonl`). "
          f"Before the ablations: {r['spend']['before_ablations']['calls']} calls, ${r['spend']['before_ablations']['usd']:.4f}; ablation runs: {r['spend']['ablation_runs']['calls']} calls, ${r['spend']['ablation_runs']['usd']:.4f}.", ""]
    for k, v in r["spend"]["by_purpose"].items():
        L.append(f"- {k}: {v['calls']} calls, ${v['usd']:.4f}")
    L += ["", "Synthetic data: the company (Orbitask) and all its documents are made up for this demo."]
    (OUT / "report.md").write_text("\n".join(L) + "\n")

    M = ["# Manual review of graded answers", "",
         f"{review['read']} graded answers checked again one by one against the documents in a second AI review, separate from the automatic grader. Agree on correct / not correct: {review['score_agree']}/{review['read']}. Agree on the exact label: {review['label_agree']}/{review['read']}.", "",
         "| Run | Question | Grade | Graded by | Agree | Note |", "|---|---|---|---|---|---|"]
    for m in review["items"]:
        M.append(f"| {m['run']} | {m['id']} | {m['grade']} | {m['grader']} | {'yes' if m['agree'] and m['label_agree'] else ('score yes, label no' if m['agree'] else 'no')} | {m['note']} |")
    (OUT / "manual_review.md").write_text("\n".join(M) + "\n")


if __name__ == "__main__":
    main()
