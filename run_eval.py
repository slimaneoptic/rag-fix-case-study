"""Answer every question of a set with one pipeline and save the raw run, then grade it.

Usage: python run_eval.py <before|after> <run number> [eval|dev]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from grade import SETS, grade_run
from rag import PIPELINES

ROOT = Path(__file__).parent


def main(pipeline_name: str, run_no: int, qset: str = "eval") -> None:
    questions = json.loads((ROOT / "eval" / SETS[qset]).read_text())
    pipe = PIPELINES[pipeline_name]()
    pipe.answer("warm-up: load the models")  # model loading is not counted in the latency
    items = []
    for q in questions:
        out = pipe.answer(q["question"])
        items.append({"id": q["id"], "category": q["category"], "question": q["question"], **out})
        print(f"{q['id']} {out['latency_s']:5.2f}s {out['answer'][:90]!r}", flush=True)
    run = {"set": qset, "pipeline": pipeline_name, "run": run_no, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "settings": pipe.settings, "items": items}
    path = ROOT / "outputs" / "runs" / f"{qset}_{pipeline_name}_run{run_no}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False))
    grade_run(path)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "eval")
