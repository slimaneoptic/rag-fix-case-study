"""The two pipelines compared in this demo.

BEFORE = a typical first RAG build seen in client projects.
AFTER  = the same knowledge base, embedding model and LLM, with the fixes applied.
Only the pipeline changes between the two, so the score difference comes from the fixes.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

import llm

KB = Path(__file__).parent / "kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

BEFORE_SETTINGS = {
    "loading": "document body text only; the title, effective date and status are not attached to chunks",
    "chunking": "fixed 500-character windows with 50-character overlap, cut anywhere (tables, rows and sentences get split)",
    "retrieval": "dense only: all-MiniLM-L6-v2 embeddings, cosine similarity, top 2 chunks",
    "rerank": "none",
    "prompt": "\"Answer the question based on the context below.\" No instruction to cite sources or to say when the answer is not in the context.",
    "llm": "google/gemini-3.8-flash, reasoning effort low (same model in both pipelines)",
}
AFTER_SETTINGS = {
    "loading": "body text plus metadata from each document header: title, effective date, status (current or superseded)",
    "chunking": "one chunk per heading section; tables are never split; sections over 1,200 characters are split only between paragraphs; each chunk is indexed with its document title and section heading",
    "retrieval": "hybrid: BM25 keyword search (top 10) + all-MiniLM-L6-v2 dense search (top 10), merged with reciprocal rank fusion (k = 60) into 12 candidates",
    "rerank": "cross-encoder/ms-marco-MiniLM-L6-v2 scores the 12 candidates; superseded documents are ranked below current ones; top 6 chunks are kept",
    "prompt": "answer only from the sources, cite a source id for every fact, say \"The documents don't say.\" when the answer is missing, prefer current over superseded documents, read table headers, show any calculation",
    "llm": "google/gemini-3.8-flash, reasoning effort low (same model in both pipelines)",
}


# ---------- shared ----------
def load_docs() -> list[dict]:
    docs = []
    for path in sorted(KB.glob("*.md")):
        raw = path.read_text()
        head, body = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S).groups()
        meta = dict(line.split(": ", 1) for line in head.splitlines())
        docs.append({"doc_id": meta["doc_id"], "meta": meta, "body": body.strip()})
    return docs


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    return embedder().encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


# ---------- BEFORE ----------
class BeforePipeline:
    name = "before"
    settings = BEFORE_SETTINGS
    SIZE, OVERLAP, TOP_K = 500, 50, 2

    def __init__(self):
        self.chunks = []
        for doc in load_docs():
            text = doc["body"]
            for start in range(0, len(text), self.SIZE - self.OVERLAP):
                self.chunks.append({"doc_id": doc["doc_id"], "text": text[start:start + self.SIZE]})
                if start + self.SIZE >= len(text):
                    break
        self.vectors = embed([c["text"] for c in self.chunks])

    def retrieve(self, question: str) -> list[dict]:
        scores = self.vectors @ embed([question])[0]
        return [self.chunks[i] for i in np.argsort(-scores)[: self.TOP_K]]

    def prompt(self, question: str, hits: list[dict]) -> list[dict]:
        context = "\n\n".join(h["text"] for h in hits)
        return [
            {"role": "system", "content": "You are a helpful assistant for Orbitask customers."},
            {"role": "user", "content": f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"},
        ]

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        t1 = time.perf_counter()
        out = llm.complete_text(self.prompt(question, hits), purpose="answer-before")
        t2 = time.perf_counter()
        return {"answer": out["text"], "model": out["model"], "cost_usd": out["cost_usd"],
                "retrieval_s": round(t1 - t0, 3), "llm_s": round(t2 - t1, 3), "latency_s": round(t2 - t0, 3),
                "retrieved": [{"doc_id": h["doc_id"], "text": h["text"]} for h in hits]}


# ---------- AFTER ----------
STOP = set("a an the of to in on for and or is are be do does did i we my our you your it its at by with from how what which when can if as this that there".split())


def tokens(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in STOP]


def blocks(lines: list[str]) -> list[str]:
    """Split a section into paragraphs and whole tables (a table is never cut)."""
    out, buf, in_table = [], [], False
    for line in lines + [""]:
        is_row = line.lstrip().startswith("|")
        if (not line.strip() and not in_table) or (in_table and not is_row) or (is_row and not in_table and buf):
            if buf:
                out.append("\n".join(buf).strip())
            buf, in_table = [], False
        if line.strip():
            buf.append(line)
            in_table = is_row
    return [b for b in out if b]


class AfterPipeline:
    name = "after"
    settings = AFTER_SETTINGS
    MAX_CHARS, FIRST_K, CANDIDATES, TOP_K, RRF_K = 1200, 10, 12, 6, 60

    def __init__(self):
        from rank_bm25 import BM25Okapi
        self.chunks = []
        for doc in load_docs():
            meta = doc["meta"]
            sections, heading, buf = [], meta["title"], []
            for line in doc["body"].splitlines():
                if line.startswith("## ") or line.startswith("# "):
                    if any(l.strip() for l in buf):
                        sections.append((heading, buf))
                    heading = line.lstrip("#").strip() if line.startswith("## ") else meta["title"]
                    buf = []
                else:
                    buf.append(line)
            if any(l.strip() for l in buf):
                sections.append((heading, buf))
            for heading, lines in sections:
                packed, size = [], 0
                for block in blocks(lines):
                    if packed and size + len(block) > self.MAX_CHARS:
                        self._add(doc, heading, packed)
                        packed, size = [], 0
                    packed.append(block)
                    size += len(block)
                if packed:
                    self._add(doc, heading, packed)
        self.bm25 = BM25Okapi([tokens(c["index_text"]) for c in self.chunks])
        self.vectors = embed([c["index_text"] for c in self.chunks])

    def _add(self, doc: dict, heading: str, packed: list[str]) -> None:
        meta = doc["meta"]
        section = meta["title"] if heading == meta["title"] else f"{meta['title']} > {heading}"
        text = "\n\n".join(packed)
        self.chunks.append({"doc_id": doc["doc_id"], "section": section, "text": text,
                            "index_text": f"{section}\n{text}", "effective_date": meta["effective_date"],
                            "status": meta["status"], "superseded_by": meta.get("superseded_by", "")})

    def retrieve(self, question: str) -> list[dict]:
        bm = self.bm25.get_scores(tokens(question))
        dense = self.vectors @ embed([question])[0]
        fused = {}
        for ranking in (np.argsort(-bm)[: self.FIRST_K], np.argsort(-dense)[: self.FIRST_K]):
            for rank, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.RRF_K + rank + 1)
        candidates = sorted(fused, key=fused.get, reverse=True)[: self.CANDIDATES]
        scores = reranker().predict([(question, self.chunks[i]["index_text"]) for i in candidates], show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda p: (self.chunks[p[0]]["status"] == "superseded", -p[1]))
        return [{**self.chunks[i], "rerank_score": round(float(s), 3)} for i, s in ranked[: self.TOP_K]]

    @staticmethod
    def source_label(h: dict) -> str:
        status = "current" if h["status"] == "current" else f"SUPERSEDED (outdated, replaced by {h['superseded_by']})"
        return f"[{h['doc_id']}] {h['section']} | effective {h['effective_date']} | status: {status}"

    def prompt(self, question: str, hits: list[dict]) -> list[dict]:
        context = "\n\n".join(f"Source {n}: {self.source_label(h)}\n{h['text']}" for n, h in enumerate(hits, 1))
        system = (
            "You answer customer questions about Orbitask using ONLY the sources provided.\n"
            "Rules:\n"
            "1. Use only facts written in the sources. Never use outside knowledge and never guess.\n"
            "2. Cite the source id in square brackets after each fact, for example [refund-policy].\n"
            "3. If the sources do not contain the answer, start your reply with \"The documents don't say.\" "
            "You may add one short sentence on what the sources do cover, without guessing.\n"
            "4. If sources disagree, follow the one with status current and the latest effective date. "
            "Use a superseded source only for questions about the past, and say it is outdated.\n"
            "5. For tables, check the column headers and the row before quoting a value.\n"
            "6. If the answer needs a calculation, show it in one line.\n"
            "7. Keep the answer short: 1 to 4 sentences."
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"}]

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        t1 = time.perf_counter()
        out = llm.complete_text(self.prompt(question, hits), purpose="answer-after")
        t2 = time.perf_counter()
        return {"answer": out["text"], "model": out["model"], "cost_usd": out["cost_usd"],
                "retrieval_s": round(t1 - t0, 3), "llm_s": round(t2 - t1, 3), "latency_s": round(t2 - t0, 3),
                "retrieved": [{"doc_id": h["doc_id"], "section": h["section"], "status": h["status"],
                               "rerank_score": h["rerank_score"], "text": h["text"]} for h in hits]}


PIPELINES = {"before": BeforePipeline, "after": AfterPipeline}
