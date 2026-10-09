"""Ablation runs: change ONE thing at a time, to measure which fix mattered.

The two main pipelines in rag.py are not changed (they were committed before the first eval run).
Each variant below is the old or the fixed pipeline with a single change; everything else is identical.

Usage: python ablation.py <variant> <run number>      variants: see VARIANTS
"""
from __future__ import annotations

import sys

import numpy as np

import rag
from rag import AFTER_SETTINGS, BEFORE_SETTINGS, AfterPipeline, BeforePipeline, embed, load_docs


class BeforeTop6(BeforePipeline):
    """Old bot, but 6 chunks instead of 2 (same 500-character chunks, same plain prompt)."""
    name = "abl-before-top6"
    TOP_K = 6
    settings = {**BEFORE_SETTINGS, "ablation": "old bot with only one change: top 6 chunks instead of top 2",
                "retrieval": "dense only: all-MiniLM-L6-v2 embeddings, cosine similarity, top 6 chunks"}


class AfterFixedChunks(AfterPipeline):
    """Fixed bot, but with the old 500-character chunks (no sections, tables can be cut, no title/section tags)."""
    name = "abl-after-fixed-chunks"
    SIZE, OVERLAP = 500, 50
    settings = {**AFTER_SETTINGS, "ablation": "fixed bot with only one change: the old chunking",
                "chunking": "fixed 500-character windows with 50-character overlap, cut anywhere; no section or title tags in the index"}

    def __init__(self):
        from rank_bm25 import BM25Okapi
        self.chunks = []
        for doc in load_docs():
            meta, text = doc["meta"], doc["body"]
            for start in range(0, len(text), self.SIZE - self.OVERLAP):
                piece = text[start:start + self.SIZE]
                self.chunks.append({"doc_id": doc["doc_id"], "section": meta["title"], "text": piece, "index_text": piece,
                                    "effective_date": meta["effective_date"], "status": meta["status"],
                                    "superseded_by": meta.get("superseded_by", "")})
                if start + self.SIZE >= len(text):
                    break
        self.bm25 = BM25Okapi([rag.tokens(c["index_text"]) for c in self.chunks])
        self.vectors = embed([c["index_text"] for c in self.chunks])


class AfterNoMetadata(AfterPipeline):
    """Fixed bot, but without document metadata: no effective date or status in the prompt, no ranking of outdated docs last.
    The document text itself is unchanged (the current refund policy still says which policy it replaces)."""
    name = "abl-after-no-metadata"
    settings = {**AFTER_SETTINGS, "ablation": "fixed bot with only one change: no document metadata",
                "loading": "body text plus title only; no effective date, no current/superseded status",
                "rerank": "cross-encoder/ms-marco-MiniLM-L6-v2 scores the 12 candidates; top 6 chunks are kept (no status ordering)"}

    def _add(self, doc: dict, heading: str, packed: list[str]) -> None:
        super()._add(doc, heading, packed)
        self.chunks[-1].update({"effective_date": "", "status": "current", "superseded_by": ""})

    @staticmethod
    def source_label(h: dict) -> str:
        return f"[{h['doc_id']}] {h['section']}"


class AfterNoRerank(AfterPipeline):
    """Fixed bot, but without the cross-encoder: the top 6 of the BM25 + embeddings fusion go straight to the LLM."""
    name = "abl-after-no-rerank"
    settings = {**AFTER_SETTINGS, "ablation": "fixed bot with only one change: no reranker",
                "rerank": "none: top 6 of the reciprocal rank fusion; superseded documents are still ranked below current ones"}

    def retrieve(self, question: str) -> list[dict]:
        bm = self.bm25.get_scores(rag.tokens(question))
        dense = self.vectors @ embed([question])[0]
        fused = {}
        for ranking in (np.argsort(-bm)[: self.FIRST_K], np.argsort(-dense)[: self.FIRST_K]):
            for rank, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.RRF_K + rank + 1)
        ranked = sorted(fused, key=lambda i: (self.chunks[i]["status"] == "superseded", -fused[i]))
        return [{**self.chunks[i], "rerank_score": round(fused[i], 5)} for i in ranked[: self.TOP_K]]


class AfterOldPrompt(AfterPipeline):
    """Fixed bot (same chunks, same 6 sources with their dates and status), but the old plain prompt:
    no rule to cite, to say "the documents don't say", or to never guess."""
    name = "abl-after-old-prompt"
    settings = {**AFTER_SETTINGS, "ablation": "fixed bot with only one change: the old prompt",
                "prompt": "\"Answer the question based on the context below.\" (the old prompt); sources keep their labels with date and status"}

    def prompt(self, question: str, hits: list[dict]) -> list[dict]:
        context = "\n\n".join(f"Source {n}: {self.source_label(h)}\n{h['text']}" for n, h in enumerate(hits, 1))
        return [
            {"role": "system", "content": "You are a helpful assistant for Orbitask customers."},
            {"role": "user", "content": f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"},
        ]


VARIANTS = {c.name: c for c in (BeforeTop6, AfterFixedChunks, AfterNoMetadata, AfterNoRerank, AfterOldPrompt)}

if __name__ == "__main__":
    rag.PIPELINES.update(VARIANTS)
    import run_eval
    run_eval.main(sys.argv[1], int(sys.argv[2]))
