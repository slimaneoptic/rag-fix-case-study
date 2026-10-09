# RAG chatbot fix: before and after

Same 12 documents, same 30 questions, same embedding model and same LLM. Only the pipeline changed.

Note: this public copy includes this report, but not the raw runs, grades, second-review files or spend log it names under `outputs/`.

## Result

| | Before, run 1 | Before, run 2 | After, run 1 | After, run 2 |
|---|---|---|---|---|
| Correct answers | 20/30 | 20/30 | 29/30 | 29/30 |
| table lookup | 5/7 | 5/7 | 7/7 | 7/7 |
| current vs outdated policy | 3/6 | 3/6 | 6/6 | 6/6 |
| multi-hop (two documents) | 2/6 | 2/6 | 5/6 | 5/6 |
| exact numbers | 5/6 | 5/6 | 6/6 | 6/6 |
| unanswerable | 5/5 | 5/5 | 5/5 | 5/5 |
| Hallucinations (wrong or outdated fact given as valid) | 3 | 3 | 0 | 0 |
| Unanswerable questions correctly declined | 5/5 | 5/5 | 5/5 | 5/5 |
| False "don't know" on answerable questions | 4 | 5 | 1 | 0 |
| Right document(s) retrieved (25 answerable) | 20/25 | 20/25 | 25/25 | 25/25 |
| Answers citing a source | 0/30 | 0/30 | 29/30 | 29/30 |
| Median latency per question | 2.03 s | 1.96 s | 2.12 s | 2.04 s |
| Mean LLM cost per question | $0.0005 | $0.0005 | $0.0009 | $0.0009 |
| Graded by rules / by LLM judge | 19 / 11 | 21 / 9 | 26 / 4 | 26 / 4 |

Missed questions: before run 1 T1, T7, C3, C4, C6, M1, M3, M4, M5, E6; before run 2 T1, T7, C3, C4, C6, M1, M3, M4, M5, E6; after run 1 M2; after run 2 M2.
Hallucinations: before run 1 C3, C4, C6; before run 2 C3, C4, C6; after: none.
Latency is retrieval + LLM call, measured after the models were loaded. Median retrieval time: before 0.005 s, after 0.14 s (BM25 + embeddings + reranker on CPU). Models used: google/gemini-3.8-flash.

Net change, question by question: 10 questions wrong in both before runs became right in both after runs (T1, T7, C3, C4, C6, M1, M3, M4, M5, E6); 1 question right in both before runs became a miss in both after runs (M2, see below). 20 + 10 - 1 = 29.

## What went wrong before, and the fix

The cause of each miss was found by reading the chunks each before answer received (retrieval is deterministic, so both before runs saw the same chunks). All fixes were applied together; the counts after each fix are for the full fix. The next section measures each fix on its own.

- **Tables and sections cut in the middle** (T1, M1, M3, M5). Fix: Heading- and table-aware chunking: one chunk per section, tables never split, each chunk tagged with its document and section title. Wrong before in both runs: 4/4; correct after the full fix in both runs: 4/4. Fixed bot with the old 500-character chunks: 27 and 27 of 30 (`abl-after-fixed-chunks`).
- **Outdated policy retrieved next to the current one, with no dates** (C3, C4, C6). Fix: Document metadata (effective date, current or superseded): outdated documents rank below current ones and are labelled in the prompt. Wrong before in both runs: 3/3; correct after the full fix in both runs: 3/3. Fixed bot without dates and status: 29 and 27 of 30 (`abl-after-no-metadata`).
- **The right document was not in the top 2 results** (T7, E6, M4). Fix: Hybrid search (keywords + embeddings), a reranker, and 6 chunks instead of 2. Wrong before in both runs: 3/3; correct after the full fix in both runs: 3/3. Fixed bot without the reranker only (hybrid search and 6 chunks kept): 28 and 28 of 30 (`abl-after-no-rerank`).
- **Where the dates and status come from:** each of the 12 synthetic documents has its effective date and status (current or superseded) in its header, written when the documents were made; the old pipeline does not use these headers. On real documents this comes from file dates, version names, or a one-time review in the first step. Here the current refund policy's own text also says which policy it replaces.
- **Prompt** (cite a source for every fact, say "The documents don't say." when the answer is missing, never guess). After: 29/30 and 29/30 answers cite a source, against 0/30 before. Honest note: the before bot already declined all 5 unanswerable questions (5/5 in both runs), so this fix did not change that score, and the fixed bot with the old prompt scored 30 and 30 of 30. The strict prompt adds citations, not accuracy, on this test set.

## Which fix mattered (ablation)

Each row changes ONE thing in the old or the fixed pipeline (`ablation.py`, committed before these runs; `rag.py` was not changed). Same 30 questions, same grader, two runs each. These runs were made after the main results, to answer "which fix mattered, or was it just more text?".

| Version | Correct (run 1 / run 2) | Outdated or wrong facts given as valid | Answers citing a source | Missed |
|---|---|---|---|---|
| Old bot (top 2 chunks) | 20 / 20 | 3 / 3 | 0 / 0 | T1, T7, C3, C4, C6, M1, M3, M4, M5, E6 |
| Fixed bot | 29 / 29 | 0 / 0 | 29 / 29 | M2 |
| Old bot, 6 chunks instead of 2 | 25 / 25 | 2 / 2 | 0 / 0 | C5, C6, M3, M4, E6 |
| Fixed bot, but the old 500-character chunks | 27 / 27 | 0 / 0 | 28 / 28 | M1, M2, M3 |
| Fixed bot, but no document dates or status | 29 / 27 | 0 / 2 | 29 / 28 | M2 / C5, C6, M2 |
| Fixed bot, but no reranker | 28 / 28 | 0 / 0 | 29 / 30 | T7, M2 |
| Fixed bot, but the old prompt | 30 / 30 | 0 / 0 | 0 / 0 | none |

What this shows:

- More text alone helps a lot: 6 chunks instead of 2 took the old bot from 20 to 25 and 25. It still gave an outdated refund rule as valid (C5, C6 in run 1; C5, C6 in run 2) and missed C5, C6, M3, M4, E6.
- Whole tables: without them the fixed bot scored 27 and 27. It missed M1 and M3 in both runs because the chunk with the Business price row was not among its 6 chunks (it also missed M2, like the full fix).
- Dates and status: without them the fixed bot scored 29 and 27; in run 2 it gave the old 2024 refund rule as a second valid rule (C5, C6). The current policy's own text says it replaces the 2024 one, but without the tags that was not enough in run 2.
- Reranker: without it, 28 and 28 (missed T7, M2).
- Prompt: with the old prompt on the same 6 chunks, 30 and 30, but no answer cites a source. On the same chunks, the strict prompt is what turned M2 into a miss (next section). The main pipelines were NOT changed after these runs: that would be tuning on the test answers.

Second check of the ablation grades: 30 answers checked again one by one against the documents in a second AI review, separate from the LLM judge: all judge-graded answers behind the findings above, plus the outdated-policy and M2 answers. Agree on correct / not correct: 30/30; on the exact label: 28/30 (two M3 answers graded "abstained" are "partial" in that check; not correct either way). File: `outputs/manual_review_ablation.json`.

## One question got worse (M2)

M2 ("Which plans have an uptime SLA, and what is the API rate limit per minute on the cheapest of them?"): the old bot answered it right in both runs (Business and Enterprise; 1,000 requests per minute on Business). The fixed bot missed it in both runs. Both bots had the same two tables (SLA uptime and API limits) and no prices in their chunks. The fixed bot had no price table in its 6 chunks, and its rule is never guess, so it would not say which plan is cheaper. Run 1 answered "The documents don't say." because the sources "do not disclose the prices of the plans"; run 2 named both plans and both limits but not the cheaper plan. When the price table is in its chunks (T7, the cheapest plan with the Salesforce integration), the same prompt does answer: Business, right in both runs. Without the reranker, the price table was not in its T7 chunks and it refused T7 the same way, in both runs. The old prompt guessed that Business is the cheaper plan and happened to be right; with the old prompt on the same 6 chunks, the fixed bot also answered M2 right in both runs. M2's gold sources are only `sla` and `api-rate-limits`, and no document says Business costs less than Enterprise (Enterprise is "Custom quote" in `pricing-plans`), so the refusal is defensible. It is still counted as a miss. Net result: 10 old misses fixed, 1 old pass lost, 20 to 29.

## Three worked examples (run 1)

### T1 (table lookup): What is the price per user of the Team plan when billed annually?

**Before (abstained):** Based on the provided context, the pricing details for the Team plan are missing/cut off from the table, so the exact price cannot be determined from the text.

**After (correct):** The Team plan costs $20 per user per month when billed annually [pricing-plans]. Annual billing is charged once a year for the whole term [pricing-plans]. Calculation: $20 × 12 months = $240 per user per year [pricing-plans].

**Why it changed:** The 500-character split cut the pricing table inside its first row (Starter), so the Team row never reached the model. Keeping tables whole put the full table in the context. Measured: the old bot with 6 chunks instead of 2 also got this one right, in both runs, so more chunks alone fix it too.

### C6 (current vs outdated policy): Is there a fee when a refund is paid back for a bank transfer payment?

**Before (wrong):** Based on the provided text, there are conflicting policies:

* One section states: **There is no fee for refunds**, including refunds of payments made by bank transfer.
* Another section states: Refunds of payments made by bank transfer carry a **$25 processing fee**.

**After (correct):** No, there is no fee for refunds, including refunds of payments made by bank transfer [refund-policy]. Approved refunds are sent to the original payment method within 5 business days [refund-policy].

**Why it changed:** The old 2024 refund policy and the current one were both retrieved, with no dates, so the bot offered both rules. With document dates and status, the outdated policy ranks below the current one and is labelled as superseded. Measured: without the dates and status, the fixed bot gave the old $25 fee as a second valid rule in 1 of 2 runs.

### M3 (multi-hop (two documents)): A Business customer with 10 users on monthly billing had 98.5% uptime last month. How much is the service credit in dollars?

**Before (partial):** Based on the provided context, the exact monthly price for the **Business** plan is missing (the plan comparison table cuts off after the Starter plan). Therefore, the exact dollar amount of the service credit cannot be determined.

However, based on the uptime table, an uptime of 98.5% qualifies for a **25% service credit** (since it is below 99.0% but at least 95.0%) applied to that month's subscription fee.

**After (correct):** The Business plan on monthly billing costs $39 per user per month [pricing-plans], making the monthly fee $390 for 10 users ($39 × 10 = $390). Because the Business uptime commitment is 99.9% [sla] and the monthly uptime of 98.5% is below 99.0% but at least 95.0%, the customer is eligible for a 25% service credit [sla]. 

Calculation: $390 × 0.25 = $97.50.

The service credit is $97.50, which is added to the account balance [sla].

**Why it changed:** The answer needs two documents: the price table and the SLA credit table. Before, the price table was cut off; after, both whole tables were retrieved and the bot showed the calculation. Measured: with the old 500-character chunks, the fixed bot still missed it in both runs.

## Exact settings

| | Before | After |
|---|---|---|
| loading | document body text only; the title, effective date and status are not attached to chunks | body text plus metadata from each document header: title, effective date, status (current or superseded) |
| chunking | fixed 500-character windows with 50-character overlap, cut anywhere (tables, rows and sentences get split) | one chunk per heading section; tables are never split; sections over 1,200 characters are split only between paragraphs; each chunk is indexed with its document title and section heading |
| retrieval | dense only: all-MiniLM-L6-v2 embeddings, cosine similarity, top 2 chunks | hybrid: BM25 keyword search (top 10) + all-MiniLM-L6-v2 dense search (top 10), merged with reciprocal rank fusion (k = 60) into 12 candidates |
| rerank | none | cross-encoder/ms-marco-MiniLM-L6-v2 scores the 12 candidates; superseded documents are ranked below current ones; top 6 chunks are kept |
| prompt | "Answer the question based on the context below." No instruction to cite sources or to say when the answer is not in the context. | answer only from the sources, cite a source id for every fact, say "The documents don't say." when the answer is missing, prefer current over superseded documents, read table headers, show any calculation |
| llm | google/gemini-3.8-flash, reasoning effort low (same model in both pipelines) | google/gemini-3.8-flash, reasoning effort low (same model in both pipelines) |

## How it was graded

Full rubric: `eval/rubric.md`. Deterministic checks first (key numbers and phrases, known outdated values, "don't know" phrases); the LLM judge (gemini-3.8-flash, low reasoning, with the whole knowledge base and the rubric) only for mixed cases, yes/no answers and two-part answers.

Second check: 21 graded answers checked again one by one against the documents in a second AI review, separately from the LLM judge (14 of them judge-graded). Agree on correct / not correct: 21/21. Agree on the exact label: 20/21. Details: `outputs/manual_review.md`.

Grader change, stated openly: the first grading pass (kept in `outputs/graded_v1/`) had two grader problems found while reading the answers: the judge only saw the gold document (so it flagged a true fact from the release notes as a hallucination), and one rule could mark a two-part answer as wrong and hallucinated from a single outdated keyword. The judge now sees the whole knowledge base, rules can only mark an answer correct or a false "don't know", everything else (and every question marked judge_only) goes to the judge, and all four runs were re-graded. Correct answers, first pass: 20, 20, 29, 29; final: 20, 20, 29, 29 (before 1, before 2, after 1, after 2). No answer and no pipeline setting was changed after the eval runs.

The 30 eval questions and both pipelines were committed to git before the first eval run. 8 separate dev questions (`eval/dev_questions.json`) were used while building.

## Cost

All LLM calls of this demo, including building, the 4 main eval runs, the 10 ablation runs and grading: 560 calls, $0.7624 in total (from OpenRouter usage data, `outputs/spend.jsonl`). Before the ablations: 199 calls, $0.2510; ablation runs: 361 calls, $0.5114.

- smoke-test: 1 calls, $0.0000
- answer-before: 133 calls, $0.0923
- judge: 107 calls, $0.3581
- answer-after: 319 calls, $0.3120

Synthetic data: the company (Orbitask) and all its documents are made up for this demo.
