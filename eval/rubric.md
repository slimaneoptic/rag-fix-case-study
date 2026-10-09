# Grading rubric

Every answer gets one verdict. Only **correct** counts toward the score.

| Verdict | Meaning |
|---|---|
| correct | Gives every key fact of the gold answer, with no claim that contradicts the current documents. Extra true details are fine. For an unanswerable question: clearly says the documents do not contain the answer and does not invent one. |
| partial | Some key facts right, but a required part is missing or vague (for example, one half of a two-part question). |
| wrong | States a wrong value or rule, answers the wrong question, gives a definite answer to an unanswerable question, or offers an outdated (superseded) rule as valid, either alone or as one of two "conflicting" options without saying which one applies now. |
| abstained | Says the documents don't say (or that the information is missing), although the answer is in the documents. |

**Hallucination** (yes/no, counted separately): the answer presents to the customer, as true or as possibly true, a fact that is wrong under the current documents: an invented fact, a misread value (wrong table row or column), or an outdated rule offered as valid. Mentioning an old rule while clearly saying it no longer applies is fine. Saying "the documents don't say" is never a hallucination.

**Main error type** (one): none, outdated (used a superseded rule), wrong_value (misread or miscalculated), invented (not in any document), incomplete (part of the answer missing), false_dont_know (said the documents don't say, but they do).

## Step 1: deterministic checks (no LLM)

Each question lists required key facts (`include`: every group must appear in at least one accepted form), known wrong values (`traps`, such as the outdated 14-day refund window) and whether it is answerable. Text is lowercased, thousands separators are removed (10,000 = 10000) and numbers must match as whole numbers (30 does not match 300).

- Answerable, all key facts present, no trap value, no "don't know" phrase -> **correct**.
- Answerable, a "don't know" phrase, no key fact and no trap value -> **abstained**.
- Unanswerable and a "don't know" phrase -> **correct**.
- Questions marked `judge_only` (yes/no or two-part answers that key phrases cannot capture) and every other case -> step 2.

## Step 2: LLM judge

google/gemini-3.8-flash (reasoning low) receives this rubric, the complete knowledge base with each document's effective date and status, the question, the gold answer and gold sources, and the answer to grade. It returns JSON: verdict, hallucination, main error type and a one-sentence reason.

## Step 3: second check

At least 10 graded answers, including judge-graded ones, are re-read one by one against the documents, separately from the grader, and marked agree / disagree in `outputs/manual_review.md`.
