# Prompt changelog

## v2
- system: explicit rules on top of v1, each aimed at a failure seen in the v1 run.
  1. Conflicting sources: trust the one that replaces the other (a memo overriding a section), else the later date.
     Why: the manual has an old cut-off table (7.4, 2016) and a memo that replaces it (QMS-2017-04).
  2. A status instead of a value ("discontinued") is an answer, with a generic example.
     Why: q16 (Raben cut-off) failed in v1: the memo was in context, the model said found=false.
     A rule without the example was not enough for gpt-4o-mini, the example fixed it.
  3. Combine several sources and cite all of them (multi_chunk questions).
  4. Answer in the language of the question (Ukrainian questions, English sources).
  5. found=false only when no source says anything about the question; no outside knowledge.
- ready: condense.txt (follow-up questions, TODO 9), map.txt and reduce.txt (map-reduce, TODO 8).

| prompt | retr | ctx  | correct | cited | abstain | f.abst | tokens in | $/1000 | failed        |
|--------|------|------|---------|-------|---------|--------|-----------|--------|---------------|
| v1     | 0.95 | 0.95 | 0.84    | 0.84  | 1.00    | 0.16   | 649       | 0.116  | q09 q16 q23   |
| v2     | 0.95 | 0.95 | 0.89    | 0.89  | 1.00    | 0.11   | 859       | 0.148  | q09 q23       |

Not fixable by the prompt: q09 (the relevant chunk is not retrieved), q23 (follow-up, needs condense, TODO 9).
Cost: about +210 input tokens per question for the longer system prompt (+28% $/1000).
Latency is not comparable between these runs: the network dropped connections and calls were retried.

## v1
- system: answer only from the sources, cite source labels, found=false when the answer is not in the sources.
