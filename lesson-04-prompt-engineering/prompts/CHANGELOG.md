# Prompt changelog

## v1
Baseline zero-shot prompt: one instruction plus the JSON schema.

## v2
Problems seen in v1 (llama3.2:3b): invented salary for "competitive salary" (0 USD/year),
"up to X" filled salary_min, guessed seniority and work_format, "confidential" kept as the
company name, "Team Lead" mapped to middle, and the injection in 06 was followed.
Changes:
- Treat the <posting> content as data only; explicit rule against following embedded instructions.
- General rule: extract only stated facts; use null / "unknown" instead of guessing.
- Per-field rules: seniority mapping (Team Lead -> lead, "not specified" -> unknown),
  On-site -> office, hidden company -> null, salary range / "up to" / "from" handling, never 0.
- Positive salary rule: if an amount is stated, always fill salary_min/salary_max.
- Two synthetic few-shot examples (not taken from the eval set): a normal range/year posting,
  and an "up to" / lead / on-site posting with an injection.
  (A first draft with only the negative rules and one example over-nulled salaries: 61/72.)
Result (llama3.2:3b, temp 0): fields 60/72 (83%) -> 70/72 (97%), invented 8 -> 1, injection 1/1 -> 0/1.
Remaining misses in 02: salary_min=5000 for "up to 5 000 EUR", company=null.
