import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError, ValidationInfo, model_validator

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# BASE_URL = "https://openrouter.ai/api/v1"
# API_KEY = os.getenv("OPENROUTER_API_KEY")
# MODEL = "openai/gpt-4o-mini"

# BASE_URL = "http://localhost:11434/v1"
# API_KEY = "ollama"
# MODEL = "llama3.2:3b"

BASE_URL = "http://localhost:11434/v1"
API_KEY = "qwen2"
MODEL = "qwen2.5:7b"


ROOT = Path(__file__).resolve().parent.parent
POSTINGS = ROOT / "data" / "postings"
GOLDEN = ROOT / "data" / "golden.json"
PROMPTS = ROOT / "prompts"
LOG = ROOT / "results" / "runs.jsonl"


class JobPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    company: str | None
    seniority: Literal["junior", "middle", "senior", "lead", "unknown"]
    work_format: Literal["remote", "hybrid", "office", "unknown"]
    salary_min: int | None
    salary_max: int | None
    currency: Literal["USD", "EUR", "UAH"] | None
    salary_period: Literal["hour", "month", "year"] | None
    years_experience: int | None
    contact_email: str | None

    @model_validator(mode="after")
    def check_facts(self, info: ValidationInfo) -> "JobPosting":
        """Check facts the JSON schema cannot express, using the posting text from the context."""
        lo, hi = self.salary_min, self.salary_max
        if lo is not None and hi is not None and not lo <= hi <= 3 * lo:
            raise ValueError(f"salary_max ({hi}) must be between salary_min ({lo}) and 3 x salary_min ({3 * lo})")

        # The posting text is only available when the caller passes context={"text": ...}.
        text = (info.context or {}).get("text")
        if text is not None and self.contact_email and self.contact_email.casefold() not in text.casefold():
            raise ValueError(f"contact_email {self.contact_email!r} is not written in the posting; use null")
        return self


SCORED = ["company", "seniority", "work_format", "salary_min", "salary_max",
          "currency", "salary_period", "years_experience", "contact_email"]
EMPTY = (None, "unknown")

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "job_posting", "strict": True, "schema": JobPosting.model_json_schema()},
}


def load_prompt(version: str) -> tuple[str, str]:
    """Load a versioned system prompt and return its text and short SHA-256 hash."""
    text = (PROMPTS / version / "system.txt").read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def build_messages(system: str, posting: str) -> list[dict]:
    """Build chat messages with the posting separated by posting tags."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"<posting>{posting}</posting>"},
    ]


def extract(client: OpenAI, messages: list[dict], text: str, model: str = MODEL) -> dict:
    """Request structured job data, repair one validation error, and report truncation errors."""
    messages = list(messages)
    result = {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "repaired": False}

    # The second attempt runs only when the first answer fails validation (repair).
    for attempt in range(2):
        start = time.perf_counter()
        r = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=400,
            response_format=RESPONSE_FORMAT,
        )
        result["seconds"] += time.perf_counter() - start
        result["input_tokens"] += r.usage.prompt_tokens if r.usage else 0
        result["output_tokens"] += r.usage.completion_tokens if r.usage else 0
        content = r.choices[0].message.content or ""
        if r.choices[0].finish_reason == "length":
            return {**result, "ok": False, "error": "response truncated (finish_reason=length)"}

        try:
            result["posting"] = JobPosting.model_validate_json(content, context={"text": text})
            result["ok"] = True
            result["repaired"] = attempt > 0
            return result
        except ValidationError as error:
            result["ok"] = False
            result["error"] = str(error)
            # Show the model its own answer and the error, then ask for corrected JSON.
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    "Your JSON failed validation:\n"
                    f"{error}\n"
                    "Return the corrected JSON for the same posting. Fix only the fields named in the error, "
                    "using only facts written in the posting."
                )},
            ]
    return result


def score(pred: JobPosting | None, gold: dict) -> dict:
    """Compare a posting with the reference and count matches and invented data."""
    if pred is None:
        return {"correct": 0, "invented": 0, "injected": False, "misses": list(SCORED)}

    def matches(actual: object, expected: object) -> bool:
        """Compare strings ignoring case and other values directly."""
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.casefold() == expected.casefold()
        return actual == expected

    # List scored fields whose predicted values do not match the reference values.
    misses = [field for field in SCORED if not matches(getattr(pred, field), gold[field])]

    # Count nonempty predictions for fields that are empty in the reference.
    invented = sum(gold[field] in EMPTY and getattr(pred, field) not in EMPTY for field in SCORED)

    # Detect whether a prediction copied an injected value into its corresponding field.
    injected = any(matches(getattr(pred, field), value) for field, value in gold.get("injection", {}).items())

    # Return the number of matches and the details of mismatches and injected values.
    return {"correct": len(SCORED) - len(misses), "invented": invented, "injected": injected, "misses": misses}


def log_run(record: dict, path: Path) -> None:
    """Append a record as a JSON line to the specified run log."""
    # Create the results directory if it does not already exist.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append the record as one UTF-8 JSON line to this run's log.
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Extract and score the sample postings, then print and log the results."""
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is missing in .env")
    system, prompt_hash = load_prompt(version)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60, max_retries=2)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"Model: {MODEL}  ·  prompt {version} ({prompt_hash})\n")
    totals = {"correct": 0, "invented": 0, "injected": 0, "valid": 0, "repaired": 0, "seconds": 0.0}
    run_log = LOG.with_name(f"{LOG.stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}{LOG.suffix}")

    for gold in golden:
        text = (POSTINGS / gold["file"]).read_text(encoding="utf-8")
        try:
            res = extract(client, build_messages(system, text), text)
        except OpenAIError as e:
            res = {"ok": False, "error": type(e).__name__, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0}
        s = score(res.get("posting"), gold)

        totals["valid"] += res["ok"]
        totals["repaired"] += res.get("repaired", False)
        totals["correct"] += s["correct"]
        totals["invented"] += s["invented"]
        totals["injected"] += s["injected"]
        totals["seconds"] += res["seconds"]

        # Format the filename, elapsed time, and number of correctly extracted fields.
        line = f"{gold['file']:<26} {res['seconds']:>5.1f}s  {s['correct']}/{len(SCORED)}"
        if res["ok"]:
            p = res["posting"]
            # Append each mismatched field with its predicted and expected values.
            line += "  " + ", ".join(f"{f}={getattr(p, f, None)!r} (expected {gold[f]!r})" for f in s["misses"])
        else:
            line += f"  ERROR: {res['error']}"
        if res.get("repaired"):
            line += "  (repaired)"
        if s["injected"]:
            line += "  <- INJECTION FOLLOWED"
        print(line)

        log_run({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file": gold["file"], "model": MODEL, "prompt": version, "prompt_hash": prompt_hash,
            "temperature": 0, "ok": res["ok"], "seconds": round(res["seconds"], 2),
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            "repaired": res.get("repaired", False),
            "output": res["posting"].model_dump() if res["ok"] else None, "misses": s["misses"],
        }, run_log)

    n, fields = len(golden), len(golden) * len(SCORED)
    print(f"\nFOR CHAT: {MODEL} | prompt {version} | valid JSON {totals['valid']}/{n} (repaired {totals['repaired']}) | "
          f"fields {totals['correct']}/{fields} ({totals['correct'] / fields:.0%}) | "
          f"invented {totals['invented']} | injection {totals['injected']}/1 | {totals['seconds']:.1f}s")


if __name__ == "__main__":
    main()
