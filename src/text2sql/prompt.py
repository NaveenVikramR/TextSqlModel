"""Single source of truth for the prompt format.

Training, evaluation and serving must all build prompts through these functions,
otherwise the fine-tuned model sees a different format at inference time.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "respond with a single SQLite query that answers the question. "
    "Respond with SQL only, no explanation."
)


def build_user_prompt(question: str, schema: str) -> str:
    return f"### Schema\n{schema.strip()}\n\n### Question\n{question.strip()}"


def build_messages(question: str, schema: str, answer: str | None = None) -> list[dict]:
    """Chat messages for chat-template models. Include `answer` for training targets."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, schema)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def build_prompt_completion(question: str, schema: str, answer: str) -> dict:
    """TRL conversational prompt-completion example.

    With this format SFTTrainer computes loss on the completion (the SQL) only, and
    the prompt is rendered exactly as at inference (chat template + generation prompt).
    """
    return {
        "prompt": build_messages(question, schema),
        "completion": [{"role": "assistant", "content": answer.strip()}],
    }


_FENCE_RE =re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str | None) -> str:
    """Pull SQL out of a model response (handles ```sql fenced blocks)."""
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()
