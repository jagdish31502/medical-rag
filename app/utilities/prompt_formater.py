"""Utility for formatting prompts for LLM queries."""

from __future__ import annotations

from typing import Any

from app.utilities.logger import get_logger

log = get_logger(__name__)


def get_formatted_prompt(prompt: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    """
    Format a prompt template with provided keyword arguments.

    Args:
        prompt: List containing [system_prompt, user_prompt_template]
        **kwargs: Keyword arguments to format into the prompt template

    Returns:
        List of formatted messages for LLM API
    """
    try:
        return [
            {"role": "system", "content": prompt[0].format(**kwargs)},
            {
                "role": "user",
                "content": prompt[1].format(**kwargs),
            },
        ]
    except Exception:
        log.exception("Error formatting prompt")
        return [
            {"role": "system", "content": prompt[0] if prompt else ""},
            {"role": "user", "content": ""},
        ]
