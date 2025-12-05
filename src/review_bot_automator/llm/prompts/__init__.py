# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Prompt templates and examples for LLM-based comment parsing.

This package contains the prompt engineering system for parsing CodeRabbit
review comments. It includes:
- Base prompt template for structured JSON extraction
- Few-shot examples demonstrating different comment formats
- Format-specific examples (diff blocks, suggestions, natural language)

The prompts are designed to work with structured output modes (JSON) to ensure
reliable parsing and validation of LLM responses.
"""

from review_bot_automator.llm.prompts.base_prompt import (
    AI_PROMPT_FALLBACK_PREAMBLE,
    PARSE_COMMENT_PROMPT,
)
from review_bot_automator.llm.prompts.examples import (
    EXAMPLE_DIFF_BLOCKS,
    EXAMPLE_MULTI_OPTION,
    EXAMPLE_NATURAL_LANGUAGE,
    EXAMPLE_SUGGESTIONS,
)

__all__ = [
    "AI_PROMPT_FALLBACK_PREAMBLE",
    "EXAMPLE_DIFF_BLOCKS",
    "EXAMPLE_MULTI_OPTION",
    "EXAMPLE_NATURAL_LANGUAGE",
    "EXAMPLE_SUGGESTIONS",
    "PARSE_COMMENT_PROMPT",
]
