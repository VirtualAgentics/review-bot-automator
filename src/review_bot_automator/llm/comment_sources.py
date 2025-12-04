# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Comment source detection for structured CodeRabbit comment blocks.

This module provides extraction of structured blocks from CodeRabbit review
comments, including:
- Diff blocks (```diff)
- Suggestion blocks (```suggestion)
- AI prompt blocks (<details><summary>... Prompt for AI ...</summary>)
- Generic <details> blocks with type classification

These extracted sources enable targeted processing of different comment
formats by downstream parsers, particularly for prioritizing AI prompt
blocks over natural language inference.

Example:
    >>> from review_bot_automator.llm.comment_sources import extract_comment_sources
    >>> sources = extract_comment_sources(comment_body)
    >>> if sources.has_ai_prompt:
    ...     print(sources.ai_prompt_blocks[0].content)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Maximum comment size to process (100KB) - larger comments are truncated
_MAX_COMMENT_SIZE: Final[int] = 100 * 1024

# =============================================================================
# Regex Patterns (Module-Level, Compiled Once)
# =============================================================================

# Diff blocks: ```diff ... ```
_DIFF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```diff\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Suggestion blocks: ```suggestion ... ```
_SUGGESTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```suggestion\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# <details>...</details> blocks (non-greedy)
# NOTE: Nested <details> are NOT supported. The non-greedy `.*?` pattern pairs
# each opening <details> tag with the FIRST encountered </details> tag, which
# will break nested structures. For example, in "<details>outer<details>inner
# </details>more</details>", only "outer<details>inner" is captured, leaving
# "more</details>" as unparsed text. This is acceptable for CodeRabbit comments
# which do not use nested collapsible sections.
_DETAILS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<details[^>]*>(.*?)</details>",
    re.DOTALL | re.IGNORECASE,
)

# <summary>...</summary> within details
_SUMMARY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<summary[^>]*>(.*?)</summary>",
    re.DOTALL | re.IGNORECASE,
)

# HTML comments <!-- ... -->
_HTML_COMMENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<!--.*?-->",
    re.DOTALL,
)

# Diff hunk headers @@ -X,Y +X,Y @@
_HUNK_HEADER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^@@\s*-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s*@@",
    re.MULTILINE,
)

# **Option X:** labels preceding suggestion blocks
_OPTION_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\*\*([^*]+)\*\*\s*:?\s*$",
    re.MULTILINE,
)

# Code fences in AI prompt content (to be stripped)
_CODE_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^```\w*\s*\n?|\n?```\s*$",
)

# =============================================================================
# Keyword Classification Sets
# =============================================================================

# AI prompt indicators (case-insensitive matching in summary text)
_AI_PROMPT_INDICATORS: Final[frozenset[str]] = frozenset(
    {
        "\U0001f916",  # Robot emoji
        "prompt for ai",
        "ai-generated",
        "ai generated",
        "generated prompt",
        "ai summary",
        "ai prompt",
        "llm generated",
    }
)

# Non-AI indicators (take precedence for classification)
_ANALYSIS_CHAIN_INDICATORS: Final[frozenset[str]] = frozenset(
    {
        "analysis chain",
        "\U0001f9e9 analysis",  # Puzzle piece emoji
    }
)

_WALKTHROUGH_INDICATORS: Final[frozenset[str]] = frozenset(
    {
        "walkthrough",
        "code walkthrough",
    }
)

_RELATED_INDICATORS: Final[frozenset[str]] = frozenset(
    {
        "related issues",
        "related prs",
        "related pull requests",
    }
)


# =============================================================================
# Enums
# =============================================================================


class DetailsBlockType(Enum):
    """Classification of <details> block types in CodeRabbit comments.

    CodeRabbit uses various collapsible sections for different purposes.
    This enum classifies them based on their summary content.

    Attributes:
        AI_PROMPT: Contains AI-generated instructions for automated tools
        ANALYSIS_CHAIN: Contains analysis reasoning chain
        WALKTHROUGH: Contains code walkthrough explanation
        RELATED_ISSUES: Contains related issues/PRs references
        OTHER: Unclassified collapsible section
        UNKNOWN: Missing or unparseable summary tag
    """

    AI_PROMPT = auto()
    ANALYSIS_CHAIN = auto()
    WALKTHROUGH = auto()
    RELATED_ISSUES = auto()
    OTHER = auto()
    UNKNOWN = auto()


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class DiffBlock:
    """Extracted diff block from a CodeRabbit comment.

    Attributes:
        content: Raw diff content without fence markers
        position: Character position in original comment
        has_hunk_header: True if content contains @@ hunk header
    """

    content: str
    position: int
    has_hunk_header: bool


@dataclass(frozen=True, slots=True)
class SuggestionBlock:
    """Extracted suggestion block from a CodeRabbit comment.

    Attributes:
        content: Raw suggestion content without fence markers
        position: Character position in original comment
        option_label: Preceding **Option X:** label if present, else None
    """

    content: str
    position: int
    option_label: str | None


@dataclass(frozen=True, slots=True)
class AIPromptBlock:
    """AI prompt block extracted from <details> section.

    These blocks contain explicit instructions from CodeRabbit designed
    for automated tools. They should be prioritized over natural language.

    Attributes:
        summary: Content of the <summary> tag
        content: Cleaned prompt content (code fences stripped)
        position: Character position in original comment
    """

    summary: str
    content: str
    position: int


@dataclass(frozen=True, slots=True)
class DetailsBlock:
    """Generic <details> block with type classification.

    Attributes:
        summary: Content of the <summary> tag
        content: Content inside details (after summary)
        block_type: Classification based on summary keywords
        position: Character position in original comment
        raw: Full raw block including HTML tags
    """

    summary: str
    content: str
    block_type: DetailsBlockType
    position: int
    raw: str


@dataclass(frozen=True, slots=True)
class CommentSources:
    """All detected source blocks from a CodeRabbit comment.

    This is the main return type of extract_comment_sources(). It contains
    tuples of all detected blocks, with convenience properties for common
    checks.

    Attributes:
        diff_blocks: All diff blocks found
        suggestion_blocks: All suggestion blocks found
        ai_prompt_blocks: AI prompt blocks (filtered from details_blocks)
        details_blocks: All <details> blocks with classification
        html_comments_stripped: Count of HTML comments removed during processing

    Example:
        >>> sources = extract_comment_sources(body)
        >>> if sources.has_ai_prompt:
        ...     prompt = sources.ai_prompt_blocks[0].content
        ...     print(f"AI instructions: {prompt}")
    """

    diff_blocks: tuple[DiffBlock, ...]
    suggestion_blocks: tuple[SuggestionBlock, ...]
    ai_prompt_blocks: tuple[AIPromptBlock, ...]
    details_blocks: tuple[DetailsBlock, ...]
    html_comments_stripped: int

    @property
    def has_any_blocks(self) -> bool:
        """Return True if any structured blocks were detected."""
        return bool(self.diff_blocks or self.suggestion_blocks or self.ai_prompt_blocks)

    @property
    def has_ai_prompt(self) -> bool:
        """Return True if at least one AI prompt block exists."""
        return bool(self.ai_prompt_blocks)

    @property
    def has_diff(self) -> bool:
        """Return True if at least one diff block exists."""
        return bool(self.diff_blocks)

    @property
    def has_suggestion(self) -> bool:
        """Return True if at least one suggestion block exists."""
        return bool(self.suggestion_blocks)

    @property
    def block_count(self) -> int:
        """Total number of structured blocks (diff + suggestion + AI prompt)."""
        return len(self.diff_blocks) + len(self.suggestion_blocks) + len(self.ai_prompt_blocks)


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _empty_sources() -> CommentSources:
    """Return an empty CommentSources instance."""
    return CommentSources(
        diff_blocks=(),
        suggestion_blocks=(),
        ai_prompt_blocks=(),
        details_blocks=(),
        html_comments_stripped=0,
    )


def _strip_html_comments(text: str) -> tuple[str, int]:
    """Strip HTML comments from text.

    This handles fingerprinting markers like:
    <!-- fingerprinting:phantom:medusa:ocelot -->

    Args:
        text: Input text potentially containing HTML comments.

    Returns:
        Tuple of (cleaned_text, comment_count).
    """
    count = 0

    def counter(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return ""

    cleaned = _HTML_COMMENT_PATTERN.sub(counter, text)
    return cleaned, count


def _classify_details_block(summary: str) -> DetailsBlockType:
    """Classify a details block based on its summary content.

    Uses keyword matching with priority:
    1. Non-AI indicators (specific known types)
    2. AI indicators
    3. Fallback to OTHER or UNKNOWN

    Args:
        summary: The content of the <summary> tag.

    Returns:
        DetailsBlockType classification.
    """
    if not summary:
        return DetailsBlockType.UNKNOWN

    summary_lower = summary.lower()

    # Check non-AI indicators first (higher priority)
    if any(ind in summary_lower for ind in _ANALYSIS_CHAIN_INDICATORS):
        return DetailsBlockType.ANALYSIS_CHAIN
    if any(ind in summary_lower for ind in _WALKTHROUGH_INDICATORS):
        return DetailsBlockType.WALKTHROUGH
    if any(ind in summary_lower for ind in _RELATED_INDICATORS):
        return DetailsBlockType.RELATED_ISSUES

    # Check AI indicators
    if any(ind in summary_lower for ind in _AI_PROMPT_INDICATORS):
        return DetailsBlockType.AI_PROMPT

    return DetailsBlockType.OTHER


def _clean_ai_prompt(content: str) -> str:
    """Clean AI prompt content extracted from details block.

    Real CodeRabbit structure wraps content in triple backticks:
    <details><summary>... Prompt...</summary>
    ```
    actual content here
    ```
    </details>

    Args:
        content: Raw content from inside the details block (after summary).

    Returns:
        Cleaned prompt text with code fences stripped.
    """
    content = content.strip()
    # Remove code fences at boundaries
    content = _CODE_FENCE_PATTERN.sub("", content)
    return content.strip()


def _extract_option_label(preceding_text: str) -> str | None:
    """Extract **Option X:** style label from text preceding a suggestion block.

    Args:
        preceding_text: Text before the suggestion block (up to 200 chars).

    Returns:
        The option label if found, else None.
    """
    matches = list(_OPTION_LABEL_PATTERN.finditer(preceding_text))
    if matches:
        last_match = matches[-1]
        label = last_match.group(1).strip().rstrip(":")
        return label if label else None
    return None


def _extract_diff_blocks(text: str) -> tuple[DiffBlock, ...]:
    """Extract all diff blocks from text.

    Args:
        text: Comment body text.

    Returns:
        Tuple of DiffBlock objects.
    """
    blocks: list[DiffBlock] = []

    for match in _DIFF_PATTERN.finditer(text):
        content = match.group(1).rstrip("\n")
        position = match.start()
        has_hunk = bool(_HUNK_HEADER_PATTERN.search(content))

        blocks.append(
            DiffBlock(
                content=content,
                position=position,
                has_hunk_header=has_hunk,
            )
        )

    return tuple(blocks)


def _extract_suggestion_blocks(text: str) -> tuple[SuggestionBlock, ...]:
    """Extract all suggestion blocks with optional preceding labels.

    Args:
        text: Comment body text.

    Returns:
        Tuple of SuggestionBlock objects.
    """
    blocks: list[SuggestionBlock] = []

    for match in _SUGGESTION_PATTERN.finditer(text):
        content = match.group(1).rstrip("\n")
        position = match.start()

        # Look for preceding **Option X:** label (within 200 chars)
        preceding_text = text[max(0, position - 200) : position]
        option_label = _extract_option_label(preceding_text)

        blocks.append(
            SuggestionBlock(
                content=content,
                position=position,
                option_label=option_label,
            )
        )

    return tuple(blocks)


def _extract_details_blocks(text: str) -> tuple[DetailsBlock, ...]:
    """Extract all <details> blocks with type classification.

    Args:
        text: Comment body text.

    Returns:
        Tuple of DetailsBlock objects with classification.
    """
    blocks: list[DetailsBlock] = []

    for match in _DETAILS_PATTERN.finditer(text):
        raw = match.group(0)
        inner_content = match.group(1)
        position = match.start()

        # Extract summary
        summary_match = _SUMMARY_PATTERN.search(inner_content)
        if summary_match:
            summary = summary_match.group(1).strip()
            # Content is everything after the summary tag
            summary_end = summary_match.end()
            content = inner_content[summary_end:].strip()
        else:
            summary = ""
            content = inner_content.strip()

        # Classify block type
        block_type = _classify_details_block(summary)

        # Clean AI prompt content if applicable
        if block_type == DetailsBlockType.AI_PROMPT:
            content = _clean_ai_prompt(content)

        blocks.append(
            DetailsBlock(
                summary=summary,
                content=content,
                block_type=block_type,
                position=position,
                raw=raw,
            )
        )

    return tuple(blocks)


# =============================================================================
# Public API
# =============================================================================


def extract_comment_sources(comment_body: str | None) -> CommentSources:
    """Extract all source blocks from a CodeRabbit comment.

    This function parses a raw comment body and extracts structured blocks:
    - Diff blocks (```diff ... ```)
    - Suggestion blocks (```suggestion ... ```)
    - AI prompt blocks (<details> with AI-related summary)
    - All <details> blocks with type classification

    Args:
        comment_body: Raw comment text, may be None or empty.

    Returns:
        CommentSources with all extracted blocks. Returns empty CommentSources
        if input is None, empty, or contains no extractable blocks.

    Note:
        - HTML comments (<!-- ... -->) are stripped before processing
        - Comments larger than 100KB are truncated with a warning
        - Nested <details> blocks are NOT supported: the non-greedy regex
          pairs each opening <details> tag with the FIRST encountered
          </details> tag, potentially breaking nested structures (e.g.,
          "<details>outer<details>inner</details>more</details>" captures
          only "outer<details>inner", leaving "more</details>" unparsed)

    Example:
        >>> sources = extract_comment_sources('''
        ... Apply this fix:
        ... ```suggestion
        ... def better_func():
        ...     pass
        ... ```
        ... ''')
        >>> len(sources.suggestion_blocks)
        1
        >>> sources.has_suggestion
        True
    """
    # Edge case: None or empty input
    if not comment_body:
        return _empty_sources()

    # Edge case: whitespace only
    if not comment_body.strip():
        return _empty_sources()

    # Edge case: very large comment
    original_length = len(comment_body)
    if original_length > _MAX_COMMENT_SIZE:
        logger.warning(
            "Comment body exceeds max size (%d > %d bytes), truncating",
            original_length,
            _MAX_COMMENT_SIZE,
        )
        comment_body = comment_body[:_MAX_COMMENT_SIZE]

    # Strip HTML comments first (they may contain fingerprinting markers)
    comment_body, html_comments_count = _strip_html_comments(comment_body)

    # Extract each block type
    diff_blocks = _extract_diff_blocks(comment_body)
    suggestion_blocks = _extract_suggestion_blocks(comment_body)
    details_blocks = _extract_details_blocks(comment_body)

    # Filter AI prompt blocks from details
    ai_prompt_blocks = tuple(
        AIPromptBlock(
            summary=db.summary,
            content=db.content,
            position=db.position,
        )
        for db in details_blocks
        if db.block_type == DetailsBlockType.AI_PROMPT
    )

    return CommentSources(
        diff_blocks=diff_blocks,
        suggestion_blocks=suggestion_blocks,
        ai_prompt_blocks=ai_prompt_blocks,
        details_blocks=details_blocks,
        html_comments_stripped=html_comments_count,
    )
