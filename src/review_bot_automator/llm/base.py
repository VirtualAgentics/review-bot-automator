# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Base data structures and interfaces for LLM-based parsing.

This module provides foundational data structures and abstract interfaces for
LLM integration. It defines:
- ParsedChange: Intermediate representation of LLM parser output
- LLMParser: Abstract base class for all parser implementations

Phase 1: Provider and parser implementations build on these foundations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedChange:
    r"""Intermediate representation of a change parsed by an LLM.

    This dataclass represents the output from an LLM parser before conversion
    to the standard Change model. It includes additional metadata specific to
    LLM parsing like confidence scores and rationale.

    Args:
        file_path: Path to the file to be modified
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (inclusive)
        new_content: The new content to apply
        change_type: Type of change ("addition", "modification", "deletion")
        confidence: LLM confidence score (0.0-1.0)
        rationale: Explanation of why this change was suggested
        risk_level: Risk assessment ("low", "medium", "high")

    Example:
        >>> change = ParsedChange(
        ...     file_path="src/example.py",
        ...     start_line=10,
        ...     end_line=12,
        ...     new_content="def new_function():\n    pass",
        ...     change_type="modification",
        ...     confidence=0.95,
        ...     rationale="Replace deprecated function with new API",
        ...     risk_level="low"
        ... )
        >>> change.confidence
        0.95
    """

    file_path: str
    start_line: int
    end_line: int
    new_content: str
    change_type: str
    confidence: float
    rationale: str
    risk_level: str = "low"

    def __post_init__(self) -> None:
        """Validate ParsedChange fields after initialization.

        Raises:
            ValueError: If any field has an invalid value
        """
        if self.start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {self.start_line}")
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.change_type not in ("addition", "modification", "deletion"):
            raise ValueError(
                f"change_type must be 'addition', 'modification', or 'deletion', "
                f"got '{self.change_type}'"
            )
        if self.risk_level not in ("low", "medium", "high"):
            raise ValueError(
                f"risk_level must be 'low', 'medium', or 'high', got '{self.risk_level}'"
            )


class LLMParser(ABC):
    """Abstract base class for LLM-powered comment parsers.

    This ABC defines the interface that all LLM parser implementations must follow.
    Parsers are responsible for:
    - Taking raw GitHub comment text as input
    - Using an LLM provider to extract code changes
    - Returning structured ParsedChange objects
    - Handling errors gracefully with optional fallback

    Subclasses must implement the parse_comment() method. Different parser
    implementations may use different prompting strategies, providers, or
    processing logic while maintaining a consistent interface.

    Examples:
        >>> class MyParser(LLMParser):
        ...     def parse_comment(self, body, file_path=None, line_number=None):
        ...         # Implementation here
        ...         return [ParsedChange(...)]
        >>> parser = MyParser()
        >>> changes = parser.parse_comment("Fix the bug at line 10")
    """

    @abstractmethod
    def parse_comment(
        self,
        comment_body: str,
        file_path: str | None = None,
        # TODO(#294): Remove line_number once all callers migrate to start_line/end_line
        line_number: int | None = None,  # Deprecated, use end_line instead
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[ParsedChange]:
        """Parse a GitHub comment to extract code changes.

        This method processes a raw comment and returns zero or more ParsedChange
        objects representing the suggested modifications. The parser should handle
        multiple comment formats:

        - Diff blocks: @@ -1,3 +1,3 @@ format
        - Suggestion blocks: ```suggestion code here```
        - Natural language: "change X to Y on line N"
        - Multi-option suggestions: **Option 1:** ... **Option 2:** ...

        Args:
            comment_body: Raw comment text from GitHub (markdown format)
            file_path: Optional file path for context (helps with ambiguous suggestions)
            line_number: Deprecated - use end_line instead. Will be removed in future version.
                See https://github.com/VirtualAgentics/review-bot-automator/issues/294
            start_line: Start of the diff range (from GitHub start_line field)
            end_line: End of the diff range (from GitHub line field)

        Returns:
            List of ParsedChange objects. Empty list if:
            - No changes detected in comment
            - Parsing failed and fallback is enabled
            - Comment contains only discussion/questions

        Raises:
            RuntimeError: If parsing fails and fallback is disabled
            ValueError: If comment_body is None or empty

        Note:
            Implementations should:
            - Handle LLM provider errors gracefully
            - Validate ParsedChange objects before returning
            - Filter low-confidence changes if configured
            - Log parsing failures for debugging
            - Support fallback to regex parsing when enabled
        """
        pass
