# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Indentation utilities for code change application.

This module provides functions to detect and restore indentation when applying
code changes extracted by the LLM parser. LLMs sometimes strip leading whitespace
from extracted code, causing IndentationError when applied to Python files.

The main function `restore_indentation()` ensures replacement code matches the
indentation of the original code being replaced.
"""

from __future__ import annotations


def get_leading_whitespace(line: str) -> str:
    r"""Extract leading whitespace from a line.

    Args:
        line: A string that may have leading whitespace.

    Returns:
        The leading whitespace characters (spaces, tabs) from the line.
        Returns empty string if line has no leading whitespace.

    Example:
        >>> get_leading_whitespace("    def foo():")
        '    '
        >>> get_leading_whitespace("no indent")
        ''
        >>> get_leading_whitespace("\t\tindented")
        '\t\t'
    """
    return line[: len(line) - len(line.lstrip())]


def restore_indentation(
    original_lines: list[str],
    replacement_lines: list[str],
    start_idx: int,
) -> list[str]:
    """Restore indentation to replacement lines based on original file context.

    When LLM parsers extract code changes, they sometimes strip leading indentation.
    This function detects the expected indentation from the original file and
    applies it to the replacement lines while preserving relative indentation
    within multi-line blocks.

    Args:
        original_lines: List of lines from the original file (without newlines).
        replacement_lines: List of replacement lines to apply (without newlines).
        start_idx: Zero-based index where replacement starts in original_lines.

    Returns:
        List of replacement lines with corrected indentation.
        Returns replacement_lines unchanged if:
        - replacement is empty (deletion)
        - replacement already has correct indentation
        - no indentation context can be determined

    Algorithm:
        1. Find expected indentation from the line being replaced (or previous line)
        2. Check if replacement already has correct indentation
        3. If not, strip all leading whitespace and re-apply expected indent
        4. Preserve relative indentation within multi-line replacements

    Example:
        >>> original = ["def foo():", "    '''Docstring.'''", "    pass"]
        >>> replacement = ["'''New docstring.'''"]  # Missing indentation
        >>> restore_indentation(original, replacement, 1)
        ["    '''New docstring.'''"]
    """
    # Handle empty replacement (deletions)
    if not replacement_lines or all(not line for line in replacement_lines):
        return replacement_lines

    # Find first non-empty replacement line to analyze
    first_content_line = next((line for line in replacement_lines if line.strip()), None)
    if first_content_line is None:
        return replacement_lines

    # Determine expected indentation from original file context
    expected_indent = _get_expected_indentation(original_lines, start_idx)

    # Check if first non-empty replacement line already has correct indentation
    # Must be exact match, not just startswith (8 spaces != 4 spaces expected)
    replacement_indent = get_leading_whitespace(first_content_line)
    if replacement_indent == expected_indent:
        return replacement_lines  # Already correctly indented

    # Calculate base indent of replacement (to preserve relative indentation)
    replacement_base = replacement_indent

    # Re-indent all lines while preserving relative indentation
    result = []
    for line in replacement_lines:
        if not line.strip():
            # Preserve empty/whitespace-only lines as-is
            result.append(line)
        else:
            # Calculate relative indentation beyond the base
            current_indent = get_leading_whitespace(line)
            if current_indent.startswith(replacement_base):
                relative_indent = current_indent[len(replacement_base) :]
            else:
                # Line has less indent than base (unusual), use no relative
                relative_indent = ""

            # Apply expected indent + relative indent + content
            stripped = line.lstrip()
            result.append(expected_indent + relative_indent + stripped)

    return result


def _get_expected_indentation(original_lines: list[str], start_idx: int) -> str:
    """Determine expected indentation from original file context.

    Priority:
    1. Use the line being replaced (at start_idx) if it has content
    2. Look backwards for the nearest non-empty line
    3. Return empty string if no context available

    Args:
        original_lines: List of lines from the original file.
        start_idx: Zero-based index where replacement starts.

    Returns:
        The expected leading whitespace string.
    """
    # Try the line being replaced first
    if start_idx < len(original_lines):
        line = original_lines[start_idx]
        if line.strip():
            return get_leading_whitespace(line)

    # Look backwards for nearest non-empty line
    for i in range(start_idx - 1, -1, -1):
        if i < len(original_lines) and original_lines[i].strip():
            return get_leading_whitespace(original_lines[i])

    # No context found - return empty
    return ""
