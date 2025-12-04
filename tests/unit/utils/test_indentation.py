# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Tests for indentation utility functions.

These tests verify that the indentation restoration logic correctly handles
various edge cases when LLM parsers strip leading whitespace from extracted code.
"""

import pytest

from review_bot_automator.utils.indentation import (
    get_leading_whitespace,
    restore_indentation,
)


class TestGetLeadingWhitespace:
    """Tests for get_leading_whitespace function."""

    @pytest.mark.parametrize(
        ("input_line", "expected"),
        [
            ("    def foo():", "    "),  # spaces
            ("\t\tdef foo():", "\t\t"),  # tabs
            ("  \t  content", "  \t  "),  # mixed whitespace
            ("no indent", ""),  # no indent
            ("", ""),  # empty line
            ("    ", "    "),  # whitespace only
        ],
    )
    def test_get_leading_whitespace(self, input_line: str, expected: str) -> None:
        """Extract leading whitespace from various line formats."""
        assert get_leading_whitespace(input_line) == expected


class TestRestoreIndentation:
    """Tests for restore_indentation function."""

    def test_single_line_missing_indent_restored(self) -> None:
        """Single line with missing indentation is restored."""
        original = ["def foo():", "    '''Docstring.'''", "    pass"]
        replacement = ["'''New docstring.'''"]  # Missing 4-space indent
        result = restore_indentation(original, replacement, 1)
        assert result == ["    '''New docstring.'''"]

    def test_single_line_correct_indent_unchanged(self) -> None:
        """Single line with correct indentation is unchanged."""
        original = ["def foo():", "    '''Docstring.'''", "    pass"]
        replacement = ["    '''New docstring.'''"]  # Already has 4-space indent
        result = restore_indentation(original, replacement, 1)
        assert result == ["    '''New docstring.'''"]

    def test_multiline_preserves_relative_indent(self) -> None:
        """Multi-line block preserves relative indentation."""
        original = [
            "class Foo:",
            "    def method(self):",
            "        if True:",
            "            pass",
        ]
        # Replacement has wrong base indent but correct relative structure
        replacement = [
            "def method(self):",
            "    if True:",
            "        return 42",
        ]
        result = restore_indentation(original, replacement, 1)
        assert result == [
            "    def method(self):",
            "        if True:",
            "            return 42",
        ]

    def test_empty_replacement_unchanged(self) -> None:
        """Empty replacement (deletion) is returned unchanged."""
        original = ["line1", "line2", "line3"]
        replacement = [""]
        result = restore_indentation(original, replacement, 1)
        assert result == [""]

    def test_all_empty_lines_unchanged(self) -> None:
        """Replacement with only empty lines is unchanged."""
        original = ["line1", "line2", "line3"]
        replacement = ["", ""]
        result = restore_indentation(original, replacement, 1)
        assert result == ["", ""]

    def test_first_line_of_file_no_indent(self) -> None:
        """First line of file with no prior context gets no indentation."""
        original = ["# Comment", "code"]
        replacement = ["# New comment"]  # No indent needed at position 0
        result = restore_indentation(original, replacement, 0)
        assert result == ["# New comment"]

    def test_uses_previous_line_when_target_empty(self) -> None:
        """Uses previous non-empty line when target line is empty."""
        original = ["def foo():", "    pass", "", "    more_code"]
        replacement = ["new_line"]
        # Position 2 is empty, should look back to position 1 for indent
        result = restore_indentation(original, replacement, 2)
        assert result == ["    new_line"]

    def test_tabs_preserved(self) -> None:
        """Tab-based indentation is preserved."""
        original = ["def foo():", "\treturn 1"]
        replacement = ["return 42"]  # Missing tab
        result = restore_indentation(original, replacement, 1)
        assert result == ["\treturn 42"]

    def test_eight_space_indent_restored(self) -> None:
        """Eight-space indentation (nested block) is restored."""
        original = [
            "class Test:",
            "    def test_method(self):",
            "        '''Docstring.'''",
            "        pass",
        ]
        replacement = ["'''New docstring.'''"]  # Missing 8 spaces
        result = restore_indentation(original, replacement, 2)
        assert result == ["        '''New docstring.'''"]

    def test_multiline_with_empty_lines_preserved(self) -> None:
        """Empty lines within multi-line replacement are preserved."""
        original = ["def foo():", "    code", "    more"]
        replacement = [
            "first_line",
            "",
            "third_line",
        ]
        result = restore_indentation(original, replacement, 1)
        assert result == [
            "    first_line",
            "",
            "    third_line",
        ]

    def test_no_context_available(self) -> None:
        """When no context is available, no indentation is added."""
        original: list[str] = []
        replacement = ["code"]
        result = restore_indentation(original, replacement, 0)
        assert result == ["code"]

    def test_start_idx_beyond_file(self) -> None:
        """Handle start_idx beyond file length gracefully."""
        original = ["line1", "line2"]
        replacement = ["new_line"]
        # start_idx=5 is beyond file, should look back for context
        result = restore_indentation(original, replacement, 5)
        # Should use last non-empty line's indent (line2 has no indent)
        assert result == ["new_line"]

    def test_deeply_nested_block(self) -> None:
        """Deeply nested code block gets correct indentation."""
        original = [
            "class Outer:",
            "    class Inner:",
            "        def method(self):",
            "            if condition:",
            "                return value",
        ]
        replacement = ["return new_value"]  # Missing 16 spaces
        result = restore_indentation(original, replacement, 4)
        assert result == ["                return new_value"]

    def test_replacement_with_extra_indent_normalized(self) -> None:
        """Replacement with wrong base indent is normalized."""
        original = ["def foo():", "    return 1"]
        replacement = ["        return 42"]  # Has 8 spaces, should have 4
        result = restore_indentation(original, replacement, 1)
        # This should detect the expected indent (4 spaces) and not match,
        # then re-indent
        assert result == ["    return 42"]

    def test_issue_287_example(self) -> None:
        """Test the exact example from Issue #287."""
        original = [
            "class TestRestoreUnsafe:",
            "    def test_restore_unsafe_segment_fails(self, tmp_path: Path) -> None:",
            '        """Test that paths with unsafe segments fail validation."""',
            "        pass",
        ]
        # LLM returned docstring without indentation
        replacement = ['"""Test that nonexistent files fail restoration."""']
        result = restore_indentation(original, replacement, 2)
        assert result == ['        """Test that nonexistent files fail restoration."""']


class TestRestoreIndentationIntegration:
    """Integration tests simulating real resolver usage."""

    def test_python_method_body_replacement(self) -> None:
        """Simulate replacing a method body in Python code."""
        original_file = [
            "class Calculator:",
            "    def add(self, a, b):",
            "        '''Add two numbers.'''",
            "        return a + b",
            "",
            "    def subtract(self, a, b):",
            "        return a - b",
        ]

        # LLM extracted replacement without indentation
        llm_replacement = [
            "'''Add two numbers and log the result.'''",
            "result = a + b",
            "print(f'Adding {a} + {b} = {result}')",
            "return result",
        ]

        result = restore_indentation(original_file, llm_replacement, 2)

        assert result == [
            "        '''Add two numbers and log the result.'''",
            "        result = a + b",
            "        print(f'Adding {a} + {b} = {result}')",
            "        return result",
        ]

    def test_yaml_indentation_preserved(self) -> None:
        """YAML file with space-based indentation."""
        original = [
            "config:",
            "  database:",
            "    host: localhost",
            "    port: 5432",
        ]
        replacement = ["host: 127.0.0.1"]  # Missing 4-space indent
        result = restore_indentation(original, replacement, 2)
        assert result == ["    host: 127.0.0.1"]


class TestRestoreIndentationEdgeCases:
    """Edge case tests for complete coverage."""

    def test_whitespace_only_lines_unchanged(self) -> None:
        """When all replacement lines are whitespace-only, return unchanged.

        Tests the behavior when no actual content exists to indent - lines
        contain only spaces/tabs but no code or text. In this case, the
        function should return the replacement unchanged since there's
        nothing meaningful to re-indent.
        """
        original = ["def foo():", "    pass"]
        # Lines with only whitespace - no actual content
        replacement = ["   ", "\t", "  \t  "]
        result = restore_indentation(original, replacement, 1)
        # Should return unchanged since there's no content to indent
        assert result == ["   ", "\t", "  \t  "]

    def test_multiline_dedented_with_wrong_base(self) -> None:
        """When a replacement line is less indented than the detected base, use no relative indent.

        Tests the fallback behavior when a line in the replacement has less
        indentation than the first content line (the detected base). Since the
        line's indent doesn't start with the base indent, relative indentation
        cannot be computed, so the line receives only the expected indent with
        no additional relative offset.
        """
        original = ["class Foo:", "    def method(self):", "        pass"]
        # First line has 8 spaces (wrong), second line has only 2 spaces
        # Expected indent is 4 spaces
        replacement = [
            "        def method(self):",  # 8 spaces (base, wrong)
            "  if True:",  # Only 2 spaces - less than 8-space base
            "        return 42",  # Back to 8 spaces
        ]
        result = restore_indentation(original, replacement, 1)
        # First line: expected 4 spaces, gets re-indented
        # Second line: 2 spaces doesn't start with 8-space base, no relative indent
        # Third line: 8 spaces matches base, relative indent is empty
        assert result == [
            "    def method(self):",
            "    if True:",
            "    return 42",
        ]
