"""Integration tests for ConflictResolver with LLM parsing.

This module tests the integration between ConflictResolver and UniversalLLMParser:
- LLM parsing with automatic regex fallback
- ParsedChange to Change conversion
- LLM metadata propagation
- Backward compatibility (resolver without LLM parser)
"""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    import pytest

from review_bot_automator.core.models import FileType
from review_bot_automator.core.resolver import ConflictResolver
from review_bot_automator.llm.base import LLMParser, ParsedChange
from review_bot_automator.llm.parallel_parser import CommentInput, ParallelLLMParser


def _parsed_change(path: str = "test.py", line: int = 10) -> ParsedChange:
    """Helper to create ParsedChange instances."""
    return ParsedChange(
        file_path=path,
        start_line=line,
        end_line=line,
        new_content="patched_code",
        change_type="modification",
        confidence=0.9,
        rationale="auto",
        risk_level="low",
    )


class TestResolverWithoutLLM:
    """Test resolver backward compatibility without LLM parser."""

    def test_resolver_without_llm_parser_uses_regex_only(self, tmp_path: Path) -> None:
        """Test that resolver works without LLM parser (backward compatible)."""
        resolver = ConflictResolver(workspace_root=tmp_path)

        assert resolver.llm_parser is None

        # Test with suggestion block (regex should work)
        comments = [
            {
                "path": "test.py",
                "line": 10,
                "body": "```suggestion\nprint('fixed')\n```",
                "html_url": "https://github.com/test",
                "user": {"login": "testuser"},
            }
        ]

        changes = resolver.extract_changes_from_comments(comments)

        assert len(changes) == 1
        assert changes[0].parsing_method == "regex"
        assert changes[0].llm_confidence is None


class TestResolverWithLLM:
    """Test resolver with LLM parser integration."""

    def test_resolver_accepts_llm_parser(self, tmp_path: Path) -> None:
        """Test that resolver accepts and stores LLM parser."""
        mock_parser = MagicMock(spec=LLMParser)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        assert resolver.llm_parser is mock_parser

    def test_llm_parsing_with_successful_extraction(self, tmp_path: Path) -> None:
        """Test LLM parsing extracts changes successfully."""
        mock_parser = MagicMock(spec=LLMParser)
        mock_parser.parse_comment.return_value = [
            ParsedChange(
                file_path="src/test.py",
                start_line=10,
                end_line=12,
                new_content="def fixed_function():\n    return True",
                change_type="modification",
                confidence=0.95,
                rationale="Fixed the function logic",
                risk_level="low",
            )
        ]

        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        comments = [
            {
                "path": "src/test.py",
                "line": 10,
                "body": "Please fix this function to return True instead of False",
                "html_url": "https://github.com/test/pr/1",
                "user": {"login": "reviewer"},
            }
        ]

        changes = resolver.extract_changes_from_comments(comments)

        assert len(changes) == 1
        change = changes[0]

        # Verify basic fields
        assert change.path == "src/test.py"
        assert change.start_line == 10
        assert change.end_line == 12
        assert "def fixed_function()" in change.content

        # Verify LLM metadata
        assert change.parsing_method == "llm"
        assert change.llm_confidence == 0.95
        assert change.change_rationale == "Fixed the function logic"
        assert change.risk_level == "low"

        # Verify comment metadata
        assert change.metadata["author"] == "reviewer"
        assert change.metadata["source"] == "llm_parsed"

        # Verify computed fields
        assert change.file_type == FileType.PYTHON
        assert change.fingerprint is not None

    def test_llm_fallback_to_regex_on_no_changes(self, tmp_path: Path) -> None:
        """Test automatic fallback to regex when LLM returns no changes."""
        mock_parser = MagicMock(spec=LLMParser)
        mock_parser.parse_comment.return_value = []  # LLM finds nothing

        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        comments = [
            {
                "path": "test.py",
                "line": 5,
                "body": "```suggestion\nprint('regex fallback')\n```",
                "html_url": "https://github.com/test",
                "user": {"login": "testuser"},
            }
        ]

        changes = resolver.extract_changes_from_comments(comments)

        # Should fall back to regex and find the suggestion block
        assert len(changes) == 1
        assert changes[0].parsing_method == "regex"
        assert changes[0].content == "print('regex fallback')"

    def test_llm_fallback_on_exception(self, tmp_path: Path) -> None:
        """Test automatic fallback to regex when LLM raises exception."""
        mock_parser = MagicMock(spec=LLMParser)
        mock_parser.parse_comment.side_effect = RuntimeError("LLM API error")

        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        comments = [
            {
                "path": "test.py",
                "line": 5,
                "body": "```suggestion\nprint('error fallback')\n```",
                "html_url": "https://github.com/test",
                "user": {"login": "testuser"},
            }
        ]

        changes = resolver.extract_changes_from_comments(comments)

        # Should fall back to regex despite LLM error
        assert len(changes) == 1
        assert changes[0].parsing_method == "regex"

    def test_llm_provider_name_extraction(self, tmp_path: Path) -> None:
        """Test extraction of LLM provider name from parser."""
        mock_parser = MagicMock(spec=LLMParser)
        mock_provider = MagicMock()
        mock_provider.model = "gpt-4o-mini"
        mock_parser.provider = mock_provider

        mock_parser.parse_comment.return_value = [
            ParsedChange(
                file_path="test.py",
                start_line=1,
                end_line=1,
                new_content="# fixed",
                change_type="modification",
                confidence=0.8,
                rationale="Added comment",
                risk_level="low",
            )
        ]

        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        comments = [
            {
                "path": "test.py",
                "line": 1,
                "body": "Add a comment here",
                "html_url": "https://github.com/test",
                "user": {"login": "testuser"},
            }
        ]

        changes = resolver.extract_changes_from_comments(comments)

        assert len(changes) == 1
        assert changes[0].llm_provider == "gpt-4o-mini"

    def test_multiple_changes_from_single_comment(self, tmp_path: Path) -> None:
        """Test LLM parser returning multiple changes from one comment."""
        mock_parser = MagicMock(spec=LLMParser)
        mock_parser.parse_comment.return_value = [
            ParsedChange(
                file_path="test.py",
                start_line=10,
                end_line=11,
                new_content="# First change",
                change_type="modification",
                confidence=0.9,
                rationale="Change 1",
                risk_level="low",
            ),
            ParsedChange(
                file_path="test.py",
                start_line=20,
                end_line=21,
                new_content="# Second change",
                change_type="modification",
                confidence=0.85,
                rationale="Change 2",
                risk_level="medium",
            ),
        ]

        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        comments = [
            {
                "path": "test.py",
                "line": 10,
                "body": "Please make these two changes",
                "html_url": "https://github.com/test",
                "user": {"login": "reviewer"},
            }
        ]

        changes = resolver.extract_changes_from_comments(comments)

        assert len(changes) == 2
        assert all(c.parsing_method == "llm" for c in changes)
        assert changes[0].start_line == 10
        assert changes[1].start_line == 20

    def test_mixed_llm_and_regex_changes(self, tmp_path: Path) -> None:
        """Test processing multiple comments with both LLM and regex parsing."""
        mock_parser = MagicMock(spec=LLMParser)

        def side_effect(comment_body: str, **kwargs: dict[str, object]) -> list[ParsedChange]:
            """Return changes only for first comment."""
            if "natural language" in comment_body:
                return [
                    ParsedChange(
                        file_path="test.py",
                        start_line=1,
                        end_line=1,
                        new_content="# LLM parsed",
                        change_type="modification",
                        confidence=0.9,
                        rationale="LLM understood",
                        risk_level="low",
                    )
                ]
            return []

        mock_parser.parse_comment.side_effect = side_effect

        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        comments = [
            {
                "path": "test.py",
                "line": 1,
                "body": "Fix this with natural language",
                "html_url": "https://github.com/test/1",
                "user": {"login": "user1"},
            },
            {
                "path": "test.py",
                "line": 10,
                "body": "```suggestion\nprint('regex')\n```",
                "html_url": "https://github.com/test/2",
                "user": {"login": "user2"},
            },
        ]

        changes = resolver.extract_changes_from_comments(comments)

        assert len(changes) == 2
        # First should be LLM-parsed
        llm_change = next(c for c in changes if c.parsing_method == "llm")
        assert llm_change.llm_confidence == 0.9

        # Second should be regex-parsed
        regex_change = next(c for c in changes if c.parsing_method == "regex")
        assert regex_change.content == "print('regex')"


class TestParsedChangeConversion:
    """Test ParsedChange to Change conversion logic."""

    def test_conversion_with_all_metadata(self, tmp_path: Path) -> None:
        """Test conversion includes all metadata fields."""
        parsed_change = ParsedChange(
            file_path="src/module.py",
            start_line=100,
            end_line=105,
            new_content="def new_impl():\n    pass",
            change_type="modification",
            confidence=0.92,
            rationale="Refactored for clarity",
            risk_level="medium",
        )

        comment = {
            "html_url": "https://github.com/repo/pr/123#comment-456",
            "user": {"login": "code_reviewer"},
        }

        resolver = ConflictResolver(workspace_root=tmp_path)
        change = resolver._convert_parsed_change_to_change(parsed_change, comment)

        # Verify all fields transferred
        assert change.path == "src/module.py"
        assert change.start_line == 100
        assert change.end_line == 105
        assert change.content == "def new_impl():\n    pass"
        assert change.llm_confidence == 0.92
        assert change.change_rationale == "Refactored for clarity"
        assert change.risk_level == "medium"
        assert change.parsing_method == "llm"

        # Verify metadata dict (LLM metadata stored in direct attributes, not metadata dict)
        assert change.metadata["author"] == "code_reviewer"
        assert change.metadata["url"] == "https://github.com/repo/pr/123#comment-456"
        assert change.metadata["source"] == "llm_parsed"

    def test_conversion_detects_file_type(self, tmp_path: Path) -> None:
        """Test conversion correctly detects file types."""
        test_cases = [
            ("file.py", FileType.PYTHON),
            ("file.ts", FileType.TYPESCRIPT),
            ("file.json", FileType.JSON),
            ("file.yaml", FileType.YAML),
            ("file.toml", FileType.TOML),
            ("file.txt", FileType.PLAINTEXT),
        ]

        resolver = ConflictResolver(workspace_root=tmp_path)

        for file_path, expected_type in test_cases:
            parsed_change = ParsedChange(
                file_path=file_path,
                start_line=1,
                end_line=1,
                new_content="test",
                change_type="modification",
                confidence=0.8,
                rationale="test",
                risk_level="low",
            )

            change = resolver._convert_parsed_change_to_change(parsed_change, {"user": {}})

            assert change.file_type == expected_type, f"Failed for {file_path}"

    def test_conversion_generates_fingerprint(self, tmp_path: Path) -> None:
        """Test conversion generates unique fingerprints."""
        resolver = ConflictResolver(workspace_root=tmp_path)

        parsed1 = ParsedChange(
            file_path="test.py",
            start_line=1,
            end_line=2,
            new_content="content1",
            change_type="modification",
            confidence=0.8,
            rationale="test",
            risk_level="low",
        )

        parsed2 = ParsedChange(
            file_path="test.py",
            start_line=1,
            end_line=2,
            new_content="content2",  # Different content
            change_type="modification",
            confidence=0.8,
            rationale="test",
            risk_level="low",
        )

        change1 = resolver._convert_parsed_change_to_change(parsed1, {"user": {}})
        change2 = resolver._convert_parsed_change_to_change(parsed2, {"user": {}})

        # Different content should produce different fingerprints
        assert change1.fingerprint != change2.fingerprint


class TestResolverParallelParsing:
    """Test ConflictResolver with ParallelLLMParser integration."""

    def test_parallel_parsing_with_valid_comments(self, tmp_path: Path) -> None:
        """Test parallel parsing extracts changes from multiple comments."""
        mock_provider = MagicMock()
        # Use proper JSON without literal newlines
        mock_provider.generate.return_value = (
            '[{"file_path": "test.py", "start_line": 10, "end_line": 12, '
            '"new_content": "def fixed():\\n    return True", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed function", "risk_level": "low"}]'
        )

        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        # Need 5+ comments to trigger parallel parsing
        comments = [
            {
                "path": "test.py",
                "line": 10 + i,
                "body": f"Fix this function {i}",
                "html_url": f"https://github.com/test/{i}",
                "user": {"login": f"user{i}"},
            }
            for i in range(5)
        ]

        changes = resolver.extract_changes_from_comments(comments, parallel_parsing=True)

        # Should parse all comments in parallel
        assert len(changes) == 5
        assert all(c.parsing_method == "llm" for c in changes)
        assert mock_provider.generate.call_count == 5

    def test_parallel_parsing_with_missing_fields(self, tmp_path: Path) -> None:
        """Test parallel parsing handles comments without required fields."""
        mock_provider = MagicMock()
        # Use proper JSON without literal newlines
        mock_provider.generate.return_value = (
            '[{"file_path": "test.py", "start_line": 10, "end_line": 10, '
            '"new_content": "fixed", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed", "risk_level": "low"}]'
        )

        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        # Need 5+ comments total, but some missing required fields
        comments = [
            {
                "path": "test.py",
                "line": 10 + i,
                "body": f"Fix this {i}",
                "html_url": f"https://github.com/test/{i}",
                "user": {"login": f"user{i}"},
            }
            for i in range(4)
        ] + [
            {
                # Missing 'line' field - should use regex fallback
                "path": "test2.py",
                "body": "```suggestion\nregex_parsed\n```",
                "html_url": "https://github.com/test/5",
                "user": {"login": "user5"},
            },
        ]

        changes = resolver.extract_changes_from_comments(comments, parallel_parsing=True)

        # First 4 parsed by LLM, last comment missing 'line' field so not parsed
        assert len(changes) == 4
        assert sum(1 for c in changes if c.parsing_method == "llm") == 4
        # Last comment missing 'line' field, so not eligible for regex parsing either
        # Only 4 LLM calls (for valid comments)
        assert mock_provider.generate.call_count == 4

    def test_parallel_parsing_no_valid_comments(self, tmp_path: Path) -> None:
        """Test parallel parsing falls back to sequential when no valid comments."""
        mock_provider = MagicMock()
        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        comments = [
            {
                # Missing 'line' field - not eligible for LLM parsing
                # Regex also needs 'line', so no changes will be extracted
                "path": "test.py",
                "body": "```suggestion\nregex_only\n```",
                "html_url": "https://github.com/test/1",
                "user": {"login": "user1"},
            },
        ]

        changes = resolver.extract_changes_from_comments(comments, parallel_parsing=True)

        # Should fall back to sequential, but no changes extracted
        # because comment is missing 'line' field (required for both LLM and regex)
        assert len(changes) == 0
        # No LLM calls since comment missing required fields
        assert mock_provider.generate.call_count == 0

    def test_parallel_parsing_with_max_workers_override(self, tmp_path: Path) -> None:
        """Test parallel parsing respects max_workers override."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = (
            '[{"file_path": "test.py", "start_line": 10, "end_line": 10, '
            '"new_content": "fixed", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed", "risk_level": "low"}]'
        )

        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=4)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        comments = [
            {
                "path": "test.py",
                "line": 10,
                "body": "Fix this",
                "html_url": "https://github.com/test/1",
                "user": {"login": "user1"},
            }
        ]

        # Override max_workers to 2
        changes = resolver.extract_changes_from_comments(
            comments, parallel_parsing=True, max_workers=2
        )

        assert len(changes) == 1
        # Verify max_workers was restored
        assert parallel_parser.max_workers == 4

    def test_parallel_parsing_with_llm_failure_fallback(self, tmp_path: Path) -> None:
        """Test parallel parsing falls back to regex when LLM returns empty results."""
        mock_provider = MagicMock()
        # Some comments succeed, some return empty
        # Use proper JSON strings without literal newlines
        llm_result1 = (
            '[{"file_path": "test.py", "start_line": 10, "end_line": 10, '
            '"new_content": "llm_parsed", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed", "risk_level": "low"}]'
        )
        llm_result2 = (
            '[{"file_path": "test.py", "start_line": 30, "end_line": 30, '
            '"new_content": "llm_parsed2", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed", "risk_level": "low"}]'
        )

        def mock_generate(prompt: str, max_tokens: int = 2000) -> str:
            match = re.search(r"regex_fallback_(\d+)", prompt)
            if match:
                return "[]"

            match = re.search(r"Fix this (\d+)", prompt)
            if not match:
                return "[]"

            idx = int(match.group(1))
            if idx == 0:
                return llm_result1
            if idx == 2:
                return llm_result2
            return "[]"

        mock_provider.generate.side_effect = mock_generate

        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        comments = [
            {
                "path": "test.py",
                "line": 10 + i * 10,
                "body": (
                    f"```suggestion\nregex_fallback_{i}\n```" if i in [1, 3, 4] else f"Fix this {i}"
                ),
                "html_url": f"https://github.com/test/{i}",
                "user": {"login": f"user{i}"},
            }
            for i in range(5)
        ]

        changes = resolver.extract_changes_from_comments(comments, parallel_parsing=True)

        # Some parsed by LLM, some fall back to regex
        assert len(changes) >= 2
        llm_changes = [c for c in changes if c.parsing_method == "llm"]
        regex_changes = [c for c in changes if c.parsing_method == "regex"]
        assert len(llm_changes) == 2  # Two successful LLM parses
        assert len(regex_changes) >= 3  # Three regex fallbacks

    def test_parallel_parsing_with_progress_callback(self, tmp_path: Path) -> None:
        """Test parallel parsing invokes progress callback."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = (
            '[{"file_path": "test.py", "start_line": 10, "end_line": 10, '
            '"new_content": "fixed", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed", "risk_level": "low"}]'
        )

        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        progress_calls: list[tuple[int, int]] = []

        def progress_callback(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        # Need 5+ comments to trigger parallel parsing
        comments = [
            {
                "path": "test.py",
                "line": 10 + i,
                "body": f"Fix this {i}",
                "html_url": f"https://github.com/test/{i}",
                "user": {"login": f"user{i}"},
            }
            for i in range(5)
        ]

        changes = resolver.extract_changes_from_comments(
            comments, parallel_parsing=True, progress_callback=progress_callback
        )

        assert len(changes) == 5
        # Progress callback should be called for each comment
        assert len(progress_calls) == 5
        assert progress_calls[-1] == (5, 5)

    def test_parallel_parsing_invalid_max_workers_warning(self, tmp_path: Path) -> None:
        """Test parallel parsing warns on invalid max_workers override."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = (
            '[{"file_path": "test.py", "start_line": 10, "end_line": 10, '
            '"new_content": "fixed", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed", "risk_level": "low"}]'
        )

        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=4)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        # Need 5+ comments to trigger parallel parsing
        comments = [
            {
                "path": "test.py",
                "line": 10 + i,
                "body": f"Fix this {i}",
                "html_url": f"https://github.com/test/{i}",
                "user": {"login": f"user{i}"},
            }
            for i in range(5)
        ]

        # Override with invalid max_workers (< 1)
        changes = resolver.extract_changes_from_comments(
            comments, parallel_parsing=True, max_workers=0
        )

        assert len(changes) == 5
        # Verify max_workers was not changed (should remain 4)
        assert parallel_parser.max_workers == 4


class TestResolverParallelInternals:
    """Unit tests targeting internal parallel parsing branches."""

    def test_extract_changes_parallel_non_parallel_parser(self, tmp_path: Path) -> None:
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=MagicMock(spec=LLMParser))
        comments: list[dict[str, Any]] = [{"path": "test.py", "body": "body"}]

        with patch.object(resolver, "_extract_changes_sequential", return_value=[]) as mock_seq:
            result = resolver._extract_changes_parallel(comments)

        assert result == []
        mock_seq.assert_called_once_with(comments)

    def test_extract_changes_parallel_no_valid_comments_calls_sequential(
        self, tmp_path: Path
    ) -> None:
        parser = ParallelLLMParser(provider=MagicMock(), max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parser)
        comments: list[dict[str, Any]] = [{"path": "test.py", "body": "missing line"}]

        with patch.object(resolver, "_extract_changes_sequential", return_value=[]) as mock_seq:
            result = resolver._extract_changes_parallel(comments)

        assert result == []
        mock_seq.assert_called_once_with(comments)

    def test_extract_changes_parallel_restores_workers_and_handles_regex(
        self, tmp_path: Path
    ) -> None:
        parser = ParallelLLMParser(provider=MagicMock(), max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parser)

        parsed_change = _parsed_change()

        def fake_parse(
            comment_inputs: list[CommentInput],
            progress_callback: Callable[[int, int], None] | None = None,
        ) -> list[list[ParsedChange]]:
            assert len(comment_inputs) == 2
            return [[parsed_change], []]

        comments: list[dict[str, Any]] = [
            {
                "path": "test.py",
                "line": 10,
                "body": "Fix this function",
                "html_url": "https://example.com/1",
                "user": {"login": "user1"},
            },
            {
                "path": "test.py",
                "line": 20,
                "body": "```suggestion\nregex_fix = True\n```",
                "html_url": "https://example.com/2",
                "user": {"login": "user2"},
            },
            {
                # Missing line ensures the code path handling invalid comments runs
                "path": "missing.py",
                "body": "```suggestion\nnoop\n```",
                "html_url": "https://example.com/3",
                "user": {"login": "user3"},
            },
        ]

        with patch.object(parser, "parse_comments", side_effect=fake_parse) as mock_parse:
            changes = resolver._extract_changes_parallel(comments, max_workers=5)

        assert parser.max_workers == 2
        mock_parse.assert_called_once()
        assert any(change.parsing_method == "llm" for change in changes)
        assert any(change.parsing_method == "regex" for change in changes)


class TestLineRangeValidation:
    """Test line range validation and swap logic in parallel/sequential parsing paths."""

    def test_parallel_parsing_swaps_invalid_line_range(self, tmp_path: Path) -> None:
        """Test that invalid line ranges (start_line > line) are swapped in parallel path.

        Covers resolver.py:286-292 - the line range validation swap logic.
        Requires 5+ comments to trigger parallel parsing path.
        """
        mock_provider = MagicMock()
        # Return valid parsed change
        mock_provider.generate.return_value = (
            '[{"file_path": "test.py", "start_line": 10, "end_line": 20, '
            '"new_content": "fixed_code", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed", "risk_level": "low"}]'
        )

        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        # Need 5+ comments to trigger parallel parsing. First comment has invalid line range.
        comments = [
            {
                "path": "test.py",
                "line": 10,  # end_line
                "start_line": 50,  # start_line > end_line - INVALID, should be swapped
                "body": "Fix this function",
                "html_url": "https://github.com/test/1",
                "user": {"login": "user1"},
            },
            {
                "path": "test.py",
                "line": 20,
                "body": "Fix this too",
                "html_url": "https://github.com/test/2",
                "user": {"login": "user2"},
            },
            {
                "path": "test.py",
                "line": 30,
                "body": "Another fix",
                "html_url": "https://github.com/test/3",
                "user": {"login": "user3"},
            },
            {
                "path": "test.py",
                "line": 40,
                "body": "Fourth fix",
                "html_url": "https://github.com/test/4",
                "user": {"login": "user4"},
            },
            {
                "path": "test.py",
                "line": 50,
                "body": "Fifth fix",
                "html_url": "https://github.com/test/5",
                "user": {"login": "user5"},
            },
        ]

        # Use parallel_parsing=True to exercise the parallel path
        changes = resolver.extract_changes_from_comments(comments, parallel_parsing=True)

        # Should extract changes (after swapping line range for first comment)
        assert len(changes) == 5
        assert all(c.parsing_method == "llm" for c in changes)

        # Verify that the normalized range (10 → 50) was actually propagated to the LLM
        # First comment had start_line=50, line=10 which should be swapped to 10→50
        prompts = [call.args[0] for call in mock_provider.generate.call_args_list]
        assert any("Line Range: 10 to 50" in prompt for prompt in prompts)

    def test_sequential_parsing_swaps_invalid_line_range(self, tmp_path: Path) -> None:
        """Test that invalid line ranges are swapped in sequential LLM parsing path.

        Covers resolver.py:533-537 - the line range validation swap logic.
        """
        mock_parser = MagicMock(spec=LLMParser)
        mock_parser.parse_comment.return_value = [
            ParsedChange(
                file_path="test.py",
                start_line=10,
                end_line=20,
                new_content="fixed_code",
                change_type="modification",
                confidence=0.9,
                rationale="Fixed",
                risk_level="low",
            )
        ]

        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        # Comment with invalid line range: start_line (100) > line/end_line (10)
        # This triggers swap logic in _extract_changes_with_llm (sequential path)
        comments = [
            {
                "path": "test.py",
                "line": 10,  # end_line
                "start_line": 100,  # start_line > end_line - INVALID, should be swapped
                "body": "Fix this function with natural language",
                "html_url": "https://github.com/test/1",
                "user": {"login": "user1"},
            }
        ]

        # Use sequential parsing (default) to exercise _extract_changes_with_llm
        changes = resolver.extract_changes_from_comments(comments, parallel_parsing=False)

        assert len(changes) == 1
        assert changes[0].parsing_method == "llm"
        # Verify parse_comment was called (sequential path)
        mock_parser.parse_comment.assert_called_once()

        # Verify parse_comment was called with the swapped (normalized) range
        # Original: start_line=100, line=10 → normalized: start_line=10, end_line=100
        _, kwargs = mock_parser.parse_comment.call_args
        assert kwargs["start_line"] == 10
        assert kwargs["end_line"] == 100

    def test_sequential_parsing_with_diff_hunk_logs_header(
        self, tmp_path: Path, caplog: "pytest.LogCaptureFixture"
    ) -> None:
        """Test that diff hunk header is logged when present in sequential path.

        Covers resolver.py:516-517 - the diff hunk header extraction logging.
        """
        mock_parser = MagicMock(spec=LLMParser)
        mock_parser.parse_comment.return_value = [
            ParsedChange(
                file_path="test.py",
                start_line=10,
                end_line=12,
                new_content="def fixed():\n    return True",
                change_type="modification",
                confidence=0.9,
                rationale="Fixed function",
                risk_level="low",
            )
        ]

        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=mock_parser)

        # Comment with diff_hunk - triggers header extraction at resolver.py:516-517
        comments = [
            {
                "path": "test.py",
                "line": 10,
                "body": "Fix this function",
                "diff_hunk": "@@ -8,5 +8,6 @@\n context line\n-old code\n+new code",
                "html_url": "https://github.com/test/1",
                "user": {"login": "user1"},
            }
        ]

        with caplog.at_level(logging.DEBUG):
            changes = resolver.extract_changes_from_comments(comments, parallel_parsing=False)

        assert len(changes) == 1
        assert changes[0].parsing_method == "llm"
        # Verify that the diff hunk header was logged
        assert "@@ -8,5 +8,6 @@" in caplog.text

    def test_parallel_parsing_with_original_line_fields(self, tmp_path: Path) -> None:
        """Test parallel parsing handles original_start_line and original_line fields.

        These fields are used when line/start_line are None but original_* fields exist.
        Requires 5+ comments to trigger parallel parsing path.
        """
        mock_provider = MagicMock()
        mock_provider.generate.return_value = (
            '[{"file_path": "test.py", "start_line": 5, "end_line": 15, '
            '"new_content": "fixed", "change_type": "modification", '
            '"confidence": 0.9, "rationale": "Fixed", "risk_level": "low"}]'
        )

        parallel_parser = ParallelLLMParser(provider=mock_provider, max_workers=2)
        resolver = ConflictResolver(workspace_root=tmp_path, llm_parser=parallel_parser)

        # Need 5+ comments for parallel parsing. First uses original_line fields.
        comments = [
            {
                "path": "test.py",
                "original_line": 15,  # Used when 'line' is missing
                "original_start_line": 5,  # Used when 'start_line' is missing
                "body": "Fix this function",
                "html_url": "https://github.com/test/1",
                "user": {"login": "user1"},
            },
            {
                "path": "test.py",
                "line": 20,
                "body": "Fix this too",
                "html_url": "https://github.com/test/2",
                "user": {"login": "user2"},
            },
            {
                "path": "test.py",
                "line": 30,
                "body": "Another fix",
                "html_url": "https://github.com/test/3",
                "user": {"login": "user3"},
            },
            {
                "path": "test.py",
                "line": 40,
                "body": "Fourth fix",
                "html_url": "https://github.com/test/4",
                "user": {"login": "user4"},
            },
            {
                "path": "test.py",
                "line": 50,
                "body": "Fifth fix",
                "html_url": "https://github.com/test/5",
                "user": {"login": "user5"},
            },
        ]

        changes = resolver.extract_changes_from_comments(comments, parallel_parsing=True)

        assert len(changes) == 5
        assert all(c.parsing_method == "llm" for c in changes)

        # Verify that original_start_line/original_line were used for the effective range
        # First comment used original_start_line=5, original_line=15
        prompts = [call.args[0] for call in mock_provider.generate.call_args_list]
        assert any("Line Range: 5 to 15" in prompt for prompt in prompts)
