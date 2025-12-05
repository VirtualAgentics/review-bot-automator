"""Tests for UniversalLLMParser.

This module tests the LLM-powered parser implementation including:
- Parser initialization and configuration
- Successful parsing with valid JSON responses
- JSON validation and error handling
- Confidence threshold filtering
- Fallback behavior (return empty list vs raise exception)
- Various comment formats (diff blocks, suggestions, natural language)
- Edge cases (malformed JSON, invalid fields, empty responses)
- Security features (secret scanning)
- Cost tracking and budget enforcement
"""

from unittest.mock import MagicMock, patch

import pytest

from review_bot_automator.llm.base import LLMParser
from review_bot_automator.llm.comment_sources import (
    AIPromptBlock,
    CommentSources,
    DiffBlock,
    SuggestionBlock,
)
from review_bot_automator.llm.cost_tracker import CostStatus, CostTracker
from review_bot_automator.llm.exceptions import LLMCostExceededError, LLMSecretDetectedError
from review_bot_automator.llm.parser import (
    CONFIDENCE_AI_PROMPT,
    UniversalLLMParser,
    _strip_json_fences,
)
from review_bot_automator.llm.providers.base import LLMProvider


class TestUniversalLLMParserProtocol:
    """Test that UniversalLLMParser conforms to LLMParser protocol."""

    def test_parser_implements_protocol(self) -> None:
        """Test that UniversalLLMParser implements LLMParser protocol."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)
        assert isinstance(parser, LLMParser)

    def test_parser_has_parse_comment_method(self) -> None:
        """Test that parser has parse_comment() method with correct signature."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)
        assert hasattr(parser, "parse_comment")
        assert callable(parser.parse_comment)


class TestUniversalLLMParserInitialization:
    """Test UniversalLLMParser initialization and configuration."""

    def test_init_with_valid_params(self) -> None:
        """Test initialization with valid parameters."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(
            provider=mock_provider,
            fallback_to_regex=False,
            confidence_threshold=0.7,
        )
        assert parser.provider is mock_provider
        assert parser.fallback_to_regex is False
        assert parser.confidence_threshold == 0.7

    def test_init_with_default_params(self) -> None:
        """Test initialization with default parameters."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)
        assert parser.fallback_to_regex is True
        assert parser.confidence_threshold == 0.5

    def test_init_with_invalid_threshold_raises(self) -> None:
        """Test that invalid confidence threshold raises ValueError."""
        mock_provider = MagicMock(spec=LLMProvider)
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            UniversalLLMParser(mock_provider, confidence_threshold=1.5)

    def test_init_with_negative_threshold_raises(self) -> None:
        """Test that negative confidence threshold raises ValueError."""
        mock_provider = MagicMock(spec=LLMProvider)
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            UniversalLLMParser(mock_provider, confidence_threshold=-0.1)

    def test_set_confidence_threshold_valid(self) -> None:
        """Test setting valid confidence threshold."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)
        parser.set_confidence_threshold(0.8)
        assert parser.confidence_threshold == 0.8

    def test_set_confidence_threshold_invalid_raises(self) -> None:
        """Test that invalid threshold in setter raises ValueError."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            parser.set_confidence_threshold(2.0)


class TestUniversalLLMParserValidation:
    """Test input validation in parse_comment."""

    def test_parse_comment_empty_body_raises(self) -> None:
        """Test that empty comment body raises ValueError."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)
        with pytest.raises(ValueError, match="cannot be None or empty"):
            parser.parse_comment("", file_path="test.py")

    def test_parse_comment_none_body_raises(self) -> None:
        """Test that None comment body raises ValueError."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)
        with pytest.raises(ValueError, match="cannot be None or empty"):
            parser.parse_comment(None, file_path="test.py")  # type: ignore[arg-type]


class TestUniversalLLMParserSuccessfulParsing:
    """Test successful parsing scenarios."""

    def test_parse_diff_block_success(self) -> None:
        """Test parsing a diff block comment successfully."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = (
            '[{"file_path": "src/auth.py", "start_line": 42, "end_line": 45, '
            '"new_content": "def authenticate(username, password):\\\\n'
            '    # Use parameterized query\\\\n    return True", '
            '"change_type": "modification", "confidence": 0.95, '
            '"rationale": "SQL injection vulnerability fix", "risk_level": "high"}]'
        )

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)
        changes = parser.parse_comment(
            "Fix SQL injection in auth:\n```diff\n...\n```",
            file_path="src/auth.py",
            line_number=42,
        )

        assert len(changes) == 1
        assert changes[0].file_path == "src/auth.py"
        assert changes[0].start_line == 42
        assert changes[0].end_line == 45
        assert changes[0].confidence == 0.95
        assert changes[0].risk_level == "high"

    def test_parse_multiple_changes(self) -> None:
        """Test parsing multiple changes from single comment."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/utils.py",
                "start_line": 10,
                "end_line": 12,
                "new_content": "# Change 1",
                "change_type": "modification",
                "confidence": 0.85,
                "rationale": "First change",
                "risk_level": "low"
            },
            {
                "file_path": "src/utils.py",
                "start_line": 20,
                "end_line": 22,
                "new_content": "# Change 2",
                "change_type": "addition",
                "confidence": 0.75,
                "rationale": "Second change",
                "risk_level": "medium"
            }
        ]"""

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)
        changes = parser.parse_comment("Apply these two changes", file_path="src/utils.py")

        assert len(changes) == 2
        assert changes[0].start_line == 10
        assert changes[1].start_line == 20

    def test_parse_empty_changes_array(self) -> None:
        """Test parsing comment with no actionable changes."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"

        parser = UniversalLLMParser(mock_provider)
        changes = parser.parse_comment("This looks good to me!", file_path="src/test.py")

        assert len(changes) == 0

    def test_parse_with_optional_context(self) -> None:
        """Test parsing with file_path and line_number context."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/main.py",
                "start_line": 100,
                "end_line": 105,
                "new_content": "# Fixed",
                "change_type": "modification",
                "confidence": 0.88,
                "rationale": "Context helps parsing",
                "risk_level": "low"
            }
        ]"""

        parser = UniversalLLMParser(mock_provider)
        changes = parser.parse_comment(
            "Fix this issue",
            file_path="src/main.py",
            line_number=100,
        )

        assert len(changes) == 1
        # Verify context was passed to provider (check call args)
        call_args = mock_provider.generate.call_args[0][0]
        assert "src/main.py" in call_args
        assert "100" in call_args


class TestUniversalLLMParserConfidenceFiltering:
    """Test confidence threshold filtering."""

    def test_filter_low_confidence_changes(self) -> None:
        """Test that changes below threshold are filtered out."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/test.py",
                "start_line": 1,
                "end_line": 2,
                "new_content": "# High confidence",
                "change_type": "modification",
                "confidence": 0.9,
                "rationale": "Clear fix",
                "risk_level": "low"
            },
            {
                "file_path": "src/test.py",
                "start_line": 10,
                "end_line": 12,
                "new_content": "# Low confidence",
                "change_type": "addition",
                "confidence": 0.4,
                "rationale": "Unclear",
                "risk_level": "medium"
            }
        ]"""

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.7)
        changes = parser.parse_comment("Apply these changes", file_path="src/test.py")

        # Only high-confidence change should pass
        assert len(changes) == 1
        assert changes[0].confidence == 0.9

    def test_all_changes_filtered(self) -> None:
        """Test when all changes are below threshold."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/test.py",
                "start_line": 1,
                "end_line": 2,
                "new_content": "# Low confidence",
                "change_type": "modification",
                "confidence": 0.3,
                "rationale": "Unclear",
                "risk_level": "low"
            }
        ]"""

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.7)
        changes = parser.parse_comment("Maybe fix this?", file_path="src/test.py")

        assert len(changes) == 0

    def test_exact_threshold_boundary(self) -> None:
        """Test change exactly at threshold is included."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/test.py",
                "start_line": 1,
                "end_line": 2,
                "new_content": "# Exact threshold",
                "change_type": "modification",
                "confidence": 0.7,
                "rationale": "At boundary",
                "risk_level": "low"
            }
        ]"""

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.7)
        changes = parser.parse_comment("Fix this", file_path="src/test.py")

        # Change at exactly threshold should be included (>= behavior)
        assert len(changes) == 1
        assert changes[0].confidence == 0.7


class TestUniversalLLMParserErrorHandling:
    """Test error handling and fallback behavior."""

    def test_invalid_json_with_fallback(self) -> None:
        """Test that invalid JSON returns empty list when fallback enabled."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "not valid json {{"

        parser = UniversalLLMParser(mock_provider, fallback_to_regex=True)
        changes = parser.parse_comment("Fix this", file_path="src/test.py")

        assert len(changes) == 0

    def test_invalid_json_without_fallback(self) -> None:
        """Test that invalid JSON raises error when fallback disabled."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "not valid json {{"

        parser = UniversalLLMParser(mock_provider, fallback_to_regex=False)
        with pytest.raises(RuntimeError, match="LLM parsing failed"):
            parser.parse_comment("Fix this", file_path="src/test.py")

    def test_non_list_response_with_fallback(self) -> None:
        """Test that non-list JSON returns empty list when fallback enabled."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = '{"error": "not an array"}'

        parser = UniversalLLMParser(mock_provider, fallback_to_regex=True)
        changes = parser.parse_comment("Fix this", file_path="src/test.py")

        assert len(changes) == 0

    def test_non_list_response_without_fallback(self) -> None:
        """Test that non-list JSON raises error when fallback disabled."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = '{"error": "not an array"}'

        parser = UniversalLLMParser(mock_provider, fallback_to_regex=False)
        with pytest.raises(RuntimeError, match="LLM parsing failed"):
            parser.parse_comment("Fix this", file_path="src/test.py")

    def test_invalid_change_format_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid change objects are skipped with warning."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/test.py",
                "start_line": 1,
                "end_line": 2,
                "new_content": "# Valid",
                "change_type": "modification",
                "confidence": 0.9,
                "rationale": "Good",
                "risk_level": "low"
            },
            {
                "file_path": "src/test.py",
                "missing_required_field": true
            }
        ]"""

        parser = UniversalLLMParser(mock_provider)
        changes = parser.parse_comment("Fix this", file_path="src/test.py")

        # Only valid change should be returned
        assert len(changes) == 1
        assert changes[0].confidence == 0.9
        # Check warning was logged
        assert "Invalid change format" in caplog.text

    def test_provider_exception_with_fallback(self) -> None:
        """Test that provider exceptions return empty list when fallback enabled."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.side_effect = RuntimeError("Provider error")

        parser = UniversalLLMParser(mock_provider, fallback_to_regex=True)
        changes = parser.parse_comment("Fix this", file_path="src/test.py")

        assert len(changes) == 0

    def test_provider_exception_without_fallback(self) -> None:
        """Test that provider exceptions raise error when fallback disabled."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.side_effect = RuntimeError("Provider error")

        parser = UniversalLLMParser(mock_provider, fallback_to_regex=False)
        with pytest.raises(RuntimeError, match="LLM parsing failed"):
            parser.parse_comment("Fix this", file_path="src/test.py")


class TestUniversalLLMParserEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_parse_with_none_file_path(self) -> None:
        """Test parsing with None file_path (should use 'unknown')."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "inferred.py",
                "start_line": 1,
                "end_line": 2,
                "new_content": "# Fixed",
                "change_type": "modification",
                "confidence": 0.8,
                "rationale": "Inferred path",
                "risk_level": "low"
            }
        ]"""

        parser = UniversalLLMParser(mock_provider)
        changes = parser.parse_comment("Fix this", file_path=None, line_number=None)

        assert len(changes) == 1
        # Verify 'unknown' was used in prompt
        call_args = mock_provider.generate.call_args[0][0]
        assert "unknown" in call_args

    def test_parse_with_very_long_comment(self) -> None:
        """Test parsing with very long comment body."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"

        parser = UniversalLLMParser(mock_provider)
        long_comment = "Fix this issue. " * 1000  # 16,000 chars
        changes = parser.parse_comment(long_comment, file_path="src/test.py")

        # Should handle long comments without error
        assert len(changes) == 0
        mock_provider.generate.assert_called_once()

    def test_parse_with_unicode_content(self) -> None:
        """Test parsing with unicode characters in content."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/test.py",
                "start_line": 1,
                "end_line": 2,
                "new_content": "# 修复错误 🔧",
                "change_type": "modification",
                "confidence": 0.9,
                "rationale": "Unicode content",
                "risk_level": "low"
            }
        ]"""

        parser = UniversalLLMParser(mock_provider)
        changes = parser.parse_comment("修复这个问题 🐛", file_path="src/test.py")

        assert len(changes) == 1
        assert "修复错误" in changes[0].new_content

    def test_parse_with_max_tokens_parameter(self) -> None:
        """Test that parser passes max_tokens to provider."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"

        parser = UniversalLLMParser(mock_provider)
        parser.parse_comment("Fix this", file_path="src/test.py")

        # Verify max_tokens=2000 was passed
        call_kwargs = mock_provider.generate.call_args[1]
        assert call_kwargs["max_tokens"] == 2000

    def test_multiple_risk_levels(self) -> None:
        """Test parsing changes with different risk levels."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/test.py",
                "start_line": 1,
                "end_line": 2,
                "new_content": "# Low risk",
                "change_type": "modification",
                "confidence": 0.9,
                "rationale": "Formatting",
                "risk_level": "low"
            },
            {
                "file_path": "src/test.py",
                "start_line": 10,
                "end_line": 15,
                "new_content": "# High risk",
                "change_type": "modification",
                "confidence": 0.95,
                "rationale": "Security fix",
                "risk_level": "high"
            }
        ]"""

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)
        changes = parser.parse_comment("Apply changes", file_path="src/test.py")

        assert len(changes) == 2
        assert changes[0].risk_level == "low"
        assert changes[1].risk_level == "high"


class TestUniversalLLMParserFallbackStats:
    """Test fallback statistics tracking."""

    def test_initial_fallback_stats_zero(self) -> None:
        """Test that initial fallback stats are zero."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        fallback_count, total_count, rate = parser.get_fallback_stats()

        assert fallback_count == 0
        assert total_count == 0
        assert rate == 0.0

    def test_successful_parse_increments_success_count(self) -> None:
        """Test that successful parsing increments success count."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider)
        parser.parse_comment("Fix this", file_path="src/test.py")

        fallback_count, total_count, rate = parser.get_fallback_stats()

        assert fallback_count == 0
        assert total_count == 1
        assert rate == 0.0

    def test_failed_parse_with_fallback_increments_fallback_count(self) -> None:
        """Test that failed parsing increments fallback count."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.side_effect = RuntimeError("LLM failed")

        parser = UniversalLLMParser(mock_provider, fallback_to_regex=True)
        result = parser.parse_comment("Fix this", file_path="src/test.py")

        assert result == []  # Empty list for fallback
        fallback_count, total_count, rate = parser.get_fallback_stats()

        assert fallback_count == 1
        assert total_count == 1
        assert rate == 1.0

    def test_fallback_rate_calculation(self) -> None:
        """Test that fallback rate is calculated correctly."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.get_total_cost.return_value = 0.0

        # First call succeeds
        mock_provider.generate.return_value = "[]"
        parser = UniversalLLMParser(mock_provider, fallback_to_regex=True)
        parser.parse_comment("Fix this", file_path="src/test.py")

        # Second call fails (triggers fallback)
        mock_provider.generate.side_effect = RuntimeError("LLM failed")
        parser.parse_comment("Fix that", file_path="src/other.py")

        fallback_count, total_count, rate = parser.get_fallback_stats()

        assert fallback_count == 1
        assert total_count == 2
        assert rate == 0.5  # 1 fallback / 2 total = 50%

    def test_reset_fallback_stats(self) -> None:
        """Test that reset_fallback_stats clears counters."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider)
        parser.parse_comment("Fix this", file_path="src/test.py")

        # Verify stats are non-zero
        _, total_count, _ = parser.get_fallback_stats()
        assert total_count == 1

        # Reset and verify
        parser.reset_fallback_stats()
        fallback_count, total_count, rate = parser.get_fallback_stats()

        assert fallback_count == 0
        assert total_count == 0
        assert rate == 0.0

    def test_invalid_json_triggers_fallback(self) -> None:
        """Test that invalid JSON response triggers fallback count."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "not valid json"
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider, fallback_to_regex=True)
        parser.parse_comment("Fix this", file_path="src/test.py")

        fallback_count, total_count, rate = parser.get_fallback_stats()

        assert fallback_count == 1
        assert total_count == 1
        assert rate == 1.0


class TestStripJsonFences:
    """Test JSON code fence stripping utility function."""

    def test_strip_json_fence(self) -> None:
        """Test stripping ```json fences from response."""
        text = '```json\n[{"key": "value"}]\n```'
        result = _strip_json_fences(text)
        assert result == '[{"key": "value"}]'

    def test_strip_plain_fence(self) -> None:
        """Test stripping plain ``` fences without json marker."""
        text = '```\n[{"key": "value"}]\n```'
        result = _strip_json_fences(text)
        assert result == '[{"key": "value"}]'

    def test_no_fence_returns_original(self) -> None:
        """Test that text without fences is returned unchanged."""
        text = '[{"key": "value"}]'
        result = _strip_json_fences(text)
        assert result == '[{"key": "value"}]'


class TestSecretScanning:
    """Test secret scanning security feature."""

    def test_secret_detected_raises_error(self) -> None:
        """Test that detected secrets raise LLMSecretDetectedError."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider, scan_for_secrets=True)

        # Comment containing what looks like an API key (test data, not real)
        secret_comment = "api_key = 'sk-1234567890abcdefghijklmnopqrstuvwxyz12345'"  # noqa: S105

        with pytest.raises(LLMSecretDetectedError):
            parser.parse_comment(secret_comment, file_path="src/config.py")

        # Provider should NOT have been called
        mock_provider.generate.assert_not_called()

    def test_secret_scanning_disabled_allows_through(self) -> None:
        """Test that disabling secret scanning allows potentially sensitive content."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider, scan_for_secrets=False)

        # This would normally trigger secret detection (test data, not real)
        secret_comment = "api_key = 'sk-1234567890abcdefghijklmnopqrstuvwxyz12345'"  # noqa: S105

        # Should not raise, secret scanning is disabled
        result = parser.parse_comment(secret_comment, file_path="src/config.py")
        assert result == []
        mock_provider.generate.assert_called_once()


class TestCostTracking:
    """Test cost tracking and budget enforcement."""

    def test_cost_exceeded_before_call_raises(self) -> None:
        """Test that exceeding budget before call raises LLMCostExceededError."""
        mock_provider = MagicMock(spec=LLMProvider)
        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.should_block_request.return_value = True
        cost_tracker.accumulated_cost = 10.0
        cost_tracker.budget = 5.0

        parser = UniversalLLMParser(
            mock_provider, cost_tracker=cost_tracker, fallback_to_regex=False
        )

        with pytest.raises(LLMCostExceededError, match="Cost budget exceeded"):
            parser.parse_comment("Fix this", file_path="src/test.py")

        # Provider should NOT have been called
        mock_provider.generate.assert_not_called()

    def test_cost_exceeded_with_fallback_returns_empty(self) -> None:
        """Test that cost exceeded during request with fallback returns empty list."""
        mock_provider = MagicMock(spec=LLMProvider)
        # First call succeeds, but add_cost raises LLMCostExceededError
        mock_provider.generate.return_value = "[]"
        mock_provider.get_total_cost.side_effect = [0.0, 10.0]  # Before and after

        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.should_block_request.return_value = False  # Allow first request

        # Simulate exception during cost tracking (after LLM call)
        def add_cost_side_effect(cost: float) -> CostStatus:
            raise LLMCostExceededError(
                "Budget exceeded after call",
                accumulated_cost=10.0,
                budget=5.0,
            )

        cost_tracker.add_cost.side_effect = add_cost_side_effect

        parser = UniversalLLMParser(
            mock_provider, cost_tracker=cost_tracker, fallback_to_regex=True
        )

        result = parser.parse_comment("Fix this", file_path="src/test.py")

        assert result == []
        # Provider WAS called (exception happens during cost tracking after the call)
        mock_provider.generate.assert_called_once()

    def test_cost_warning_threshold_logged(self) -> None:
        """Test that cost warning is logged when threshold is reached."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"
        mock_provider.get_total_cost.side_effect = [0.0, 4.5]  # Before and after

        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.should_block_request.return_value = False
        cost_tracker.add_cost.return_value = CostStatus.WARNING
        cost_tracker.get_warning_message.return_value = "Warning: 80% of budget used"

        parser = UniversalLLMParser(mock_provider, cost_tracker=cost_tracker)

        with patch("review_bot_automator.llm.parser.logger") as mock_logger:
            parser.parse_comment("Fix this", file_path="src/test.py")
            mock_logger.warning.assert_called_with("Warning: 80% of budget used")

    def test_cost_tracking_records_request_cost(self) -> None:
        """Test that cost tracking records the incremental request cost."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"
        mock_provider.get_total_cost.side_effect = [0.0, 0.05]  # Before: $0.00, After: $0.05

        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.should_block_request.return_value = False
        cost_tracker.add_cost.return_value = CostStatus.OK

        parser = UniversalLLMParser(mock_provider, cost_tracker=cost_tracker)
        parser.parse_comment("Fix this", file_path="src/test.py")

        # Should have added the incremental cost ($0.05 - $0.00 = $0.05)
        cost_tracker.add_cost.assert_called_once_with(0.05)

    def test_cost_exceeded_without_fallback_raises(self) -> None:
        """Test that cost exceeded during request without fallback raises error."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "[]"
        mock_provider.get_total_cost.side_effect = [0.0, 10.0]

        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.should_block_request.return_value = False

        # Simulate exception during cost tracking
        def add_cost_side_effect(cost: float) -> CostStatus:
            raise LLMCostExceededError(
                "Budget exceeded after call",
                accumulated_cost=10.0,
                budget=5.0,
            )

        cost_tracker.add_cost.side_effect = add_cost_side_effect

        parser = UniversalLLMParser(
            mock_provider, cost_tracker=cost_tracker, fallback_to_regex=False
        )

        with pytest.raises(LLMCostExceededError, match="Budget exceeded after call"):
            parser.parse_comment("Fix this", file_path="src/test.py")


class TestJsonFenceInLLMResponse:
    """Test handling of JSON fences in LLM responses."""

    def test_parse_response_with_json_fence(self) -> None:
        """Test that parser strips JSON fences from LLM response."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """```json
[
    {
        "file_path": "src/test.py",
        "start_line": 1,
        "end_line": 2,
        "new_content": "# Fixed",
        "change_type": "modification",
        "confidence": 0.9,
        "rationale": "LLM wrapped response in fence",
        "risk_level": "low"
    }
]
```"""
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider)
        changes = parser.parse_comment("Fix this", file_path="src/test.py")

        assert len(changes) == 1
        assert changes[0].file_path == "src/test.py"


class TestBuildSourceContext:
    """Test _build_source_context method for source detection context building."""

    def test_no_blocks_returns_fallback_message(self) -> None:
        """Test fallback message when no structured blocks detected."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        # Create empty CommentSources
        sources = CommentSources(
            diff_blocks=(),
            suggestion_blocks=(),
            ai_prompt_blocks=(),
            details_blocks=(),
            html_comments_stripped=0,
        )

        result = parser._build_source_context(sources)

        assert "No structured blocks detected" in result
        assert "natural language parsing" in result
        assert "lower confidence" in result

    def test_ai_prompt_block_short_content(self) -> None:
        """Test AI prompt block with short content (no truncation)."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        # Create CommentSources with AI prompt < 200 chars
        short_content = "In src/foo.py around line 50, rename bar to baz"
        sources = CommentSources(
            diff_blocks=(),
            suggestion_blocks=(),
            ai_prompt_blocks=(
                AIPromptBlock(summary="AI Prompt", content=short_content, position=0),
            ),
            details_blocks=(),
            html_comments_stripped=0,
        )

        result = parser._build_source_context(sources)

        assert "AI Prompt block(s) detected" in result
        assert f"confidence >= {CONFIDENCE_AI_PROMPT}" in result
        assert short_content in result
        assert "..." not in result  # No truncation

    def test_ai_prompt_block_long_content_truncated(self) -> None:
        """Test AI prompt block with long content gets truncated."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        # Create CommentSources with AI prompt > 200 chars
        long_content = "x" * 250  # Longer than _AI_PROMPT_PREVIEW_LENGTH (200)
        sources = CommentSources(
            diff_blocks=(),
            suggestion_blocks=(),
            ai_prompt_blocks=(
                AIPromptBlock(summary="AI Prompt", content=long_content, position=0),
            ),
            details_blocks=(),
            html_comments_stripped=0,
        )

        result = parser._build_source_context(sources)

        assert "AI Prompt block(s) detected" in result
        assert "..." in result  # Content was truncated
        assert long_content not in result  # Full content not present

    def test_ai_prompt_block_empty_content(self) -> None:
        """Test AI prompt block with empty content doesn't show instructions."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        sources = CommentSources(
            diff_blocks=(),
            suggestion_blocks=(),
            ai_prompt_blocks=(AIPromptBlock(summary="AI Prompt", content="", position=0),),
            details_blocks=(),
            html_comments_stripped=0,
        )

        result = parser._build_source_context(sources)

        assert "AI Prompt block(s) detected" in result
        assert "AI Instructions:" not in result  # Empty content not shown

    def test_suggestion_blocks_detected(self) -> None:
        """Test suggestion blocks are included in context."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        sources = CommentSources(
            diff_blocks=(),
            suggestion_blocks=(
                SuggestionBlock(content="new code", position=0, option_label=None),
                SuggestionBlock(content="more code", position=50, option_label="Option 1"),
            ),
            ai_prompt_blocks=(),
            details_blocks=(),
            html_comments_stripped=0,
        )

        result = parser._build_source_context(sources)

        assert "2 suggestion block(s) detected" in result

    def test_diff_blocks_with_hunk_headers(self) -> None:
        """Test diff blocks with hunk headers."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        sources = CommentSources(
            diff_blocks=(
                DiffBlock(content="@@ -1,2 +1,3 @@\n-old\n+new", position=0, has_hunk_header=True),
            ),
            suggestion_blocks=(),
            ai_prompt_blocks=(),
            details_blocks=(),
            html_comments_stripped=0,
        )

        result = parser._build_source_context(sources)

        assert "1 diff block(s) detected" in result
        assert "(with hunk headers)" in result

    def test_diff_blocks_without_hunk_headers(self) -> None:
        """Test diff blocks without hunk headers."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        sources = CommentSources(
            diff_blocks=(DiffBlock(content="-old\n+new", position=0, has_hunk_header=False),),
            suggestion_blocks=(),
            ai_prompt_blocks=(),
            details_blocks=(),
            html_comments_stripped=0,
        )

        result = parser._build_source_context(sources)

        assert "1 diff block(s) detected" in result
        assert "(with hunk headers)" not in result

    def test_multiple_block_types_combined(self) -> None:
        """Test context with multiple block types."""
        mock_provider = MagicMock(spec=LLMProvider)
        parser = UniversalLLMParser(mock_provider)

        sources = CommentSources(
            diff_blocks=(
                DiffBlock(content="@@ -1 +1 @@\n-old\n+new", position=0, has_hunk_header=True),
                DiffBlock(content="-foo\n+bar", position=50, has_hunk_header=False),
            ),
            suggestion_blocks=(
                SuggestionBlock(content="suggestion", position=100, option_label=None),
            ),
            ai_prompt_blocks=(AIPromptBlock(summary="AI", content="instructions", position=150),),
            details_blocks=(),
            html_comments_stripped=0,
        )

        result = parser._build_source_context(sources)

        assert "AI Prompt block(s) detected" in result
        assert "suggestion block(s) detected" in result
        assert "diff block(s) detected" in result
        # Should have hunk headers note since at least one diff has them
        assert "(with hunk headers)" in result


class TestParseCommentSourceDetection:
    """Test source detection integration in parse_comment."""

    def test_parse_comment_logs_ai_prompt_detection(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that AI prompt detection is logged."""
        import logging

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = """[
            {
                "file_path": "src/test.py",
                "start_line": 50,
                "end_line": 52,
                "new_content": "renamed_function()",
                "change_type": "modification",
                "confidence": 0.95,
                "rationale": "AI prompt instructed rename",
                "risk_level": "low"
            }
        ]"""
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider)

        # Comment body with AI prompt block
        comment_with_ai_prompt = """
<details>
<summary>🤖 Prompt for AI Agents</summary>

In src/test.py around line 50, rename function foo to renamed_function.

</details>
"""
        with caplog.at_level(logging.INFO):
            parser.parse_comment(
                comment_with_ai_prompt, file_path="src/test.py", start_line=50, end_line=52
            )

        # Check that AI prompt detection was logged
        assert any("AI Prompt block detected" in record.message for record in caplog.records)


class TestAIPromptFallbackReparse:
    """Test AI prompt fallback re-parse mechanism (Issue #301).

    When the initial parse returns no changes above threshold but an AI prompt
    block is detected, the parser should automatically re-parse with an enhanced
    prompt emphasizing the AI prompt content.
    """

    def test_fallback_triggers_on_low_confidence_with_ai_prompt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that fallback activates when initial parse has low confidence.

        Scenario:
        - Initial LLM response returns changes below threshold
        - Comment contains AI prompt block
        - Fallback should trigger with enhanced prompt
        """
        import logging

        mock_provider = MagicMock(spec=LLMProvider)
        # First call: returns low confidence change (below 0.5 threshold)
        # Second call (fallback): returns high confidence change
        mock_provider.generate.side_effect = [
            # First parse - low confidence
            """[{
                "file_path": "src/test.py",
                "start_line": 50,
                "end_line": 52,
                "new_content": "# Low confidence",
                "change_type": "modification",
                "confidence": 0.3,
                "rationale": "Unclear",
                "risk_level": "low"
            }]""",
            # Fallback parse - high confidence
            """[{
                "file_path": "src/test.py",
                "start_line": 50,
                "end_line": 52,
                "new_content": "renamed_function()",
                "change_type": "modification",
                "confidence": 0.95,
                "rationale": "AI prompt instructed rename",
                "risk_level": "low"
            }]""",
        ]
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)

        # Comment with AI prompt block
        comment_with_ai_prompt = """
<details>
<summary>🤖 Prompt for AI Agents</summary>

In src/test.py around line 50, rename function foo to renamed_function.

</details>
"""
        with caplog.at_level(logging.INFO):
            changes = parser.parse_comment(
                comment_with_ai_prompt, file_path="src/test.py", start_line=50, end_line=52
            )

        # Fallback should have been triggered
        assert any(
            "attempting enhanced AI prompt fallback" in record.message for record in caplog.records
        )

        # Should have made two LLM calls
        assert mock_provider.generate.call_count == 2

        # Second call should include the fallback preamble
        second_call_prompt = mock_provider.generate.call_args_list[1][0][0]
        assert "FALLBACK MODE" in second_call_prompt
        assert "PRIORITY EXTRACTION SOURCE" in second_call_prompt

        # Should have returned the high-confidence change from fallback
        assert len(changes) == 1
        assert changes[0].confidence == 0.95
        assert "renamed_function" in changes[0].new_content

    def test_fallback_limited_to_single_attempt(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that re-parsing is limited to single attempt to prevent loops.

        Scenario:
        - Initial parse returns low confidence
        - Fallback parse also returns low confidence
        - Should NOT trigger another fallback (only one re-parse allowed)
        """
        import logging

        mock_provider = MagicMock(spec=LLMProvider)
        # Both calls return low confidence - should NOT trigger third call
        # Use side_effect with list to explicitly simulate two separate LLM calls
        mock_provider.generate.side_effect = [
            # First parse - low confidence
            """[{
                "file_path": "src/test.py",
                "start_line": 50,
                "end_line": 52,
                "new_content": "# Low confidence initial",
                "change_type": "modification",
                "confidence": 0.3,
                "rationale": "Unclear from initial parse",
                "risk_level": "low"
            }]""",
            # Fallback parse - also low confidence
            """[{
                "file_path": "src/test.py",
                "start_line": 50,
                "end_line": 52,
                "new_content": "# Low confidence fallback",
                "change_type": "modification",
                "confidence": 0.3,
                "rationale": "Still unclear from fallback",
                "risk_level": "low"
            }]""",
        ]
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)

        comment_with_ai_prompt = """
<details>
<summary>🤖 Prompt for AI Agents</summary>

In src/test.py around line 50, rename function foo.

</details>
"""
        with caplog.at_level(logging.INFO):
            changes = parser.parse_comment(
                comment_with_ai_prompt, file_path="src/test.py", start_line=50, end_line=52
            )

        # Should have made exactly two LLM calls (initial + one fallback)
        assert mock_provider.generate.call_count == 2

        # No changes above threshold (both attempts returned low confidence)
        assert len(changes) == 0

    def test_fallback_skipped_when_no_ai_prompt(self) -> None:
        """Test that fallback is NOT triggered when no AI prompt block exists.

        Scenario:
        - Initial parse returns low confidence (or no changes above threshold)
        - Comment does NOT contain AI prompt block
        - Fallback should NOT trigger
        """
        mock_provider = MagicMock(spec=LLMProvider)
        # Returns low confidence - normally would trigger fallback
        mock_provider.generate.return_value = """[{
            "file_path": "src/test.py",
            "start_line": 50,
            "end_line": 52,
            "new_content": "# Low confidence",
            "change_type": "modification",
            "confidence": 0.3,
            "rationale": "Unclear",
            "risk_level": "low"
        }]"""
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)

        # Comment WITHOUT AI prompt block
        comment_without_ai_prompt = "Maybe fix this function? Not sure."

        changes = parser.parse_comment(
            comment_without_ai_prompt, file_path="src/test.py", start_line=50, end_line=52
        )

        # Should have made only ONE LLM call (no fallback)
        assert mock_provider.generate.call_count == 1

        # No changes above threshold
        assert len(changes) == 0

    def test_fallback_skipped_when_changes_accepted(self) -> None:
        """Test that fallback is NOT triggered when changes are already accepted.

        Scenario:
        - Initial parse returns changes above threshold
        - Even if AI prompt block exists, fallback should NOT trigger
        """
        mock_provider = MagicMock(spec=LLMProvider)
        # Returns high confidence change
        mock_provider.generate.return_value = """[{
            "file_path": "src/test.py",
            "start_line": 50,
            "end_line": 52,
            "new_content": "# High confidence",
            "change_type": "modification",
            "confidence": 0.95,
            "rationale": "Clear instruction",
            "risk_level": "low"
        }]"""
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)

        # Comment with AI prompt block
        comment_with_ai_prompt = """
<details>
<summary>🤖 Prompt for AI Agents</summary>

In src/test.py around line 50, do something.

</details>
"""
        changes = parser.parse_comment(
            comment_with_ai_prompt, file_path="src/test.py", start_line=50, end_line=52
        )

        # Should have made only ONE LLM call (no fallback needed)
        assert mock_provider.generate.call_count == 1

        # Change was accepted
        assert len(changes) == 1
        assert changes[0].confidence == 0.95

    def test_fallback_resets_for_each_comment(self) -> None:
        """Test that fallback flag resets for each new parse_comment call.

        Scenario:
        - First comment: triggers fallback
        - Second comment: should also be able to trigger fallback (not blocked)
        """
        mock_provider = MagicMock(spec=LLMProvider)
        # All calls return low confidence to trigger fallback
        mock_provider.generate.return_value = """[{
            "file_path": "src/test.py",
            "start_line": 50,
            "end_line": 52,
            "new_content": "# Low confidence",
            "change_type": "modification",
            "confidence": 0.3,
            "rationale": "Unclear",
            "risk_level": "low"
        }]"""
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)

        comment_with_ai_prompt = """
<details>
<summary>🤖 Prompt for AI Agents</summary>

Some AI instruction.

</details>
"""
        # First comment
        parser.parse_comment(
            comment_with_ai_prompt, file_path="src/test.py", start_line=50, end_line=52
        )

        # Second comment
        parser.parse_comment(
            comment_with_ai_prompt, file_path="src/other.py", start_line=10, end_line=15
        )

        # Each comment should trigger initial + fallback = 2 calls
        # Total: 2 comments * 2 calls = 4 calls
        assert mock_provider.generate.call_count == 4

    def test_fallback_handles_invalid_json_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that fallback handles invalid JSON response gracefully.

        Scenario:
        - Initial parse returns low confidence
        - Fallback parse returns invalid JSON
        - Should not crash, return empty list
        """
        import logging

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.side_effect = [
            # First parse - low confidence
            """[{
                "file_path": "src/test.py",
                "start_line": 50,
                "end_line": 52,
                "new_content": "# Low confidence",
                "change_type": "modification",
                "confidence": 0.3,
                "rationale": "Unclear",
                "risk_level": "low"
            }]""",
            # Fallback parse - invalid JSON
            "not valid json {{",
        ]
        mock_provider.get_total_cost.return_value = 0.0

        parser = UniversalLLMParser(mock_provider, confidence_threshold=0.5)

        comment_with_ai_prompt = """
<details>
<summary>🤖 Prompt for AI Agents</summary>

Some instruction.

</details>
"""
        with caplog.at_level(logging.WARNING):
            changes = parser.parse_comment(
                comment_with_ai_prompt, file_path="src/test.py", start_line=50, end_line=52
            )

        # Should have made two calls
        assert mock_provider.generate.call_count == 2

        # Should return empty list (no valid changes)
        assert len(changes) == 0

        # Should have logged warning about invalid JSON
        assert any("invalid JSON" in record.message for record in caplog.records)

    def test_fallback_tracks_cost(self) -> None:
        """Test that fallback LLM call cost is tracked."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate.side_effect = [
            # First parse - low confidence
            """[{
                "file_path": "src/test.py",
                "start_line": 50,
                "end_line": 52,
                "new_content": "# Low confidence",
                "change_type": "modification",
                "confidence": 0.3,
                "rationale": "Unclear",
                "risk_level": "low"
            }]""",
            # Fallback parse - high confidence
            """[{
                "file_path": "src/test.py",
                "start_line": 50,
                "end_line": 52,
                "new_content": "renamed()",
                "change_type": "modification",
                "confidence": 0.95,
                "rationale": "Renamed",
                "risk_level": "low"
            }]""",
        ]
        # Costs: initial 0 -> 0.01, fallback 0.01 -> 0.02
        mock_provider.get_total_cost.side_effect = [0.0, 0.01, 0.01, 0.02]

        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.should_block_request.return_value = False
        cost_tracker.add_cost.return_value = CostStatus.OK

        parser = UniversalLLMParser(mock_provider, cost_tracker=cost_tracker)

        comment_with_ai_prompt = """
<details>
<summary>🤖 Prompt for AI Agents</summary>

Rename the function.

</details>
"""
        parser.parse_comment(
            comment_with_ai_prompt, file_path="src/test.py", start_line=50, end_line=52
        )

        # Cost tracker should have been called twice (initial + fallback)
        assert cost_tracker.add_cost.call_count == 2

        # First call: initial parse cost (0.01 - 0.0 = 0.01)
        assert cost_tracker.add_cost.call_args_list[0][0][0] == 0.01

        # Second call: fallback parse cost (0.02 - 0.01 = 0.01)
        assert cost_tracker.add_cost.call_args_list[1][0][0] == 0.01
