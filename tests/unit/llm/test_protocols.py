"""Tests for LLM base protocols and abstract classes.

This module tests that the provider protocol and parser ABC are correctly
defined and enforce the expected interface contracts.
"""

import pytest

from review_bot_automator.llm.base import LLMParser, ParsedChange
from review_bot_automator.llm.providers.base import LLMProvider


class TestLLMProviderProtocol:
    """Test LLMProvider protocol definition and conformance checking."""

    def test_protocol_has_generate_method(self) -> None:
        """Test that LLMProvider protocol requires generate() method."""
        # Protocol should have generate in its interface
        assert hasattr(LLMProvider, "generate")

    def test_protocol_has_count_tokens_method(self) -> None:
        """Test that LLMProvider protocol requires count_tokens() method."""
        assert hasattr(LLMProvider, "count_tokens")

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that LLMProvider can be checked at runtime with isinstance."""

        class ValidProvider:
            """Minimal valid provider for testing."""

            def generate(self, prompt: str, max_tokens: int = 2000) -> str:
                return "response"

            def count_tokens(self, text: str) -> int:
                return len(text.split())

            def get_total_cost(self) -> float:
                return 0.0

        provider = ValidProvider()
        assert isinstance(provider, LLMProvider)

    def test_protocol_rejects_incomplete_implementation(self) -> None:
        """Test that incomplete provider doesn't conform to protocol."""

        class IncompleteProvider:
            """Provider missing count_tokens."""

            def generate(self, prompt: str, max_tokens: int = 2000) -> str:
                return "response"

        provider = IncompleteProvider()
        assert not isinstance(provider, LLMProvider)

    def test_protocol_allows_additional_methods(self) -> None:
        """Test that providers can have extra methods beyond protocol."""

        class ExtendedProvider:
            """Provider with additional methods."""

            def generate(self, prompt: str, max_tokens: int = 2000) -> str:
                return "response"

            def count_tokens(self, text: str) -> int:
                return len(text.split())

            def get_total_cost(self) -> float:
                return 0.0

            def get_cost_estimate(self) -> float:
                """Extra method not in protocol."""
                return 0.01

        provider = ExtendedProvider()
        assert isinstance(provider, LLMProvider)
        assert hasattr(provider, "get_cost_estimate")


class TestLLMParserABC:
    """Test LLMParser abstract base class."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """Test that LLMParser cannot be instantiated without implementation."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMParser()  # type: ignore[abstract]

    def test_abc_requires_parse_comment_implementation(self) -> None:
        """Test that subclass must implement parse_comment()."""

        class IncompleteParser(LLMParser):
            """Parser missing parse_comment implementation."""

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteParser()  # type: ignore[abstract]

    def test_valid_parser_implementation(self) -> None:
        """Test that complete parser implementation can be instantiated."""

        class ValidParser(LLMParser):
            """Minimal valid parser."""

            def parse_comment(
                self,
                comment_body: str,
                file_path: str | None = None,
                line_number: int | None = None,
                *,
                start_line: int | None = None,
                end_line: int | None = None,
            ) -> list[ParsedChange]:
                return []

        parser = ValidParser()
        assert isinstance(parser, LLMParser)
        result = parser.parse_comment("test comment")
        assert result == []

    def test_parser_can_have_additional_methods(self) -> None:
        """Test that parser implementations can add extra methods."""

        class ExtendedParser(LLMParser):
            """Parser with additional methods."""

            def parse_comment(
                self,
                comment_body: str,
                file_path: str | None = None,
                line_number: int | None = None,
                *,
                start_line: int | None = None,
                end_line: int | None = None,
            ) -> list[ParsedChange]:
                return []

            def reset_state(self) -> None:
                """Extra method not in ABC."""

        parser = ExtendedParser()
        assert isinstance(parser, LLMParser)
        assert hasattr(parser, "reset_state")
        parser.reset_state()  # Should not raise


class TestParsedChangeValidation:
    """Test ParsedChange dataclass validation."""

    def test_valid_parsed_change(self) -> None:
        """Test creating a valid ParsedChange."""
        change = ParsedChange(
            file_path="test.py",
            start_line=10,
            end_line=15,
            new_content="def foo(): pass",
            change_type="modification",
            confidence=0.95,
            rationale="Refactor for clarity",
            risk_level="low",
        )
        assert change.file_path == "test.py"
        assert change.start_line == 10
        assert change.end_line == 15
        assert change.confidence == 0.95

    def test_start_line_validation(self) -> None:
        """Test that start_line must be >= 1."""
        with pytest.raises(ValueError, match="start_line must be >= 1"):
            ParsedChange(
                file_path="test.py",
                start_line=0,  # Invalid
                end_line=5,
                new_content="code",
                change_type="modification",
                confidence=0.9,
                rationale="Test",
            )

    def test_end_line_validation(self) -> None:
        """Test that end_line must be >= start_line."""
        with pytest.raises(ValueError, match="end_line.*must be >= start_line"):
            ParsedChange(
                file_path="test.py",
                start_line=10,
                end_line=5,  # Invalid: less than start_line
                new_content="code",
                change_type="modification",
                confidence=0.9,
                rationale="Test",
            )

    def test_confidence_range_validation(self) -> None:
        """Test that confidence must be in [0.0, 1.0]."""
        with pytest.raises(ValueError, match="confidence must be in"):
            ParsedChange(
                file_path="test.py",
                start_line=1,
                end_line=5,
                new_content="code",
                change_type="modification",
                confidence=1.5,  # Invalid: > 1.0
                rationale="Test",
            )

    def test_change_type_validation(self) -> None:
        """Test that change_type must be valid enum value."""
        with pytest.raises(ValueError, match="change_type must be"):
            ParsedChange(
                file_path="test.py",
                start_line=1,
                end_line=5,
                new_content="code",
                change_type="invalid_type",  # Invalid
                confidence=0.9,
                rationale="Test",
            )

    def test_risk_level_validation(self) -> None:
        """Test that risk_level must be valid enum value."""
        with pytest.raises(ValueError, match="risk_level must be"):
            ParsedChange(
                file_path="test.py",
                start_line=1,
                end_line=5,
                new_content="code",
                change_type="modification",
                confidence=0.9,
                rationale="Test",
                risk_level="critical",  # Invalid
            )

    def test_default_risk_level(self) -> None:
        """Test that risk_level defaults to 'low'."""
        change = ParsedChange(
            file_path="test.py",
            start_line=1,
            end_line=5,
            new_content="code",
            change_type="modification",
            confidence=0.9,
            rationale="Test",
            # risk_level not specified
        )
        assert change.risk_level == "low"

    def test_parsed_change_is_immutable(self) -> None:
        """Test that ParsedChange is frozen (immutable)."""
        change = ParsedChange(
            file_path="test.py",
            start_line=1,
            end_line=5,
            new_content="code",
            change_type="modification",
            confidence=0.9,
            rationale="Test",
        )
        with pytest.raises(AttributeError):
            change.confidence = 0.8  # type: ignore[misc]
