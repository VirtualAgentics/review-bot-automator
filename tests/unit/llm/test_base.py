"""Unit tests for LLM base data structures.

Tests the ParsedChange dataclass for Phase 0 (Foundation).
"""

import pytest

from review_bot_automator.llm.base import ParsedChange


class TestParsedChangeValid:
    """Test ParsedChange with valid inputs."""

    def test_valid_modification(self) -> None:
        """Test creating a valid ParsedChange for modification."""
        change = ParsedChange(
            file_path="src/example.py",
            start_line=10,
            end_line=12,
            new_content="def new_function():\n    pass",
            change_type="modification",
            confidence=0.95,
            rationale="Replace deprecated API",
            risk_level="low",
        )

        assert change.file_path == "src/example.py"
        assert change.start_line == 10
        assert change.end_line == 12
        assert change.new_content == "def new_function():\n    pass"
        assert change.change_type == "modification"
        assert change.confidence == 0.95
        assert change.rationale == "Replace deprecated API"
        assert change.risk_level == "low"

    def test_valid_addition(self) -> None:
        """Test creating a valid ParsedChange for addition."""
        change = ParsedChange(
            file_path="src/new.py",
            start_line=1,
            end_line=1,
            new_content="# New file content",
            change_type="addition",
            confidence=0.80,
            rationale="Add new functionality",
            risk_level="medium",
        )

        assert change.change_type == "addition"
        assert change.risk_level == "medium"

    def test_valid_deletion(self) -> None:
        """Test creating a valid ParsedChange for deletion."""
        change = ParsedChange(
            file_path="src/deprecated.py",
            start_line=50,
            end_line=100,
            new_content="",
            change_type="deletion",
            confidence=0.65,
            rationale="Remove deprecated code",
            risk_level="high",
        )

        assert change.change_type == "deletion"
        assert change.new_content == ""
        assert change.risk_level == "high"

    def test_default_risk_level(self) -> None:
        """Test that risk_level defaults to 'low'."""
        change = ParsedChange(
            file_path="src/safe.py",
            start_line=1,
            end_line=1,
            new_content="# Safe change",
            change_type="modification",
            confidence=0.99,
            rationale="Safe formatting change",
        )

        assert change.risk_level == "low"

    def test_confidence_at_boundaries(self) -> None:
        """Test confidence at 0.0 and 1.0 boundaries."""
        change_min = ParsedChange(
            file_path="src/test.py",
            start_line=1,
            end_line=1,
            new_content="# Low confidence",
            change_type="modification",
            confidence=0.0,
            rationale="Very uncertain",
        )
        assert change_min.confidence == 0.0

        change_max = ParsedChange(
            file_path="src/test.py",
            start_line=1,
            end_line=1,
            new_content="# High confidence",
            change_type="modification",
            confidence=1.0,
            rationale="Completely certain",
        )
        assert change_max.confidence == 1.0


class TestParsedChangeValidation:
    """Test ParsedChange validation in __post_init__."""

    def test_start_line_less_than_one_raises_error(self) -> None:
        """Test that start_line < 1 raises ValueError."""
        with pytest.raises(ValueError, match="start_line must be >= 1"):
            ParsedChange(
                file_path="src/test.py",
                start_line=0,
                end_line=5,
                new_content="# Invalid",
                change_type="modification",
                confidence=0.9,
                rationale="Invalid line number",
            )

    def test_end_line_less_than_start_line_raises_error(self) -> None:
        """Test that end_line < start_line raises ValueError."""
        with pytest.raises(ValueError, match="end_line .* must be >= start_line"):
            ParsedChange(
                file_path="src/test.py",
                start_line=10,
                end_line=5,
                new_content="# Invalid",
                change_type="modification",
                confidence=0.9,
                rationale="Invalid line range",
            )

    def test_confidence_below_zero_raises_error(self) -> None:
        """Test that confidence < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be in \\[0.0, 1.0\\]"):
            ParsedChange(
                file_path="src/test.py",
                start_line=1,
                end_line=1,
                new_content="# Invalid",
                change_type="modification",
                confidence=-0.1,
                rationale="Negative confidence",
            )

    def test_confidence_above_one_raises_error(self) -> None:
        """Test that confidence > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be in \\[0.0, 1.0\\]"):
            ParsedChange(
                file_path="src/test.py",
                start_line=1,
                end_line=1,
                new_content="# Invalid",
                change_type="modification",
                confidence=1.5,
                rationale="Confidence over 100%",
            )

    def test_invalid_change_type_raises_error(self) -> None:
        """Test that invalid change_type raises ValueError."""
        with pytest.raises(ValueError, match="change_type must be"):
            ParsedChange(
                file_path="src/test.py",
                start_line=1,
                end_line=1,
                new_content="# Invalid",
                change_type="invalid_type",  # type: ignore[arg-type]
                confidence=0.9,
                rationale="Invalid change type",
            )

    def test_invalid_risk_level_raises_error(self) -> None:
        """Test that invalid risk_level raises ValueError."""
        with pytest.raises(ValueError, match="risk_level must be"):
            ParsedChange(
                file_path="src/test.py",
                start_line=1,
                end_line=1,
                new_content="# Invalid",
                change_type="modification",
                confidence=0.9,
                rationale="Invalid risk level",
                risk_level="critical",
            )


class TestParsedChangeImmutability:
    """Test that ParsedChange is immutable (frozen=True)."""

    def test_cannot_modify_file_path(self) -> None:
        """Test that file_path cannot be modified."""
        change = ParsedChange(
            file_path="src/test.py",
            start_line=1,
            end_line=1,
            new_content="# Test",
            change_type="modification",
            confidence=0.9,
            rationale="Test change",
        )

        with pytest.raises((AttributeError, TypeError)):
            change.file_path = "src/other.py"  # type: ignore[misc]

    def test_cannot_modify_confidence(self) -> None:
        """Test that confidence cannot be modified."""
        change = ParsedChange(
            file_path="src/test.py",
            start_line=1,
            end_line=1,
            new_content="# Test",
            change_type="modification",
            confidence=0.9,
            rationale="Test change",
        )

        with pytest.raises((AttributeError, TypeError)):
            change.confidence = 0.5  # type: ignore[misc]


class TestParsedChangeEquality:
    """Test ParsedChange equality comparison."""

    def test_identical_changes_are_equal(self) -> None:
        """Test that two identical ParsedChanges are equal."""
        change1 = ParsedChange(
            file_path="src/test.py",
            start_line=1,
            end_line=1,
            new_content="# Test",
            change_type="modification",
            confidence=0.9,
            rationale="Test change",
            risk_level="low",
        )
        change2 = ParsedChange(
            file_path="src/test.py",
            start_line=1,
            end_line=1,
            new_content="# Test",
            change_type="modification",
            confidence=0.9,
            rationale="Test change",
            risk_level="low",
        )

        assert change1 == change2

    def test_different_confidence_not_equal(self) -> None:
        """Test that changes with different confidence are not equal."""
        change1 = ParsedChange(
            file_path="src/test.py",
            start_line=1,
            end_line=1,
            new_content="# Test",
            change_type="modification",
            confidence=0.9,
            rationale="Test change",
        )
        change2 = ParsedChange(
            file_path="src/test.py",
            start_line=1,
            end_line=1,
            new_content="# Test",
            change_type="modification",
            confidence=0.8,
            rationale="Test change",
        )

        assert change1 != change2
