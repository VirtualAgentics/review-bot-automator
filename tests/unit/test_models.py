"""Unit tests for data models in review_bot_automator.core.models."""

from dataclasses import is_dataclass

import pytest

from review_bot_automator.core.models import (
    Change,
    ChangeType,
    Conflict,
    FileType,
    Resolution,
    ResolutionResult,
)


def test_filetype_enum_members() -> None:
    """Ensure expected FileType enum members exist."""
    members = {m.name for m in FileType}
    assert {"PYTHON", "TYPESCRIPT", "JSON", "YAML", "TOML", "PLAINTEXT"} <= members


def test_change_dataclass_fields_and_equality() -> None:
    """Validate Change dataclass structure and equality semantics."""
    assert is_dataclass(Change)

    c1 = Change(
        path="a.json",
        start_line=1,
        end_line=3,
        content='{"k":"v"}',
        metadata={"author": "bot"},
        fingerprint="fp1",
        file_type=FileType.JSON,
    )
    c2 = Change(
        path="a.json",
        start_line=1,
        end_line=3,
        content='{"k":"v"}',
        metadata={"author": "bot"},
        fingerprint="fp1",
        file_type=FileType.JSON,
    )
    c3 = Change(
        path="a.json",
        start_line=1,
        end_line=3,
        content='{"k":"v2"}',
        metadata={"author": "bot"},
        fingerprint="fp2",
        file_type=FileType.JSON,
    )

    assert c1 == c2
    assert c1 != c3
    assert c1.file_type is FileType.JSON
    assert isinstance(c1.metadata, dict)


def test_conflict_dataclass() -> None:
    """Validate Conflict dataclass creation and fields."""
    ch = Change(
        path="file.yaml",
        start_line=10,
        end_line=12,
        content="name: test",
        metadata={},
        fingerprint="abc",
        file_type=FileType.YAML,
    )
    conflict = Conflict(
        file_path="file.yaml",
        line_range=(10, 12),
        changes=[ch],
        conflict_type="partial",
        severity="low",
        overlap_percentage=33.3,
    )

    assert is_dataclass(Conflict)
    assert conflict.file_path == "file.yaml"
    assert conflict.line_range == (10, 12)
    assert conflict.changes and conflict.changes[0] == ch
    assert conflict.conflict_type in {
        "exact",
        "major",
        "partial",
        "multiple",
        "key_conflict",
        "section_conflict",
    }
    assert conflict.severity in {"low", "medium", "high"}
    assert 0.0 <= conflict.overlap_percentage <= 100.0


def test_resolution_and_result_dataclasses() -> None:
    """Validate Resolution and ResolutionResult containers."""
    ch_applied = Change(
        path="config.toml",
        start_line=2,
        end_line=4,
        content='[tool]\nname="x"',
        metadata={},
        fingerprint="fp3",
        file_type=FileType.TOML,
    )
    res = Resolution(
        strategy="priority",
        applied_changes=[ch_applied],
        skipped_changes=[],
        success=True,
        message="ok",
    )
    result = ResolutionResult(
        applied_count=1,
        conflict_count=0,
        success_rate=100.0,
        resolutions=[res],
        conflicts=[],
    )

    assert is_dataclass(Resolution)
    assert is_dataclass(ResolutionResult)
    assert result.applied_count == 1
    assert result.conflict_count == 0
    assert result.success_rate == 100.0
    assert result.resolutions[0].success is True
    assert result.resolutions[0].strategy == "priority"


def test_change_with_llm_fields() -> None:
    """Test Change dataclass with new LLM fields (Phase 0)."""
    # Test with all LLM fields populated
    change_llm = Change(
        path="src/example.py",
        start_line=10,
        end_line=12,
        content="def new_function():\n    pass",
        metadata={"author": "coderabbit", "llm_confidence": 0.95, "parsing_method": "llm"},
        fingerprint="llm_fp1",
        file_type=FileType.PYTHON,
        llm_confidence=0.95,
        llm_provider="claude-cli",
        parsing_method="llm",
        change_rationale="Modernize API usage",
        risk_level="low",
    )

    assert change_llm.llm_confidence == 0.95
    assert change_llm.llm_provider == "claude-cli"
    assert change_llm.parsing_method == "llm"
    assert change_llm.change_rationale == "Modernize API usage"
    assert change_llm.risk_level == "low"


def test_change_backward_compatibility_with_defaults() -> None:
    """Test that Change works without LLM fields (backward compatibility)."""
    # Old code that doesn't know about LLM fields should still work
    change_old = Change(
        path="src/old.py",
        start_line=1,
        end_line=3,
        content="# Old code",
        metadata={"author": "human"},
        fingerprint="old_fp",
        file_type=FileType.PYTHON,
    )

    # LLM fields should have safe defaults
    assert change_old.llm_confidence is None
    assert change_old.llm_provider is None
    assert change_old.parsing_method == "regex"  # default
    assert change_old.change_rationale is None
    assert change_old.risk_level is None


def test_change_with_partial_llm_fields() -> None:
    """Test Change with only some LLM fields specified."""
    change_partial = Change(
        path="src/partial.py",
        start_line=5,
        end_line=10,
        content="# Partial LLM data",
        metadata={"author": "bot"},
        fingerprint="partial_fp",
        file_type=FileType.PYTHON,
        llm_confidence=0.80,
        parsing_method="llm",
        # llm_provider, change_rationale, risk_level not specified
    )

    assert change_partial.llm_confidence == 0.80
    assert change_partial.parsing_method == "llm"
    assert change_partial.llm_provider is None
    assert change_partial.change_rationale is None
    assert change_partial.risk_level is None


def test_change_metadata_with_llm_fields() -> None:
    """Test ChangeMetadata TypedDict with new LLM fields."""
    from review_bot_automator.core.models import ChangeMetadata

    # Type-checked metadata with LLM fields
    metadata: ChangeMetadata = {
        "author": "coderabbit",
        "url": "https://github.com/org/repo/pull/123",
        "llm_confidence": 0.92,
        "parsing_method": "llm",
    }

    change = Change(
        path="src/typed.py",
        start_line=1,
        end_line=1,
        content="# Typed",
        metadata=metadata,
        fingerprint="typed_fp",
        file_type=FileType.PYTHON,
    )

    assert change.metadata.get("llm_confidence") == 0.92
    assert change.metadata.get("parsing_method") == "llm"


def test_change_validation_invalid_confidence_too_low() -> None:
    """Test Change validation rejects llm_confidence < 0.0."""
    with pytest.raises(ValueError, match="llm_confidence must be between 0.0 and 1.0"):
        Change(
            path="test.py",
            start_line=1,
            end_line=1,
            content="# test",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PYTHON,
            llm_confidence=-0.1,
        )


def test_change_validation_invalid_confidence_too_high() -> None:
    """Test Change validation rejects llm_confidence > 1.0."""
    with pytest.raises(ValueError, match="llm_confidence must be between 0.0 and 1.0"):
        Change(
            path="test.py",
            start_line=1,
            end_line=1,
            content="# test",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PYTHON,
            llm_confidence=1.5,
        )


def test_change_validation_empty_provider() -> None:
    """Test Change validation rejects empty string llm_provider."""
    with pytest.raises(ValueError, match="llm_provider must not be empty string"):
        Change(
            path="test.py",
            start_line=1,
            end_line=1,
            content="# test",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PYTHON,
            llm_provider="",
        )


def test_change_validation_invalid_risk_level() -> None:
    """Test Change validation rejects invalid risk_level values."""
    with pytest.raises(ValueError, match="risk_level must be one of"):
        Change(
            path="test.py",
            start_line=1,
            end_line=1,
            content="# test",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PYTHON,
            risk_level="critical",  # Invalid - only low/medium/high allowed
        )


def test_change_validation_valid_edge_cases() -> None:
    """Test Change validation accepts valid edge case values."""
    # Confidence = 0.0 should be valid
    change_zero_conf = Change(
        path="test.py",
        start_line=1,
        end_line=1,
        content="# test",
        metadata={},
        fingerprint="fp1",
        file_type=FileType.PYTHON,
        llm_confidence=0.0,
    )
    assert change_zero_conf.llm_confidence == 0.0

    # Confidence = 1.0 should be valid
    change_perfect_conf = Change(
        path="test.py",
        start_line=1,
        end_line=1,
        content="# test",
        metadata={},
        fingerprint="fp2",
        file_type=FileType.PYTHON,
        llm_confidence=1.0,
    )
    assert change_perfect_conf.llm_confidence == 1.0

    # All valid risk levels
    for risk in ["low", "medium", "high"]:
        change_risk = Change(
            path="test.py",
            start_line=1,
            end_line=1,
            content="# test",
            metadata={},
            fingerprint=f"fp_{risk}",
            file_type=FileType.PYTHON,
            risk_level=risk,
        )
        assert change_risk.risk_level == risk


def test_change_validation_none_values_allowed() -> None:
    """Test Change validation allows None for optional LLM fields."""
    # None values should be valid for all optional LLM fields
    change = Change(
        path="test.py",
        start_line=1,
        end_line=1,
        content="# test",
        metadata={},
        fingerprint="fp1",
        file_type=FileType.PYTHON,
        llm_confidence=None,
        llm_provider=None,
        risk_level=None,
    )
    assert change.llm_confidence is None
    assert change.llm_provider is None
    assert change.risk_level is None


def test_change_type_default_is_modification() -> None:
    """Test Change defaults to change_type='modification' (Issue #312)."""
    change = Change(
        path="test.py",
        start_line=1,
        end_line=1,
        content="# test",
        metadata={},
        fingerprint="fp1",
        file_type=FileType.PYTHON,
    )
    assert change.change_type == "modification"


@pytest.mark.parametrize("change_type", ["addition", "modification", "deletion"])
def test_change_type_valid_values(change_type: ChangeType) -> None:
    """Test Change accepts all valid change_type values (Issue #312)."""
    change = Change(
        path="test.py",
        start_line=1,
        end_line=1,
        content="# test",
        metadata={},
        fingerprint=f"fp_{change_type}",
        file_type=FileType.PYTHON,
        change_type=change_type,
    )
    assert change.change_type == change_type


def test_change_type_invalid_value_raises() -> None:
    """Test Change validation rejects invalid change_type values (Issue #312)."""
    with pytest.raises(ValueError, match="change_type must be one of"):
        Change(
            path="test.py",
            start_line=1,
            end_line=1,
            content="# test",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PYTHON,
            change_type="replace",  # type: ignore[arg-type]  # Invalid - only addition/modification/deletion allowed
        )
