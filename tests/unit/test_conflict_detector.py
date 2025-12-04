"""Test conflict detection functionality."""

from review_bot_automator.analysis.conflict_detector import ConflictDetector
from review_bot_automator.core.models import Change, Conflict, FileType


class TestDetectOverlap:
    """Tests for detect_overlap method."""

    def test_detect_exact_overlap(self) -> None:
        """Test detection of exact line range overlaps."""
        detector = ConflictDetector()

        change1 = Change(
            path="test.py",
            start_line=10,
            end_line=15,
            content="content1",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        change2 = Change(
            path="test.py",
            start_line=10,
            end_line=15,
            content="content2",
            metadata={},
            fingerprint="test2",
            file_type=FileType.PYTHON,
        )

        assert detector.detect_overlap(change1, change2) == "exact"

    def test_detect_partial_overlap(self) -> None:
        """Test detection of partial line range overlaps."""
        detector = ConflictDetector()

        change1 = Change(
            path="test.py",
            start_line=10,
            end_line=15,
            content="content1",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        change2 = Change(
            path="test.py",
            start_line=12,
            end_line=18,
            content="content2",
            metadata={},
            fingerprint="test2",
            file_type=FileType.PYTHON,
        )

        overlap = detector.detect_overlap(change1, change2)
        assert overlap in ["major", "partial", "minor"]

    def test_detect_no_overlap(self) -> None:
        """Test detection when no overlap exists."""
        detector = ConflictDetector()

        change1 = Change(
            path="test.py",
            start_line=10,
            end_line=15,
            content="content1",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        change2 = Change(
            path="test.py",
            start_line=20,
            end_line=25,
            content="content2",
            metadata={},
            fingerprint="test2",
            file_type=FileType.PYTHON,
        )

        assert detector.detect_overlap(change1, change2) is None

    def test_detect_major_overlap(self) -> None:
        """Test detection of major overlap (>= 80%)."""
        detector = ConflictDetector()

        # Range 1-10 and 2-11: overlap 9, total 11, ~82%
        change1 = Change(
            path="test.py",
            start_line=1,
            end_line=10,
            content="content1",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        change2 = Change(
            path="test.py",
            start_line=2,
            end_line=11,
            content="content2",
            metadata={},
            fingerprint="test2",
            file_type=FileType.PYTHON,
        )

        assert detector.detect_overlap(change1, change2) == "major"

    def test_detect_minor_overlap(self) -> None:
        """Test detection of minor overlap (< 50%)."""
        detector = ConflictDetector()

        # Range 1-10 and 8-20: overlap 3, total 20, 15%
        change1 = Change(
            path="test.py",
            start_line=1,
            end_line=10,
            content="content1",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        change2 = Change(
            path="test.py",
            start_line=8,
            end_line=20,
            content="content2",
            metadata={},
            fingerprint="test2",
            file_type=FileType.PYTHON,
        )

        assert detector.detect_overlap(change1, change2) == "minor"


class TestIsSemanticDuplicate:
    """Tests for is_semantic_duplicate method."""

    def test_detect_semantic_duplicate_json(self) -> None:
        """Test detection of semantically identical JSON changes."""
        detector = ConflictDetector()

        change1 = Change(
            path="test.json",
            start_line=10,
            end_line=15,
            content='{"name": "test"}',
            metadata={},
            fingerprint="test1",
            file_type=FileType.JSON,
        )
        change2 = Change(
            path="test.json",
            start_line=10,
            end_line=15,
            content='{\n  "name": "test"\n}',
            metadata={},
            fingerprint="test2",
            file_type=FileType.JSON,
        )

        assert detector.is_semantic_duplicate(change1, change2) is True

    def test_not_semantic_duplicate_different_content(self) -> None:
        """Test non-duplicates with different content."""
        detector = ConflictDetector()

        change1 = Change(
            path="test.py",
            start_line=10,
            end_line=15,
            content="def foo(): pass",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        change2 = Change(
            path="test.py",
            start_line=10,
            end_line=15,
            content="def bar(): pass",
            metadata={},
            fingerprint="test2",
            file_type=FileType.PYTHON,
        )

        assert detector.is_semantic_duplicate(change1, change2) is False


class TestIsStructuredContent:
    """Tests for _is_structured_content method."""

    def test_json_object_detected(self) -> None:
        """Test JSON object detection."""
        detector = ConflictDetector()
        assert detector._is_structured_content('{"key": "value"}') is True

    def test_json_array_detected(self) -> None:
        """Test JSON array detection."""
        detector = ConflictDetector()
        assert detector._is_structured_content("[1, 2, 3]") is True

    def test_yaml_detected(self) -> None:
        """Test YAML detection with colon and dash."""
        detector = ConflictDetector()
        assert detector._is_structured_content("key: value\n- item") is True

    def test_yaml_with_pipe_detected(self) -> None:
        """Test YAML detection with colon and pipe."""
        detector = ConflictDetector()
        assert detector._is_structured_content("data: |\n  multi\n  line") is True

    def test_plain_text_not_structured(self) -> None:
        """Test plain text is not detected as structured."""
        detector = ConflictDetector()
        assert detector._is_structured_content("Just plain text") is False


class TestCompareStructuredContent:
    """Tests for _compare_structured_content method."""

    def test_json_match(self) -> None:
        """Test matching JSON content."""
        detector = ConflictDetector()
        result = detector._compare_structured_content(
            '{"name": "test", "value": 1}', '{"value": 1, "name": "test"}'
        )
        assert result is True

    def test_json_mismatch(self) -> None:
        """Test non-matching JSON content."""
        detector = ConflictDetector()
        result = detector._compare_structured_content('{"name": "test1"}', '{"name": "test2"}')
        assert result is False

    def test_yaml_match(self) -> None:
        """Test matching YAML content."""
        detector = ConflictDetector()
        result = detector._compare_structured_content(
            "name: test\nvalue: 1", "name: test\nvalue: 1"
        )
        assert result is True

    def test_yaml_mismatch(self) -> None:
        """Test non-matching YAML content."""
        detector = ConflictDetector()
        result = detector._compare_structured_content("name: test1", "name: test2")
        assert result is False

    def test_invalid_both_returns_false(self) -> None:
        """Test that invalid content for both JSON and YAML returns False."""
        detector = ConflictDetector()
        # Content that fails both JSON and YAML parsing
        result = detector._compare_structured_content(
            "not valid {json or yaml:", "also not valid {json or yaml:"
        )
        assert result is False

    def test_nested_json_match(self) -> None:
        """Test matching nested JSON structures."""
        detector = ConflictDetector()
        result = detector._compare_structured_content(
            '{"outer": {"inner": [1, 2, 3]}}', '{"outer": {"inner": [1, 2, 3]}}'
        )
        assert result is True


class TestAnalyzeConflictImpact:
    """Tests for analyze_conflict_impact method."""

    def test_empty_changes(self) -> None:
        """Test impact analysis with empty changes."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact({"changes": []})
        assert result == {"impact": "none", "severity": "low"}

    def test_no_changes_key(self) -> None:
        """Test impact analysis with missing changes key."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact({})
        assert result == {"impact": "none", "severity": "low"}

    def test_security_keywords_detected(self) -> None:
        """Test detection of security-related keywords."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact(
            {"changes": [{"content": "Update authentication token handling"}]}
        )
        assert result["security_related"] is True
        assert result["impact"] == "high"
        assert result["severity"] == "critical"

    def test_syntax_keywords_detected(self) -> None:
        """Test detection of syntax/bug-related keywords."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact(
            {"changes": [{"content": "Fix syntax error in parser"}]}
        )
        assert result["syntax_related"] is True
        assert result["impact"] == "medium"
        assert result["severity"] == "high"

    def test_code_block_change_type(self) -> None:
        """Test code block change type detection."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact(
            {"changes": [{"content": "```python\ndef foo(): pass\n```"}]}
        )
        assert "code_block" in result["change_types"]

    def test_diff_format_change_type(self) -> None:
        """Test diff format change type detection."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact({"changes": [{"content": "+added line"}]})
        assert "diff" in result["change_types"]

    def test_diff_format_minus_change_type(self) -> None:
        """Test diff format with minus prefix."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact({"changes": [{"content": "-removed line"}]})
        assert "diff" in result["change_types"]

    def test_text_change_type(self) -> None:
        """Test plain text change type detection."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact(
            {"changes": [{"content": "Just a plain text comment"}]}
        )
        assert "text" in result["change_types"]

    def test_multiple_changes_medium_impact(self) -> None:
        """Test that more than 2 changes results in medium impact."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact(
            {
                "changes": [
                    {"content": "change 1"},
                    {"content": "change 2"},
                    {"content": "change 3"},
                ]
            }
        )
        assert result["impact"] == "medium"
        assert result["severity"] == "medium"
        assert result["change_count"] == 3

    def test_single_change_low_impact(self) -> None:
        """Test that single non-security/syntax change has low impact."""
        detector = ConflictDetector()
        result = detector.analyze_conflict_impact(
            {"changes": [{"content": "Update documentation"}]}
        )
        assert result["impact"] == "low"
        assert result["severity"] == "low"

    def test_all_security_keywords(self) -> None:
        """Test all security keywords are detected."""
        detector = ConflictDetector()
        keywords = ["security", "vulnerability", "auth", "token", "key", "password"]
        for keyword in keywords:
            result = detector.analyze_conflict_impact(
                {"changes": [{"content": f"Contains {keyword} info"}]}
            )
            assert result["security_related"] is True, f"Keyword '{keyword}' not detected"

    def test_all_syntax_keywords(self) -> None:
        """Test all syntax keywords are detected."""
        detector = ConflictDetector()
        keywords = ["error", "fix", "bug", "issue", "syntax"]
        for keyword in keywords:
            result = detector.analyze_conflict_impact(
                {"changes": [{"content": f"Contains {keyword} info"}]}
            )
            assert result["syntax_related"] is True, f"Keyword '{keyword}' not detected"


class TestGenerateConflictFingerprint:
    """Tests for generate_conflict_fingerprint method."""

    def test_empty_changes_returns_empty(self) -> None:
        """Test fingerprint for conflict with no changes."""
        detector = ConflictDetector()
        result = detector.generate_conflict_fingerprint({"changes": []})
        assert result == ""

    def test_missing_changes_key_returns_empty(self) -> None:
        """Test fingerprint for conflict with missing changes key."""
        detector = ConflictDetector()
        result = detector.generate_conflict_fingerprint({})
        assert result == ""

    def test_single_change_fingerprint(self) -> None:
        """Test fingerprint generation for single change."""
        detector = ConflictDetector()
        result = detector.generate_conflict_fingerprint(
            {"changes": [{"path": "test.py", "start_line": 1, "end_line": 10, "content": "code"}]}
        )
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_multiple_changes_fingerprint(self) -> None:
        """Test fingerprint generation for multiple changes."""
        detector = ConflictDetector()
        result = detector.generate_conflict_fingerprint(
            {
                "changes": [
                    {"path": "test.py", "start_line": 1, "end_line": 10, "content": "code1"},
                    {"path": "test.py", "start_line": 20, "end_line": 30, "content": "code2"},
                ]
            }
        )
        assert len(result) == 16

    def test_fingerprint_order_independence(self) -> None:
        """Test that fingerprint is independent of change order."""
        detector = ConflictDetector()
        change1 = {"path": "a.py", "start_line": 1, "end_line": 10, "content": "code1"}
        change2 = {"path": "b.py", "start_line": 20, "end_line": 30, "content": "code2"}

        fp1 = detector.generate_conflict_fingerprint({"changes": [change1, change2]})
        fp2 = detector.generate_conflict_fingerprint({"changes": [change2, change1]})

        assert fp1 == fp2

    def test_different_changes_different_fingerprint(self) -> None:
        """Test that different changes produce different fingerprints."""
        detector = ConflictDetector()
        fp1 = detector.generate_conflict_fingerprint(
            {"changes": [{"path": "test.py", "content": "code1"}]}
        )
        fp2 = detector.generate_conflict_fingerprint(
            {"changes": [{"path": "test.py", "content": "code2"}]}
        )
        assert fp1 != fp2


class TestGenerateChangeFingerprint:
    """Tests for _generate_change_fingerprint method."""

    def test_full_change_fingerprint(self) -> None:
        """Test fingerprint for change with all fields."""
        detector = ConflictDetector()
        change = {
            "path": "test.py",
            "start_line": 10,
            "end_line": 20,
            "content": "def foo(): pass",
        }
        result = detector._generate_change_fingerprint(change)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_partial_change_fingerprint(self) -> None:
        """Test fingerprint for change with missing optional fields."""
        detector = ConflictDetector()
        change = {"content": "code"}
        result = detector._generate_change_fingerprint(change)
        assert len(result) == 16

    def test_empty_change_fingerprint(self) -> None:
        """Test fingerprint for empty change dict."""
        detector = ConflictDetector()
        result = detector._generate_change_fingerprint({})
        assert len(result) == 16

    def test_consistent_fingerprint(self) -> None:
        """Test that same change produces same fingerprint."""
        detector = ConflictDetector()
        change = {"path": "test.py", "content": "code"}
        fp1 = detector._generate_change_fingerprint(change)
        fp2 = detector._generate_change_fingerprint(change)
        assert fp1 == fp2


class TestDetectConflictPatterns:
    """Tests for detect_conflict_patterns method."""

    def test_empty_conflicts_list(self) -> None:
        """Test pattern detection with empty conflicts list."""
        detector = ConflictDetector()
        result = detector.detect_conflict_patterns([])
        assert result["total_conflicts"] == 0
        assert result["file_conflicts"] == {}
        assert result["conflict_types"] == {}
        assert result["severity_distribution"] == {}
        assert result["common_patterns"] == []

    def test_single_conflict(self) -> None:
        """Test pattern detection with single conflict."""
        detector = ConflictDetector()
        change = Change(
            path="test.py",
            start_line=1,
            end_line=10,
            content="code",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        conflict = Conflict(
            file_path="test.py",
            line_range=(1, 10),
            changes=[change],
            conflict_type="exact",
            severity="high",
            overlap_percentage=100.0,
        )
        result = detector.detect_conflict_patterns([conflict])
        assert result["total_conflicts"] == 1
        assert result["file_conflicts"]["test.py"] == 1
        assert result["conflict_types"]["exact"] == 1
        assert result["severity_distribution"]["high"] == 1

    def test_multiple_files_conflicts(self) -> None:
        """Test pattern detection with conflicts in multiple files."""
        detector = ConflictDetector()
        change1 = Change(
            path="file1.py",
            start_line=1,
            end_line=10,
            content="code1",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        change2 = Change(
            path="file2.py",
            start_line=1,
            end_line=10,
            content="code2",
            metadata={},
            fingerprint="test2",
            file_type=FileType.PYTHON,
        )
        conflict1 = Conflict(
            file_path="file1.py",
            line_range=(1, 10),
            changes=[change1],
            conflict_type="exact",
            severity="high",
            overlap_percentage=100.0,
        )
        conflict2 = Conflict(
            file_path="file2.py",
            line_range=(1, 10),
            changes=[change2],
            conflict_type="partial",
            severity="medium",
            overlap_percentage=60.0,
        )
        result = detector.detect_conflict_patterns([conflict1, conflict2])
        assert result["total_conflicts"] == 2
        assert result["file_conflicts"]["file1.py"] == 1
        assert result["file_conflicts"]["file2.py"] == 1
        assert result["conflict_types"]["exact"] == 1
        assert result["conflict_types"]["partial"] == 1

    def test_high_exact_overlap_pattern(self) -> None:
        """Test detection of high_exact_overlap pattern (>50% exact)."""
        detector = ConflictDetector()
        conflicts = []
        for i in range(3):
            change = Change(
                path=f"file{i}.py",
                start_line=1,
                end_line=10,
                content="code",
                metadata={},
                fingerprint=f"test{i}",
                file_type=FileType.PYTHON,
            )
            conflicts.append(
                Conflict(
                    file_path=f"file{i}.py",
                    line_range=(1, 10),
                    changes=[change],
                    conflict_type="exact",
                    severity="low",
                    overlap_percentage=100.0,
                )
            )
        result = detector.detect_conflict_patterns(conflicts)
        assert "high_exact_overlap" in result["common_patterns"]

    def test_high_severity_conflicts_pattern(self) -> None:
        """Test detection of high_severity_conflicts pattern (>30% high severity)."""
        detector = ConflictDetector()
        conflicts = []
        # 2 high severity out of 3 (67% > 30%)
        severities = ["high", "high", "low"]
        for i, severity in enumerate(severities):
            change = Change(
                path=f"file{i}.py",
                start_line=1,
                end_line=10,
                content="code",
                metadata={},
                fingerprint=f"test{i}",
                file_type=FileType.PYTHON,
            )
            conflicts.append(
                Conflict(
                    file_path=f"file{i}.py",
                    line_range=(1, 10),
                    changes=[change],
                    conflict_type="partial",
                    severity=severity,
                    overlap_percentage=60.0,
                )
            )
        result = detector.detect_conflict_patterns(conflicts)
        assert "high_severity_conflicts" in result["common_patterns"]

    def test_no_patterns_when_below_threshold(self) -> None:
        """Test no patterns detected when thresholds not met."""
        detector = ConflictDetector()
        conflicts = []
        # 1 exact out of 3 (33% < 50%), 1 high out of 3 (33% > 30% but close)
        types_severities = [
            ("exact", "low"),
            ("partial", "low"),
            ("minor", "high"),
        ]
        for i, (ctype, severity) in enumerate(types_severities):
            change = Change(
                path=f"file{i}.py",
                start_line=1,
                end_line=10,
                content="code",
                metadata={},
                fingerprint=f"test{i}",
                file_type=FileType.PYTHON,
            )
            conflicts.append(
                Conflict(
                    file_path=f"file{i}.py",
                    line_range=(1, 10),
                    changes=[change],
                    conflict_type=ctype,
                    severity=severity,
                    overlap_percentage=60.0,
                )
            )
        result = detector.detect_conflict_patterns(conflicts)
        # high_exact_overlap not in patterns (33% < 50%)
        assert "high_exact_overlap" not in result["common_patterns"]
        # high_severity_conflicts is in patterns (33% > 30%)
        assert "high_severity_conflicts" in result["common_patterns"]

    def test_same_file_multiple_conflicts(self) -> None:
        """Test counting multiple conflicts in the same file."""
        detector = ConflictDetector()
        change1 = Change(
            path="test.py",
            start_line=1,
            end_line=10,
            content="code1",
            metadata={},
            fingerprint="test1",
            file_type=FileType.PYTHON,
        )
        change2 = Change(
            path="test.py",
            start_line=20,
            end_line=30,
            content="code2",
            metadata={},
            fingerprint="test2",
            file_type=FileType.PYTHON,
        )
        conflict1 = Conflict(
            file_path="test.py",
            line_range=(1, 10),
            changes=[change1],
            conflict_type="exact",
            severity="high",
            overlap_percentage=100.0,
        )
        conflict2 = Conflict(
            file_path="test.py",
            line_range=(20, 30),
            changes=[change2],
            conflict_type="exact",
            severity="high",
            overlap_percentage=100.0,
        )
        result = detector.detect_conflict_patterns([conflict1, conflict2])
        assert result["file_conflicts"]["test.py"] == 2


class TestConflictDetectorCache:
    """Tests for ConflictDetector cache functionality."""

    def test_cache_initialized_empty(self) -> None:
        """Test that conflict cache is initialized empty."""
        detector = ConflictDetector()
        assert detector.conflict_cache == {}

    def test_cache_can_store_data(self) -> None:
        """Test that cache can store data."""
        detector = ConflictDetector()
        detector.conflict_cache["test_key"] = {"result": "cached"}
        assert detector.conflict_cache["test_key"] == {"result": "cached"}
