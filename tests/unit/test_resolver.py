"""Test the main ConflictResolver class."""

from pathlib import Path
from unittest.mock import Mock, patch

from review_bot_automator import Change, ConflictResolver, FileType
from review_bot_automator.utils.text import normalize_content


class TestConflictResolver:
    """Test the ConflictResolver class."""

    def test_init(self) -> None:
        """Test resolver initialization."""
        resolver = ConflictResolver()
        assert resolver.config == {}
        assert resolver.conflict_detector is not None
        assert FileType.JSON in resolver.handlers
        assert FileType.YAML in resolver.handlers
        assert FileType.TOML in resolver.handlers

    def test_detect_file_type(self) -> None:
        """Test file type detection."""
        resolver = ConflictResolver()

        assert resolver.detect_file_type("test.json") == FileType.JSON
        assert resolver.detect_file_type("test.yaml") == FileType.YAML
        assert resolver.detect_file_type("test.yml") == FileType.YAML
        assert resolver.detect_file_type("test.toml") == FileType.TOML
        assert resolver.detect_file_type("test.py") == FileType.PYTHON
        assert resolver.detect_file_type("test.ts") == FileType.TYPESCRIPT
        assert resolver.detect_file_type("test.txt") == FileType.PLAINTEXT

    def test_generate_fingerprint(self) -> None:
        """Test fingerprint generation."""
        resolver = ConflictResolver()

        fp1 = resolver.generate_fingerprint("test.py", 10, 15, "content")
        fp2 = resolver.generate_fingerprint("test.py", 10, 15, "content")
        fp3 = resolver.generate_fingerprint("test.py", 10, 15, "different")

        assert fp1 == fp2  # Same content should generate same fingerprint
        assert fp1 != fp3  # Different content should generate different fingerprint

    def test_normalize_content(self) -> None:
        """
        Verify normalize_content trims leading and trailing whitespace from each line and
            removes blank lines while preserving line order.
        """
        content = "  line1  \n  line2  \n  \n  line3  "
        normalized = normalize_content(content)
        expected = "line1\nline2\nline3"
        assert normalized == expected

    def test_extract_changes_from_comments(self) -> None:
        """Test extracting changes from comments."""
        resolver = ConflictResolver()

        comments = [
            {
                "path": "test.json",
                "body": '```suggestion\n{\n  "name": "test"\n}\n```',
                "start_line": 1,
                "line": 3,
                "html_url": "https://github.com/test",
                "user": {"login": "coderabbit"},
            }
        ]

        changes = resolver.extract_changes_from_comments(comments)

        assert len(changes) == 1
        change = changes[0]
        assert change.path == "test.json"
        assert change.start_line == 1
        assert change.end_line == 3
        assert change.content == '{\n  "name": "test"\n}'
        assert change.file_type == FileType.JSON
        assert change.metadata["author"] == "coderabbit"

    def test_detect_conflicts(self) -> None:
        """
        Verify that ConflictResolver groups overlapping changes in the same file into a single
            conflict.

        Sets up two overlapping JSON Changes on "test.json" and asserts that detect_conflicts
            returns a single conflict covering both changes, that the conflict's file_path is
            "test.json", and that the conflict_type is either "major" or "partial".
        """
        resolver = ConflictResolver()

        changes = [
            Change(
                path="test.json",
                start_line=10,
                end_line=15,
                content='{"key": "value1"}',
                metadata={},
                fingerprint="fp1",
                file_type=FileType.JSON,
            ),
            Change(
                path="test.json",
                start_line=12,
                end_line=18,
                content='{"key": "value2"}',
                metadata={},
                fingerprint="fp2",
                file_type=FileType.JSON,
            ),
        ]

        conflicts = resolver.detect_conflicts(changes)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.file_path == "test.json"
        assert len(conflict.changes) == 2
        assert conflict.conflict_type in ["major", "partial"]

    def test_has_line_overlap(self) -> None:
        """Test line overlap detection."""
        resolver = ConflictResolver()

        change1 = Change("test.py", 10, 15, "content1", {}, "fp1", FileType.PYTHON)
        change2 = Change("test.py", 12, 18, "content2", {}, "fp2", FileType.PYTHON)
        change3 = Change("test.py", 20, 25, "content3", {}, "fp3", FileType.PYTHON)

        assert resolver._has_line_overlap(change1, change2) is True
        assert resolver._has_line_overlap(change1, change3) is False

    def test_classify_conflict_type(self) -> None:
        """Test conflict type classification."""
        resolver = ConflictResolver()

        change1 = Change("test.py", 10, 15, "content1", {}, "fp1", FileType.PYTHON)
        change2 = Change("test.py", 10, 15, "content2", {}, "fp2", FileType.PYTHON)
        change3 = Change("test.py", 12, 18, "content3", {}, "fp3", FileType.PYTHON)

        assert resolver._classify_conflict_type(change1, [change2]) == "exact"
        assert resolver._classify_conflict_type(change1, [change3]) in ["major", "partial"]

    def test_assess_conflict_severity(self) -> None:
        """Test conflict severity assessment."""
        resolver = ConflictResolver()

        # Security-related change
        security_change = Change("test.py", 10, 15, "security fix", {}, "fp1", FileType.PYTHON)
        assert resolver._assess_conflict_severity(security_change, []) == "high"

        # Syntax error fix
        syntax_change = Change("test.py", 10, 15, "fix error", {}, "fp1", FileType.PYTHON)
        assert resolver._assess_conflict_severity(syntax_change, []) == "medium"

        # Regular change
        regular_change = Change("test.py", 10, 15, "regular change", {}, "fp1", FileType.PYTHON)
        assert resolver._assess_conflict_severity(regular_change, []) == "low"

    def test_calculate_overlap_percentage(self) -> None:
        """Test overlap percentage calculation."""
        resolver = ConflictResolver()

        change1 = Change("test.py", 10, 15, "content1", {}, "fp1", FileType.PYTHON)
        change2 = Change("test.py", 12, 18, "content2", {}, "fp2", FileType.PYTHON)

        percentage = resolver._calculate_overlap_percentage(change1, [change2])
        assert 0 <= percentage <= 100

    @patch("review_bot_automator.core.resolver.GitHubCommentExtractor")
    def test_resolve_pr_conflicts(self, mock_extractor: Mock) -> None:
        """Test resolving PR conflicts."""
        resolver = ConflictResolver()

        # Mock GitHub extractor
        mock_extractor.return_value.fetch_pr_comments.return_value = []

        result = resolver.resolve_pr_conflicts("owner", "repo", 123)

        assert result.applied_count == 0
        assert result.conflict_count == 0
        assert result.success_rate == 0
        assert result.resolutions == []
        assert result.conflicts == []

    @patch("review_bot_automator.core.resolver.GitHubCommentExtractor")
    def test_analyze_conflicts(self, mock_extractor: Mock) -> None:
        """Test analyzing conflicts."""
        resolver = ConflictResolver()

        # Mock GitHub extractor
        mock_extractor.return_value.fetch_pr_comments.return_value = []

        conflicts = resolver.analyze_conflicts("owner", "repo", 123)

        assert conflicts == []

    def test_separate_changes_by_conflict_status_no_conflicts(self) -> None:
        """Test separating changes when there are no conflicts."""
        from review_bot_automator.core.models import Conflict

        resolver = ConflictResolver()

        # Create non-conflicting changes
        changes = [
            Change("test.py", 10, 15, "content1", {}, "fp1", FileType.PYTHON),
            Change("test.py", 20, 25, "content2", {}, "fp2", FileType.PYTHON),
            Change("test.py", 30, 35, "content3", {}, "fp3", FileType.PYTHON),
        ]

        conflicts: list[Conflict] = []

        conflicting, non_conflicting = resolver.separate_changes_by_conflict_status(
            changes, conflicts
        )

        # All changes should be non-conflicting
        assert len(conflicting) == 0
        assert len(non_conflicting) == 3
        assert {c.fingerprint for c in non_conflicting} == {"fp1", "fp2", "fp3"}

    def test_separate_changes_by_conflict_status_all_conflicting(self) -> None:
        """Test separating changes when all changes are conflicting."""
        from review_bot_automator.core.models import Conflict

        resolver = ConflictResolver()

        # Create overlapping changes
        change1 = Change("test.py", 10, 15, "content1", {}, "fp1", FileType.PYTHON)
        change2 = Change("test.py", 12, 18, "content2", {}, "fp2", FileType.PYTHON)
        change3 = Change("test.py", 14, 20, "content3", {}, "fp3", FileType.PYTHON)

        changes = [change1, change2, change3]

        # Create conflict containing all changes
        conflict = Conflict(
            file_path="test.py",
            line_range=(10, 20),
            changes=[change1, change2, change3],
            conflict_type="multiple",
            severity="low",
            overlap_percentage=50.0,
        )
        conflicts = [conflict]

        conflicting, non_conflicting = resolver.separate_changes_by_conflict_status(
            changes, conflicts
        )

        # All changes should be conflicting
        assert len(conflicting) == 3
        assert len(non_conflicting) == 0
        assert {c.fingerprint for c in conflicting} == {"fp1", "fp2", "fp3"}

    def test_separate_changes_by_conflict_status_mixed(self) -> None:
        """Test separating changes with mix of conflicting and non-conflicting."""
        from review_bot_automator.core.models import Conflict

        resolver = ConflictResolver()

        # Create mixed changes: some conflicting, some not
        change1 = Change("test.py", 10, 15, "content1", {}, "fp1", FileType.PYTHON)
        change2 = Change("test.py", 12, 18, "content2", {}, "fp2", FileType.PYTHON)
        change3 = Change("test.py", 30, 35, "content3", {}, "fp3", FileType.PYTHON)
        change4 = Change("test.py", 50, 55, "content4", {}, "fp4", FileType.PYTHON)

        changes = [change1, change2, change3, change4]

        # Create conflict for change1 and change2 only
        conflict = Conflict(
            file_path="test.py",
            line_range=(10, 18),
            changes=[change1, change2],
            conflict_type="partial",
            severity="low",
            overlap_percentage=40.0,
        )
        conflicts = [conflict]

        conflicting, non_conflicting = resolver.separate_changes_by_conflict_status(
            changes, conflicts
        )

        # change1 and change2 should be conflicting, change3 and change4 non-conflicting
        assert len(conflicting) == 2
        assert len(non_conflicting) == 2
        assert {c.fingerprint for c in conflicting} == {"fp1", "fp2"}
        assert {c.fingerprint for c in non_conflicting} == {"fp3", "fp4"}

    def test_separate_changes_by_conflict_status_multiple_conflicts(self) -> None:
        """Test separating changes with multiple separate conflicts."""
        from review_bot_automator.core.models import Conflict

        resolver = ConflictResolver()

        # Create changes with multiple separate conflict groups
        change1 = Change("test.py", 10, 15, "content1", {}, "fp1", FileType.PYTHON)
        change2 = Change("test.py", 12, 18, "content2", {}, "fp2", FileType.PYTHON)
        change3 = Change("test.py", 30, 35, "content3", {}, "fp3", FileType.PYTHON)
        change4 = Change("test.py", 50, 55, "content4", {}, "fp4", FileType.PYTHON)
        change5 = Change("test.py", 52, 58, "content5", {}, "fp5", FileType.PYTHON)

        changes = [change1, change2, change3, change4, change5]

        # Create two separate conflicts
        conflict1 = Conflict(
            file_path="test.py",
            line_range=(10, 18),
            changes=[change1, change2],
            conflict_type="partial",
            severity="low",
            overlap_percentage=40.0,
        )
        conflict2 = Conflict(
            file_path="test.py",
            line_range=(50, 58),
            changes=[change4, change5],
            conflict_type="partial",
            severity="low",
            overlap_percentage=30.0,
        )
        conflicts = [conflict1, conflict2]

        conflicting, non_conflicting = resolver.separate_changes_by_conflict_status(
            changes, conflicts
        )

        # change1, change2, change4, change5 should be conflicting; change3 non-conflicting
        assert len(conflicting) == 4
        assert len(non_conflicting) == 1
        assert {c.fingerprint for c in conflicting} == {"fp1", "fp2", "fp4", "fp5"}
        assert {c.fingerprint for c in non_conflicting} == {"fp3"}

    def test_separate_changes_by_conflict_status_empty_inputs(self) -> None:
        """Test separating changes with empty inputs."""

        resolver = ConflictResolver()

        # Test empty changes list
        conflicting, non_conflicting = resolver.separate_changes_by_conflict_status([], [])
        assert len(conflicting) == 0
        assert len(non_conflicting) == 0

        # Test empty conflicts list with changes
        changes = [
            Change("test.py", 10, 15, "content1", {}, "fp1", FileType.PYTHON),
        ]
        conflicting, non_conflicting = resolver.separate_changes_by_conflict_status(changes, [])
        assert len(conflicting) == 0
        assert len(non_conflicting) == 1

    def test_apply_changes_with_rollback_no_rollback_manager(self) -> None:
        """Test apply_changes_with_rollback when rollback_manager is None."""
        resolver = ConflictResolver()

        # When rollback_manager is None, should call apply_changes directly
        applied, skipped, failed = resolver.apply_changes_with_rollback([], validate=True)

        assert len(applied) == 0
        assert len(skipped) == 0
        assert len(failed) == 0

    @patch("review_bot_automator.core.rollback.RollbackManager")
    def test_apply_changes_with_rollback_initialization_failure(
        self, mock_rollback_manager: Mock
    ) -> None:
        """Test apply_changes_with_rollback falls back when RollbackManager init fails."""
        # Make RollbackManager raise ValueError on initialization
        mock_rollback_manager.side_effect = ValueError("Git not available")

        resolver = ConflictResolver()

        # Should fall back to apply_changes when RollbackManager fails to initialize
        applied, skipped, failed = resolver.apply_changes_with_rollback([], validate=True)

        assert len(applied) == 0
        assert len(skipped) == 0
        assert len(failed) == 0

    # ========================================================================
    # Phase 1: Tests for apply_changes() method
    # ========================================================================

    def test_apply_changes_success(self, temp_workspace: Path) -> None:
        """Test apply_changes successfully applies valid changes."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        # Create a valid change
        change = Change(
            path=str(test_file),
            start_line=2,
            end_line=2,
            content="line2_modified",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
        )

        applied, skipped, failed = resolver.apply_changes([change], validate=True)

        # Verify the change was applied
        assert len(applied) == 1
        assert len(skipped) == 0
        assert len(failed) == 0
        assert applied[0].fingerprint == "fp1"

        # Verify file content was modified
        content = test_file.read_text()
        assert "line2_modified" in content

    def test_apply_changes_with_validation(self, temp_workspace: Path) -> None:
        """Test apply_changes validates changes before applying."""
        resolver = ConflictResolver(workspace_root=temp_workspace)

        # Create a change for a non-existent file
        change = Change(
            path=str(temp_workspace / "nonexistent.txt"),
            start_line=1,
            end_line=1,
            content="test",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
        )

        applied, skipped, failed = resolver.apply_changes([change], validate=True)

        # Change should be skipped due to validation failure
        assert len(applied) == 0
        assert len(skipped) == 1
        assert len(failed) == 0
        assert skipped[0].fingerprint == "fp1"

    def test_apply_changes_skip_validation(self, temp_workspace: Path) -> None:
        """Test apply_changes with validate=False skips validation step."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        # Create a valid change
        change = Change(
            path=str(test_file),
            start_line=2,
            end_line=2,
            content="line2_modified",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
        )

        # Apply without validation
        applied, skipped, failed = resolver.apply_changes([change], validate=False)

        # Change should still be applied (validation skipped)
        assert len(applied) == 1
        assert len(skipped) == 0
        assert len(failed) == 0

    def test_apply_changes_with_failed_change(self, temp_workspace: Path) -> None:
        """Test apply_changes tracks changes that fail to apply."""
        resolver = ConflictResolver(workspace_root=temp_workspace)

        # Mock _apply_change to return False (simulating application failure)
        with patch.object(resolver, "_apply_change", return_value=False):
            change = Change(
                path="test.txt",
                start_line=1,
                end_line=1,
                content="test",
                metadata={},
                fingerprint="fp1",
                file_type=FileType.PLAINTEXT,
            )

            applied, skipped, failed = resolver.apply_changes([change], validate=False)

            # Change should be in failed list
            assert len(applied) == 0
            assert len(skipped) == 0
            assert len(failed) == 1
            assert failed[0][0].fingerprint == "fp1"
            assert "unspecified failure" in failed[0][1].lower()

    def test_apply_changes_with_exception(self, temp_workspace: Path) -> None:
        """Test apply_changes handles exceptions during application."""
        resolver = ConflictResolver(workspace_root=temp_workspace)

        # Mock _apply_change to raise an exception
        with patch.object(resolver, "_apply_change", side_effect=OSError("Disk full")):
            change = Change(
                path="test.txt",
                start_line=1,
                end_line=1,
                content="test",
                metadata={},
                fingerprint="fp1",
                file_type=FileType.PLAINTEXT,
            )

            applied, skipped, failed = resolver.apply_changes([change], validate=False)

            # Change should be in failed list with exception details
            assert len(applied) == 0
            assert len(skipped) == 0
            assert len(failed) == 1
            assert failed[0][0].fingerprint == "fp1"
            assert "OSError" in failed[0][1]
            assert "Disk full" in failed[0][1]

    # ========================================================================
    # Phase 2: Tests for _validate_change() method
    # ========================================================================

    def test_validate_change_valid(self, temp_workspace: Path) -> None:
        """Test _validate_change returns True for a valid change."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(test_file),
            start_line=1,
            end_line=2,
            content="modified",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
        )

        is_valid, reason = resolver._validate_change(change)

        assert is_valid is True
        assert reason == ""

    def test_validate_change_invalid_file_path(self, temp_workspace: Path) -> None:
        """Test _validate_change rejects invalid file paths."""
        resolver = ConflictResolver(workspace_root=temp_workspace)

        # Create change with path traversal attempt
        change = Change(
            path="../../../etc/passwd",
            start_line=1,
            end_line=1,
            content="malicious",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
        )

        is_valid, reason = resolver._validate_change(change)

        assert is_valid is False
        assert "Invalid or unsafe file path" in reason

    def test_validate_change_file_not_found(self, temp_workspace: Path) -> None:
        """Test _validate_change rejects changes to non-existent files."""
        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(temp_workspace / "nonexistent.txt"),
            start_line=1,
            end_line=1,
            content="test",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
        )

        is_valid, reason = resolver._validate_change(change)

        assert is_valid is False
        assert "does not exist" in reason.lower()

    # NOTE: Symlink detection test skipped - the resolve_file_path function
    # resolves symlinks before the symlink check runs, so symlinks are not
    # currently rejected by _validate_change. This may be a design decision
    # to allow symlinks or a potential security gap to address in future work.

    def test_validate_change_with_handler_validation(self, temp_workspace: Path) -> None:
        """Test _validate_change delegates to handler's validate_change if available."""
        # Create a JSON file
        json_file = temp_workspace / "test.json"
        json_file.write_text('{"key": "value"}')

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(json_file),
            start_line=1,
            end_line=1,
            content='{"key": "modified"}',
            metadata={},
            fingerprint="fp1",
            file_type=FileType.JSON,
        )

        # This should use the JSON handler's validation
        is_valid, reason = resolver._validate_change(change)

        # For this test, we just verify it doesn't crash and returns a result
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)

    def test_validate_change_handler_exception(self, temp_workspace: Path) -> None:
        """Test _validate_change handles handler validation exceptions."""
        test_file = temp_workspace / "test.json"
        test_file.write_text('{"key": "value"}')

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(test_file),
            start_line=1,
            end_line=1,
            content="invalid json",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.JSON,
        )

        # Mock the handler to raise an exception
        handler = resolver.handlers.get(FileType.JSON)
        if handler:
            with patch.object(handler, "validate_change", side_effect=ValueError("Invalid JSON")):
                is_valid, reason = resolver._validate_change(change)

                assert is_valid is False
                assert "Validation error" in reason


class TestPlaintextChangeType:
    """Test plaintext change_type handling in resolver._apply_plaintext_change."""

    def test_apply_plaintext_change_deletion(self, temp_workspace: Path) -> None:
        """Test plaintext deletion removes specified lines."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(test_file),
            start_line=2,
            end_line=2,
            content="ignored content",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
            change_type="deletion",
        )

        result = resolver._apply_plaintext_change(change)
        assert result is True

        content = test_file.read_text()
        assert "line1" in content
        assert "line2" not in content
        assert "line3" in content

    def test_apply_plaintext_change_addition(self, temp_workspace: Path) -> None:
        """Test plaintext addition inserts lines after specified position."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("line1\nline2\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(test_file),
            start_line=1,
            end_line=1,
            content="inserted",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
            change_type="addition",
        )

        result = resolver._apply_plaintext_change(change)
        assert result is True

        content = test_file.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "line1"
        assert "inserted" in lines[1]
        assert lines[2] == "line2"

    def test_apply_plaintext_change_modification(self, temp_workspace: Path) -> None:
        """Test plaintext modification (default) replaces specified lines."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("line1\nold_line\nline3\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(test_file),
            start_line=2,
            end_line=2,
            content="new_line",
            metadata={},
            fingerprint="fp1",
            file_type=FileType.PLAINTEXT,
            change_type="modification",
        )

        result = resolver._apply_plaintext_change(change)
        assert result is True

        content = test_file.read_text()
        assert "line1" in content
        assert "new_line" in content
        assert "old_line" not in content
        assert "line3" in content

    def test_apply_plaintext_change_invalid_type_returns_false(self, temp_workspace: Path) -> None:
        """Test plaintext handler returns False for invalid change_type."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("line1\nline2\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        # Use object.__new__ to bypass Change validation for testing
        change = object.__new__(Change)
        object.__setattr__(change, "path", str(test_file))
        object.__setattr__(change, "start_line", 1)
        object.__setattr__(change, "end_line", 1)
        object.__setattr__(change, "content", "new")
        object.__setattr__(change, "metadata", {})
        object.__setattr__(change, "fingerprint", "fp1")
        object.__setattr__(change, "file_type", FileType.PLAINTEXT)
        object.__setattr__(change, "change_type", "invalid_type")

        result = resolver._apply_plaintext_change(change)
        assert result is False

    def test_apply_plaintext_change_addition_at_top_of_file(self, temp_workspace: Path) -> None:
        """Test addition change type with end_line=0 (insert at top of file).

        This tests the indent_ref bounds fix at resolver.py:1391 where end_idx - 1
        could be negative when end_line is 0.
        """
        test_file = temp_workspace / "test.txt"
        test_file.write_text("existing_line1\nexisting_line2\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(test_file),
            start_line=0,
            end_line=0,
            content="new_line_at_top",
            metadata={},
            fingerprint="fp_top",
            file_type=FileType.PLAINTEXT,
            change_type="addition",
        )

        result = resolver._apply_plaintext_change(change)
        assert result is True

        content = test_file.read_text()
        # New line should be at the top, followed by original content
        assert "new_line_at_top" in content
        assert "existing_line1" in content
        assert "existing_line2" in content

    def test_apply_plaintext_change_start_line_beyond_file_returns_false(
        self, temp_workspace: Path
    ) -> None:
        """Test that start_line beyond file length returns False."""
        test_file = temp_workspace / "short.txt"
        test_file.write_text("line1\nline2\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(test_file),
            start_line=100,  # Beyond file length
            end_line=100,
            content="new_content",
            metadata={},
            fingerprint="fp_beyond",
            file_type=FileType.PLAINTEXT,
            change_type="modification",
        )

        result = resolver._apply_plaintext_change(change)
        assert result is False

    def test_apply_plaintext_change_end_line_beyond_file_clamps(self, temp_workspace: Path) -> None:
        """Test that end_line beyond file length is clamped to file end."""
        test_file = temp_workspace / "clamp.txt"
        test_file.write_text("line1\nline2\nline3\n")

        resolver = ConflictResolver(workspace_root=temp_workspace)

        change = Change(
            path=str(test_file),
            start_line=2,
            end_line=100,  # Beyond file length, should clamp to 3
            content="replacement",
            metadata={},
            fingerprint="fp_clamp",
            file_type=FileType.PLAINTEXT,
            change_type="modification",
        )

        result = resolver._apply_plaintext_change(change)
        assert result is True

        content = test_file.read_text()
        # Line 1 preserved, lines 2-3 replaced
        assert "line1" in content
        assert "replacement" in content
        assert "line2" not in content
        assert "line3" not in content
