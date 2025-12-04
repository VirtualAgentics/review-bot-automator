"""Tests for BaseHandler functionality, particularly restore_file edge cases."""

from pathlib import Path

import pytest

from review_bot_automator.core.models import Change, Conflict
from review_bot_automator.handlers.base import BaseHandler


class ConcreteHandler(BaseHandler):
    """Concrete implementation of BaseHandler for testing."""

    def can_handle(self, file_path: str) -> bool:
        """Handle all files for testing."""
        return True

    def apply_change(self, path: str, content: str, start_line: int, end_line: int) -> bool:
        """Stub implementation."""
        return True

    def validate_change(
        self, path: str, content: str, start_line: int, end_line: int
    ) -> tuple[bool, str]:
        """Stub implementation."""
        return True, "Valid"

    def detect_conflicts(self, path: str, changes: list[Change]) -> list[Conflict]:
        """Stub implementation."""
        return []


class TestBaseHandlerInit:
    """Tests for BaseHandler initialization."""

    def test_default_workspace_root(self) -> None:
        """Test default workspace root is current working directory."""
        handler = ConcreteHandler()
        assert handler.workspace_root == Path.cwd().resolve()

    def test_custom_workspace_root_str(self, tmp_path: Path) -> None:
        """Test custom workspace root from string."""
        handler = ConcreteHandler(workspace_root=str(tmp_path))
        assert handler.workspace_root == tmp_path.resolve()

    def test_custom_workspace_root_path(self, tmp_path: Path) -> None:
        """Test custom workspace root from Path."""
        handler = ConcreteHandler(workspace_root=tmp_path)
        assert handler.workspace_root == tmp_path.resolve()


class TestBackupFile:
    """Tests for backup_file method."""

    def test_backup_creates_file(self, tmp_path: Path) -> None:
        """Test that backup_file creates a backup."""
        handler = ConcreteHandler(workspace_root=tmp_path)
        original = tmp_path / "test.txt"
        original.write_text("original content", encoding="utf-8")

        backup_path = handler.backup_file(str(original))

        assert Path(backup_path).exists()
        assert Path(backup_path).read_text(encoding="utf-8") == "original content"

    def test_backup_invalid_path_raises(self, tmp_path: Path) -> None:
        """Test that invalid path raises ValueError."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        with pytest.raises(ValueError, match="Invalid file path"):
            handler.backup_file("../../../etc/passwd")

    def test_backup_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Test that non-existent file raises FileNotFoundError."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        with pytest.raises(FileNotFoundError):
            handler.backup_file(str(tmp_path / "nonexistent.txt"))

    def test_backup_directory_raises(self, tmp_path: Path) -> None:
        """Test that directory raises ValueError."""
        handler = ConcreteHandler(workspace_root=tmp_path)
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        with pytest.raises(ValueError, match="not a regular file"):
            handler.backup_file(str(subdir))


class TestRestoreFile:
    """Tests for restore_file edge cases."""

    def test_restore_success(self, tmp_path: Path) -> None:
        """Test successful file restoration."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        # Create original and backup files
        original = tmp_path / "test.txt"
        original.write_text("modified content", encoding="utf-8")
        backup = tmp_path / "test.txt.backup"
        backup.write_text("original content", encoding="utf-8")

        result = handler.restore_file(str(backup), str(original))

        assert result is True
        assert original.read_text(encoding="utf-8") == "original content"
        assert not backup.exists()

    def test_restore_different_directories_fails(self, tmp_path: Path) -> None:
        """Test that restore fails when backup and original are in different directories."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        # Create subdirectories
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Create files in different directories
        original = dir1 / "test.txt"
        original.write_text("content", encoding="utf-8")
        backup = dir2 / "test.txt.backup"
        backup.write_text("backup content", encoding="utf-8")

        result = handler.restore_file(str(backup), str(original))

        assert result is False
        # Verify files unchanged
        assert backup.exists()

    def test_restore_outside_workspace_fails(self, tmp_path: Path) -> None:
        """Test that restore fails when paths are outside workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        handler = ConcreteHandler(workspace_root=workspace)

        # Create files outside workspace
        original = outside / "test.txt"
        original.write_text("content", encoding="utf-8")
        backup = outside / "test.txt.backup"
        backup.write_text("backup content", encoding="utf-8")

        result = handler.restore_file(str(backup), str(original))

        assert result is False

    def test_restore_backup_not_file_fails(self, tmp_path: Path) -> None:
        """Test that restore fails when backup is not a regular file."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        # Create a directory instead of a file
        backup = tmp_path / "backup_dir"
        backup.mkdir()
        original = tmp_path / "test.txt"
        original.write_text("content", encoding="utf-8")

        result = handler.restore_file(str(backup), str(original))

        assert result is False

    def test_restore_path_validation_fallback(self, tmp_path: Path) -> None:
        """Test relaxed path validation fallback for safe segment paths."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        # Create files with valid names
        original = tmp_path / "test_file.txt"
        original.write_text("modified", encoding="utf-8")
        backup = tmp_path / "test_file.txt.backup"
        backup.write_text("original", encoding="utf-8")

        result = handler.restore_file(str(backup), str(original))

        assert result is True
        assert original.read_text(encoding="utf-8") == "original"

    def test_restore_nonexistent_files_fails(self, tmp_path: Path) -> None:
        """Test that restore fails when both files don't exist."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        # Both backup and original don't exist
        result = handler.restore_file(str(tmp_path / "test.txt.backup"), str(tmp_path / "test.txt"))
        assert result is False

    def test_restore_nonexistent_backup_fails(self, tmp_path: Path) -> None:
        """Test that restore fails when backup doesn't exist."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        original = tmp_path / "test.txt"
        original.write_text("content", encoding="utf-8")
        nonexistent_backup = tmp_path / "nonexistent.backup"

        result = handler.restore_file(str(nonexistent_backup), str(original))

        assert result is False

    def test_restore_both_paths_invalid_fails(self, tmp_path: Path) -> None:
        """Test that restore fails when both paths are invalid."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        result = handler.restore_file("../../../etc/shadow.backup", "../../../etc/shadow")

        assert result is False

    def test_restore_original_path_invalid_fails(self, tmp_path: Path) -> None:
        """Test that restore fails when only original path is invalid."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        # Create valid backup
        backup = tmp_path / "test.txt.backup"
        backup.write_text("backup content", encoding="utf-8")

        result = handler.restore_file(str(backup), "../../../etc/passwd")

        assert result is False

    def test_restore_backup_path_invalid_fails(self, tmp_path: Path) -> None:
        """Test that restore fails when only backup path is invalid."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        # Create valid original
        original = tmp_path / "test.txt"
        original.write_text("content", encoding="utf-8")

        result = handler.restore_file("../../../etc/passwd.backup", str(original))

        assert result is False


class TestBackupAndRestoreIntegration:
    """Integration tests for backup and restore workflow."""

    def test_backup_then_restore_roundtrip(self, tmp_path: Path) -> None:
        """Test complete backup and restore workflow."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        # Create original file
        original = tmp_path / "config.json"
        original_content = '{"key": "value"}'
        original.write_text(original_content, encoding="utf-8")

        # Backup
        backup_path = handler.backup_file(str(original))

        # Modify original
        original.write_text('{"key": "modified"}', encoding="utf-8")
        assert original.read_text(encoding="utf-8") != original_content

        # Restore
        result = handler.restore_file(backup_path, str(original))

        assert result is True
        assert original.read_text(encoding="utf-8") == original_content
        assert not Path(backup_path).exists()

    def test_multiple_backups_collision_handling(self, tmp_path: Path) -> None:
        """Test that multiple backup attempts handle collisions."""
        handler = ConcreteHandler(workspace_root=tmp_path)

        original = tmp_path / "test.txt"
        original.write_text("content", encoding="utf-8")

        # Create first backup
        backup1 = handler.backup_file(str(original))

        # Re-create original (since backup_file doesn't delete original)
        original.write_text("content2", encoding="utf-8")

        # Create second backup - should handle collision
        backup2 = handler.backup_file(str(original))

        assert backup1 != backup2
        assert Path(backup1).exists()
        assert Path(backup2).exists()
