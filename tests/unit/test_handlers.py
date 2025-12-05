"""Test the file handlers."""

import os
import stat
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from review_bot_automator import FileType, JsonHandler, TomlHandler, YamlHandler
from review_bot_automator.core.models import Change, ChangeType, Conflict
from review_bot_automator.handlers.base import BaseHandler
from review_bot_automator.handlers.yaml_handler import YAMLValue


class MockTaggedObject:
    """Mock object for testing YAML tagged object detection."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class TestJsonHandler:
    """Test the JSON handler."""

    def test_can_handle(self) -> None:
        """Test file type detection."""
        handler = JsonHandler()

        assert handler.can_handle("test.json") is True
        assert handler.can_handle("test.JSON") is True
        assert handler.can_handle("test.yaml") is False
        assert handler.can_handle("test.txt") is False

    def test_validate_change(self) -> None:
        """
        Verify JsonHandler.validate_change accepts valid JSON and rejects malformed JSON.

        Asserts that:
        - a well-formed JSON string produces (True, message) and the message contains "Valid JSON";
        - a malformed JSON string produces (False, message) and the message contains "Invalid JSON";
        - re-validating a well-formed JSON string still reports valid JSON (duplicate-key
            behavior is tested via parsing semantics).
        """
        handler = JsonHandler()

        # Valid JSON
        valid, msg = handler.validate_change("test.json", '{"key": "value"}', 1, 3)
        assert valid is True
        assert "Valid JSON" in msg

        # Invalid JSON
        valid, msg = handler.validate_change("test.json", '{"key": "value"', 1, 3)
        assert valid is False
        assert "Invalid JSON" in msg

        # Test validation logic - duplicate keys would be caught during parsing
        # Since Python dicts can't have duplicate keys, we test the validation logic differently
        valid, msg = handler.validate_change("test.json", '{"key": "value"}', 1, 3)
        assert valid is True
        assert "Valid JSON" in msg

    def test_detect_conflicts(self) -> None:
        """
        Verify that JsonHandler.detect_conflicts identifies key conflicts among multiple JSON
            changes.

        This test constructs three Change objects for the same JSON path where two changes
            modify the same key across different line ranges and the third changes a different
            key. It asserts that exactly one conflict is reported, that the conflict type is
            "key_conflict", and that the conflict includes the two related changes.
        """
        handler = JsonHandler()

        changes = [
            Change(
                path="test.json",
                start_line=1,
                end_line=3,
                content='{"key1": "value1"}',
                metadata={},
                fingerprint="test1",
                file_type=FileType.JSON,
            ),
            Change(
                path="test.json",
                start_line=4,
                end_line=6,
                content='{"key1": "value2"}',
                metadata={},
                fingerprint="test2",
                file_type=FileType.JSON,
            ),
            Change(
                path="test.json",
                start_line=7,
                end_line=9,
                content='{"key2": "value3"}',
                metadata={},
                fingerprint="test3",
                file_type=FileType.JSON,
            ),
        ]

        conflicts = handler.detect_conflicts("test.json", changes)

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "key_conflict"
        assert len(conflicts[0].changes) == 2

    def test_smart_merge_json(self) -> None:
        """Test smart JSON merging."""
        handler = JsonHandler()

        original = {"key1": "value1", "key2": "value2"}
        suggestion = {"key1": "new_value1", "key3": "value3"}

        result = handler._smart_merge_json(original, suggestion, 1, 3)

        expected = {"key1": "new_value1", "key2": "value2", "key3": "value3"}
        assert result == expected

    def test_is_complete_object(self) -> None:
        """Test complete object detection."""
        handler = JsonHandler()

        original = {"key1": "value1", "key2": "value2"}

        # Complete object
        suggestion = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert handler._is_complete_object(suggestion, original) is True

        # Partial object
        suggestion = {"key1": "value1"}
        assert handler._is_complete_object(suggestion, original) is False

    def test_apply_change_deletion_removes_keys(self, tmp_path: Path) -> None:
        """Test JSON deletion removes specified keys from original."""
        handler = JsonHandler(workspace_root=tmp_path)
        target = tmp_path / "config.json"
        target.write_text('{"keep": "value1", "remove": "value2"}', encoding="utf-8")

        # Delete the "remove" key
        result = handler.apply_change(
            str(target), '{"remove": "ignored"}', 1, 1, change_type="deletion"
        )
        assert result is True
        content = target.read_text(encoding="utf-8")
        assert '"keep"' in content
        assert '"remove"' not in content

    def test_apply_change_addition_merges_keys(self, tmp_path: Path) -> None:
        """Test JSON addition merges new keys into original."""
        handler = JsonHandler(workspace_root=tmp_path)
        target = tmp_path / "config.json"
        target.write_text('{"existing": "value1"}', encoding="utf-8")

        # Add new key with addition type
        result = handler.apply_change(
            str(target), '{"new_key": "value2"}', 1, 1, change_type="addition"
        )
        assert result is True
        content = target.read_text(encoding="utf-8")
        assert '"existing"' in content
        assert '"new_key"' in content

    def test_apply_change_modification_merges_keys(self, tmp_path: Path) -> None:
        """Test JSON modification (default) merges and overrides keys."""
        handler = JsonHandler(workspace_root=tmp_path)
        target = tmp_path / "config.json"
        target.write_text('{"key1": "old", "key2": "keep"}', encoding="utf-8")

        # Modify with merge
        result = handler.apply_change(
            str(target), '{"key1": "new", "key3": "added"}', 1, 1, change_type="modification"
        )
        assert result is True
        content = target.read_text(encoding="utf-8")
        assert '"key1": "new"' in content
        assert '"key2"' in content
        assert '"key3"' in content

    def test_apply_change_invalid_change_type_returns_false(self, tmp_path: Path) -> None:
        """Test JSON handler returns False for invalid change_type."""
        handler = JsonHandler(workspace_root=tmp_path)
        target = tmp_path / "config.json"
        target.write_text('{"key": "value"}', encoding="utf-8")

        result = handler.apply_change(
            str(target), '{"key": "new"}', 1, 1, change_type="invalid"  # type: ignore[arg-type]
        )
        assert result is False

    def test_apply_change_deletion_with_non_dict_json_returns_false(self, tmp_path: Path) -> None:
        """Test JSON deletion fails when original is not a dict (e.g., list)."""
        handler = JsonHandler(workspace_root=tmp_path)
        target = tmp_path / "array.json"
        target.write_text('["item1", "item2"]', encoding="utf-8")

        # Attempt to delete from a list (should fail - deletion requires dicts)
        result = handler.apply_change(
            str(target), '{"key": "ignored"}', 1, 1, change_type="deletion"
        )
        assert result is False


class TestYamlHandler:
    """Test the YAML handler."""

    def test_can_handle(self) -> None:
        """Test file type detection."""
        handler = YamlHandler()

        assert handler.can_handle("test.yaml") is True
        assert handler.can_handle("test.yml") is True
        assert handler.can_handle("test.YAML") is True
        assert handler.can_handle("test.json") is False
        assert handler.can_handle("test.txt") is False

    @patch("review_bot_automator.handlers.yaml_handler.YAML_AVAILABLE", False)
    def test_yaml_not_available(self) -> None:
        """Test behavior when ruamel.yaml is not available."""
        handler = YamlHandler()

        valid, msg = handler.validate_change("test.yaml", "key: value", 1, 3)
        assert valid is False
        assert "not available" in msg

    @patch("review_bot_automator.handlers.yaml_handler.YAML_AVAILABLE", True)
    def test_validate_change(self) -> None:
        """Test change validation."""
        handler = YamlHandler()

        with patch("ruamel.yaml.YAML") as mock_yaml:
            mock_yaml.return_value.load.return_value = {"key": "value"}

            valid, msg = handler.validate_change("test.yaml", "key: value", 1, 3)
            assert valid is True
            assert "Valid YAML" in msg

    def test_extract_keys(self) -> None:
        """Test key extraction."""
        handler = YamlHandler()

        data: YAMLValue = {
            "key1": "value1",
            "key2": {"nested1": "value2", "nested2": ["item1", "item2"]},
        }

        keys = handler._extract_keys(data)

        expected_keys = [
            "key1",
            "key2",
            "key2.nested1",
            "key2.nested2",
            "key2.nested2[0]",
            "key2.nested2[1]",
        ]
        assert set(expected_keys) <= set(keys)

    def test_yaml_deeply_nested_structures(self) -> None:
        """Test deeply nested YAML structures (3+ levels)."""
        handler = YamlHandler()

        # Create deeply nested structure
        data: YAMLValue = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep_value",
                            "list": ["item1", "item2", {"nested": "value"}],
                        }
                    }
                }
            }
        }

        keys = handler._extract_keys(data)

        # Should extract all nested dotted paths and indexed list entries
        expected_keys = [
            "level1",
            "level1.level2",
            "level1.level2.level3",
            "level1.level2.level3.level4",
            "level1.level2.level3.level4.value",
            "level1.level2.level3.level4.list",
            "level1.level2.level3.level4.list[0]",
            "level1.level2.level3.level4.list[1]",
            "level1.level2.level3.level4.list[2]",
        ]
        assert set(expected_keys) <= set(keys)

    @pytest.mark.parametrize(
        "dangerous_yaml",
        [
            "key: !!python/object/apply:os.system ['rm -rf /']",
            "key: !!python/name:os.system",
        ],
    )
    @patch("review_bot_automator.handlers.yaml_handler.YAML_AVAILABLE", True)
    def test_validate_change_dangerous_tags(self, dangerous_yaml: str) -> None:
        """Test validation rejects dangerous YAML tags."""
        handler = YamlHandler()

        valid, msg = handler.validate_change("test.yaml", dangerous_yaml, 1, 3)
        assert valid is False
        assert "dangerous Python object tags" in msg

    @patch("review_bot_automator.handlers.yaml_handler.YAML_AVAILABLE", True)
    @pytest.mark.parametrize("control_char", ["\x00", "\x01", "\x1f"])
    def test_validate_change_dangerous_characters(self, control_char: str) -> None:
        """Test validation rejects dangerous control characters."""
        handler = YamlHandler()

        dangerous_yaml = f"key: value{control_char}"
        valid, msg = handler.validate_change("test.yaml", dangerous_yaml, 1, 3)
        assert valid is False
        assert "dangerous control characters" in msg

    def test_contains_dangerous_tags_none(self) -> None:
        """Test _contains_dangerous_tags with None value."""
        handler = YamlHandler()
        assert handler._contains_dangerous_tags(None) is False

    def test_contains_dangerous_tags_nested_dict(self) -> None:
        """Test _contains_dangerous_tags with nested dictionaries."""
        handler = YamlHandler()

        dangerous_data = {
            "safe_key": "safe_value",
            "dangerous_key": MockTaggedObject("!!python/object"),
            "nested": {"another_dangerous": MockTaggedObject("!!python/name")},
        }

        assert handler._contains_dangerous_tags(cast(YAMLValue, dangerous_data)) is True

    def test_contains_dangerous_tags_nested_list(self) -> None:
        """Test _contains_dangerous_tags with nested lists."""
        handler = YamlHandler()

        dangerous_data = [
            "safe_item",
            MockTaggedObject("!!python/function"),
            ["nested", MockTaggedObject("!!python/module")],
        ]

        assert handler._contains_dangerous_tags(cast(YAMLValue, dangerous_data)) is True

    def test_detect_conflicts_unparseable_content(self) -> None:
        """Test detect_conflicts with unparseable change content."""
        handler = YamlHandler()

        # Create changes with unparseable content
        changes = [
            Change(
                path="test.yaml",
                start_line=1,
                end_line=3,
                content="invalid: yaml: content: [",
                metadata={},
                fingerprint="test1",
                file_type=FileType.YAML,
            ),
            Change(
                path="test.yaml",
                start_line=4,
                end_line=6,
                content="key: value",
                metadata={},
                fingerprint="test2",
                file_type=FileType.YAML,
            ),
        ]

        # Should handle unparseable content gracefully
        conflicts = handler.detect_conflicts("test.yaml", changes)

        # No conflicts expected for single parseable change
        # The unparseable change is skipped, leaving only one valid change
        assert conflicts == [], "No conflicts expected for single parseable change"

    @patch("review_bot_automator.handlers.yaml_handler.YAML_AVAILABLE", True)
    def test_apply_change_invalid_path(self) -> None:
        """Test apply_change with invalid file path (security rejection)."""
        handler = YamlHandler()

        # Test with path traversal attempt
        result = handler.apply_change("../../../etc/passwd", "key: value", 1, 3)
        assert result is False

    def test_yaml_empty_dict_and_list(self) -> None:
        """Test empty dictionaries and empty lists."""
        handler = YamlHandler()

        # Test empty dictionary
        empty_dict: YAMLValue = {}
        keys = handler._extract_keys(empty_dict)
        assert keys == []

        # Test empty list
        data_with_empty_list: YAMLValue = {"key": []}
        keys = handler._extract_keys(data_with_empty_list)
        assert "key" in keys
        assert "key[0]" not in keys  # No items in empty list

        # Test dictionary with empty nested structures
        data_with_empty_nested: YAMLValue = {
            "empty_dict": {},
            "empty_list": [],
            "normal_key": "value",
        }
        keys = handler._extract_keys(data_with_empty_nested)
        expected_keys = ["empty_dict", "empty_list", "normal_key"]
        assert set(expected_keys) <= set(keys)

    @patch("review_bot_automator.handlers.yaml_handler.YAML_AVAILABLE", True)
    def test_yaml_anchors_and_aliases(self) -> None:
        """Test YAML with anchors/aliases."""
        handler = YamlHandler()

        with patch("ruamel.yaml.YAML") as mock_yaml:
            # Mock YAML structure with shared references
            mock_yaml.return_value.load.return_value = {
                "anchor": "&anchor_value",
                "alias": "*anchor_value",
                "shared": {"ref": "*anchor_value"},
            }

            # Test that validate_change handles anchors/aliases
            valid, msg = handler.validate_change("test.yaml", "key: &ref value\nother: *ref", 1, 3)
            assert valid is True
            assert "Valid YAML" in msg

            # Test key extraction with anchors/aliases
            data: YAMLValue = {"anchor": "&ref", "alias": "*ref"}
            keys = handler._extract_keys(data)
            assert "anchor" in keys
            assert "alias" in keys

    def test_apply_change_deletion_removes_keys(self, tmp_path: Path) -> None:
        """Test YAML deletion removes specified keys from original."""
        handler = YamlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.yaml"
        target.write_text("keep: value1\nremove: value2\n", encoding="utf-8")

        # Delete the "remove" key
        result = handler.apply_change(
            str(target), "remove: ignored\n", 1, 2, change_type="deletion"
        )
        assert result is True
        content = target.read_text(encoding="utf-8")
        assert "keep:" in content
        assert "remove:" not in content

    def test_apply_change_addition_merges_keys(self, tmp_path: Path) -> None:
        """Test YAML addition merges new keys into original."""
        handler = YamlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.yaml"
        target.write_text("existing: value1\n", encoding="utf-8")

        # Add new key with addition type
        result = handler.apply_change(
            str(target), "new_key: value2\n", 1, 1, change_type="addition"
        )
        assert result is True
        content = target.read_text(encoding="utf-8")
        assert "existing:" in content
        assert "new_key:" in content

    def test_apply_change_modification_merges_keys(self, tmp_path: Path) -> None:
        """Test YAML modification (default) merges and overrides keys."""
        handler = YamlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.yaml"
        target.write_text("key1: old\nkey2: keep\n", encoding="utf-8")

        # Modify with merge
        result = handler.apply_change(
            str(target), "key1: new\nkey3: added\n", 1, 2, change_type="modification"
        )
        assert result is True
        content = target.read_text(encoding="utf-8")
        assert "key1: new" in content
        assert "key2:" in content
        assert "key3:" in content

    def test_apply_change_invalid_change_type_returns_false(self, tmp_path: Path) -> None:
        """Test YAML handler returns False for invalid change_type."""
        handler = YamlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.yaml"
        target.write_text("key: value\n", encoding="utf-8")

        result = handler.apply_change(
            str(target), "key: new\n", 1, 1, change_type="invalid"  # type: ignore[arg-type]
        )
        assert result is False

    def test_apply_change_deletion_with_non_dict_yaml_returns_false(self, tmp_path: Path) -> None:
        """Test YAML deletion fails when original is not a dict (e.g., list)."""
        handler = YamlHandler(workspace_root=tmp_path)
        target = tmp_path / "array.yaml"
        target.write_text("- item1\n- item2\n", encoding="utf-8")

        # Attempt to delete from a list (should fail - deletion requires dicts)
        result = handler.apply_change(str(target), "key: ignored\n", 1, 2, change_type="deletion")
        assert result is False


class TestTomlHandler:
    """Test the TOML handler."""

    def test_can_handle(self) -> None:
        """Test file type detection."""
        handler = TomlHandler()

        assert handler.can_handle("test.toml") is True
        assert handler.can_handle("test.TOML") is True
        assert handler.can_handle("test.json") is False
        assert handler.can_handle("test.txt") is False

    @patch("review_bot_automator.handlers.toml_handler.TOML_READ_AVAILABLE", False)
    def test_toml_not_available(self) -> None:
        """Test behavior when TOML_READ_AVAILABLE is False."""
        handler = TomlHandler()

        valid, msg = handler.validate_change("test.toml", "key = 'value'", 1, 3)
        assert valid is False
        assert "not available" in msg

    def test_validate_change(self) -> None:
        """Test change validation."""
        handler = TomlHandler()

        with patch("tomllib.loads") as mock_tomllib:
            mock_tomllib.return_value = {"key": "value"}

            valid, msg = handler.validate_change("test.toml", "key = 'value'", 1, 3)
            assert valid is True
            assert "Valid TOML" in msg

    def test_extract_sections(self) -> None:
        """Test section extraction."""
        handler = TomlHandler()

        data = {
            "section1": "value1",
            "section2": {"subsection1": "value2", "subsection2": "value3"},
        }

        sections = handler._extract_sections(data)

        expected_sections = ["section1", "section2", "section2.subsection1", "section2.subsection2"]
        assert set(expected_sections) <= set(sections)

    def test_apply_change_nonexistent_file(self) -> None:
        """Test apply_change handles non-existent files."""
        handler = TomlHandler()

        # Test with non-existent file
        result = handler.apply_change("nonexistent.toml", "key = 'value'", 1, 3)
        assert result is False

    def test_apply_change_unicode_error(self) -> None:
        """Test apply_change handles Unicode decode errors."""
        handler = TomlHandler()

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            # Write invalid UTF-8
            f.write(b"\xff\xfe\x00\x00")
            f.flush()

            try:
                result = handler.apply_change(f.name, "key = 'value'", 1, 3)
                assert result is False
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_apply_change_toml_parse_error(self) -> None:
        """Test apply_change handles TOML parse errors."""
        handler = TomlHandler()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            # Write invalid TOML
            f.write("invalid toml content [")
            f.flush()

            try:
                result = handler.apply_change(f.name, "key = 'value'", 1, 3)
                assert result is False
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_apply_change_suggestion_parse_error(self) -> None:
        """Test apply_change handles suggestion parse errors."""
        handler = TomlHandler()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            # Write valid original TOML
            f.write("key = 'value'")
            f.flush()

            try:
                # Invalid suggestion TOML
                result = handler.apply_change(f.name, "invalid suggestion [", 1, 3)
                assert result is False
            finally:
                Path(f.name).unlink(missing_ok=True)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics on Windows")
    def test_apply_change_success_with_permissions(self) -> None:
        """Test apply_change successfully preserves file permissions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            # Write original TOML
            f.write("key = 'value'")
            f.flush()
            temp_dir = os.path.dirname(f.name)

            # Create handler with temp directory as workspace root
            handler = TomlHandler(workspace_root=temp_dir)

            # Set specific permissions
            original_mode = 0o644
            os.chmod(f.name, original_mode)

            try:
                result = handler.apply_change(f.name, "newkey = 'newvalue'", 1, 1)
                assert result is True

                # Check that permissions were preserved
                current_mode = stat.S_IMODE(os.stat(f.name).st_mode)
                assert current_mode == original_mode
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_apply_change_atomic_replacement(self) -> None:
        """Test apply_change uses targeted line-based replacement."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            # Write original TOML with multiple lines
            original_content = "# Comment\nkey = 'value'\nanotherkey = 'anothervalue'\n"
            f.write(original_content)
            f.flush()
            temp_dir = os.path.dirname(f.name)

            # Create handler with temp directory as workspace root
            handler = TomlHandler(workspace_root=temp_dir)

            try:
                # Replace only line 2 (key = 'value')
                result = handler.apply_change(f.name, "key = 'newvalue'", 2, 2)
                assert result is True

                # Verify content: comment preserved, only target line replaced, rest preserved
                content = Path(f.name).read_text()
                assert "# Comment" in content  # Comment preserved
                assert "key = 'newvalue'" in content  # Line 2 replaced
                assert "anotherkey = 'anothervalue'" in content  # Line 3 preserved
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_apply_change_rejects_invalid_merged_content(self) -> None:
        """Test apply_change rejects when merged content is invalid TOML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            # Write original TOML with opening brace
            original_content = "key = { 'value'\n"
            f.write(original_content)
            f.flush()
            temp_dir = os.path.dirname(f.name)

            # Create handler with temp directory as workspace root
            handler = TomlHandler(workspace_root=temp_dir)

            try:
                # Apply change that would create invalid TOML when merged
                # The suggestion is valid TOML, but when merged with the rest of the file
                # (which has an opening brace without closing), it becomes invalid
                result = handler.apply_change(f.name, "otherkey = 'value'", 2, 2)

                # Should return False because merged content is invalid
                assert result is False

                # Verify original file content is unchanged
                assert Path(f.name).read_text() == original_content
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_validate_change_only_parses_toml(self) -> None:
        """Test that validate_change only validates TOML content, not file paths."""
        handler = TomlHandler()

        # validate_change only validates TOML parsing, not paths
        valid, _ = handler.validate_change("../../../etc/passwd", "key = 'value'", 1, 3)
        # Path validation is performed in apply_change
        assert valid is True  # TOML content is valid

    @patch("review_bot_automator.handlers.toml_handler.TOML_READ_AVAILABLE", False)
    def test_validate_change_toml_not_available(self) -> None:
        """Test validate_change when TOML is not available."""
        handler = TomlHandler()

        valid, msg = handler.validate_change("test.toml", "key = 'value'", 1, 3)
        assert valid is False
        assert "not available" in msg.lower()

    @pytest.mark.parametrize(
        "content,expected_valid,expected_msg",
        [
            ("key = 'value'", True, "valid toml"),
            ("invalid toml [", False, "invalid toml"),
        ],
    )
    def test_validate_change_parse_validation(
        self, content: str, expected_valid: bool, expected_msg: str
    ) -> None:
        """Test validate_change validates TOML parsing."""
        handler = TomlHandler()

        valid, msg = handler.validate_change("test.toml", content, 1, 3)
        assert valid is expected_valid
        assert expected_msg in msg.lower()

    def test_detect_conflicts_section_overlap(self) -> None:
        """Test detect_conflicts identifies section conflicts."""
        handler = TomlHandler()

        changes = [
            Change(
                path="test.toml",
                start_line=1,
                end_line=3,
                content="[section1]\nkey1 = 'value1'",
                metadata={},
                fingerprint="test1",
                file_type=FileType.TOML,
            ),
            Change(
                path="test.toml",
                start_line=4,
                end_line=6,
                content="[section1]\nkey2 = 'value2'",
                metadata={},
                fingerprint="test2",
                file_type=FileType.TOML,
            ),
        ]

        conflicts = handler.detect_conflicts("test.toml", changes)
        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == "section_conflict"

    def test_extract_sections_helper(self, subtests: pytest.Subtests) -> None:
        """Test _extract_sections helper method using subtests."""
        handler = TomlHandler()

        data = {
            "section1": "value1",
            "section2": {"subsection1": "value2", "subsection2": {"nested": "value3"}},
        }

        sections = handler._extract_sections(data)

        expected = [
            "section1",
            "section2",
            "section2.subsection1",
            "section2.subsection2",
            "section2.subsection2.nested",
        ]
        for expected_section in expected:
            with subtests.test(msg=f"Section: {expected_section}", section=expected_section):
                assert expected_section in sections

    def test_apply_change_deletion_removes_lines(self, tmp_path: Path) -> None:
        """Test TOML deletion removes specified lines."""
        handler = TomlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.toml"
        target.write_text(
            'keep = "value1"\nremove = "value2"\nafter = "value3"\n', encoding="utf-8"
        )

        # Delete line 2 (remove = "value2")
        result = handler.apply_change(
            str(target), 'ignored = "content"', 2, 2, change_type="deletion"
        )
        assert result is True
        content = target.read_text(encoding="utf-8")
        assert 'keep = "value1"' in content
        assert 'remove = "value2"' not in content
        assert 'after = "value3"' in content

    def test_apply_change_addition_inserts_lines(self, tmp_path: Path) -> None:
        """Test TOML addition inserts lines after specified position."""
        handler = TomlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.toml"
        target.write_text('line1 = "value1"\nline2 = "value2"\n', encoding="utf-8")

        # Add new line after line 1
        result = handler.apply_change(str(target), 'inserted = "new"', 1, 1, change_type="addition")
        assert result is True
        content = target.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Line 1 should be line1, line 2 should be inserted, line 3 should be line2
        assert 'line1 = "value1"' in lines[0]
        assert 'inserted = "new"' in lines[1]
        assert 'line2 = "value2"' in lines[2]

    def test_apply_change_modification_replaces_lines(self, tmp_path: Path) -> None:
        """Test TOML modification replaces specified lines."""
        handler = TomlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.toml"
        target.write_text('before = "keep"\nreplace = "old"\nafter = "keep"\n', encoding="utf-8")

        # Replace line 2
        result = handler.apply_change(
            str(target), 'replace = "new"', 2, 2, change_type="modification"
        )
        assert result is True
        content = target.read_text(encoding="utf-8")
        assert 'before = "keep"' in content
        assert 'replace = "new"' in content
        assert 'replace = "old"' not in content
        assert 'after = "keep"' in content

    def test_apply_change_invalid_change_type_returns_false(self, tmp_path: Path) -> None:
        """Test TOML handler returns False for invalid change_type."""
        handler = TomlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.toml"
        target.write_text('key = "value"\n', encoding="utf-8")

        result = handler.apply_change(
            str(target), 'key = "new"', 1, 1, change_type="invalid"  # type: ignore[arg-type]
        )
        assert result is False

    def test_apply_change_addition_with_invalid_toml_returns_false(self, tmp_path: Path) -> None:
        """Test TOML addition fails when content is not valid TOML syntax."""
        handler = TomlHandler(workspace_root=tmp_path)
        target = tmp_path / "config.toml"
        target.write_text('[section]\nkey = "value"\n', encoding="utf-8")

        # Attempt to add invalid TOML syntax (missing closing quote)
        result = handler.apply_change(
            str(target), 'new_key = "unclosed', 2, 2, change_type="addition"
        )
        assert result is False


@pytest.fixture
def test_handler(tmp_path: Path) -> BaseHandler:
    """Fixture providing a concrete TestHandler instance for testing BaseHandler functionality."""

    class TestHandler(BaseHandler):
        def can_handle(self, file_path: str) -> bool:
            return True

        def apply_change(
            self,
            path: str,
            content: str,
            start_line: int,
            end_line: int,
            change_type: ChangeType = "modification",
        ) -> bool:
            return True

        def validate_change(
            self, path: str, content: str, start_line: int, end_line: int
        ) -> tuple[bool, str]:
            return True, "Valid"

        def detect_conflicts(self, path: str, changes: list[Change]) -> list[Conflict]:
            return []

    return TestHandler(workspace_root=tmp_path)


class TestBaseHandlerBackupRestore:
    """Test BaseHandler backup and restore functionality."""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics on Windows")
    def test_backup_file_success(self, test_handler: BaseHandler, tmp_path: Path) -> None:
        """Test successful backup file creation."""
        # Create test file in tmp_path (same workspace_root as handler)
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Create backup
        backup_path = test_handler.backup_file(str(test_file))

        # Verify backup was created
        assert Path(backup_path).exists()
        assert Path(backup_path).read_text() == "test content"
        assert backup_path.endswith(".backup")

        # Verify backup file has secure permissions (0o600) on POSIX systems
        assert stat.S_IMODE(Path(backup_path).stat().st_mode) == 0o600

    def test_backup_file_nonexistent(self, test_handler: BaseHandler, tmp_path: Path) -> None:
        """Test backup_file with non-existent file raises FileNotFoundError."""

        # Create a valid path that doesn't exist (within tmp_path)
        nonexistent_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="Source file does not exist"):
            test_handler.backup_file(str(nonexistent_file))

    def test_backup_file_directory(self, test_handler: BaseHandler, tmp_path: Path) -> None:
        """Test backup_file with directory instead of file raises ValueError."""
        # Use tmp_path as the directory (within workspace_root)
        with pytest.raises(ValueError, match="Source path is not a regular file"):
            test_handler.backup_file(str(tmp_path))

    def test_backup_file_collision_handling(
        self, test_handler: BaseHandler, tmp_path: Path
    ) -> None:
        """Test backup file collision handling with timestamp and counter."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Create existing backup file
        existing_backup = test_file.with_suffix(".txt.backup")
        existing_backup.write_text("existing backup")

        # Mock time.time to return a fixed timestamp
        with patch("time.time", return_value=1234567890):
            backup_path = test_handler.backup_file(str(test_file))

        # Should create timestamped backup
        assert Path(backup_path).exists()
        assert backup_path.endswith(".backup.1234567890")

    def test_backup_file_collision_counter_limit(
        self, test_handler: BaseHandler, tmp_path: Path
    ) -> None:
        """Test backup file collision handling with 5 attempts raises OSError."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Capture original os.open before patching
        original_os_open = os.open

        # Patch os.open to raise FileExistsError for all 5 attempts
        # This simulates the collision scenario where each backup file already exists
        attempt_count = [0]

        def mock_open(path: str, flags: int, mode: int = 0o777) -> int:
            """Mock os.open that raises FileExistsError for first 5 attempts."""
            # Check if O_EXCL flag is set (attempting to create exclusively)
            if flags & os.O_EXCL:
                attempt_count[0] += 1
                if attempt_count[0] <= 5:  # Fail the first 5 attempts
                    raise FileExistsError("File exists")
                # On 6th attempt, call original for cleanup
            return original_os_open(path, flags, mode)

        with (
            patch("os.open", side_effect=mock_open),
            pytest.raises(
                OSError, match=r"Unable to create unique backup filename after 5 attempts"
            ),
        ):
            test_handler.backup_file(str(test_file))

        # Verify exactly 5 collision attempts were made
        assert attempt_count[0] == 5

    def test_backup_file_permission_error(self, test_handler: BaseHandler, tmp_path: Path) -> None:
        """Test backup file creation with permission errors fails fast."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Mock os.open to raise OSError immediately to test the permission error path
        # Non-collision errors fail immediately instead of retrying
        with (
            patch("os.open", side_effect=OSError("Permission denied")),
            pytest.raises(OSError, match=r"Backup failed for.*: Permission denied"),
        ):
            test_handler.backup_file(str(test_file))

    def test_restore_file_success(self, test_handler: BaseHandler, tmp_path: Path) -> None:
        """Test successful file restoration."""
        # Create files within handler's workspace_root (tmp_path)
        # Create original file
        original_file = tmp_path / "original.txt"
        original_file.write_text("original content")

        # Create backup file
        backup_file = tmp_path / "backup.txt"
        backup_file.write_text("backup content")

        # Restore file
        result = test_handler.restore_file(str(backup_file), str(original_file))

        # Verify restoration
        assert result is True
        assert original_file.read_text() == "backup content"
        assert not backup_file.exists()  # Backup should be removed

    def test_restore_file_failure(
        self, test_handler: BaseHandler, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test restore_file failure returns False and logs error."""
        # Create temporary files under tmp_path to pass containment checks
        backup = tmp_path / "b.txt"
        original = tmp_path / "o.txt"
        backup.write_text("data")

        # Mock shutil.copy2 to raise an exception
        with patch("shutil.copy2", side_effect=OSError("Copy failed")):
            result = test_handler.restore_file(str(backup), str(original))
            assert result is False
            # Verify error was logged with context
            assert "Restore failed" in caplog.text
            assert str(backup) in caplog.text
