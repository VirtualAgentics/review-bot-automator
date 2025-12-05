# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""JSON handler for applying CodeRabbit suggestions with AST validation.

This handler provides JSON-aware suggestion application with duplicate key detection,
smart merging, and structural validation to prevent issues like the package.json
duplication problem.
"""

import contextlib
import json
import logging
import os
import stat
import tempfile
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

from review_bot_automator.core.models import Change, Conflict
from review_bot_automator.handlers.base import BaseHandler
from review_bot_automator.security.input_validator import InputValidator
from review_bot_automator.utils.path_utils import resolve_file_path

JsonValue: TypeAlias = dict[str, Any] | list[Any] | str | int | float | bool | None


class JsonHandler(BaseHandler):
    """Handler for JSON files with duplicate key detection and smart merging."""

    def __init__(self, workspace_root: str | PathLike[str] | None = None) -> None:
        """Initialize the JsonHandler instance and configure a module-level logger.

        Args:
            workspace_root: Root directory for validating absolute paths.
                If None, defaults to current working directory.
        """
        super().__init__(workspace_root)
        self.logger = logging.getLogger(__name__)

    def can_handle(self, file_path: str) -> bool:
        """Determine whether a file path refers to a JSON file.

        Returns:
            bool: `True` if the given `file_path` ends with ".json" (case-insensitive),
                `False` otherwise.
        """
        return file_path.lower().endswith(".json")

    def apply_change(
        self,
        path: str,
        content: str,
        start_line: int,
        end_line: int,
        change_type: str = "modification",
    ) -> bool:
        """Apply a JSON change to a file based on change_type.

        This method implements security validation to prevent path traversal attacks and
        ensures safe JSON processing. Path validation occurs before any file operations.

        Security Features:
            - Path traversal protection using InputValidator.validate_file_path()
            - Duplicate key detection to prevent JSON structure corruption
            - Safe JSON parsing with proper error handling
            - File existence verification before processing

        Args:
            path (str): Filesystem path to the target JSON file. Must pass security validation.
            content (str): Suggested JSON content; may be a complete JSON object or a
                partial fragment.
            start_line (int): Start line of the suggestion in the original file (used as
                contextual hint for merging).
            end_line (int): End line of the suggestion in the original file (used as
                contextual hint for merging).
            change_type (str): Type of change to apply. One of:
                - "modification": Merge suggestion into original JSON (default).
                - "addition": Same as modification for JSON (keys are added/merged).
                - "deletion": Remove keys specified in content from the JSON file.

        Returns:
            bool: `True` if the file was successfully updated with the merged JSON,
                `False` otherwise.

        Example:
            >>> handler = JsonHandler()
            >>> # Valid path and JSON
            >>> handler.apply_change("config.json", '{"key": "value"}', 1, 1)
            True
            >>> # Path traversal attempt - rejected
            >>> handler.apply_change("../../../etc/passwd", '{"key": "value"}', 1, 1)
            False

        Warning:
            This method validates file paths to prevent directory traversal attacks.
            Invalid paths are rejected before any file operations occur.

        See Also:
            InputValidator.validate_file_path: Path security validation
            validate_change: Content validation for JSON structure
        """
        # Validate file path to prevent path traversal attacks
        # Use workspace root for absolute path containment check
        if not InputValidator.validate_file_path(
            path, allow_absolute=True, base_dir=str(self.workspace_root)
        ):
            self.logger.error(f"Invalid file path rejected: {path}")
            return False

        # Resolve path relative to workspace_root with strict containment to workspace
        try:
            file_path = resolve_file_path(
                path,
                self.workspace_root,
                allow_absolute=True,
                validate_workspace=True,
                enforce_containment=True,
            )
        except (ValueError, OSError) as e:
            self.logger.error(f"Path resolution failed for {path}: {type(e).__name__}: {e}")
            return False

        # Reject symlinks in the target path or any parent within the workspace
        try:
            for p in (file_path, *file_path.parents):
                if p == self.workspace_root:
                    break  # don't traverse above workspace root
                if p.is_symlink():
                    self.logger.error(f"Symlink detected in path component: {p}")
                    return False
        except OSError as e:
            self.logger.error(f"Symlink check failed for {file_path}: {e}")
            return False

        # Parse original file (accept any valid JSON, not only dict)
        try:
            original_content = file_path.read_text(encoding="utf-8")
            try:
                original_value = self._loads_strict(original_content)
            except (json.JSONDecodeError, ValueError) as pe:
                self.logger.error(f"Error parsing JSON file {file_path}: {type(pe).__name__}: {pe}")
                return False
        except (OSError, UnicodeDecodeError) as e:
            self.logger.error(f"Error reading JSON file {file_path}: {type(e).__name__}: {e}")
            return False

        # Parse suggestion - support both dict and non-dict JSON
        try:
            suggestion_value = self._loads_strict(content)
        except (json.JSONDecodeError, ValueError):
            # Invalid JSON - try partial suggestion fallback only if original is dict
            if isinstance(original_value, dict):
                return self._apply_partial_suggestion(
                    file_path, original_value, content, start_line, end_line
                )
            else:
                self.logger.error("Partial suggestion not supported for non-dict JSON")
                return False

        # Validate change_type
        valid_change_types = {"addition", "modification", "deletion"}
        if change_type not in valid_change_types:
            self.logger.error(
                f"Invalid change_type {change_type!r}; must be one of {valid_change_types}"
            )
            return False

        # Handle change_type and dict vs non-dict JSON
        merged_data: dict[str, Any] | list[Any] | str | int | float | bool | None
        if change_type == "deletion":
            # Deletion: remove keys specified in content from original
            if isinstance(suggestion_value, dict) and isinstance(original_value, dict):
                merged_data = {k: v for k, v in original_value.items() if k not in suggestion_value}
            else:
                self.logger.error("JSON deletion requires both original and content to be dicts")
                return False
        elif isinstance(suggestion_value, dict) and isinstance(original_value, dict):
            # Dict+dict: merge (both modification and addition)
            merged_data = self._smart_merge_json(
                original_value, suggestion_value, start_line, end_line
            )
        else:
            # Any non-dict on either side -> full replacement with suggestion
            merged_data = suggestion_value

        # Write with atomic operation
        original_mode = None
        temp_path: Path | None = None

        try:
            # Capture original file permissions if file exists
            if file_path.exists():
                original_mode = os.stat(file_path).st_mode

            # Create temp file in same directory
            temp_dir = file_path.parent
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=temp_dir,
                prefix=f".{file_path.name}.tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                # Write JSON content
                temp_file.write(json.dumps(merged_data, indent=2, ensure_ascii=False) + "\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Ensure data written to disk

            # Apply original permissions if we have them
            if original_mode is not None:
                os.chmod(temp_path, stat.S_IMODE(original_mode))

            # Atomically replace the original file
            os.replace(temp_path, file_path)

            # Best-effort directory fsync for durability
            try:
                parent_dir = file_path.parent
                dir_fd = os.open(parent_dir, os.O_DIRECTORY | os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (OSError, AttributeError) as e:
                # Ignore fsync errors (may not be supported on all platforms)
                self.logger.debug(f"Directory fsync failed (non-fatal): {e}")

            return True

        except OSError as e:
            temp_path_str = str(temp_path) if temp_path is not None else "N/A"
            self.logger.error(
                f"Error writing JSON file {path}: {e} (temp: {temp_path_str}, target: {file_path})"
            )
            return False

        finally:
            # Always clean up temp file if it exists
            if temp_path is not None and temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()  # Ignore cleanup errors

    def validate_change(
        self, path: str, content: str, start_line: int, end_line: int
    ) -> tuple[bool, str]:
        """Validate a JSON suggestion without applying changes.

        This method performs security validation on the file path before validating JSON content.
        It ensures that path traversal attacks are prevented and JSON structure is valid.

        Security Features:
            - Path traversal protection using InputValidator.validate_file_path()
            - JSON syntax validation with proper error reporting
            - Duplicate key detection to prevent structure corruption
            - Safe JSON parsing without side effects

        Args:
            path (str): File path of the JSON being validated. Must pass security validation.
            content (str): JSON text to validate.
            start_line (int): Start line of the suggested change (contextual; not used for
                validation).
            end_line (int): End line of the suggested change (contextual; not used for
                validation).

        Returns:
            tuple[bool, str]: `True, "Valid JSON"` if `content` is valid JSON and contains no
                duplicate keys; otherwise `False` with an error message (either
                `"Duplicate keys detected"` or `"Invalid JSON: <error>"`).

        Example:
            >>> handler = JsonHandler()
            >>> # Valid JSON
            >>> handler.validate_change("config.json", '{"key": "value"}', 1, 1)
            (True, "Valid JSON")
            >>> # Invalid JSON syntax
            >>> handler.validate_change("config.json", '{"key": value}', 1, 1)
            (False, "Invalid JSON: Expecting value: line 1 column 10 (char 9)")
            >>> # Path traversal attempt - rejected
            >>> handler.validate_change("../../../etc/passwd", '{"key": "value"}', 1, 1)
            (False, "Invalid file path: path traversal detected")

        Warning:
            This method validates file paths to prevent directory traversal attacks.
            Invalid paths are rejected before JSON validation occurs.

        See Also:
            InputValidator.validate_file_path: Path security validation
            apply_change: Apply validated changes to files
        """
        # Validate file path to prevent path traversal attacks
        # Use workspace root for absolute path containment check
        if not InputValidator.validate_file_path(
            path, allow_absolute=True, base_dir=str(self.workspace_root)
        ):
            return False, "Invalid file path: path traversal detected"

        try:
            self._loads_strict(content)
            return True, "Valid JSON"
        except (json.JSONDecodeError, ValueError) as e:
            return False, f"Invalid JSON: {e}"

    def detect_conflicts(self, path: str, changes: list[Change]) -> list[Conflict]:
        """Identify key-based conflicts among JSON changes for a given file.

        Args:
            path (str): Path to the JSON file being analyzed.
            changes (list[Change]): List of Change objects whose `content` is expected to be
                JSON; changes with unparsable JSON are ignored.

        Returns:
            list[Conflict]: Conflicts where multiple changes target the same top-level JSON key.
                Each Conflict contains the file path, a line range spanning the minimum start line
                to maximum end line of involved changes, the list of conflicting changes, a
                `conflict_type` of `"key_conflict"`, a `severity` of `"medium"`, and an
                `overlap_percentage` quantifying how much the changes' line ranges overlap.
        """
        conflicts: list[Conflict] = []

        # Group changes by key
        key_changes: dict[str, list[Change]] = {}
        for change in changes:
            try:
                data = self._loads_strict(change.content)
                if isinstance(data, dict):
                    for key in data:
                        if key not in key_changes:
                            key_changes[key] = []
                        key_changes[key].append(change)
            except (json.JSONDecodeError, ValueError):
                continue

        # Find conflicts (multiple changes to same key)
        for _key, key_change_list in key_changes.items():
            if len(key_change_list) > 1:
                # Calculate actual overlap percentage
                overlap_percentage = self._calculate_overlap_percentage(key_change_list)

                # Compute min/max lines to handle unsorted changes
                min_start = min(c.start_line for c in key_change_list)
                max_end = max(c.end_line for c in key_change_list)

                conflicts.append(
                    Conflict(
                        file_path=path,
                        line_range=(min_start, max_end),
                        changes=key_change_list,
                        conflict_type="key_conflict",
                        severity="medium",
                        overlap_percentage=overlap_percentage,
                    )
                )

        return conflicts

    def _calculate_overlap_percentage(self, changes: list[Change]) -> float:
        """Calculate the percentage of overlapping line coverage among multiple changes.

        Args:
            changes (list[Change]): List of Change objects; each must have `start_line` and
                `end_line` attributes defining the inclusive line range of the change.

        Returns:
            float: Percentage of lines that are covered by more than one change, in the range
                0.0 to 100.0. Returns 0.0 if fewer than two changes are provided or if no lines
                are covered.
        """
        if not changes:
            return 0.0

        # Line-sweep algorithm: convert each inclusive range to two events (start:+1, end+1:-1),
        # sort events, sweep while tracking active count; accumulate union when active>0 and
        # overlap when active>1. Using end+1 models inclusive ranges correctly.
        events: list[tuple[int, int]] = []
        for c in changes:
            start = min(c.start_line, c.end_line)
            end = max(c.start_line, c.end_line)
            events.append((start, +1))
            events.append((end + 1, -1))  # inclusive end
        events.sort()

        active = 0
        prev: int | None = None
        union_len = 0
        overlap_len = 0

        for pos, delta in events:
            if prev is not None:
                span = pos - prev
                if active > 0:
                    union_len += span
                if active > 1:
                    overlap_len += span
            active += delta
            prev = pos

        return 0.0 if union_len <= 0 else (overlap_len / union_len) * 100.0

    def _smart_merge_json(
        self, original: dict[str, Any], suggestion: dict[str, Any], start_line: int, end_line: int
    ) -> dict[str, Any]:
        """Merge a suggested JSON object into the original JSON.

        Args:
            original (dict[str, Any]): The existing JSON object from the file.
            suggestion (dict[str, Any]): The suggested JSON fragment to apply.
            start_line (int): The starting line number of the suggestion in the file (1-based).
            end_line (int): The ending line number of the suggestion in the file (1-based).

        Returns:
            dict[str, Any]: The resulting merged JSON object; the suggestion is returned intact
                if it is considered a complete replacement, otherwise keys from the suggestion
                overwrite or are added to the original.
        """
        # Strategy 1: If suggestion is complete object, use it
        if self._is_complete_object(suggestion, original):
            return suggestion

        # Strategy 2: If suggestion is partial, merge specific keys
        result = original.copy()
        for key, value in suggestion.items():
            result[key] = value

        return result

    def _is_complete_object(self, suggestion: dict[str, Any], original: dict[str, Any]) -> bool:
        """Determine whether the suggestion should be treated as a complete replacement.

        This uses a heuristic: the suggestion is considered complete if it contains at least all
        top-level keys present in the original.

        Returns:
            True if the suggestion contains at least all top-level keys from the original, False
                otherwise.
        """
        # Heuristic: suggestion has all top-level keys from original
        original_keys = set(original.keys())
        suggestion_keys = set(suggestion.keys())
        return suggestion_keys >= original_keys

    def _apply_partial_suggestion(
        self,
        file_path: Path,
        original_data: dict[str, Any],
        suggestion: str,
        start_line: int,
        end_line: int,
    ) -> bool:
        """Attempt to apply a partial (non-parseable) JSON suggestion to an existing JSON file.

        Args:
            file_path (Path): Path to the JSON file being modified.
            original_data (dict[str, Any]): Parsed JSON content of the original file.
            suggestion (str): Suggested content that is not valid/complete JSON.
            start_line (int): Start line number of the suggested change in the file.
            end_line (int): End line number of the suggested change in the file.

        Returns:
            bool: `False` if the suggestion was not applied. The method currently logs a warning
                and does not perform a merge.
        """
        # For now, fall back to plain text replacement
        # This is a simplified approach - in practice, you might want more sophisticated
        # parsing of partial JSON structures
        self.logger.warning("Partial JSON suggestion detected, using fallback method")
        return False

    def _loads_strict(self, s: str) -> JsonValue:
        """Parse JSON and raise ValueError on duplicate keys.

        Args:
            s: JSON string to parse.

        Returns:
            Parsed JSON data (dict, list, or primitive).

        Raises:
            json.JSONDecodeError: If the JSON is malformed.
            ValueError: If duplicate keys are detected.
        """

        def _no_dupes_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            obj: dict[str, Any] = {}
            for k, v in pairs:
                if k in obj:
                    raise ValueError(f"Duplicate key detected: {k}")
                obj[k] = v
            return obj

        result: JsonValue = json.loads(s, object_pairs_hook=_no_dupes_object_pairs_hook)
        return result
