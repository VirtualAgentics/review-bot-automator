# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""TOML handler for applying CodeRabbit suggestions with AST validation.

This handler provides TOML-aware suggestion application with structure validation.
"""

import contextlib
import logging
import os
import stat
import tempfile
import warnings
from os import PathLike
from pathlib import Path
from typing import Any

from review_bot_automator.core.models import Change, ChangeType, Conflict
from review_bot_automator.handlers.base import BaseHandler
from review_bot_automator.security.input_validator import InputValidator
from review_bot_automator.utils.path_utils import resolve_file_path

try:
    import tomllib

    TOML_READ_AVAILABLE = True
except ImportError:
    TOML_READ_AVAILABLE = False


class TomlHandler(BaseHandler):
    """Handler for TOML files with structure validation.

    Note: Suggestions must be self-contained, complete TOML documents. Partial
    fragments (e.g., a key without its table header) will be rejected during
    validation. This validation occurs before the merge operation.

    Example valid suggestion::

        [tool.poetry]
        name = "my-project"
        version = "1.0.0"

    Example invalid suggestion (fragment without table header)::

        name = "my-project"  # Invalid: missing [tool.poetry] header
    """

    _warned_about_temp_workspace: bool

    def __init__(self, workspace_root: str | PathLike[str] | None = None) -> None:
        """Initialize the TOML handler.

        Initializes a module-level logger and verifies TOML parsing/writing availability.
        NOTE: This handler requires Python 3.11+ `tomllib` for read-only TOML parsing.
        No TOML serialization/write library is needed because writes use deterministic
        line-based replacements with atomic file handling. No additional write-only
        dependencies are required.

        Args:
            workspace_root: Root directory for validating absolute paths.
                If None, defaults to current working directory.

        Returns:
            None

        Raises:
            TypeError: If workspace_root has an invalid type.
            OSError: If workspace_root path cannot be accessed.
        """
        super().__init__(workspace_root)
        self.logger = logging.getLogger(__name__)
        self._warned_about_temp_workspace = False
        if not TOML_READ_AVAILABLE:
            self.logger.warning("TOML parsing support unavailable. Install tomllib (Python 3.11+)")

    def can_handle(self, file_path: str) -> bool:
        """Determine whether the handler supports the given file path.

        Args:
            file_path (str): The file path to evaluate.

        Returns:
            bool: True if the file path ends with ".toml" (case-insensitive); False otherwise.
        """
        return file_path.lower().endswith(".toml")

    def apply_change(
        self,
        path: str,
        content: str,
        start_line: int,
        end_line: int,
        change_type: ChangeType = "modification",
    ) -> bool:
        """Apply a TOML change to the file based on change_type.

        Performs secure path validation, validates the suggestion as proper TOML, then applies
        the change according to its type. This preserves all original formatting, comments, and
        section ordering outside the target region.

        Args:
            path (str): Filesystem path to the TOML file to modify. Must pass security validation.
            content (str): TOML-formatted suggestion content. For modifications and additions,
                this is inserted. For deletions, this is ignored.
            start_line (int): One-based start line of the region (inclusive).
            end_line (int): One-based end line of the region (inclusive).
            change_type (str): Type of change to apply. One of:
                - "modification": Replace lines start_line to end_line with content (default).
                - "addition": Insert content AFTER end_line, preserving all existing lines.
                - "deletion": Remove lines start_line to end_line, content is ignored.

        Returns:
            bool: True if the change was applied successfully; False if validation
                fails (e.g., path traversal detected, invalid TOML syntax).
                When returning False, the method logs an error message with details.

        Raises:
            ValueError: If line range is invalid (start_line < 1, end_line < start_line,
                or end_line > total_lines).
        """
        # Parse temp bypass configuration once
        env_val = (os.getenv("ALLOW_TEMP_OUTSIDE_WORKSPACE") or "").strip().lower()
        allow_temp_outside = env_val in {"1", "true", "yes", "on"}
        path_obj = Path(path)
        tempdir = Path(tempfile.gettempdir())
        is_temp_file = (
            path_obj.is_absolute() and allow_temp_outside and self._is_temp_file(path_obj, tempdir)
        )

        # Validate file path to prevent path traversal attacks
        # Use workspace root for absolute path containment check
        validation_passed = InputValidator.validate_file_path(
            path, allow_absolute=True, base_dir=str(self.workspace_root)
        )

        # Early return: validation failed and no bypass applies
        if not validation_passed and not is_temp_file:
            self.logger.error(f"Invalid file path rejected: {path}")
            return False

        # Early bypass: validation failed but temp file bypass is enabled
        if not validation_passed and is_temp_file:
            self.logger.debug(
                f"Allowing temp file outside workspace: {path} (ALLOW_TEMP_OUTSIDE_WORKSPACE=True)"
            )

        if not TOML_READ_AVAILABLE:
            self.logger.error("TOML parsing support unavailable. Install tomllib (Python 3.11+)")
            return False

        # Resolve path: temp bypass or workspace containment
        # Security model:
        # - Normal case: enforce containment within workspace_root
        # - Temp bypass: if allowed and path is in temp dir, resolve directly (explicit risk)
        if is_temp_file:
            file_path = path_obj.resolve(strict=False)
        else:
            file_path = resolve_file_path(
                path,
                self.workspace_root,
                allow_absolute=True,
                validate_workspace=True,
                enforce_containment=True,
            )

        # Check for symlinks in the target path and all parent components before any file I/O
        if file_path.is_symlink():
            self.logger.error(f"Symlink detected, rejecting path for security: {file_path}")
            return False

        # Check all parent directories for symlinks
        for parent in file_path.parents:
            if parent.is_symlink():
                self.logger.error(
                    f"Symlink detected in path hierarchy, rejecting for security: {parent}"
                )
                return False

        # Read original file as lines to preserve formatting
        try:
            original_content = file_path.read_text(encoding="utf-8")
            original_lines = original_content.splitlines(keepends=True)
        except (OSError, UnicodeDecodeError) as e:
            self.logger.error(f"Error reading original TOML file {file_path}: {e}")
            return False

        # Validate original file is proper TOML
        try:
            tomllib.loads(original_content)
        except tomllib.TOMLDecodeError as e:
            self.logger.error(f"Error parsing original TOML {file_path}: {e}")
            return False

        # Validate line range bounds (1-based indices)
        total_lines = len(original_lines)
        if start_line < 1:
            raise ValueError(f"Invalid line range for {file_path}: start_line={start_line} (<1)")
        if end_line < start_line:
            raise ValueError(
                f"Invalid line range for {file_path}: end_line={end_line} < start_line={start_line}"
            )
        if end_line > total_lines:
            raise ValueError(
                f"Invalid line range for {file_path}: "
                f"end_line={end_line} > total_lines={total_lines}"
            )

        # Validate change_type
        valid_change_types = {"addition", "modification", "deletion"}
        if change_type not in valid_change_types:
            self.logger.error(
                f"Invalid change_type {change_type!r}; must be one of {valid_change_types}"
            )
            return False

        # Apply change based on change_type
        if change_type == "deletion":
            # Deletion: remove lines start_line to end_line, ignore content
            new_lines = original_lines[: start_line - 1] + original_lines[end_line:]
        elif change_type == "addition":
            # Addition: insert content AFTER end_line, preserving all existing lines
            # Validate suggestion is proper TOML
            try:
                tomllib.loads(content)
            except tomllib.TOMLDecodeError as e:
                self.logger.error(f"Error parsing TOML suggestion: {e}")
                return False
            formatted_suggestion = self._format_suggestion_for_insertion(content)
            new_lines = original_lines[:end_line] + formatted_suggestion + original_lines[end_line:]
        else:  # modification (default)
            # Modification: replace lines start_line to end_line with content
            # Validate suggestion is proper TOML
            try:
                tomllib.loads(content)
            except tomllib.TOMLDecodeError as e:
                self.logger.error(f"Error parsing TOML suggestion: {e}")
                return False
            formatted_suggestion = self._format_suggestion_for_insertion(content)
            new_lines = (
                original_lines[: start_line - 1]  # Lines before the target region
                + formatted_suggestion  # New content
                + original_lines[end_line:]  # Lines after the target region
            )

        # Join lines and write atomically with preserved permissions
        merged_content = "".join(new_lines)

        # Validate that merged content parses as valid TOML before writing
        try:
            tomllib.loads(merged_content)
        except tomllib.TOMLDecodeError as e:
            self.logger.error(
                f"Error parsing merged TOML content after applying suggestion to {file_path}: {e}"
            )
            return False

        return self._write_atomically(file_path, merged_content)

    def _write_atomically(self, file_path: Path, content: str) -> bool:
        """Write content to file atomically with permission preservation.

        This method writes content to a target file using an atomic write pattern
        that preserves the original file's permissions and ensures data durability.

        Args:
            file_path: Target file path to write to.
            content: Content to write to the file.

        Returns:
            bool: True if write succeeded, False on error.
        """
        # Write with atomic operation
        original_mode = None
        temp_path = None
        dir_fd = None

        try:
            # Capture original file mode if it exists
            if file_path.exists():
                original_mode = os.stat(file_path).st_mode

            # Create temporary file in the same directory as the target file
            temp_dir = file_path.parent
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=temp_dir,
                prefix=f".{file_path.name}.tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Ensure data is written to disk

            # Apply original file permissions (mask to permission bits) to temp file if we have it
            if original_mode is not None:
                masked_mode = stat.S_IMODE(original_mode)
                os.chmod(temp_path, masked_mode)

            # Atomically replace the original file
            os.replace(temp_path, file_path)

            # Ensure directory durability by fsyncing the parent directory
            try:
                dir_fd = os.open(temp_dir, os.O_RDONLY)
                os.fsync(dir_fd)
            except OSError:
                # Log at warning level so filesystem/permission issues are visible in production
                self.logger.warning("directory fsync failed for %s", temp_dir, exc_info=True)
            finally:
                if dir_fd is not None:
                    os.close(dir_fd)

            return True
        except (OSError, UnicodeEncodeError) as e:
            self.logger.error(f"Error writing TOML file {file_path}: {e}")
            return False
        finally:
            # Clean up temporary file if it exists
            if temp_path is not None and temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()

    def validate_change(
        self, path: str, content: str, start_line: int, end_line: int
    ) -> tuple[bool, str]:
        """Validate the provided TOML suggestion and report whether it parses.

        Args:
            path (str): File path the suggestion targets (informational).
            content (str): TOML text to validate.
            start_line (int): Starting line number (1-based) of the suggested range.
            end_line (int): Ending line number (1-based) of the suggested range.

        Returns:
            tuple[bool, str]: Tuple of (is_valid, message). When valid, returns
                (True, "Valid TOML"); otherwise returns (False, "<error message>").
        """
        if not TOML_READ_AVAILABLE:
            return False, "tomllib not available"

        try:
            tomllib.loads(content)
            return True, "Valid TOML"
        except tomllib.TOMLDecodeError as e:
            return False, f"Invalid TOML: {e}"

    def detect_conflicts(self, path: str, changes: list[Change]) -> list[Conflict]:
        """Identify conflicting changes that target the same TOML sections.

        Parses each change's TOML to collect affected section paths, groups by section, and
        creates a Conflict when multiple changes modify the same section. Unparseable changes
        are skipped (a warning is logged). The conflict `line_range` spans from the first to the
        last change, and the overlap percentage is computed from the union/intersection of
        line ranges.

        Args:
            path (str): File path reported on each Conflict.
            changes (list[Change]): List of Change objects to analyze.

        Returns:
            list[Conflict]: List of detected conflicts; empty list if no conflicts are found.

        Raises:
            None
        """
        conflicts: list[Conflict] = []

        # Group changes by section
        section_changes: dict[str, list[Change]] = {}
        for change in changes:
            try:
                data = tomllib.loads(change.content)
                sections = self._extract_sections(data)
                for section in sections:
                    if section not in section_changes:
                        section_changes[section] = []
                    section_changes[section].append(change)
            except tomllib.TOMLDecodeError as e:
                self.logger.warning("Failed to parse TOML change (path=%s): %s", path, e)
                continue

        # Find conflicts (multiple changes to same section)
        for _section, section_change_list in section_changes.items():
            if len(section_change_list) > 1:
                # Calculate actual overlap percentage
                overlap_percentage = self._calculate_overlap_percentage(section_change_list)

                min_start = min(c.start_line for c in section_change_list)
                max_end = max(c.end_line for c in section_change_list)
                conflicts.append(
                    Conflict(
                        file_path=path,
                        line_range=(min_start, max_end),
                        changes=section_change_list,
                        conflict_type="section_conflict",
                        severity="medium",
                        overlap_percentage=overlap_percentage,
                    )
                )

        return conflicts

    def _calculate_overlap_percentage(self, changes: list[Change]) -> float:
        """Compute the percentage of line-range overlap among multiple changes.

        Args:
            changes (list[Change]): List of Change objects each containing `start_line` and
                `end_line`.

        Returns:
            float: Overlap percentage between 0.0 and 100.0. Returns 0.0 if fewer than two changes
                or if there is no overlapping range.
        """
        if len(changes) < 2:
            return 0.0

        # Gather all line ranges
        starts = [change.start_line for change in changes]
        ends = [change.end_line for change in changes]

        # Calculate intersection (overlapping lines)
        intersection_start = max(starts)
        intersection_end = min(ends)
        intersection_lines = max(0, intersection_end - intersection_start + 1)

        # Calculate union (total span)
        union_start = min(starts)
        union_end = max(ends)
        union_lines = union_end - union_start + 1

        if union_lines == 0:
            return 0.0

        return (intersection_lines / union_lines) * 100.0

    def _format_suggestion_for_insertion(self, content: str) -> list[str]:
        """Format a TOML suggestion string into a list of lines ready for insertion.

        Ensures the suggestion has proper line endings for seamless insertion into the original
        file's line array. Preserves the suggestion's internal formatting while ensuring
        compatibility with the splitlines(keepends=True) format.

        Args:
            content (str): TOML-formatted suggestion content to format.

        Returns:
            list[str]: List of lines with preserved line endings, ready for insertion.
        """
        # Split suggestion into lines, preserving line endings
        lines = content.splitlines(keepends=True)

        # If the last line doesn't have a line ending, add one for proper formatting
        if lines and not lines[-1].endswith(("\n", "\r\n", "\r")):
            lines[-1] += "\n"

        return lines

    def _is_temp_file(self, path: Path, tempdir: Path) -> bool:
        """Check if a path is within the system temporary directory.

        Args:
            path: The path to check.
            tempdir: The system temporary directory path.

        Returns:
            bool: True if the path is relative to the temp directory, False otherwise.
        """
        try:
            path.relative_to(tempdir)
            # Check if ALLOW_TEMP_OUTSIDE_WORKSPACE is set and emit warning once
            if not self._warned_about_temp_workspace:
                env_val = (os.getenv("ALLOW_TEMP_OUTSIDE_WORKSPACE") or "").strip().lower()
                if env_val in {"1", "true", "yes", "on"}:
                    self._warned_about_temp_workspace = True
                    warnings.warn(
                        "ALLOW_TEMP_OUTSIDE_WORKSPACE is set. "
                        "This bypasses workspace containment "
                        "and is a SECURITY RISK in production. "
                        "Disable this environment variable in "
                        "production environments.",
                        category=RuntimeWarning,
                        stacklevel=3,
                    )
            return True
        except ValueError:
            return False

    def _extract_sections(self, data: dict[str, Any], prefix: str = "") -> list[str]:
        """Collects dot-separated section paths from a nested TOML mapping.

        Parameters:
            data (dict[str, Any]): Parsed TOML data (mapping of keys to values or nested tables).
            prefix (str): Optional section path prefix used when descending into nested tables.

        Returns:
            list[str]: A list of section path strings (dot-separated) for each table and leaf key
                in `data`.
        """
        sections = []

        for key, value in data.items():
            if isinstance(value, dict):
                current_section = f"{prefix}.{key}" if prefix else key
                sections.append(current_section)
                sections.extend(self._extract_sections(value, current_section))
            else:
                current_section = f"{prefix}.{key}" if prefix else key
                sections.append(current_section)

        return sections
