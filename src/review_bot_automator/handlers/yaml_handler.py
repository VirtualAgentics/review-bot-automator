# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""YAML handler for applying CodeRabbit suggestions with AST validation.

This handler provides YAML-aware suggestion application with structure validation
and comment preservation using ruamel.yaml.
"""

from __future__ import annotations

import contextlib
import logging
import os
import stat
import tempfile
from os import PathLike
from pathlib import Path
from typing import ClassVar, TypeAlias

from review_bot_automator.core.models import Change, ChangeType, Conflict
from review_bot_automator.handlers.base import BaseHandler
from review_bot_automator.security.input_validator import InputValidator
from review_bot_automator.utils.path_utils import resolve_file_path

# Type alias for YAML values - recursive to capture nested dicts/lists
YAMLValue: TypeAlias = dict[str, "YAMLValue"] | list["YAMLValue"] | str | int | float | bool | None

try:
    from ruamel.yaml import YAML

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class YamlHandler(BaseHandler):
    """Handler for YAML files with comment preservation and structure validation."""

    # Dangerous YAML tags that could lead to code execution through Python object serialization
    DANGEROUS_TAGS: ClassVar[tuple[str, ...]] = (
        "!!python/object",
        "!!python/object/new",
        "!!python/object/apply",
        "!!python/name",
        "!!python/module",
        "!!python/function",
        "!!python/apply",
    )

    def __init__(self, workspace_root: str | PathLike[str] | None = None) -> None:
        """Initialize the YAML handler.

        Args:
            workspace_root: Root directory for validating absolute paths.
                If None, defaults to current working directory.
        """
        super().__init__(workspace_root)
        self.logger = logging.getLogger(__name__)
        if not YAML_AVAILABLE:
            self.logger.warning("ruamel.yaml not available. Install with: pip install ruamel.yaml")

    def can_handle(self, file_path: str) -> bool:
        """Determine whether this handler should process the given file path.

        Returns:
            `true` if the path ends with `.yaml` or `.yml` (case-insensitive), `false` otherwise.
        """
        return file_path.lower().endswith((".yaml", ".yml"))

    def apply_change(
        self,
        path: str,
        content: str,
        start_line: int,
        end_line: int,
        change_type: ChangeType = "modification",
    ) -> bool:
        """Apply a YAML change to a file based on change_type.

        Parses the target file and the provided YAML suggestion, performs a high-level merge
        (preserving quotes and comments where possible) guided by the given start and end line
        positions, and writes the merged result back to the file. If parsing or writing fails, no
        changes are written.

        Args:
            path (str): Filesystem path to the YAML file to update.
            content (str): YAML text containing the suggestion to apply.
            start_line (int): Starting line number in the original file used to guide merging.
            end_line (int): Ending line number in the original file used to guide merging.
            change_type (str): Type of change to apply. One of:
                - "modification": Merge suggestion into original YAML (default).
                - "addition": Same as modification for YAML (keys are added/merged).
                - "deletion": Remove keys specified in content from the YAML file.

        Returns:
            bool: `True` if the merged YAML was written successfully, `False` otherwise.
        """
        # Validate file path to prevent path traversal attacks
        # Use workspace root for absolute path containment check
        if not InputValidator.validate_file_path(
            path, allow_absolute=True, base_dir=str(self.workspace_root)
        ):
            self.logger.error(f"Invalid file path rejected: {path}")
            return False

        if not YAML_AVAILABLE:
            self.logger.error("ruamel.yaml not available. Install with: pip install ruamel.yaml")
            return False

        # Resolve path relative to workspace_root and enforce containment within workspace
        file_path = resolve_file_path(
            path,
            self.workspace_root,
            allow_absolute=True,
            validate_workspace=True,
            enforce_containment=True,
        )

        # Check for symlinks in the target path and all parent components before any file I/O
        # Single traversal over the path and its parents to avoid duplicate probes
        for component in (file_path, *file_path.parents):
            try:
                # Probe if the component is a symlink (detects even broken symlinks)
                if component.is_symlink():
                    self.logger.error(
                        f"Symlink detected in path hierarchy, rejecting for security: {component}"
                    )
                    return False
            except OSError:
                # If probing fails, treat as unsafe
                self.logger.error(
                    f"Error probing filesystem component (possible symlink), rejecting: {component}"
                )
                return False

        # Parse original file
        try:
            original_content = file_path.read_text(encoding="utf-8")
            # Step 1: validate structure safely (no object construction)
            safe_yaml = YAML(typ="safe")
            _ = safe_yaml.load(original_content)
            # Step 2: re-parse with round-trip loader for formatting preservation
            yaml_rt = YAML(typ="rt")
            yaml_rt.preserve_quotes = True
            original_data = yaml_rt.load(original_content)
        except (OSError, ValueError) as e:
            self.logger.error(f"Error parsing original YAML: {e}")
            return False
        except Exception as e:  # ruamel.yaml raises base Exception
            self.logger.error(f"Error parsing original YAML: {e}")
            return False

        # Parse suggestion
        try:
            # Step 1: validate safely to prevent object construction
            safe_yaml_suggestion = YAML(typ="safe")
            parsed_safe = safe_yaml_suggestion.load(content)

            # Structural check: reject if dangerous tags found in parsed data
            if self._contains_dangerous_tags(parsed_safe):
                raise ValueError("YAML contains dangerous Python object tags in structure")

            # Defense-in-depth: reject dangerous Python YAML tags before round-trip parse
            lowered = content.lower()
            if (
                "!!python" in lowered
                or "tag:yaml.org,2002:python" in lowered
                or any(tag.lower() in lowered for tag in self.DANGEROUS_TAGS)
            ):
                raise ValueError("YAML contains dangerous Python object tags")

            # Step 2: round-trip parse for structure with formatting support
            yaml_suggestion_rt = YAML(typ="rt")
            suggestion_data = yaml_suggestion_rt.load(content)
        except ValueError as e:
            self.logger.error(f"Error parsing YAML suggestion: {e}")
            return False
        except Exception as e:  # ruamel.yaml raises base Exception
            self.logger.error(f"Error parsing YAML suggestion: {e}")
            return False

        # Validate change_type
        valid_change_types = {"addition", "modification", "deletion"}
        if change_type not in valid_change_types:
            self.logger.error(
                f"Invalid change_type {change_type!r}; must be one of {valid_change_types}"
            )
            return False

        # Apply change based on change_type.
        # Note: YAML uses key-based merging (unlike line-based formats like TOML/plaintext).
        # For YAML, "addition" and "modification" both result in key merging (suggestion keys
        # overwrite/add to original). Only "deletion" is distinct (removes keys from original).
        merged_data: YAMLValue
        if change_type == "deletion":
            # Deletion: remove keys specified in content from original
            # Preserve CommentedMap semantics (comments/ordering) by copying and deleting keys
            if isinstance(suggestion_data, dict) and isinstance(original_data, dict):
                # Copy original to preserve CommentedMap type and metadata
                merged_data = original_data.copy()
                for key in suggestion_data:
                    if key in merged_data:
                        del merged_data[key]
            else:
                self.logger.error("YAML deletion requires both original and content to be dicts")
                return False
        else:
            # modification and addition: merge suggestion into original
            merged_data = self._smart_merge_yaml(
                original_data, suggestion_data, start_line, end_line
            )

        # Write atomically with comment preservation
        try:
            yaml_rt = YAML(typ="rt")
            yaml_rt.preserve_quotes = True
            orig_mode = os.stat(file_path).st_mode if file_path.exists() else None
            tmp_path = None
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=file_path.parent,
                prefix=f".{file_path.name}.tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                yaml_rt.dump(merged_data, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            if orig_mode is not None:
                os.chmod(tmp_path, stat.S_IMODE(orig_mode))
            os.replace(tmp_path, file_path)
            # Best-effort directory fsync
            try:
                dir_fd = os.open(str(file_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (OSError, AttributeError):
                self.logger.debug("Directory fsync failed (non-fatal)", exc_info=True)
            return True
        except (OSError, ValueError) as e:
            self.logger.error(f"Error writing YAML: {e}")
            return False
        except Exception as e:  # ruamel.yaml raises base Exception
            self.logger.error(f"Error writing YAML: {e}")
            return False
        finally:
            if "tmp_path" in locals() and tmp_path and tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()

    def validate_change(
        self, path: str, content: str, start_line: int, end_line: int
    ) -> tuple[bool, str]:
        r"""Validate a YAML suggestion string and report whether it parses successfully.

        This method implements comprehensive security validation to prevent YAML deserialization
        attacks and other malicious content. It uses a defense-in-depth approach with multiple
        layers of validation.

        Security Features:
            - Dangerous control character detection (null bytes, etc.)
            - Python object serialization tag detection (!!python/object, !!python/name, etc.)
            - Structural analysis of parsed YAML for hidden dangerous tags
            - Safe YAML parsing using ruamel.yaml's safe loader

        Parameters:
            path (str): File path associated with the suggestion (used for context only).
            content (str): YAML text to validate.
            start_line (int): Start line of the suggested change in the file (provided for
                context; not used to alter validation).
            end_line (int): End line of the suggested change in the file (provided for context;
                not used to alter validation).

        Returns:
            tuple[bool, str]: `True` and "Valid YAML" if `content` parses as YAML; `False` and an
                error message otherwise.

        Example:
            >>> handler = YamlHandler()
            >>> # Valid YAML
            >>> handler.validate_change("config.yaml", "key: value", 1, 1)
            (True, "Valid YAML")
            >>> # Dangerous Python object - rejected
            >>> handler.validate_change("config.yaml",
            ...     "key: !!python/object/apply:os.system ['rm -rf /']", 1, 1)
            (False, "YAML contains dangerous Python object tags")
            >>> # Null byte - rejected
            >>> handler.validate_change("config.yaml", "key: value\x00", 1, 1)
            (False, "Invalid YAML: contains dangerous control characters")

        Warning:
            This method rejects YAML content that could lead to arbitrary code execution
            through Python object deserialization. Always validate YAML content before
            processing to prevent security vulnerabilities.

        See Also:
            _contains_dangerous_characters: Detects dangerous control characters
            _contains_dangerous_tags: Detects dangerous Python tags in parsed data
        """
        if not YAML_AVAILABLE:
            return False, "ruamel.yaml not available"

        # Check for null bytes and other dangerous control characters first
        if self._contains_dangerous_characters(content):
            return False, "Invalid YAML: contains dangerous control characters"

        # Check for dangerous YAML tags that could lead to code execution
        # Defense-in-depth: keep substring check as first line of defense
        content_lower = content.lower()
        for tag in self.DANGEROUS_TAGS:
            if tag.lower() in content_lower:
                return False, "YAML contains dangerous Python object tags"

        # Parse with safe loader and perform structural tag checks
        try:
            yaml = YAML(typ="safe")
            parsed_data = yaml.load(content)

            # If parsing succeeded, check for dangerous tags in the parsed structure
            if self._contains_dangerous_tags(parsed_data):
                return False, "YAML contains dangerous Python object tags"

            return True, "Valid YAML"
        except ValueError as e:
            return False, f"Invalid YAML: {e}"
        except Exception as e:  # ruamel.yaml raises base Exception
            return False, f"Invalid YAML: {e}"

    def detect_conflicts(self, path: str, changes: list[Change]) -> list[Conflict]:
        """Identify key-level conflicts among a set of YAML changes.

        Parses each change's YAML content to extract key paths, groups changes that target the
        same key path, and returns a Conflict for each key that is modified by more than one
        change.

        Args:
            path (str): File path the changes apply to; used as the Conflict.file_path.
            changes (list[Change]): List of Change objects whose `.content` contains YAML
                snippets and which provide `start_line`/`end_line` for conflict ranges.

        Returns:
            list[Conflict]: A list of Conflict objects describing keys modified by multiple
                changes. Each Conflict uses `conflict_type` "key_conflict", `severity` "medium",
                and a computed `overlap_percentage`; `line_range` spans min start to max end.
        """
        conflicts: list[Conflict] = []

        # Group changes by key path
        key_changes: dict[str, list[Change]] = {}
        for change in changes:
            try:
                yaml_safe = YAML(typ="safe")
                data = yaml_safe.load(change.content)
                keys = self._extract_keys(data)
                for key in keys:
                    if key not in key_changes:
                        key_changes[key] = []
                    key_changes[key].append(change)
            except Exception as e:  # ruamel.yaml raises base Exception
                self.logger.warning("Failed to parse YAML change (path=%s): %s", path, e)
                continue

        # Find conflicts (multiple changes to same key)
        for _key, key_change_list in key_changes.items():
            if len(key_change_list) > 1:
                overlap_percentage = self._calculate_overlap_percentage(key_change_list)
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

    def _smart_merge_yaml(
        self, original: YAMLValue, suggestion: YAMLValue, start_line: int, end_line: int
    ) -> YAMLValue:
        """Merge two YAML structures, giving precedence to the suggestion.

        Parameters:
            original (Any): The existing YAML-parsed data.
            suggestion (Any): The YAML-parsed suggestion to apply.
            start_line (int): Start line number of the suggested change (used to indicate the
                targeted range).
            end_line (int): End line number of the suggested change (used to indicate the
                targeted range).

        Returns:
            Any: The merged YAML structure. If both inputs are dicts, returns a dict where
                suggestion keys override original keys; if both are lists, returns the suggestion
                list; if types differ, returns the suggestion.
        """
        if isinstance(original, dict) and isinstance(suggestion, dict):
            # Merge dictionaries
            result = original.copy()
            for key, value in suggestion.items():
                result[key] = value
            return result
        elif isinstance(original, list) and isinstance(suggestion, list):
            # For lists, we might want to append or replace based on context
            # For now, simple replacement
            return suggestion
        else:
            # Different types - use suggestion
            return suggestion

    def _extract_keys(self, data: YAMLValue, prefix: str = "") -> list[str]:
        """Recursively collect key paths from parsed YAML data.

        Produces dot-separated paths for mappings and bracketed indices for sequences. For
        example, a mapping {"a": {"b": 1}} yields "a" and "a.b"; a sequence [1, {"x": 2}] yields
        "[0]" and "[1].x".

        Parameters:
            data (Any): Parsed YAML structure (mappings as dict, sequences as list, scalars as
                leaf values).
            prefix (str): Optional starting path to prepend (no leading or trailing separators).

        Returns:
            list[str]: A list of key path strings found in `data`.
        """
        keys = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_key = f"{prefix}.{key}" if prefix else key
                keys.append(current_key)
                keys.extend(self._extract_keys(value, current_key))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                keys.append(current_key)
                keys.extend(self._extract_keys(item, current_key))

        return keys

    def _contains_dangerous_characters(self, content: str) -> bool:
        """Check if content contains dangerous control characters.

        Parameters:
            content (str): Raw content to check.

        Returns:
            bool: True if dangerous characters are found, False otherwise.
        """
        for char in content:
            code_point = ord(char)
            # C0 controls (except safe whitespace)
            if code_point < 32 and char not in "\n\r\t":
                return True
            # C1 controls
            if 0x80 <= code_point <= 0x9F:
                return True
            # Dangerous Unicode characters
            if code_point in {0x200B, 0x200C, 0x200D, 0x202E, 0xFEFF}:
                return True
        return False

    def _contains_dangerous_tags(self, data: YAMLValue) -> bool:
        """Check if parsed YAML data contains dangerous Python-specific tags.

        Parameters:
            data (YAMLValue): Parsed YAML data structure.

        Returns:
            bool: True if dangerous tags are found, False otherwise.
        """
        if data is None:
            return False

        # Check if this is a tagged value (ruamel.yaml preserves tag information)
        if hasattr(data, "tag") and data.tag:
            tag_str = str(data.tag)
            if tag_str.startswith("!!python"):
                return True

        # Recursively check nested structures
        if isinstance(data, dict):
            for value in data.values():
                if self._contains_dangerous_tags(value):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._contains_dangerous_tags(item):
                    return True

        return False
