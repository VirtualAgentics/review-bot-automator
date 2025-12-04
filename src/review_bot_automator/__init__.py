# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Review Bot Automator.

An intelligent, automated conflict resolution system for GitHub PR comments.
"""

__version__ = "2.0.2"
__author__ = "VirtualAgentics"
__email__ = "bdc@virtualagentics.ai"

from .analysis.conflict_detector import ConflictDetector
from .config.presets import PresetConfig
from .core.models import Change, Conflict, FileType, Resolution, ResolutionResult
from .core.resolver import ConflictResolver
from .handlers.json_handler import JsonHandler
from .handlers.toml_handler import TomlHandler
from .handlers.yaml_handler import YamlHandler
from .integrations.github import GitHubCommentExtractor
from .security import InputValidator, SecretFinding, SecretScanner
from .strategies.priority_strategy import PriorityStrategy

__all__ = [
    "Change",
    "Conflict",
    "ConflictDetector",
    "ConflictResolver",
    "FileType",
    "GitHubCommentExtractor",
    "InputValidator",
    "JsonHandler",
    "PresetConfig",
    "PriorityStrategy",
    "Resolution",
    "ResolutionResult",
    "SecretFinding",
    "SecretScanner",
    "TomlHandler",
    "YamlHandler",
]
