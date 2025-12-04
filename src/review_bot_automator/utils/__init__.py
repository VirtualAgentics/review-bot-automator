# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Utility modules for the conflict resolver.

This package contains utility functions and helpers used across
the conflict resolver components.
"""

from review_bot_automator.utils.indentation import (
    get_leading_whitespace,
    restore_indentation,
)

__all__ = [
    "get_leading_whitespace",
    "restore_indentation",
]
