# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Universal LLM-powered comment parser.

This module implements the core parsing logic that uses LLM providers to extract
code changes from CodeRabbit review comments. The parser:
- Accepts any comment format (diff blocks, suggestions, natural language)
- Uses prompt templates to guide the LLM
- Returns structured ParsedChange objects
- Handles errors gracefully with optional fallback
- Filters results by confidence threshold
- Supports cost budget enforcement

The parser is provider-agnostic and works with any LLMProvider implementation.
"""

from __future__ import annotations

import json
import logging
import re
import threading

from review_bot_automator.llm.base import LLMParser, ParsedChange
from review_bot_automator.llm.cost_tracker import CostStatus, CostTracker
from review_bot_automator.llm.exceptions import LLMCostExceededError, LLMSecretDetectedError
from review_bot_automator.llm.prompts import PARSE_COMMENT_PROMPT
from review_bot_automator.llm.providers.base import LLMProvider
from review_bot_automator.security.secret_scanner import SecretScanner

logger = logging.getLogger(__name__)

# Pattern to match markdown JSON code fences
_JSON_FENCE_PATTERN = re.compile(
    r"^```(?:json)?\s*\n(.*?)\n```\s*$",
    re.DOTALL,
)


def _strip_json_fences(text: str) -> str:
    r"""Strip markdown code fences from JSON response.

    LLMs sometimes wrap JSON responses in ```json ... ``` markers despite
    being asked not to. This function extracts the JSON content.

    Args:
        text: Raw text that may contain markdown code fences.

    Returns:
        The JSON content with fences stripped, or original text if no fences found.

    Example:
        >>> _strip_json_fences("```json\\n[{...}]\\n```")
        '[{...}]'
        >>> _strip_json_fences("[{...}]")
        '[{...}]'
    """
    text = text.strip()
    match = _JSON_FENCE_PATTERN.match(text)
    if match:
        return match.group(1).strip()
    return text


class UniversalLLMParser(LLMParser):
    """LLM-powered universal comment parser.

    This parser uses LLM providers to extract code changes from any CodeRabbit
    comment format. It handles:
    - Diff blocks (```diff with @@ headers)
    - Suggestion blocks (```suggestion)
    - Natural language descriptions
    - Multi-option suggestions

    The parser validates LLM output against the ParsedChange schema and filters
    by confidence threshold to ensure quality results.

    Examples:
        >>> from review_bot_automator.llm.providers.openai_api import OpenAIAPIProvider
        >>> provider = OpenAIAPIProvider(api_key="sk-...")
        >>> parser = UniversalLLMParser(provider, confidence_threshold=0.7)
        >>> changes = parser.parse_comment("Apply this fix: ...", file_path="test.py")

    Attributes:
        provider: LLM provider instance for text generation
        fallback_to_regex: If True, return empty list on failure (enables fallback)
        confidence_threshold: Minimum confidence score (0.0-1.0) to accept changes
    """

    def __init__(
        self,
        provider: LLMProvider,
        fallback_to_regex: bool = True,
        confidence_threshold: float = 0.5,
        max_tokens: int = 2000,
        cost_tracker: CostTracker | None = None,
        scan_for_secrets: bool = True,
    ) -> None:
        """Initialize universal LLM parser.

        Args:
            provider: LLM provider instance implementing LLMProvider protocol
            fallback_to_regex: If True, return empty list on failure to trigger
                regex fallback. If False, raise exception on parsing errors.
            confidence_threshold: Minimum confidence (0.0-1.0) to accept a change.
                Lower threshold accepts more changes but with potentially lower quality.
                Recommended: 0.5 for balanced results, 0.7 for high quality only.
            max_tokens: Maximum tokens for LLM response generation. Default 2000 is
                sufficient for most PR comments while keeping costs low. Increase for
                very long comments with many changes.
            cost_tracker: Optional CostTracker for budget enforcement. If provided,
                requests are blocked when budget is exceeded and LLMCostExceededError
                is raised.
            scan_for_secrets: Security setting (default: True - secure). When enabled,
                scans comment bodies for secrets before sending to external LLM APIs
                and raises LLMSecretDetectedError if any are found. Disable only for
                testing or trusted local-only LLMs.

        Raises:
            ValueError: If confidence_threshold is not in [0.0, 1.0]
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
            )

        self.provider = provider
        self.fallback_to_regex = fallback_to_regex
        self.confidence_threshold = confidence_threshold
        self.max_tokens = max_tokens
        self.cost_tracker = cost_tracker
        self.scan_for_secrets = scan_for_secrets

        # Fallback tracking counters (thread-safe)
        self._fallback_count = 0
        self._llm_success_count = 0
        self._stats_lock = threading.Lock()

        logger.info(
            "Initialized UniversalLLMParser: fallback=%s, threshold=%s, max_tokens=%s, "
            "cost_tracker=%s, secret_scan=%s",
            fallback_to_regex,
            confidence_threshold,
            max_tokens,
            "enabled" if cost_tracker else "disabled",
            "enabled" if scan_for_secrets else "disabled",
        )

    def parse_comment(
        self,
        comment_body: str,
        file_path: str | None = None,
        # TODO(#294): Remove line_number once all callers migrate to start_line/end_line
        line_number: int | None = None,  # Deprecated, use end_line instead
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[ParsedChange]:
        """Parse comment using LLM to extract code changes.

        This method:
        1. Builds a prompt with comment body and context
        2. Sends prompt to LLM provider
        3. Parses JSON response into ParsedChange objects
        4. Validates and filters by confidence threshold
        5. Returns structured changes or empty list on failure

        Args:
            comment_body: Raw comment text from GitHub (markdown format)
            file_path: Optional file path for context (helps LLM with ambiguous comments)
            line_number: Deprecated - use end_line instead. Will be removed in future version.
                See https://github.com/VirtualAgentics/review-bot-automator/issues/294
            start_line: Start of the diff range (from GitHub start_line field)
            end_line: End of the diff range (from GitHub line field)

        Returns:
            List of ParsedChange objects meeting confidence threshold.
            Empty list if:
            - No changes found in comment
            - LLM parsing failed and fallback_to_regex=True
            - All changes filtered out by confidence threshold

        Raises:
            RuntimeError: If parsing fails and fallback_to_regex=False
            ValueError: If comment_body is None or empty
            LLMSecretDetectedError: If secrets are detected in comment_body
                and scan_for_secrets=True (default)
            LLMCostExceededError: If cost budget is exceeded

        Note:
            The method logs all parsing failures for debugging. Check logs
            if you're not getting expected results.
        """
        if not comment_body:
            raise ValueError("comment_body cannot be None or empty")

        # Scan for secrets BEFORE sending to external LLM API
        if self.scan_for_secrets:
            findings = SecretScanner.scan_content(comment_body, stop_on_first=True)
            if findings:
                # Log count only - no tainted secret type data flows to logs
                logger.error(
                    "Secret detected in comment body (count=%d), blocking LLM request - "
                    "refusing to send to external API",
                    len(findings),
                )
                raise LLMSecretDetectedError(
                    "Secret detected in comment content",
                    findings=findings,
                    details={"file_path": file_path, "line_number": line_number},
                )

        # Check budget before making LLM API call
        if self.cost_tracker and self.cost_tracker.should_block_request():
            # Cache values to avoid multiple lock acquisitions
            accumulated = self.cost_tracker.accumulated_cost
            budget = self.cost_tracker.budget or 0.0
            raise LLMCostExceededError(
                f"Cost budget exceeded: ${accumulated:.4f}/${budget:.4f}",
                accumulated_cost=accumulated,
                budget=budget,
            )

        try:
            # TODO(#294): Remove backward compat logic once line_number is removed
            # Handle backward compatibility: line_number maps to end_line
            # Use 0 as sentinel for "unknown" to maintain consistent integer typing
            # in the prompt template (0 is invalid for 1-indexed line numbers)
            effective_start = start_line if start_line is not None else 0
            effective_end = (
                end_line
                if end_line is not None
                else (line_number if line_number is not None else 0)
            )

            # Build prompt with context (now includes start_line and end_line)
            prompt = PARSE_COMMENT_PROMPT.format(
                comment_body=comment_body,
                file_path=file_path or "unknown",
                start_line=effective_start,
                end_line=effective_end,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Parsing comment: file=%s, start_line=%s, end_line=%s, body_length=%d",
                    file_path,
                    effective_start,
                    effective_end,
                    len(comment_body),
                )

                # Log the context section of the prompt (Issue #285 investigation)
                # Extract just the context part to see what the LLM receives
                context_start_idx = prompt.find("## Context Information")
                context_end_idx = prompt.find("## Comment Body")
                if context_start_idx != -1 and context_end_idx != -1:
                    context_section = prompt[context_start_idx:context_end_idx].strip()
                    logger.debug("Prompt context section:\n%s", context_section)

            # Track cost before call to calculate incremental cost
            previous_cost = self.provider.get_total_cost() if self.cost_tracker else 0.0

            # Generate response from LLM
            response = self.provider.generate(prompt, max_tokens=self.max_tokens)

            # Track cost after successful call
            if self.cost_tracker:
                current_cost = self.provider.get_total_cost()
                request_cost = current_cost - previous_cost
                status = self.cost_tracker.add_cost(request_cost)

                # Log warning at threshold (only once)
                if status == CostStatus.WARNING:
                    warning_msg = self.cost_tracker.get_warning_message()
                    if warning_msg:
                        logger.warning(warning_msg)

            logger.debug("LLM response length: %d characters", len(response))

            # Strip markdown code fences if present (LLMs sometimes add them)
            json_text = _strip_json_fences(response)

            # Parse JSON response
            try:
                changes_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(
                    f"LLM returned invalid JSON: {json_text[:200]}... "
                    f"(truncated, total {len(json_text)} chars)"
                )
                raise RuntimeError(f"Invalid JSON from LLM: {e}") from e

            # Validate response is a list
            if not isinstance(changes_data, list):
                logger.error(f"LLM returned non-list: {type(changes_data).__name__}")
                raise RuntimeError(f"LLM must return JSON array, got {type(changes_data).__name__}")

            # Convert to ParsedChange objects with validation
            parsed_changes = []
            for idx, change_dict in enumerate(changes_data):
                try:
                    # ParsedChange.__post_init__ validates all fields
                    change = ParsedChange(**change_dict)

                    # Filter by confidence threshold
                    if change.confidence < self.confidence_threshold:
                        logger.info(
                            f"Filtered change {idx+1}/{len(changes_data)}: "
                            f"confidence {change.confidence:.2f} < "
                            f"threshold {self.confidence_threshold}"
                        )
                        continue

                    parsed_changes.append(change)
                    logger.debug(
                        "Parsed change %d/%d: %s:%d-%d (confidence=%.2f, risk=%s)",
                        idx + 1,
                        len(changes_data),
                        change.file_path,
                        change.start_line,
                        change.end_line,
                        change.confidence,
                        change.risk_level,
                    )

                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Invalid change format from LLM at index {idx}: {change_dict}. "
                        f"Error: {e}"
                    )
                    continue

            logger.info(
                f"LLM parsed {len(parsed_changes)}/{len(changes_data)} changes "
                f"(threshold={self.confidence_threshold})"
            )

            # Track successful LLM parse (thread-safe)
            with self._stats_lock:
                self._llm_success_count += 1
            return parsed_changes

        except LLMCostExceededError:
            # Handle cost budget exceeded explicitly - don't wrap in RuntimeError
            if self.fallback_to_regex:
                logger.info("Cost budget exceeded; returning empty list for regex fallback")
                with self._stats_lock:
                    self._fallback_count += 1
                return []
            else:
                raise

        except Exception as e:
            logger.error(f"LLM parsing failed: {type(e).__name__}: {e}")

            if self.fallback_to_regex:
                logger.info("Returning empty list to trigger regex fallback")
                with self._stats_lock:
                    self._fallback_count += 1
                return []
            else:
                raise RuntimeError(f"LLM parsing failed: {e}") from e

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold dynamically.

        Useful for:
        - Adjusting quality requirements per file type
        - Lowering threshold for exploratory parsing
        - Raising threshold for production changes

        Args:
            threshold: New confidence threshold (0.0-1.0)

        Raises:
            ValueError: If threshold not in valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        old_threshold = self.confidence_threshold
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold: {old_threshold} -> {threshold}")

    def get_fallback_stats(self) -> tuple[int, int, float]:
        """Get fallback statistics for monitoring.

        Returns:
            Tuple of (fallback_count, total_count, fallback_rate):
            - fallback_count: Number of times regex fallback was triggered
            - total_count: Total number of parse attempts (success + fallback)
            - fallback_rate: Ratio of fallbacks to total (0.0-1.0)

        Example:
            >>> parser = UniversalLLMParser(provider)
            >>> # After some parsing...
            >>> fallbacks, total, rate = parser.get_fallback_stats()
            >>> print(f"Fallback rate: {rate:.1%}")
            Fallback rate: 15.0%
        """
        with self._stats_lock:
            total = self._llm_success_count + self._fallback_count
            rate = self._fallback_count / total if total > 0 else 0.0
            return (self._fallback_count, total, rate)

    def reset_fallback_stats(self) -> None:
        """Reset fallback statistics counters.

        Useful for testing or starting fresh tracking for a new PR.
        """
        with self._stats_lock:
            self._fallback_count = 0
            self._llm_success_count = 0
        logger.debug("Reset fallback statistics counters")
