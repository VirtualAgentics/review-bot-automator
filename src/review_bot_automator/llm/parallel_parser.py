# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Parallel LLM comment parser implementation.

This module provides parallel processing capabilities for LLM comment parsing,
enabling significant speedup for large PRs with many review comments. It includes:
- RateLimiter: Thread-safe rate limiting to prevent API throttling
- ParallelLLMParser: Parallel implementation extending UniversalLLMParser
- Circuit breaker integration for resilience

Phase 5 - Issue #223: Parallel Comment Parsing Implementation
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from review_bot_automator.llm.base import ParsedChange
from review_bot_automator.llm.cost_tracker import CostTracker
from review_bot_automator.llm.parser import UniversalLLMParser
from review_bot_automator.llm.providers.base import LLMProvider
from review_bot_automator.llm.resilience.circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CommentInput:
    """Structured input for comment parsing.

    Attributes:
        body: Raw comment text from GitHub
        file_path: Optional file path for context
        line_number: Deprecated - use end_line instead. Will be removed in future version.
            See https://github.com/VirtualAgentics/review-bot-automator/issues/294
        start_line: Start of the diff range (from GitHub start_line field)
        end_line: End of the diff range (from GitHub line field)
    """

    body: str
    file_path: str | None = None
    # TODO(#294): Remove line_number once all callers migrate to start_line/end_line
    line_number: int | None = None  # Deprecated, use end_line instead
    start_line: int | None = None
    end_line: int | None = None


@runtime_checkable
class SupportsCircuitBreaker(Protocol):
    """Protocol for LLM providers that support circuit breaker functionality."""

    @property
    def circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        pass  # Required for Protocol definition (CodeQL compatibility)  # pragma: no cover

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get circuit breaker instance."""
        pass  # Required for Protocol definition (CodeQL compatibility)  # pragma: no cover


class RateLimiter:
    """Thread-safe rate limiter for API calls.

    Ensures requests don't exceed specified rate (requests per second)
    to prevent API throttling.

    Attributes:
        rate: Maximum requests per second
        min_interval: Minimum time between requests (1.0 / rate)

    Example:
        >>> limiter = RateLimiter(rate=10.0)  # 10 requests/second
        >>> limiter.wait_if_needed()  # Blocks if needed to maintain rate
    """

    def __init__(self, rate: float) -> None:
        """Initialize rate limiter.

        Args:
            rate: Maximum requests per second (must be > 0)

        Raises:
            ValueError: If rate is not positive
        """
        if rate <= 0.0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.min_interval = 1.0 / rate
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0

        logger.debug(
            f"Initialized RateLimiter: {rate} requests/second "
            f"(min_interval={self.min_interval:.3f}s)"
        )

    def wait_if_needed(self) -> None:
        """Wait if necessary to maintain rate limit.

        This method is thread-safe and ensures that consecutive calls
        are spaced at least min_interval seconds apart.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            delay = max(0.0, self.min_interval - elapsed)
            self._last_call_time = now + delay

        if delay > 0.0:
            time.sleep(delay)


class ParallelLLMParser(UniversalLLMParser):
    """Parallel implementation of LLM comment parser.

    Extends UniversalLLMParser to process multiple comments concurrently
    using ThreadPoolExecutor. Includes rate limiting to prevent API throttling
    and progress tracking for UI feedback. When ``fallback_to_regex`` is True the
    parallel API is best-effort—comment-level failures become empty results so
    the rest of the batch can continue. When the flag is False, exceptions are
    propagated to the caller just like :meth:`parse_comment`.

    Attributes:
        max_workers: Maximum number of worker threads (1-32 recommended)
        rate_limit: Maximum requests per second (default: 10.0)
        _rate_limiter: Internal rate limiter instance

    Example:
        >>> from review_bot_automator.llm.providers.openai_api import OpenAIAPIProvider
        >>> provider = OpenAIAPIProvider(api_key="sk-...")
        >>> parser = ParallelLLMParser(
        ...     provider=provider,
        ...     max_workers=4,
        ...     rate_limit=10.0
        ... )
        >>> comments = [
        ...     CommentInput(body="Fix bug", file_path="test.py", line_number=10),
        ...     CommentInput(body="Add feature", file_path="test.py", line_number=20),
        ... ]
        >>> results = parser.parse_comments(comments)
        >>> len(results)  # Returns list of ParsedChange lists
        2
    """

    # Security: Hard limit on worker threads to prevent resource exhaustion
    MAX_ALLOWED_WORKERS = 32

    def __init__(
        self,
        provider: LLMProvider,
        max_workers: int = 4,
        rate_limit: float = 10.0,  # requests per second
        fallback_to_regex: bool = True,
        confidence_threshold: float = 0.5,
        max_tokens: int = 2000,
        cost_tracker: CostTracker | None = None,
        scan_for_secrets: bool = True,
    ) -> None:
        """Initialize parallel LLM parser.

        Args:
            provider: LLM provider instance for text generation
            max_workers: Maximum number of worker threads (1-32)
            rate_limit: Maximum requests per second to prevent API throttling
            fallback_to_regex: If True, return empty list on failure (enables fallback)
            confidence_threshold: Minimum confidence score (0.0-1.0) to accept changes
            max_tokens: Maximum tokens for LLM response
            cost_tracker: Optional CostTracker for budget enforcement. The tracker
                is thread-safe and will be shared across worker threads.
            scan_for_secrets: If True (default), scan comment bodies for secrets
                before sending to external LLM APIs. Raises LLMSecretDetectedError
                if secrets are detected.

        Raises:
            ValueError: If max_workers < 1, max_workers > 32, or rate_limit <= 0
        """
        super().__init__(
            provider=provider,
            fallback_to_regex=fallback_to_regex,
            confidence_threshold=confidence_threshold,
            max_tokens=max_tokens,
            cost_tracker=cost_tracker,
            scan_for_secrets=scan_for_secrets,
        )
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_workers > self.MAX_ALLOWED_WORKERS:
            raise ValueError(
                f"max_workers cannot exceed {self.MAX_ALLOWED_WORKERS} (got {max_workers})"
            )
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self._rate_limiter = RateLimiter(rate_limit)

    def parse_comments(
        self,
        comments: list[CommentInput],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[ParsedChange]]:
        """Parse multiple comments in parallel.

        Processes comments concurrently using ThreadPoolExecutor, with rate limiting
        to prevent API throttling. Results are returned in the same order as input.

        Args:
            comments: List of comment inputs to parse. Empty lists return [] immediately.
            progress_callback: Optional callback(completed, total) for progress updates.
                Called after each comment completes parsing. Exceptions in callback
                are caught and logged to prevent breaking the parser.

        Returns:
            List of ParsedChange lists, one per input comment (preserves order).
            Failed comments return empty list. Empty input returns [].

        Note:
            - Circuit breaker state is checked before parallel execution
            - Individual comment failures don't stop other comments
            - Rate limiting prevents API throttling
            - Progress callback is thread-safe and exception-safe
        """
        if not comments:
            logger.debug("parse_comments called with empty input; returning []")
            return []

        # Check circuit breaker state before parallel execution
        if self._should_fallback_to_sequential():
            logger.warning("Circuit breaker is OPEN, falling back to sequential parsing")
            return self._parse_sequential(comments, progress_callback)

        total = len(comments)
        logger.info(f"Parsing {total} comments in parallel (max_workers={self.max_workers})")

        # Use dict-based collection to preserve order without type: ignore
        results: dict[int, list[ParsedChange]] = {}
        completed_count = 0
        completed_lock = threading.Lock()

        def parse_single_comment(idx: int, comment: CommentInput) -> tuple[int, list[ParsedChange]]:
            """Parse a single comment with rate limiting.

            Returns:
                Tuple of (index, parsed_changes) to preserve order
            """
            nonlocal completed_count

            try:
                # Apply rate limiting before making API call
                self._rate_limiter.wait_if_needed()

                # Parse using parent class method
                parsed_changes = self.parse_comment(
                    comment_body=comment.body,
                    file_path=comment.file_path,
                    line_number=comment.line_number,
                    start_line=comment.start_line,
                    end_line=comment.end_line,
                )

                # Update progress (minimize lock hold time)
                with completed_lock:
                    completed_count += 1
                    local_count = completed_count

                # Call progress callback outside lock to avoid blocking
                if progress_callback:
                    try:
                        progress_callback(local_count, total)
                    except Exception as e:
                        logger.warning(
                            f"Progress callback raised exception: {type(e).__name__}: {e}"
                        )

                logger.debug(
                    f"Comment {idx+1}/{total} parsed: {len(parsed_changes)} changes extracted"
                )

                return (idx, parsed_changes)

            except Exception as e:
                logger.error(f"Comment {idx+1}/{total} parsing failed: {type(e).__name__}: {e}")

                # Update progress even on failure (minimize lock hold time)
                with completed_lock:
                    completed_count += 1
                    local_count = completed_count

                # Call progress callback outside lock to avoid blocking
                if progress_callback:
                    try:
                        progress_callback(local_count, total)
                    except Exception as e2:
                        logger.warning(
                            f"Progress callback raised exception: {type(e2).__name__}: {e2}"
                        )

                if not self.fallback_to_regex:
                    raise

                # Return empty list for failed comment
                return (idx, [])

        # Submit all comments for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create futures mapping to preserve order
            future_to_index: dict[Future[tuple[int, list[ParsedChange]]], int] = {}

            for idx, comment in enumerate(comments):
                future = executor.submit(parse_single_comment, idx, comment)
                future_to_index[future] = idx

            # Collect results as they complete (order doesn't matter for collection)
            for future in as_completed(future_to_index):
                try:
                    idx, parsed_changes = future.result()
                    results[idx] = parsed_changes
                except Exception as e:
                    idx = future_to_index[future]
                    if self.fallback_to_regex:
                        logger.error(f"Unexpected error processing comment {idx+1}: {e}")
                        results[idx] = []
                    else:
                        raise

        # Build the final ordered list from the dictionary
        ordered_results = [results.get(i, []) for i in range(total)]

        successful = sum(1 for r in ordered_results if r)
        logger.info(f"Parallel parsing complete: {successful}/{total} comments parsed successfully")

        return ordered_results

    def _should_fallback_to_sequential(self) -> bool:
        """Check if circuit breaker requires sequential fallback.

        Returns:
            True if circuit breaker is open and we should fall back to sequential parsing
        """
        # Check if provider has circuit breaker (ResilientLLMProvider)
        return (
            isinstance(self.provider, SupportsCircuitBreaker)
            and self.provider.circuit_state == CircuitState.OPEN
        )

    def _parse_sequential(
        self,
        comments: list[CommentInput],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[ParsedChange]]:
        """Parse comments sequentially (fallback when circuit breaker is open).

        When ``fallback_to_regex`` is False this method preserves the exact error
        semantics of :meth:`parse_comment` (any exception is re-raised). When the
        flag is True it switches to a best-effort mode where exceptions are caught
        and converted into empty results so regex-based fallback logic can run later.

        Args:
            comments: List of comment inputs to parse
            progress_callback: Optional progress callback

        Returns:
            List of ParsedChange lists, one per input comment. Entries may be empty
            either because no changes were produced or because an exception was
            suppressed when ``fallback_to_regex`` is True.

        Raises:
            RuntimeError: If parse_comment raises and fallback_to_regex=False
            ValueError: If parse_comment raises and fallback_to_regex=False
        """
        results: list[list[ParsedChange]] = []
        total = len(comments)

        logger.info(f"Parsing {total} comments sequentially (circuit breaker fallback)")

        for idx, comment in enumerate(comments):
            try:
                parsed_changes = self.parse_comment(
                    comment_body=comment.body,
                    file_path=comment.file_path,
                    line_number=comment.line_number,
                    start_line=comment.start_line,
                    end_line=comment.end_line,
                )
                results.append(parsed_changes)

                if progress_callback:
                    try:
                        progress_callback(idx + 1, total)
                    except Exception as e:
                        logger.warning(
                            f"Progress callback raised exception: {type(e).__name__}: {e}"
                        )

            except Exception as e:
                # Preserve parse_comment error semantics
                # If fallback_to_regex is False, parse_comment would have raised,
                # so we should re-raise here to maintain behavior parity
                if not self.fallback_to_regex:
                    logger.error(
                        f"Sequential parsing failed for comment {idx+1} "
                        f"(fallback_to_regex=False): {type(e).__name__}: {e}"
                    )
                    raise

                # fallback_to_regex=True: convert exception to empty list for fallback
                logger.error(
                    f"Sequential parsing failed for comment {idx+1}: {type(e).__name__}: {e}"
                )
                results.append([])

                if progress_callback:
                    try:
                        progress_callback(idx + 1, total)
                    except Exception as e2:
                        logger.warning(
                            f"Progress callback raised exception: {type(e2).__name__}: {e2}"
                        )

        return results
