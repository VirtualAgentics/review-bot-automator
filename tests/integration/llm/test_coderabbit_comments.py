# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Integration tests for CodeRabbit comment parsing.

These tests use real comment samples from PR #286 to validate
the full parsing pipeline with actual CodeRabbit formatting.

The comments were extracted from:
https://github.com/VirtualAgentics/review-bot-automator/pull/286/comments
"""

from __future__ import annotations

import pytest

from review_bot_automator.llm.comment_sources import (
    DetailsBlockType,
    extract_comment_sources,
)

# =============================================================================
# Real PR #286 Comment Fixtures
# =============================================================================

# Comment 1: Line ordering validation nitpick
# Contains: diff block with @@ hunk header AND AI prompt block
PR_286_LINE_ORDERING_COMMENT = """_🧹 Nitpick_ | _🔵 Trivial_

**Consider validating line range ordering.**

When both `start_line` and `end_line` are present, there's no validation that `start_line <= end_line`. While GitHub should provide valid ranges, defensive validation would prevent potential downstream issues if the API returns unexpected data.



Consider adding validation after line 282:

```diff
 start_line = comment.get("start_line") or comment.get("original_start_line")
+
+# Validate line ordering if both are present
+if start_line is not None and line is not None and start_line > line:
+    self.logger.warning(
+        f"Invalid line range in comment for {path}: start_line={start_line} > end_line={line}, "
+        "skipping comment"
+    )
+    continue

 if path and body and line:
```

<details>
<summary>🤖 Prompt for AI Agents</summary>

```
In src/review_bot_automator/core/resolver.py around lines 282 to 292, add
defensive validation for the comment line range: when both start_line and
end_line are present (and convertible to ints), ensure start_line <= end_line by
normalizing them (e.g., cast to int, compute min/max) and use min as start_line
and max as end_line before constructing CommentInput; if either value is missing
or not an int, fall back to the existing behavior (use line for
end_line/start_line as currently done) and ensure the normalized values are
passed into CommentInput.
```

</details>

<!-- This is an auto-generated comment by CodeRabbit -->

✅ Addressed in commit abf8450"""

# Comment 2: Deprecated line_number field
# Contains: AI prompt block only (no diff/suggestion)
PR_286_DEPRECATED_FIELD_COMMENT = """_🧹 Nitpick_ | _🔵 Trivial_

**Technical debt: deprecated line_number field.**

The comment indicates `line_number` is kept for backward compatibility. Consider creating a follow-up issue to remove this deprecated field once all callers have migrated to using `start_line` and `end_line`.




Do you want me to create a tracking issue for removing the deprecated `line_number` field?

<details>
<summary>🤖 Prompt for AI Agents</summary>

```
In src/review_bot_automator/core/resolver.py around line 289, the code still
emits the deprecated `line_number` field for backward compatibility; create a
short follow-up/tracking issue to remove `line_number` once all callers use
`start_line`/`end_line`, mark the field as deprecated in code with a TODO and
link to the issue, update documentation/comments to mention migration path, run
a project-wide search to list and update remaining callers to use
`start_line`/`end_line`, and only then remove the `line_number` emission and
associated tests in a follow-up PR referenced by the tracking issue.
```

</details>

<!-- This is an auto-generated comment by CodeRabbit -->

✅ Addressed in commit 2ca1fa0"""

# Comment 3: Redundant check removal
# Contains: diff block AND AI prompt block
PR_286_REDUNDANT_CHECK_COMMENT = """_🧹 Nitpick_ | _🔵 Trivial_

**Redundant check: effective_line is guaranteed to exist.**

The check `if not effective_line` on line 510 is redundant. At line 281-284 in the parallel path and here in the sequential path, we've already verified that `line or original_line` exists. If neither exists, we wouldn't reach this point.



Consider removing the redundant check:

```diff
-effective_line = line or original_line
-if not effective_line:
-    return []
-
 # Calculate effective start and end lines for LLM context
 effective_start = start_line or original_start_line
 effective_end = line or original_line
```

<details>
<summary>🤖 Prompt for AI Agents</summary>

```
In src/review_bot_automator/core/resolver.py around lines 509 to 511, the guard
`if not effective_line: return []` is redundant because `effective_line = line
or original_line` has already been validated on the other code path; remove the
`if not effective_line` check and its early return so execution continues using
`effective_line` without changing surrounding logic.
```

</details>

<!-- This is an auto-generated reply by CodeRabbit -->

✅ Confirmed as addressed by @VirtualAgentics

<!-- This is an auto-generated comment by CodeRabbit -->"""

# Comment 4: Debug logging nitpick with fingerprinting marker
# Contains: AI prompt block AND HTML fingerprinting comment
PR_286_DEBUG_LOGGING_COMMENT = """_🧹 Nitpick_ | _🔵 Trivial_

**Debug logging is helpful but consider log level.**

The context section logging is useful for debugging Issue #285, but once resolved, consider removing or guarding with `logger.isEnabledFor(logging.DEBUG)` to avoid the string slice computation overhead in production.

<details>
<summary>🤖 Prompt for AI Agents</summary>

```
In src/review_bot_automator/llm/parser.py around lines 245 to 251, the code
unconditionally computes prompt.find and slices the prompt to build the
context_section before calling logger.debug; wrap the index lookups, slice and
logger.debug call inside a guard that first checks
logger.isEnabledFor(logging.DEBUG) so the string slicing and find operations
only occur when DEBUG logging is enabled; ensure you import logging if not
already and use logger.isEnabledFor(logging.DEBUG) as the condition, then
perform the existing context_start_idx/context_end_idx checks and debug log
inside that block.
```

</details>

<!-- fingerprinting:phantom:medusa:ocelot -->

<!-- This is an auto-generated comment by CodeRabbit -->

✅ Addressed in commit abf8450"""


@pytest.mark.integration
class TestRealCodeRabbitComments:
    """Integration tests with actual PR #286 CodeRabbit comments."""

    def test_parse_pr286_line_ordering_comment_detects_ai_prompt(self) -> None:
        """Parse real CodeRabbit comment about line range ordering.

        Verifies:
        - AI prompt block is detected
        - Diff block with hunk header is detected
        - Content extraction is accurate
        """
        sources = extract_comment_sources(PR_286_LINE_ORDERING_COMMENT)

        # AI prompt block should be detected
        assert sources.has_ai_prompt
        assert len(sources.ai_prompt_blocks) == 1
        ai_block = sources.ai_prompt_blocks[0]
        assert "🤖 Prompt for AI Agents" in ai_block.summary

        # AI prompt content should contain key instructions
        assert "src/review_bot_automator/core/resolver.py" in ai_block.content
        assert "lines 282 to 292" in ai_block.content
        assert "start_line <= end_line" in ai_block.content

        # Diff block should be detected (without @@ hunk header in this case)
        assert sources.has_diff
        assert len(sources.diff_blocks) == 1
        diff = sources.diff_blocks[0]
        # This diff block has context lines but no @@ header
        assert not diff.has_hunk_header
        assert 'comment.get("start_line")' in diff.content

    def test_parse_pr286_deprecated_field_comment_ai_only(self) -> None:
        """Parse real CodeRabbit comment about deprecated field.

        This comment has AI prompt but no diff/suggestion blocks.
        Verifies:
        - AI prompt is detected even without code blocks
        - Content extraction captures full instructions
        """
        sources = extract_comment_sources(PR_286_DEPRECATED_FIELD_COMMENT)

        # AI prompt should be detected
        assert sources.has_ai_prompt
        assert len(sources.ai_prompt_blocks) == 1

        # No diff or suggestion blocks
        assert not sources.has_diff
        assert not sources.has_suggestion

        # AI prompt content should be complete
        ai_content = sources.ai_prompt_blocks[0].content
        assert "line 289" in ai_content
        assert "deprecated" in ai_content
        assert "follow-up/tracking issue" in ai_content

    def test_parse_pr286_redundant_check_comment_multiple_blocks(self) -> None:
        """Parse real CodeRabbit comment about redundant check.

        Contains both diff block AND AI prompt block.
        Verifies:
        - Both block types are detected
        - HTML comments are stripped
        """
        sources = extract_comment_sources(PR_286_REDUNDANT_CHECK_COMMENT)

        # Both AI prompt and diff should be present
        assert sources.has_ai_prompt
        assert sources.has_diff

        # AI prompt content
        ai_content = sources.ai_prompt_blocks[0].content
        assert "lines 509 to 511" in ai_content
        assert "redundant" in ai_content

        # Diff block should show deletion lines
        diff_content = sources.diff_blocks[0].content
        assert "-effective_line = line or original_line" in diff_content
        assert "-if not effective_line:" in diff_content

        # HTML comments should be stripped
        assert sources.html_comments_stripped >= 1

    def test_parse_pr286_fingerprinting_comment_strips_markers(self) -> None:
        """Parse comment with fingerprinting HTML comments.

        Verifies:
        - AI prompt is detected
        - Fingerprinting markers (<!-- fingerprinting:... -->) are stripped
        """
        sources = extract_comment_sources(PR_286_DEBUG_LOGGING_COMMENT)

        # AI prompt should be detected
        assert sources.has_ai_prompt
        ai_content = sources.ai_prompt_blocks[0].content
        assert "logger.isEnabledFor" in ai_content
        assert "logging.DEBUG" in ai_content

        # Fingerprinting markers should have been stripped
        # The comment contains: <!-- fingerprinting:phantom:medusa:ocelot -->
        assert sources.html_comments_stripped >= 1

    def test_extract_sources_batch_real_comments(self) -> None:
        """Extract sources from batch of real PR #286 comments.

        Verifies consistent behavior across multiple comment formats.
        """
        comments = [
            PR_286_LINE_ORDERING_COMMENT,
            PR_286_DEPRECATED_FIELD_COMMENT,
            PR_286_REDUNDANT_CHECK_COMMENT,
            PR_286_DEBUG_LOGGING_COMMENT,
        ]

        results = [extract_comment_sources(c) for c in comments]

        # All should have AI prompts
        assert all(r.has_ai_prompt for r in results)

        # All should have exactly 1 AI prompt block
        assert all(len(r.ai_prompt_blocks) == 1 for r in results)

        # Total diff blocks: comments 1, 3 have diffs
        diff_counts = [len(r.diff_blocks) for r in results]
        assert diff_counts == [1, 0, 1, 0]

        # Total block count (diff + suggestion + ai)
        total_blocks = [r.block_count for r in results]
        assert total_blocks == [2, 1, 2, 1]


@pytest.mark.integration
class TestAIPromptContentExtraction:
    """Tests for accurate AI prompt content extraction."""

    def test_ai_prompt_content_preserves_file_paths(self) -> None:
        """Verify file paths in AI prompts are preserved."""
        sources = extract_comment_sources(PR_286_LINE_ORDERING_COMMENT)
        content = sources.ai_prompt_blocks[0].content

        # Full file path should be preserved
        assert "src/review_bot_automator/core/resolver.py" in content

    def test_ai_prompt_content_preserves_line_numbers(self) -> None:
        """Verify line number references in AI prompts are preserved."""
        sources = extract_comment_sources(PR_286_REDUNDANT_CHECK_COMMENT)
        content = sources.ai_prompt_blocks[0].content

        # Line number references should be preserved
        assert "lines 509 to 511" in content

    def test_ai_prompt_content_strips_code_fences(self) -> None:
        """Verify code fences inside AI prompt are stripped.

        CodeRabbit wraps AI prompt content in triple backticks:
        <details><summary>...</summary>
        ```
        actual content
        ```
        </details>
        """
        sources = extract_comment_sources(PR_286_LINE_ORDERING_COMMENT)
        content = sources.ai_prompt_blocks[0].content

        # Code fences should be stripped, leaving clean content
        assert not content.startswith("```")
        assert not content.endswith("```")
        # But the actual instructions should remain
        assert "In src/review_bot_automator" in content


@pytest.mark.integration
class TestDetailsBlockClassification:
    """Tests for correct classification of <details> blocks."""

    def test_ai_prompt_details_classified_correctly(self) -> None:
        """Verify AI prompt blocks are classified as AI_PROMPT type."""
        sources = extract_comment_sources(PR_286_LINE_ORDERING_COMMENT)

        # Should have details blocks
        assert len(sources.details_blocks) == 1
        details = sources.details_blocks[0]

        # Should be classified as AI_PROMPT
        assert details.block_type == DetailsBlockType.AI_PROMPT
        assert "🤖 Prompt for AI Agents" in details.summary

    def test_emoji_in_summary_detected(self) -> None:
        """Verify robot emoji in summary triggers AI prompt detection."""
        sources = extract_comment_sources(PR_286_DEBUG_LOGGING_COMMENT)

        # Robot emoji (🤖) should trigger AI prompt detection
        assert sources.has_ai_prompt
        assert "🤖" in sources.ai_prompt_blocks[0].summary
