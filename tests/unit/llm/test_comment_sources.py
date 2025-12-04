# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Tests for comment source detection module.

This module tests the extraction and classification of structured blocks
from CodeRabbit review comments.
"""

from __future__ import annotations

import pytest

from review_bot_automator.llm.comment_sources import (
    AIPromptBlock,
    DetailsBlock,
    DetailsBlockType,
    DiffBlock,
    SuggestionBlock,
    extract_comment_sources,
)


class TestExtractCommentSourcesEdgeCases:
    """Test edge case handling in extract_comment_sources."""

    def test_none_input_returns_empty(self) -> None:
        """Test that None input returns empty CommentSources."""
        sources = extract_comment_sources(None)
        assert sources.diff_blocks == ()
        assert sources.suggestion_blocks == ()
        assert sources.ai_prompt_blocks == ()
        assert sources.details_blocks == ()
        assert sources.html_comments_stripped == 0
        assert not sources.has_any_blocks

    def test_empty_string_returns_empty(self) -> None:
        """Test that empty string returns empty CommentSources."""
        sources = extract_comment_sources("")
        assert not sources.has_any_blocks
        assert sources.block_count == 0

    def test_whitespace_only_returns_empty(self) -> None:
        """Test that whitespace-only input returns empty CommentSources."""
        sources = extract_comment_sources("   \n\t\n   ")
        assert not sources.has_any_blocks

    def test_plain_text_returns_empty(self) -> None:
        """Test that plain text without blocks returns empty sources."""
        sources = extract_comment_sources("Please fix this bug in the authentication flow.")
        assert not sources.has_diff
        assert not sources.has_suggestion
        assert not sources.has_ai_prompt
        assert sources.block_count == 0

    def test_large_comment_truncated(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that comments exceeding 100KB are truncated with warning."""
        # Create a comment larger than 100KB
        large_comment = "x" * (100 * 1024 + 1000)
        sources = extract_comment_sources(large_comment)

        # Should have logged a warning
        assert "exceeds max size" in caplog.text
        assert not sources.has_any_blocks


class TestExtractDiffBlocks:
    """Test diff block extraction."""

    def test_single_diff_block(self) -> None:
        """Test extraction of a single diff block."""
        body = """Fix this issue:
```diff
-old line
+new line
```
"""
        sources = extract_comment_sources(body)
        assert sources.has_diff
        assert len(sources.diff_blocks) == 1
        assert sources.diff_blocks[0].content == "-old line\n+new line"
        assert sources.diff_blocks[0].position > 0

    def test_diff_block_with_hunk_header(self) -> None:
        """Test detection of diff hunk headers."""
        body = """Apply this:
```diff
@@ -10,5 +10,6 @@
-old code
+new code
```
"""
        sources = extract_comment_sources(body)
        assert len(sources.diff_blocks) == 1
        assert sources.diff_blocks[0].has_hunk_header is True

    def test_diff_block_without_hunk_header(self) -> None:
        """Test diff block without @@ header."""
        body = """Simple change:
```diff
-remove this
+add this
```
"""
        sources = extract_comment_sources(body)
        assert sources.diff_blocks[0].has_hunk_header is False

    def test_multiple_diff_blocks(self) -> None:
        """Test extraction of multiple diff blocks."""
        body = """Multiple changes:
```diff
-first old
+first new
```

Some text between.

```diff
-second old
+second new
```
"""
        sources = extract_comment_sources(body)
        assert len(sources.diff_blocks) == 2
        assert "first" in sources.diff_blocks[0].content
        assert "second" in sources.diff_blocks[1].content

    def test_diff_block_case_insensitive(self) -> None:
        """Test that DIFF and Diff are also matched."""
        body = """```DIFF
-old
+new
```
"""
        sources = extract_comment_sources(body)
        assert len(sources.diff_blocks) == 1


class TestExtractSuggestionBlocks:
    """Test suggestion block extraction."""

    def test_single_suggestion_block(self) -> None:
        """Test extraction of a single suggestion block."""
        body = """Apply this fix:
```suggestion
def fixed_function():
    return True
```
"""
        sources = extract_comment_sources(body)
        assert sources.has_suggestion
        assert len(sources.suggestion_blocks) == 1
        assert "def fixed_function" in sources.suggestion_blocks[0].content

    def test_suggestion_with_option_label(self) -> None:
        """Test extraction of option label preceding suggestion."""
        body = """Choose one:

**Option 1:**
```suggestion
def option_one():
    pass
```

**Option 2:**
```suggestion
def option_two():
    pass
```
"""
        sources = extract_comment_sources(body)
        assert len(sources.suggestion_blocks) == 2
        assert sources.suggestion_blocks[0].option_label == "Option 1"
        assert sources.suggestion_blocks[1].option_label == "Option 2"

    def test_suggestion_without_option_label(self) -> None:
        """Test suggestion block without preceding option label."""
        body = """Just do this:
```suggestion
simple_fix()
```
"""
        sources = extract_comment_sources(body)
        assert sources.suggestion_blocks[0].option_label is None

    def test_suggestion_block_case_insensitive(self) -> None:
        """Test that SUGGESTION and Suggestion are also matched."""
        body = """```SUGGESTION
code here
```
"""
        sources = extract_comment_sources(body)
        assert len(sources.suggestion_blocks) == 1


class TestExtractAIPromptBlocks:
    """Test AI prompt block extraction."""

    def test_ai_prompt_with_robot_emoji(self) -> None:
        """Test extraction of AI prompt with robot emoji in summary."""
        body = """<details>
<summary>\U0001f916 Prompt for AI Agents</summary>

```
In src/foo.py around line 50, rename function bar to baz.
```

</details>"""
        sources = extract_comment_sources(body)
        assert sources.has_ai_prompt
        assert len(sources.ai_prompt_blocks) == 1
        assert "rename function bar to baz" in sources.ai_prompt_blocks[0].content
        assert "\U0001f916" in sources.ai_prompt_blocks[0].summary

    def test_ai_prompt_content_cleaned(self) -> None:
        """Test that code fences are stripped from AI prompt content."""
        body = """<details>
<summary>\U0001f916 Prompt for AI Agents</summary>

```
The actual instruction without fences.
```

</details>"""
        sources = extract_comment_sources(body)
        content = sources.ai_prompt_blocks[0].content
        assert "```" not in content
        assert "The actual instruction" in content

    def test_ai_prompt_real_pr286_format(self) -> None:
        """Test with real PR #286 format from CodeRabbit."""
        body = """_\U0001f9f9 Nitpick_ | _\U0001f535 Trivial_

**Consider validating line range ordering.**

<details>
<summary>\U0001f916 Prompt for AI Agents</summary>

```
In src/review_bot_automator/core/resolver.py around lines 282 to 292, add
defensive validation for the comment line range: when both start_line and
end_line are present, ensure start_line <= end_line by normalizing them.
```

</details>

<!-- This is an auto-generated comment by CodeRabbit -->
"""
        sources = extract_comment_sources(body)
        assert sources.has_ai_prompt
        assert "resolver.py" in sources.ai_prompt_blocks[0].content
        assert "defensive validation" in sources.ai_prompt_blocks[0].content
        # HTML comment should be stripped
        assert sources.html_comments_stripped == 1


class TestDetailsBlockClassification:
    """Test classification of <details> blocks."""

    def test_classify_analysis_chain(self) -> None:
        """Test classification of Analysis chain blocks."""
        body = """<details>
<summary>\U0001f9e9 Analysis chain</summary>

Extended analysis goes here...

</details>"""
        sources = extract_comment_sources(body)
        assert len(sources.details_blocks) == 1
        assert sources.details_blocks[0].block_type == DetailsBlockType.ANALYSIS_CHAIN
        # Should NOT be in ai_prompt_blocks
        assert len(sources.ai_prompt_blocks) == 0

    def test_classify_walkthrough(self) -> None:
        """Test classification of Walkthrough blocks."""
        body = """<details>
<summary>Code walkthrough</summary>

Step by step explanation...

</details>"""
        sources = extract_comment_sources(body)
        assert sources.details_blocks[0].block_type == DetailsBlockType.WALKTHROUGH

    def test_classify_related_issues(self) -> None:
        """Test classification of Related issues/PRs blocks."""
        body = """<details>
<summary>Related PRs</summary>

- #123: Previous fix
- #456: Related feature

</details>"""
        sources = extract_comment_sources(body)
        assert sources.details_blocks[0].block_type == DetailsBlockType.RELATED_ISSUES

    def test_classify_other(self) -> None:
        """Test classification of unrecognized details blocks."""
        body = """<details>
<summary>Some random collapsible section</summary>

Content here

</details>"""
        sources = extract_comment_sources(body)
        assert sources.details_blocks[0].block_type == DetailsBlockType.OTHER

    def test_classify_unknown_no_summary(self) -> None:
        """Test classification when summary tag is missing."""
        body = """<details>

Content without summary

</details>"""
        sources = extract_comment_sources(body)
        assert sources.details_blocks[0].block_type == DetailsBlockType.UNKNOWN

    def test_multiple_details_blocks_different_types(self) -> None:
        """Test multiple details blocks with different classifications."""
        body = """<details>
<summary>\U0001f916 Prompt for AI Agents</summary>

AI instructions here.

</details>

<details>
<summary>\U0001f9e9 Analysis chain</summary>

Analysis reasoning here.

</details>"""
        sources = extract_comment_sources(body)
        assert len(sources.details_blocks) == 2
        assert len(sources.ai_prompt_blocks) == 1  # Only AI_PROMPT filtered
        types = {db.block_type for db in sources.details_blocks}
        assert DetailsBlockType.AI_PROMPT in types
        assert DetailsBlockType.ANALYSIS_CHAIN in types

    def test_nested_details_limitation_documented(self) -> None:
        """Test that nested <details> blocks do NOT parse correctly.

        This test documents the intentional limitation: the non-greedy regex
        pairs each opening <details> tag with the FIRST encountered </details>
        tag, breaking nested structures.

        For example, in:
            <details>outer<details>inner</details>more</details>

        Only "outer<details>inner" is captured as content, with "more</details>"
        left as unparsed text. This is acceptable because CodeRabbit comments
        do not use nested collapsible sections.
        """
        # Nested structure: outer details containing inner details
        body = """<details>
<summary>Outer section</summary>

Some outer content
<details>
<summary>Inner section</summary>

Inner content here

</details>
More outer content after inner

</details>"""
        sources = extract_comment_sources(body)

        # Due to the limitation, only ONE block is extracted (not two properly nested)
        # The non-greedy regex matches from <details> to the first </details>
        assert len(sources.details_blocks) == 1

        # The extracted block captures content up to the FIRST </details>
        # which means it includes the inner <details> opening tag but stops
        # at the inner </details>, leaving "More outer content..." unparsed
        block = sources.details_blocks[0]
        assert block.summary == "Outer section"
        # Content includes the raw inner <details> tag since it wasn't matched
        assert "<details>" in block.content
        assert "Inner section" in block.content
        # The "More outer content after inner" is NOT in the captured content
        # because the regex stopped at the first </details>
        assert "More outer content after inner" not in block.content


class TestHTMLCommentStripping:
    """Test HTML comment stripping functionality."""

    def test_single_html_comment_stripped(self) -> None:
        """Test stripping of single HTML comment."""
        body = """Some text
<!-- This is a comment -->
More text
```diff
-old
+new
```
"""
        sources = extract_comment_sources(body)
        assert sources.html_comments_stripped == 1
        assert sources.has_diff

    def test_multiple_html_comments_stripped(self) -> None:
        """Test stripping of multiple HTML comments."""
        body = """<!-- comment 1 -->
Text here
<!-- comment 2 -->
<!-- comment 3 -->
"""
        sources = extract_comment_sources(body)
        assert sources.html_comments_stripped == 3

    def test_fingerprinting_marker_stripped(self) -> None:
        """Test stripping of CodeRabbit fingerprinting markers."""
        body = """<details>
<summary>\U0001f916 Prompt for AI Agents</summary>

Instructions here.

</details>

<!-- fingerprinting:phantom:medusa:ocelot -->
"""
        sources = extract_comment_sources(body)
        assert sources.html_comments_stripped == 1
        assert sources.has_ai_prompt


class TestPositionTracking:
    """Test that position tracking is correct."""

    def test_diff_block_position(self) -> None:
        """Test that diff block position is correctly tracked."""
        body = "prefix text\n```diff\n-old\n+new\n```"
        sources = extract_comment_sources(body)
        # Position should be after "prefix text\n"
        assert sources.diff_blocks[0].position == len("prefix text\n")

    def test_multiple_blocks_positions_ascending(self) -> None:
        """Test that positions of multiple blocks are in ascending order."""
        body = """```diff
-first
```

```diff
-second
```
"""
        sources = extract_comment_sources(body)
        positions = [block.position for block in sources.diff_blocks]
        assert positions == sorted(positions)
        assert positions[0] < positions[1]


class TestCommentSourcesProperties:
    """Test CommentSources dataclass properties."""

    def test_has_any_blocks_true(self) -> None:
        """Test has_any_blocks returns True when blocks exist."""
        body = "```diff\n-old\n+new\n```"
        sources = extract_comment_sources(body)
        assert sources.has_any_blocks is True

    def test_has_any_blocks_false(self) -> None:
        """Test has_any_blocks returns False when no blocks exist."""
        sources = extract_comment_sources("Plain text only")
        assert sources.has_any_blocks is False

    def test_block_count(self) -> None:
        """Test block_count returns correct total."""
        body = """```diff
-old
+new
```

```suggestion
new code
```

<details>
<summary>\U0001f916 Prompt for AI Agents</summary>

Instructions.

</details>
"""
        sources = extract_comment_sources(body)
        assert sources.block_count == 3  # 1 diff + 1 suggestion + 1 AI prompt

    def test_individual_has_properties(self) -> None:
        """Test has_diff, has_suggestion, has_ai_prompt properties."""
        # Only diff
        sources_diff = extract_comment_sources("```diff\n-x\n```")
        assert sources_diff.has_diff is True
        assert sources_diff.has_suggestion is False
        assert sources_diff.has_ai_prompt is False

        # Only suggestion
        sources_sugg = extract_comment_sources("```suggestion\nx\n```")
        assert sources_sugg.has_diff is False
        assert sources_sugg.has_suggestion is True


class TestMixedContent:
    """Test extraction from comments with mixed content types."""

    def test_diff_and_ai_prompt(self) -> None:
        """Test comment with both diff block and AI prompt."""
        body = """```diff
-old code
+new code
```

<details>
<summary>\U0001f916 Prompt for AI Agents</summary>

Replace old code with new code in the main function.

</details>
"""
        sources = extract_comment_sources(body)
        assert sources.has_diff
        assert sources.has_ai_prompt
        assert len(sources.diff_blocks) == 1
        assert len(sources.ai_prompt_blocks) == 1

    def test_suggestion_and_analysis_chain(self) -> None:
        """Test comment with suggestion and non-AI details block."""
        body = """```suggestion
def better():
    pass
```

<details>
<summary>\U0001f9e9 Analysis chain</summary>

Reasoning behind the suggestion.

</details>
"""
        sources = extract_comment_sources(body)
        assert sources.has_suggestion
        assert not sources.has_ai_prompt  # Analysis chain is not AI prompt
        assert len(sources.details_blocks) == 1
        assert sources.details_blocks[0].block_type == DetailsBlockType.ANALYSIS_CHAIN


class TestDataclassImmutability:
    """Test that dataclasses are properly frozen."""

    def test_diff_block_frozen(self) -> None:
        """Test that DiffBlock cannot be mutated."""
        block = DiffBlock(content="test", position=0, has_hunk_header=False)
        with pytest.raises(AttributeError):
            block.content = "modified"  # type: ignore[misc]

    def test_suggestion_block_frozen(self) -> None:
        """Test that SuggestionBlock cannot be mutated."""
        block = SuggestionBlock(content="test", position=0, option_label=None)
        with pytest.raises(AttributeError):
            block.position = 100  # type: ignore[misc]

    def test_ai_prompt_block_frozen(self) -> None:
        """Test that AIPromptBlock cannot be mutated."""
        block = AIPromptBlock(summary="test", content="content", position=0)
        with pytest.raises(AttributeError):
            block.summary = "modified"  # type: ignore[misc]

    def test_details_block_frozen(self) -> None:
        """Test that DetailsBlock cannot be mutated."""
        block = DetailsBlock(
            summary="test",
            content="content",
            block_type=DetailsBlockType.OTHER,
            position=0,
            raw="<details>test</details>",
        )
        with pytest.raises(AttributeError):
            block.content = "modified"  # type: ignore[misc]

    def test_comment_sources_frozen(self) -> None:
        """Test that CommentSources cannot be mutated."""
        sources = extract_comment_sources("test")
        with pytest.raises(AttributeError):
            sources.html_comments_stripped = 99  # type: ignore[misc]
