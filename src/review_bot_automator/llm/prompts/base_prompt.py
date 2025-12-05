# SPDX-License-Identifier: MIT
# Copyright (c) 2025 VirtualAgentics
"""Base prompt template for parsing CodeRabbit review comments.

This module contains the main prompt used to extract structured code changes
from GitHub PR review comments. The prompt is designed for:
- Handling multiple comment formats (diff blocks, suggestions, natural language)
- Producing structured JSON output that maps to ParsedChange objects
- Providing clear examples and validation rules
- Supporting context injection (file path, line numbers)

Template Placeholders:
    {comment_body}: Raw comment text to analyze
    {file_path}: Target file path for the change
    {start_line}: Start of the diff range (0 = unknown)
    {end_line}: End of the diff range (0 = unknown)
    {detected_sources}: Formatted string describing detected comment formats.
        Populated by UniversalLLMParser._build_source_context() using
        extract_comment_sources() from comment_sources module. Example values:
        "✓ 1 AI Prompt block(s) detected - HIGHEST PRIORITY (confidence >= 0.95)",
        "✓ 2 diff block(s) detected (with hunk headers)",
        "No structured blocks detected. Relying on natural language parsing".
"""

PARSE_COMMENT_PROMPT: str = """You are a code change extractor analyzing GitHub PR review \
comments from CodeRabbit AI.

Your task: Extract ALL suggested code changes from the comment below and return them \
as a JSON array.

## Supported Comment Formats

CodeRabbit comments can contain changes in multiple formats:

1. **Diff blocks**: Standard unified diff format
   ```diff
   @@ -1,3 +1,3 @@
    def example():
   -    return "old"
   +    return "new"
   ```

2. **Suggestion blocks**: Markdown code suggestions
   ```suggestion
   def new_function():
       return "updated"
   ```

3. **Natural language**: Prose descriptions of changes
   - "Change the timeout from 30 to 60 seconds on line 15"
   - "Replace the deprecated API call with the new method"
   - "Add error handling for the database connection"

4. **Multi-option suggestions**: Multiple alternatives
   **Option 1:** Use async/await
   **Option 2:** Use callbacks
   **Option 3:** Use promises

## Priority Data Sources

CodeRabbit comments may contain multiple data formats. Use this priority order:

1. **🤖 Prompt for AI Agents** (HIGHEST PRIORITY - confidence >= 0.95)
   - Found in: `<details><summary>🤖 Prompt for AI Agents</summary>...</details>`
   - Contains explicit, structured instructions for automated tools
   - Extract: file path, line range, exact action to take
   - These instructions are authoritative - follow them precisely
   - Example: "In src/foo.py around line 50, rename function bar to baz"

2. **Suggestion blocks** (` ```suggestion `) - confidence >= 0.92
   - Explicit code replacement - use exactly as provided
   - Cross-reference with AI Prompt for context if both exist

3. **Diff blocks** (` ```diff `)
   - With @@ hunk headers: confidence >= 0.90 (typically ~0.95)
   - Without hunk headers: confidence 0.70-0.85 (less precise line numbers)
   - When both diff AND AI Prompt exist, use AI Prompt for intent, diff for exact changes
   - Cross-reference to validate line numbers match

4. **Natural language** (LOWEST PRIORITY)
   - Use only when no structured formats are present
   - Lower confidence (< 0.75) for inferred changes

**Conflict Resolution**: When AI Prompt contradicts diff/suggestion blocks:
- Prioritize AI Prompt for understanding the INTENT of the change
- Use diff/suggestion blocks for the EXACT code changes
- Flag any line-number or intent discrepancies in the rationale field

## Detected Sources in This Comment

{detected_sources}

## Context Information

File: {file_path}
Line Range: {start_line} to {end_line}

NOTE: A value of 0 means the line number is unknown. When line context is missing (0),
rely on diff block line numbers, natural language references, or set confidence < 0.5.

IMPORTANT: When line numbers are provided (non-zero), this comment targets lines
{start_line}-{end_line} in the file. The changes you extract should have start_line
and end_line within or very close to this range.

## Comment Body

```
{comment_body}
```

## Output Format

Return a JSON array of change objects. Each change must have:

```json
[
  {{
    "file_path": "path/to/file.py",
    "start_line": 10,
    "end_line": 15,
    "new_content": "the actual code to apply",
    "change_type": "modification",
    "confidence": 0.95,
    "rationale": "why this change is suggested",
    "risk_level": "low"
  }}
]
```

## Field Requirements

1. **file_path** (string):
   - Use the file path from context if available
   - Extract from comment if mentioned explicitly
   - Use "unknown" only if truly ambiguous

2. **start_line** (integer >= 1):
   - When context line range is available (non-zero), MUST be within or near that range
   - When context is 0 (unknown), extract from diff block or natural language
   - Extract precise line from diff block content (count lines from context start)
   - For natural language, infer from phrases like "on line N"
   - If uncertain, set confidence < 0.5

3. **end_line** (integer >= start_line):
   - When context line range is available (non-zero), MUST be within or near that range
   - When context is 0 (unknown), calculate from diff block or suggestion length
   - Must be >= start_line
   - For single-line changes, end_line = start_line

4. **new_content** (string):
   - The exact code to apply, preserving indentation
   - For deletions, use empty string ""
   - Do NOT include markdown backticks
   - Preserve exact formatting including newlines

5. **change_type** (enum):
   - "addition": New code being added
   - "modification": Existing code being changed
   - "deletion": Code being removed

6. **confidence** (float 0.0-1.0):
   - 0.9-1.0: Clear diff block or suggestion with line numbers
   - 0.7-0.9: Natural language with specific line references
   - 0.5-0.7: Natural language with context clues
   - 0.0-0.5: Ambiguous suggestion, missing context
   - Return low confidence if line numbers are inferred

7. **rationale** (string):
   - Explain WHY this change is being suggested
   - Extract from comment context
   - Include keywords: "bug", "performance", "security", "style", etc.

8. **risk_level** (enum):
   - "low": Formatting, comments, documentation, minor refactor
   - "medium": Logic changes, new features, API changes
   - "high": Security fixes, breaking changes, data migrations

## Parsing Rules

1. **Extract ALL formats**: Don't ignore any format type
2. **Preserve code exactly**: Keep indentation, whitespace, and formatting
3. **Handle multi-line**: Diff blocks and suggestions often span multiple lines
4. **Confidence matters**: Set low confidence if line numbers are unclear
5. **Empty array is valid**: Return [] if no actionable changes found
6. **No markdown in code**: Strip ```suggestion and ```diff markers
7. **Context awareness**: Use file_path and line_number context when available
8. **Be conservative**: Low confidence is better than wrong extraction

## Examples of Confidence Levels

**High confidence (0.95)**: Diff block with explicit line numbers
```diff
@@ -10,3 +10,5 @@ def process():
```

**Medium confidence (0.75)**: Natural language with specific line
"Change line 42 to use async def instead of def"

**Low confidence (0.4)**: Vague natural language
"The function should handle errors better"
→ Return low confidence or skip if too ambiguous

## Edge Cases

- **No changes**: Return empty array []
- **Only questions/discussion**: Return []
- **Multiple options**: Extract each as separate change
- **Ambiguous line numbers**: Set confidence < 0.5
- **Missing file context**: Use "unknown" for file_path, low confidence

## Output Rules

1. Return ONLY the JSON array, no markdown, no explanation
2. Ensure valid JSON (no trailing commas, proper escaping)
3. All strings must be properly escaped
4. Array can be empty []
5. Do not include the ```json markers

Begin extraction now."""
