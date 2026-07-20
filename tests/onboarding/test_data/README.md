# Test Data for Model Onboarding

This directory contains test cases for evaluating model framework capabilities. Test cases are used for both baseline evaluation and DSPy optimization.

## File Structure

- **simple_tasks.json** - Easy, single-step tasks (100+ cases)
- **complex_workflows.json** - Multi-step tasks requiring planning (100+ cases)
- **error_recovery.json** - Validation error scenarios with fixes (100+ cases)
- **repl_exploration.json** - Tasks requiring REPL exploration (50+ cases)
- **edge_cases.json** - Corner cases and tricky scenarios (50+ cases)
- **tool_selection.json** - Runtime tool selection scenarios (50+ cases)
- **working_context.json** - Multi-session context usage (50+ cases)

**Target**: 500+ diverse test cases across all capabilities

## Test Case Format

```json
{
  "test_cases": [
    {
      "id": "unique_identifier",
      "capability": "code_generation | message_streams | repl_usage | validation_retry | tool_selection | working_context",
      "task": "Clear task description for model",
      "expected_behavior": {
        "contains": ["pattern1", "pattern2"],
        "not_contains": ["anti_pattern"],
        "validation_passes": true,
        "tool_used": "code_generation | repl_command | abort",
        "message_present": true,
        "variable_expansion": true,
        "should_use_repl": true,
        "stores_in_context": true
      },
      "difficulty": "easy | medium | hard",
      "tags": ["tag1", "tag2", "tag3"]
    }
  ]
}
```

## Collecting Test Cases

### From Real Usage

As the framework is used:
1. Capture successful generations with traces
2. Extract task, generated code, and outcome
3. Add to appropriate test data file
4. Review and tag for difficulty/capability

### Manual Creation

When creating test cases manually:
1. Cover diverse scenarios within each capability
2. Include both positive (correct) and negative (error) examples
3. Vary difficulty (easy: 40%, medium: 40%, hard: 20%)
4. Ensure good distribution across tags

### Capability Distribution

Target test case count per capability:

| Capability | Easy | Medium | Hard | Total |
|-----------|------|--------|------|-------|
| code_generation | 60 | 60 | 30 | 150 |
| validation_retry | 40 | 50 | 10 | 100 |
| repl_usage | 20 | 25 | 5 | 50 |
| message_streams | 30 | 20 | 10 | 60 |
| tool_selection | 20 | 20 | 10 | 50 |
| working_context | 15 | 25 | 10 | 50 |
| replanning | 10 | 25 | 5 | 40 |
| **Total** | **195** | **225** | **80** | **500** |

## Tag Categories

**Code Generation Tags**:
- assignment, tool_call, loop, conditional, error_handling
- async_await, list_comprehension, dict_operations
- state_mutation, dataclass, type_annotations

**Error Recovery Tags**:
- import_error, lambda_error, exec_error, reflection_error
- tool_replacement, function_replacement, direct_assignment

**REPL Tags**:
- exploration, verification, data_preview, unknown_structure
- conditional, large_data, tool_availability

**Message Tags**:
- variable_expansion, progress_report, status_update, summary

**Tool Selection Tags**:
- code_ready, need_exploration, impossible_task, wrong_approach

**Context Tags**:
- multi_step, repl_storage, dynamic_value, cross_session

## Validation Rules

Before adding a test case:

1. **Uniqueness**: ID is unique across all files
2. **Clarity**: Task description is clear and unambiguous
3. **Completeness**: All expected_behavior fields are filled
4. **Difficulty**: Difficulty rating is accurate
5. **Tags**: At least 2 relevant tags are assigned
6. **Format**: JSON is valid and follows schema

## Usage in Optimization

Test cases are split:
- **Training set** (70%): Used for DSPy optimization
- **Validation set** (15%): Used during optimization for early stopping
- **Test set** (15%): Held-out for final evaluation

Split should be stratified by capability and difficulty.

## Contributing Test Cases

To add new test cases:

1. Choose appropriate file based on capability
2. Follow JSON format exactly
3. Assign unique ID (format: `{capability}_{number}`)
4. Fill all required fields
5. Test that JSON is valid
6. Add to end of test_cases array

## Example: Full Test Case

```json
{
  "id": "code_gen_042",
  "capability": "code_generation",
  "task": "Iterate through self.documents and call self.tools.analyze on each, storing results in self.results",
  "expected_behavior": {
    "contains": [
      "for doc in self.documents:",
      "await self.tools.analyze",
      "self.results"
    ],
    "not_contains": ["import", "lambda", "exec"],
    "validation_passes": true,
    "tool_used": "code_generation"
  },
  "agent_state": {
    "documents": ["doc1.txt", "doc2.txt"],
    "results": []
  },
  "expected_code_pattern": "for.*in self\\.documents:.*await self\\.tools\\.analyze.*self\\.results",
  "difficulty": "medium",
  "tags": ["loop", "tool_call", "async_await", "state_mutation"]
}
```

## Current Status

- ✅ simple_tasks.json: 5 cases (target: 100+)
- ✅ error_recovery.json: 4 cases (target: 100+)
- ✅ repl_exploration.json: 4 cases (target: 50+)
- ⏳ complex_workflows.json: 0 cases (target: 100+)
- ⏳ edge_cases.json: 0 cases (target: 50+)
- ⏳ tool_selection.json: 0 cases (target: 50+)
- ⏳ working_context.json: 0 cases (target: 50+)

**Total**: 13 / 500+ cases

Test case collection is an ongoing process that accelerates as the framework is used and models are onboarded.
