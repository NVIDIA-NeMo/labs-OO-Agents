# Framework Capability Tests

This directory contains tests for evaluating model performance on framework-level capabilities. These tests are used for model onboarding optimization.

## Test Files

### Core Capabilities

- **test_code_generation.py** - Tests valid code generation following planning language rules
- **test_message_streams.py** - Tests message and reasoning stream generation with variable expansion
- **test_repl_behavior.py** - Tests appropriate REPL usage for exploration
- **test_validation_retry.py** - Tests learning from validation errors
- **test_tool_calls.py** - Tests runtime tool selection (code_generation, repl, abort)
- **test_working_context.py** - Tests persistent working context usage
- **test_replanning.py** - Tests replanning behavior (when implemented)

### Test Data

- **test_data/** - JSON files with diverse test cases for training and evaluation

## Metrics

Each test file tracks specific metrics:

| Test File | Primary Metrics | Target |
|-----------|-----------------|--------|
| test_code_generation.py | AST validation pass rate | >90% |
| test_message_streams.py | Message presence, expansion validity | >90%, >85% |
| test_repl_behavior.py | REPL usage appropriateness | >75% |
| test_validation_retry.py | Retry success rate | >80% first, >95% cumulative |
| test_tool_calls.py | Tool selection accuracy | >90% |
| test_working_context.py | Context usage patterns | >60% multi-step |
| test_replanning.py | Replan appropriateness | >70% |

## Running Tests

```bash
# Run all onboarding tests
pytest tests/onboarding/

# Run specific capability test
pytest tests/onboarding/test_code_generation.py

# Run with coverage
pytest tests/onboarding/ --cov=nooa

# Generate capability scorecard
python -m nooa.onboarding.evaluator --output report.html
```

## Test Data Format

Test cases in `test_data/*.json` follow this format:

```json
{
  "test_cases": [
    {
      "id": "unique_id",
      "capability": "code_generation",
      "task": "Task description for model",
      "expected_behavior": {
        "contains": ["pattern1", "pattern2"],
        "not_contains": ["anti_pattern"],
        "validation_passes": true,
        "tool_used": "code_generation"
      },
      "difficulty": "easy|medium|hard",
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

## Adding New Tests

1. Create test file in `tests/onboarding/test_new_capability.py`
2. Define test class with metrics
3. Add test cases to `test_data/new_capability.json`
4. Document metrics in this README
5. Update evaluator to include new capability

## Integration with Optimization

These tests are used by the onboarding optimizer:

1. **Baseline Evaluation**: Run tests with seed prompts
2. **DSPy Optimization**: Use test results to optimize prompts
3. **Validation**: Verify improvement on held-out test set
4. **Deployment**: Deploy optimized prompts for model

See `docs v2/19-model-onboarding.md` for complete onboarding workflow.
