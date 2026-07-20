# Fix GitLab Issue

Fetch a GitLab issue, create a branch, analyze, implement, and create a merge request.

## Arguments

GitLab issue number: `$ARGUMENTS` (e.g., `123` for issue #123)

## Workflow

### 1. Fetch Issue

```bash
glab issue view $ARGUMENTS
```

Output goes to stdout for immediate parsing. For very long issues, redirect to file: `glab issue view $ARGUMENTS > /tmp/issue-$ARGUMENTS.txt`

Extract: title, description, labels, and any linked MRs.

### 2. Prepare Repository

Ensure clean state on main:
```bash
git checkout main && git pull origin main
git status --porcelain  # Must be empty
```

If uncommitted changes exist, ask user to commit or stash first.

### 3. Create Branch

Determine branch type from issue labels/title:
- Bug/fix/error keywords → `fix/`
- Otherwise → `feat/`

Create branch: `{type}/{issue-number}-{sanitized-title}`
- Example: `fix/123-login-error` or `feat/456-user-profile`

```bash
git checkout -b "$BRANCH_NAME"
```

### 4. Analyze and Plan

1. **Investigate** the codebase using semantic search
2. **Identify** affected files and root cause
3. **Create plan** - For complex issues (3+ files or architectural changes), save to `docs/scratch/issue-{N}-plan.md`

Present summary to user:
```
## Issue #N: [Title]

**Proposed Solution**: [2-3 sentences]

**Files to Modify**:
- file.py (reason)

**Complexity**: Low/Medium/High

Approve? (yes/no)
```

Wait for user approval before proceeding.

### 5. Implement

1. Follow the plan, implementing changes incrementally
2. Verify changes address the issue
3. Check for linting errors: `ruff check .`

### 6. Test

Run relevant unit tests before committing:

```bash
pytest tests/ -v --tb=short  # Or scope to affected tests
```

If tests fail, fix issues and re-run. Don't proceed until tests pass.

### 7. Commit

Single commit with conventional format:

```bash
git add .
git commit -m "{type}: {title} (#{issue})"
```

Example: `fix: resolve login validation error (#123)`

### 8. Push and Create MR

```bash
git push -u origin "$BRANCH_NAME"
glab mr create --title "{type}: {title} (#{issue})" --description "Fixes #{issue}

{brief summary}"
```

### 9. Summary

Report completion:
```
## Complete

✅ Issue: #{N} - [Title]
✅ Branch: {branch}
✅ MR: !{mr_number}

**Changes**: [summary]

View MR: `glab mr view {mr_number}`
```

## Error Handling

| Error | Resolution |
|-------|------------|
| `glab` not authenticated | Run `glab auth login` |
| Issue not found | Verify issue number and project access |
| Branch exists | Ask user: delete and recreate, or use existing? |
| Uncommitted changes | Ask user to commit/stash first |
| Plan rejected | Get feedback, update plan, re-present |
| Tests fail | Fix issues, re-run tests, don't proceed until green |
| MR creation fails | Check if MR exists, verify push succeeded |

## Notes

- **Single commit**: Squash all changes into one commit
- **Conventional commits**: `fix:` or `feat:` prefix with issue reference
- **User approval**: Always wait for approval before implementing
- **Plan documents**: Only create for complex issues (optional for simple fixes)
