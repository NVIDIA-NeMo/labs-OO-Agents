# Address GitLab MR Comments

Help the user address all reviewer comments on a GitLab Merge Request by reading the feedback and fixing the code.

## Current Context

- Current branch: !`git branch --show-current 2>/dev/null`
- Git status: !`git status --short 2>/dev/null | head -5`

## Arguments

MR number (optional): `$ARGUMENTS`

If no MR number provided, find the MR for the current branch.

## Instructions

### Step 1: Get MR Information

```bash
# If MR number provided:
glab mr view <mr-number>

# If no MR number, find MR for current branch:
glab mr list --source-branch=$(git branch --show-current)
```

### Step 2: Fetch MR with All Comments

```bash
# Get the project id
glab repo view -F json | jq '.id'

# Use the project id and the mr number to get the total number of pages
glab api --include "projects/<project-id>/merge_requests/<mr-number>/notes" | grep X-Total-Pages

# Get MR details with all comments and discussions (iterate through all the pages from 1 to "X-Total-Pages")
glab mr view <mr-number> --comments --page <page-number>

# Get full MR diff for context
glab mr diff <mr-number>
```

### Step 3: Analyze and Categorize Comments

For each comment/discussion:
1. Identify if it's actionable (code change needed) or just discussion
2. Note the file and line number if applicable
3. Understand what change is being requested

Present a summary table:
| # | File | Line | Reviewer | Comment Summary | Action Needed |
|---|------|------|----------|-----------------|---------------|

### Step 4: Address Each Comment

For each actionable comment:
1. Read the relevant file(s)
2. Understand the context and the requested change
3. Make the fix using the Edit tool
4. Explain what was changed and why

### Step 5: Commit and Rebase

After addressing all comments:
1. Create a commit with the fixes
2. **Always rebase on main before pushing**:
   ```bash
   git fetch origin main
   git rebase origin/main
   ```
3. Force push (with lease) if needed after rebase

### Step 6: Check CI Status

After pushing, check if CI is passing:
```bash
glab ci status
```

If CI is failing, investigate and fix any issues.

### Step 7: Summary

Provide:
1. Summary of all changes made
2. Commit hash and message
3. CI status
4. Remind user to mark discussions as resolved in GitLab

## Output Format

```
## MR #<number>: <title>

### Comments to Address

1. **[file.py:42]** @reviewer: "Comment text..."
   - **Action**: Description of what needs to be done
   - **Status**: Fixed / Needs clarification / Won't fix

### Changes Made

- `file.py`: Description of change
- `other.py`: Description of change

### Suggested Commit Message

fix: address MR review comments

- Fixed X in file.py (per @reviewer)
- Updated Y in other.py (per @reviewer2)

### Git Status

- Rebased on: main (commit <hash>)
- Pushed: Yes/No
- CI Status: Passing/Failing/Pending

### Next Steps

1. Mark discussions as resolved in GitLab
2. Request re-review if needed
```

## Error Handling

- If `glab` not authenticated: suggest `glab auth login`
- If no MR found for branch: ask user for MR number
- If comment is unclear: ask for clarification before making changes
