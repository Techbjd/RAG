---
name: Git Commit Assistant
description: Review git changes, flag what the user missed, propose atomic Conventional Commits, and create the commits after approval. Use when the user asks to commit, asks how to commit, or types @git-commit.
---

# Git Commit Assistant

Make clean, reviewable commits in the **Conventional Commits** format and catch
mistakes before they enter history.

## Format

```text
<type>[optional scope][!]: <imperative, lowercase subject>

[optional body: why the change was made]
[optional BREAKING CHANGE: ... footer]
```

| Type | Use for |
| --- | --- |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `chore` | Maintenance/tooling, no product change |
| `refactor` | Restructure without behavior change |
| `test` | Tests only |
| `perf` | Performance improvement |
| `build` / `ci` | Build system or CI |

- `(scope)` names the affected area, e.g. `feat(db)`, `fix(rerank)`.
- `!` marks a **breaking change** and pairs with a `BREAKING CHANGE:` footer.
- Avoid `wip:` for finished work; use it only for an explicit interim save.
- Keep the subject under ~72 characters. The body explains *why*, not *what*.

## Workflow

1. **Inspect**: run `git status --short`, `git diff`, `git diff --cached`, and
   `git log -8 --oneline` to see the changes and match the repo's existing style.
2. **Catch what the user missed**:
   - untracked new files that belong with the change (modules, config, tests);
   - junk, secrets, `.env`, or generated artifacts that must stay out;
   - deleted files still referenced in code, config, or docs;
   - tests/lint not run — run the project's test command before committing;
   - debug leftovers, `TODO`/`wip` markers, inconsistent naming or formatting;
   - README/config drift when behavior changed.
3. **Group by logical change, not by file**. Each commit must stand on its own:
   the project should still build and pass tests after every commit. Files that
   only make sense together stay in the same commit. Do not make one commit per
   file, and do not mix unrelated `feat`/`fix`/`docs`/`chore` changes.
4. **Propose a commit plan before committing**: list each commit's message and
   the files it will contain, plus anything intentionally excluded. Explain
   any suggestion that changes what the user asked for.
5. **Confirm**, then commit: stage only the planned paths with `git add <paths>`
   (never blind `git add -A`), check `git diff --cached --check`, run tests, and
   commit. Repeat for each group in an order where each commit works.
6. **Report** the resulting hashes and subjects, and note what was left out.

## Rules

- Never commit secrets, credentials, `.env`, or generated files.
- Do not push unless the user explicitly asks.
- Do not rewrite history (amend/rebase/reset) unless the user explicitly asks.
- If there is nothing to commit, say so instead of creating an empty commit.
- When the user says "just commit", apply the best plan without asking, but
  still report what was grouped and excluded.

## Invocation

The user can type `@git-commit` with a request, for example:

```text
@git-commit commit my changes the standard way
```

The skill is also advertised automatically when a commit-related task appears,
so the agent can suggest it before the user asks.
