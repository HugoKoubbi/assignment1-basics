# Contributing

This repository follows a simple branch-and-pull-request workflow. The `spring-2025` branch is the stable base for the assignment, so new code should be developed on a separate branch and reviewed before it is merged.

## Start a change

Open the repository in a terminal and update the base branch before beginning:

```bash
git switch spring-2025
git pull --ff-only
git status
```

`git switch` selects the correct base branch. `git pull --ff-only` downloads new commits without creating an unexpected merge commit. `git status` confirms that the working directory is clean.

Create a branch whose name describes the work:

```bash
git switch -c feature/descriptive-name
```

One branch should contain one exercise, feature, or coherent correction. This keeps the history and the pull request easy to understand.

## Develop and test

While editing the code in VS Code, inspect the work regularly:

```bash
git status
git diff
```

`git status` shows which files changed. `git diff` shows the exact unstaged changes. Run the relevant tests before committing; for example:

```bash
uv run pytest -q
uv run ruff check .
```

The first command runs the automated tests. The second checks the code for common style and quality problems. A virtual environment created by `uv` keeps the project's Python packages separate from the rest of the computer.

## Commit and push

Stage only the files that belong to the change, review them, and create a clear commit:

```bash
git add <changed-files>
git diff --staged
git commit -m "Describe the change clearly"
git push -u origin feature/descriptive-name
```


git add cs336_basics/ablation.py
git add cs336_basics/model.py
git add cs336_basics/train.py
gid add cs336_basics/inference.py
git diff --staged
git commit -m "put ablation studies"
git push -u origin feature/model-components-v2


`git add` chooses what will enter the commit. `git diff --staged` provides a final review. `git commit` records a meaningful checkpoint. The first `git push -u` publishes the branch and connects it to GitHub; later pushes on the same branch need only `git push`.

## Open and merge the pull request

On GitHub, open a pull request with `spring-2025` as the base branch and the feature branch as the compare branch. Explain what changed, why it changed, and how it was tested. If corrections are needed, make new commits locally and push them; the pull request updates automatically. Merge only when the automated checks pass and the changes have been reviewed.

After the merge, update the local base branch and remove the old local branch:

```bash
git switch spring-2025
git pull --ff-only
git branch -d feature/descriptive-name
git fetch --prune
```

Before every commit, read `git status` and `git diff`. Do not commit secrets, virtual environments, large generated files, datasets, or model checkpoints unless the project explicitly requires them.
