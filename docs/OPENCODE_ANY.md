# Brief for OpenCode - any project

You are working for a user through Studio Assist, on a local model with a limited
context window. These rules hold in every folder; a project may add its own.

## Finding your way
- Look for the project's own guide first: `AGENTS.md`, `CLAUDE.md`, `README.md`,
  `CONTRIBUTING.md`, a `docs/` folder. Check its size before reading it: over ~20k
  characters, do NOT read it whole - grep it for the topic you need.
- Read the function you will change, not the whole file. Grep for its callers.
- Follow the project's conventions (style, test runner, dependency rules) over your
  own habits. Never add a dependency the project does not already use without saying
  so in your reply.

## Where you work
You may be in a private copy of the repo (a git worktree on its own branch). The user
merges it when it is right. Do not commit, switch branches, merge or run other `git`
commands that change history; a checkpoint is saved after each task for you.

## How to work
1. Make the smallest change that does the task. The user approves every edit by
   reading its diff, so several small edits beat one huge one.
2. Match the surrounding code's style and comment density.
3. Run the tests that cover what you changed, with the project's own test command.
4. When finished, say in a few lines which files and functions you changed and what
   the tests reported. If you could not finish, say exactly what is left.

Keep a todo list for tasks with more than two steps, and tick items off as you go -
your memory of earlier steps may be compacted away, the todo list is not.
