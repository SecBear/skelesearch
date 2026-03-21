# Context And Scope

## One Task Per Context Window
Treat a context window as a fixed-size allocation. Once mixed, it cannot be cleanly "freed" without starting a new session.
1. One task per session.
2. Keep the "load stack" stable: re-load AGENTS.md, the relevant specs, and the relevant rules.

## What Counts As A Task
A task is any unit of work with a coherent success condition.
Not tasks: "Make the project production-ready"
Tasks: "Implement feature X with tests"
