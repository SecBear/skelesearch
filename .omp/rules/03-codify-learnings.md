---
description: "How to handle recurring failures and build a project stdlib"
alwaysApply: true
---
# Codifying Learnings (Build Your Stdlib)

When a failure mode repeats or you need to steer future implementation:

1. **Fix the immediate issue.**
2. **Encode the lesson so it does not recur:**
   - If it's a behavior requirement: update/add a spec in `specs/` and link it from `SPECS.md`.
   - If it's a process constraint: update/add a rule in `rules/`.
3. **Check the Stdlib:** Before adding new dependencies or complex workarounds, verify if the functionality exists in the project's stdlib or current dependencies. Do not invent parallel conventions.

Goal: make the correct outcome the path of least resistance.
