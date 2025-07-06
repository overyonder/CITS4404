---
trigger: always_on
---

# Cascade's Core Principles & Project Plan

This document outlines the core principles for our development workflow and the master plan for the Neural Pong project. These guidelines are based on lessons learned from our recent collaboration and are designed to ensure accuracy, efficiency, and alignment with your goals.

---

## Part 1: Core Principles for Effective and Error-Free Development

### Tool Usage Best Practices

0. Command running (e.g. Cargo check), often breaks, so save this until you've almost completed everything else.

1. REMEMBER: THE SHELL ENVIRONMENT IS GIT BASH ON WINDOWS, SO USE PATHS THAT SUIT (e.g. `/c/Users/user/_dev/CITS4404/rust/subfolder/_____.rs`). If commands fail, consider whether you might need to use a different format, or wait until the shell loads before issuing the command. The shell output should be readable, if not, alert the user.

2. **Principle: Always Read Before Writing.**

   - **Action:** Before any file modification (`replace_file_content`), I will read the _entire_ file content or the specific, relevant function using `view_line_range` or `view_code_item`, noting that by default it only reads the first 400 lines unless I specify otherwise.
   - **Reasoning:** Relying on outlines or partial reads has repeatedly caused my edits to fail because the `TargetContent` was not exact. This led to mangled code and a cascade of syntax errors, as we saw in [tui/ui.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/tui/ui.rs:0:0-0:0). A full read ensures my understanding of the code is complete and my edits are precise.

3. **Principle: Verify Context Before Changing APIs or Logic.**

   - **Action:** Before altering any public-facing code (like a struct name or function signature), I will use `grep_search` or `codebase_search` to find all its usages across the entire project.
   - **Reasoning:** Making changes in isolation creates compilation errors in other files. A codebase-wide search reveals the full impact of a change, allowing me to perform a comprehensive, multi-file fix in a single, logical step, preventing unresolved import errors and other downstream issues.

4. **Principle: Trust, but Verify, Linter Feedback.**
   - **Action:** I will treat the latest list of lint/compiler errors you provide as the absolute source of truth for the current state of the codebase.
   - **Reasoning:** My internal state can become stale. Ignoring fresh linter feedback led to a frustrating loop of fixing problems that had already been fixed or were no longer relevant. The compiler does not lie.

### Preferred Workflow Methodology

1.  **Step 1: Plan and Prioritize.**

    - **Action:** I will begin each major task by updating the master plan. I will break down your requests into a clear, ordered checklist.
    - **Reasoning:** A clear plan prevents reactive, ad-hoc changes. It ensures we tackle root causes first (e.g., fixing [config.rs](cci:7://file:///c:/Users/user/_dev/CITS4404/rust/src/config.rs:0:0-0:0) before the TUI files that depend on it) and provides a shared understanding of the path forward.

2.  **Step 2: Investigate Systematically.**

    - **Action:** I will execute the principles of "Read Before Writing" and "Verify Context." I will gather all necessary information _before_ formulating a solution. I will not guess or assume.
    - **Reasoning:** Hasty implementation without full context was the primary driver of our recent failures. A dedicated investigation phase prevents wasted effort and incorrect edits.

3.  **Step 3: Execute with Precision.**

    - **Action:** I will apply changes in a single, logical, and comprehensive tool call whenever possible. For multi-file changes, I will group them into one response.
    - **Reasoning:** Atomic changes are easier to track and debug. This avoids leaving the codebase in a broken intermediate state.

4.  **Step 4: Verify and Iterate.**
    - **Action:** After applying a change, I will immediately request a build or check command (`cargo check`) to confirm the fix and get immediate feedback. I will use the new list of errors to guide my next step.
    - **Reasoning:** This creates a tight feedback loop, ensuring each step successfully builds upon the last and preventing the accumulation of errors.

REMEMBER: Do not make code changes until all files have been reviewed, and do not run intermittent console commands (like Cargo check) to avoid hangs.
