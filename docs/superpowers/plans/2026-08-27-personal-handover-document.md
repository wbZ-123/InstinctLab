# Personal Foothold Project Handover Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a self-contained Chinese personal handover manual in Markdown and DOCX, with polished architecture/state-flow diagrams and a copy-ready AI onboarding prompt.

**Architecture:** Treat the current Git commit and source code as the source of truth, reconcile them with existing project documents and latest training artifacts, then author one canonical Markdown document. Generate three diagram assets from version-controlled Graphviz sources and convert the canonical Markdown into DOCX so both formats carry the same factual content.

**Tech Stack:** Markdown, Graphviz DOT/PNG, Pandoc or python-docx, Git, pytest/log inspection.

## Global Constraints

- Write in Chinese for the project owner’s personal use.
- Preserve real repository links, local paths, run names, dependency commits, and operational commands.
- Distinguish implemented code, experimentally validated behavior, unfinished work, and proposed-but-not-implemented designs.
- Use commit `aab3f77` and the current working tree as the primary source of truth.
- Do not claim that training quality, stair behavior, Recovery, or hardware deployment is complete without evidence.

---

### Task 1: Establish the factual baseline

**Files:**
- Read: `PROJECT_CONTEXT.md`
- Read: `docs/foothold_project_status.md`
- Read: current planner, reward, learning, state-machine, training, and play sources
- Inspect: latest training/checkpoint/log state

- [ ] Record the current branch, commit, dependency commits, test status, latest run, key parameters, implemented capabilities, and unresolved risks.
- [ ] Reconcile stale statements in older docs against current source.

### Task 2: Create visual assets

**Files:**
- Create: `docs/assets/foothold_handover_architecture.dot`
- Create: `docs/assets/foothold_handover_architecture.png`
- Create: `docs/assets/foothold_handover_planning_flow.dot`
- Create: `docs/assets/foothold_handover_planning_flow.png`
- Create: `docs/assets/foothold_handover_state_machine.dot`
- Create: `docs/assets/foothold_handover_state_machine.png`

- [ ] Draw a system architecture diagram covering perception, high-level planner, geometry/safety, swing reference, motor PPO, and simulation feedback.
- [ ] Draw one complete HOLD planning transaction including coordinate frames, terrain Z query, reward, preflight, locking, and execution.
- [ ] Draw the contact-aware gait state machine and Recovery paths without implying unimplemented behavior.
- [ ] Render readable high-resolution PNG files and inspect their dimensions/content.

### Task 3: Author the canonical Markdown handover

**Files:**
- Create: `docs/FOOTHOLD_PROJECT_HANDOVER_PERSONAL.md`

- [ ] Write the executive summary and original-Hiking comparison table.
- [ ] Separate completed work, validated evidence, known problems, unfinished work, and abandoned/proposed designs.
- [ ] Map the major source files by responsibility and data flow.
- [ ] Embed all diagrams and explain the planner/PPO/motor-policy closed loop.
- [ ] Include exact training, resume, play, TensorBoard, test, Git, and clean-machine reproduction commands.
- [ ] Include a prioritized successor roadmap and a full copy-ready AI onboarding prompt.

### Task 4: Generate the DOCX

**Files:**
- Create: `docs/FOOTHOLD_PROJECT_HANDOVER_PERSONAL.docx`

- [ ] Convert the canonical Markdown while embedding all three PNG diagrams.
- [ ] Apply readable Chinese headings, table styling, page margins, code formatting, and image sizing.

### Task 5: Verify the deliverables

**Files:**
- Verify: `docs/FOOTHOLD_PROJECT_HANDOVER_PERSONAL.md`
- Verify: `docs/FOOTHOLD_PROJECT_HANDOVER_PERSONAL.docx`

- [ ] Scan the Markdown for placeholders, contradictions, broken local links, and missing required sections.
- [ ] Parse the DOCX to verify headings, tables, embedded images, and key phrases.
- [ ] Run `git diff --check` and report final file paths and verification evidence.
