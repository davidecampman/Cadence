---
name: technical-writing
version: 1.0.0
description: Write clear technical documentation, READMEs, API docs, and specifications
author: cadence
dependencies: []
tags:
  - writing
  - documentation
  - readme
  - specs
---

# Instructions

When writing technical documentation, follow this process:

1. **Gather context first** — Read the relevant code, configuration files, and any existing docs before writing a single word. Never document what you haven't read.
2. **Identify the audience** — Beginner users need step-by-step instructions. Experienced developers need reference material. Match the depth and assumed knowledge to the audience.
3. **Structure before you write** — Outline headings and sections first. A good doc structure: Overview → Prerequisites → Quick Start → Reference → Troubleshooting.
4. **Be concrete** — Every concept should have at least one working example. Prefer showing over explaining.
5. **Test your own instructions** — If you write setup steps, mentally walk through them. Flag any step that requires knowledge not provided in the doc.

Writing standards:
- **Active voice**: "Run `npm install`", not "Dependencies should be installed by running..."
- **Present tense**: "The function returns a string", not "The function will return a string"
- **One idea per sentence** — Split compound sentences
- **Code blocks for all code** — Use language hints (` ```python `, ` ```bash `, etc.)
- **No filler phrases** — Remove "simply", "just", "easy", "obviously", "note that"

Document types and their key sections:
- **README**: badges, one-liner description, quick start, full install, usage, contributing
- **API reference**: endpoint, method, auth, parameters (name/type/required/description), request example, response example, error codes
- **Architecture doc**: problem statement, design decisions with rationale, component diagram, data flow, tradeoffs considered
- **Runbook**: trigger condition, impact, diagnosis steps, resolution steps, escalation path

## Examples

- Write a README for this Python package
- Document the REST API endpoints in this FastAPI app
- Create an architecture decision record for this database schema change
- Write a user guide for this CLI tool
