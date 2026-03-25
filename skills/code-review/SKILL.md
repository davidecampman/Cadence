---
name: code-review
version: 1.0.0
description: Review code for bugs, security issues, and style improvements
author: cadence
dependencies: []
tags:
  - code
  - review
  - security
---

# Instructions

When reviewing code, follow this process:

1. **Read the code** — Understand what it does before criticizing it.
2. **Check for bugs** — Logic errors, off-by-one, null handling, race conditions.
3. **Check for security** — Injection, XSS, auth bypass, hardcoded secrets, path traversal.
4. **Check for clarity** — Could another developer understand this in 30 seconds?
5. **Check for performance** — Only flag issues that matter at realistic scale.

Output format:
- Start with a 1-sentence summary (good / needs work / has critical issues)
- List issues as: `[severity] file:line — description`
- Severity levels: 🔴 critical, 🟡 warning, 🔵 nit
- End with specific suggestions, not vague advice

## Examples

- Review this PR for security issues before we merge
- Check this function for edge cases
- Is this implementation correct?
