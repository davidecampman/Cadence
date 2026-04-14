---
name: security-audit
version: 1.0.0
description: Deep security review covering OWASP Top 10, secrets, dependencies, and auth flows
author: cadence
dependencies: []
tags:
  - security
  - audit
  - owasp
  - vulnerabilities
---

# Instructions

When conducting a security audit, follow this process:

1. **Map the attack surface first** — Identify all entry points: HTTP endpoints, CLI args, file inputs, env vars, IPC. List them before looking for vulnerabilities.
2. **Work top-down** — Start with critical issues (auth bypass, RCE, data exposure) before style/low-severity findings.
3. **Show exploitability** — For each finding, briefly describe how an attacker would exploit it, not just that the issue exists.
4. **Check for secrets** — Scan for hardcoded credentials, API keys, private keys, and tokens. Flag any string matching `[A-Za-z0-9+/]{32,}` in non-test code.
5. **Never suggest security theater** — Only recommend controls that actually reduce risk. Don't suggest adding rate limiting to an internal-only endpoint.

Checklist — check each category:

**Injection**
- SQL injection (string concatenation in queries, missing parameterization)
- Command injection (shell=True with user input, os.system with interpolation)
- Path traversal (user-controlled file paths without normalization)
- Template injection (user input rendered in Jinja2/Mako/etc.)

**Authentication & Authorization**
- Missing authentication on sensitive endpoints
- Broken authorization (user A can access user B's resources)
- Weak session tokens (short, predictable, not rotated)
- JWT: algorithm confusion (alg:none), weak secrets, missing expiry check

**Data Exposure**
- PII in logs or error messages
- Sensitive data in URLs (tokens, passwords in query params)
- Overly verbose error messages revealing internals
- Unencrypted sensitive data at rest

**Dependencies**
- Pinned vs. floating dependency versions
- Known CVEs in direct dependencies (check against NIST NVD or OSV)
- Transitive dependency risk

**Configuration**
- Debug mode enabled in production paths
- CORS wildcard (`*`) on credentialed endpoints
- Missing security headers (CSP, HSTS, X-Frame-Options)
- World-readable secrets files or environment variables

Output format:
- Start with a **Risk Summary**: one line per severity level (Critical/High/Medium/Low/Info) with counts
- List each finding as: `[SEVERITY] Category — Description`
  - Severity levels: 🔴 Critical, 🟠 High, 🟡 Medium, 🔵 Low, ⚪ Info
  - Include: location (file:line), impact, reproduction steps, remediation
- End with **Remediation Priority**: ordered list of fixes by risk reduction / effort ratio

## Examples

- Audit this Flask API for OWASP Top 10 vulnerabilities
- Review the authentication flow in this Node.js app
- Check these dependencies for known CVEs
- Find any hardcoded secrets or credentials in this codebase
