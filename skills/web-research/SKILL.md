---
name: web-research
version: 1.0.0
description: Conduct structured web research with source tracking and synthesis
author: cadence
dependencies: []
tags:
  - research
  - web
  - synthesis
---

# Instructions

When conducting web research, follow this process:

1. **Decompose the question** — Break the research goal into 3-5 focused sub-queries before fetching anything.
2. **Search broadly, then deep** — Start with broad queries to identify the best sources, then fetch primary sources directly.
3. **Track every source** — Record the URL, title, and a 1-sentence summary for each page you read.
4. **Cross-reference** — If two sources conflict, note it explicitly. Prefer primary sources (official docs, research papers, government data) over aggregators.
5. **Synthesize, don't paste** — Summarize findings in your own words. Never copy large blocks of text verbatim.

Output format:
- Start with a **TL;DR** (2-3 sentences summarizing the answer)
- Follow with a **Findings** section with organized bullet points or sections
- End with a **Sources** section listing every URL consulted with a 1-sentence description

Quality rules:
- Minimum 3 distinct sources for any factual claim
- Flag anything you could not verify with `[UNVERIFIED]`
- Note the publication date of key sources — prefer sources from the last 2 years unless the topic is stable

## Examples

- Research the current state of LLM context window sizes across major providers
- What are the best practices for database indexing in PostgreSQL?
- Summarize recent developments in quantum computing (last 6 months)
- Find competing approaches to rate limiting in distributed systems
