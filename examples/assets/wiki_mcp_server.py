# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny MCP server for the quickstart 11 example.

Exposes a single ``search_wiki`` tool that returns canned knowledge-base
articles.  Runs over **stdio** so no network setup is needed — the agent
launches it as a subprocess automatically.

Usage (standalone test):
    uv run python examples/assets/wiki_mcp_server.py
"""

from mcp.server import FastMCP

mcp = FastMCP("wiki")

# -- Canned knowledge base ---------------------------------------------------

_ARTICLES = {
    "deployment": {
        "title": "Deployment Best Practices",
        "content": (
            "1. Use blue-green deployments for zero-downtime releases.\n"
            "2. Always run smoke tests after deployment.\n"
            "3. Keep rollback procedures documented and tested.\n"
            "4. Use infrastructure-as-code (Terraform, Pulumi) for reproducibility.\n"
            "5. Monitor error rates for 15 minutes after every deploy."
        ),
    },
    "testing": {
        "title": "Testing Guidelines",
        "content": (
            "1. Write unit tests for business logic, integration tests for APIs.\n"
            "2. Aim for 80% code coverage on critical paths.\n"
            "3. Use property-based testing for data transformations.\n"
            "4. Run tests in CI on every push — never merge red.\n"
            "5. Prefer real databases in integration tests over mocks."
        ),
    },
    "code review": {
        "title": "Code Review Checklist",
        "content": (
            "1. Check for security issues (injection, auth bypass, secrets in code).\n"
            "2. Verify error handling covers edge cases.\n"
            "3. Look for performance regressions (N+1 queries, missing indexes).\n"
            "4. Ensure new code has adequate test coverage.\n"
            "5. Confirm documentation is updated for public API changes."
        ),
    },
    "observability": {
        "title": "Observability & Monitoring",
        "content": (
            "1. Instrument all services with structured logging (JSON).\n"
            "2. Use distributed tracing (OpenTelemetry) for cross-service requests.\n"
            "3. Set up alerts on error rate, latency p99, and saturation.\n"
            "4. Create dashboards for each team's top-level SLOs.\n"
            "5. Regularly review and prune noisy alerts."
        ),
    },
}


@mcp.tool()
def search_wiki(query: str) -> str:
    """Search the internal wiki for articles matching the query.

    Args:
        query: Search terms (e.g. "deployment", "testing best practices")

    Returns:
        Matching article text, or a not-found message.
    """
    if not query or not query.strip():
        return f"Please provide a search query. Available topics: {', '.join(_ARTICLES.keys())}."
    query_lower = query.strip().lower()
    matches = []
    for key, article in _ARTICLES.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            matches.append(f"## {article['title']}\n{article['content']}")

    if matches:
        return "\n\n".join(matches)
    return f"No articles found for '{query}'. Available topics: {', '.join(_ARTICLES.keys())}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
