# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Demo showing referenced types in action."""

from typing import Annotated

from pydantic import BaseModel, Field

from nooa.agentdoc import doc


class QueryRequest(BaseModel):
    """A database query request."""

    sql: Annotated[str, Field(description="SQL query to execute")]
    params: Annotated[dict[str, str], Field(default_factory=dict, description="Query parameters")]
    limit: Annotated[int, Field(default=100, ge=1, le=1000, description="Maximum rows to return")]


class QueryResult(BaseModel):
    """Result of a database query."""

    rows: Annotated[list[dict], Field(description="Query result rows")]
    row_count: Annotated[int, Field(description="Number of rows returned")]
    execution_time: Annotated[float, Field(description="Query execution time in seconds")]


class CacheEntry(BaseModel):
    """A cached value with metadata."""

    key: str
    value: str
    ttl_seconds: int = 3600


class DatabaseTool:
    """Database operations namespace.

    Provides query, insert, and transaction operations.
    Maintains connection pool and query statistics.
    """

    connection_string: str
    query_count: int = 0
    last_query: str | None = None

    def query(self, request: QueryRequest) -> QueryResult:
        """Execute a SQL query and return results.

        Uses parameterized queries to prevent SQL injection.
        Updates query_count and last_query state after execution.
        """
        return QueryResult(rows=[], row_count=0, execution_time=0)

    def insert(
        self,
        table: Annotated[str, "Target table name"],
        data: Annotated[dict, "Row data to insert"],
    ) -> Annotated[int, "ID of inserted row"]:
        """Insert a row into a table.

        Automatically escapes values and handles type conversion.
        Raises ValueError if table doesn't exist.
        """
        return 0

    def get_stats(self) -> dict[str, int]:
        """Get query statistics."""
        return {"query_count": self.query_count}


class CacheTool:
    """In-memory cache namespace."""

    entries: dict[str, CacheEntry] = {}
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> str | None:
        """Retrieve a value from cache."""
        pass

    def set(self, key: str, value: str, entry: CacheEntry) -> None:
        """Store a value in cache."""
        pass


if __name__ == "__main__":
    print("=" * 80)
    print("DatabaseTool with Referenced Types")
    print("=" * 80)
    print(doc(DatabaseTool))
    print()
    print()
    print("=" * 80)
    print("CacheTool with Referenced Types")
    print("=" * 80)
    print(doc(CacheTool))
