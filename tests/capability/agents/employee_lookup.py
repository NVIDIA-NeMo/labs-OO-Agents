# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Employee lookup agent for multi-step subagent chaining testing.

This agent tests whether the LLM can correctly chain two subagents
with a data dependency between them:
1. Using EmployeeDirectory to search by name → returns full record with employee_id
2. Extracting employee_id from the result
3. Using PayrollSystem to get salary by employee_id (NOT by name)
"""

from typing import Annotated, TypedDict

from nooa import Agent

# ============================================================================
# Typed return structures
# ============================================================================


class EmployeeRecord(TypedDict):
    """Full employee record from directory lookup."""

    employee_id: str
    name: str
    department: str
    title: str


class SalaryInfo(TypedDict):
    """Salary information from payroll system."""

    employee_id: str
    base_salary: int
    bonus: int
    total: int


class LookupResult(TypedDict):
    """Result from the employee salary lookup."""

    employee_name: str
    employee_id: str
    salary: int


# ============================================================================
# Mock data for deterministic testing
# ============================================================================

EMPLOYEE_DATABASE = {
    "john smith": EmployeeRecord(
        employee_id="E1001",
        name="John Smith",
        department="Engineering",
        title="Senior Developer",
    ),
    "jane doe": EmployeeRecord(
        employee_id="E1002",
        name="Jane Doe",
        department="Marketing",
        title="Marketing Manager",
    ),
    "bob wilson": EmployeeRecord(
        employee_id="E1003",
        name="Bob Wilson",
        department="Sales",
        title="Sales Representative",
    ),
}

SALARY_DATABASE = {
    "E1001": SalaryInfo(employee_id="E1001", base_salary=120000, bonus=15000, total=135000),
    "E1002": SalaryInfo(employee_id="E1002", base_salary=95000, bonus=20000, total=115000),
    "E1003": SalaryInfo(employee_id="E1003", base_salary=75000, bonus=25000, total=100000),
}


# ============================================================================
# Subagent 1: Employee Directory - search by name
# ============================================================================


class EmployeeDirectory(Agent):
    """Directory service for looking up employee information by name."""

    async def search_by_name(
        self, name: Annotated[str, "The name of the employee (e.g., 'John Smith')"]
    ) -> EmployeeRecord | None:
        """Search for an employee by name (case-insensitive)."""
        return EMPLOYEE_DATABASE.get(name.lower())


# ============================================================================
# Subagent 2: Payroll System - lookup by employee_id ONLY
# ============================================================================


class PayrollSystem(Agent):
    """Payroll system for salary information."""

    async def get_salary(
        self, employee_id: Annotated[str, "The employee ID (e.g., 'E1001')"]
    ) -> SalaryInfo | None:
        """Get salary information for an employee by their ID."""
        return SALARY_DATABASE.get(employee_id)


# ============================================================================
# Main agent - must chain subagents with data dependency
# ============================================================================


class EmployeeSalaryAgent(Agent):
    """Agent that looks up employee salaries by name."""

    # Subagent classes - agent must chain these correctly
    EmployeeDirectory = EmployeeDirectory
    PayrollSystem = PayrollSystem

    async def get_employee_salary(
        self, employee_name: Annotated[str, "The name of the employee (e.g., 'John Smith')"]
    ) -> LookupResult:
        """Look up the salary for an employee by their name."""
        ...
