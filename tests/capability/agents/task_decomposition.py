# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task Decomposition Ability Test Agent. Helper Method Creation & Reuse."""

import re  # noqa: F401 — for LLM exec_globals
from datetime import date, datetime, time, timedelta, timezone, tzinfo  # noqa: F401

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CodeActConfig
from nooa.tools import MethodWriting


class TaskDecompositionTestAgent(Agent):
    """You are an agent that must parse, normalize and validate database records."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writing = MethodWriting()
        self._users = [
            {
                "id": 1,
                "name": "John Doe",
                "email": "john.doe@example.com",
                "created": "Jan 14 2026 09:41:05 +0100",
            },
            {
                "id": 2,
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "created": "Jan 12 2025 14:22:18 -0800",
            },
            {
                "id": 3,
                "name": "Michael Johnson",
                "email": "michael.johnson@example.com",
                "created": "Jan 10 2024 08:45:32 +0000",
            },
            {
                "id": 4,
                "name": "Emily Davis",
                "email": "emily.davis@example.com",
                "created": "Jan 08 2023 19:17:55 +0900",
            },
            {
                "id": 5,
                "name": "Robert Wilson",
                "email": "robert.wilson@example.com",
                "created": "Jan 05 2022 11:33:41 -0500",
            },
        ]
        self._products = [
            {
                "id": 1,
                "name": "Wireless Keyboard",
                "description": "Ergonomic wireless keyboard\n with backlit keys and USB-C charging",
                "created": "Jan 13 2026 10:30:15 +0100",
            },
            {
                "id": 2,
                "name": "Gaming Mouse",
                "description": "High precision gaming mouse\n\nwith customizable RGB lighting",
                "created": "Jan 11 2025 16:45:22 -0800",
            },
            {
                "id": 3,
                "name": "USB-C Hub",
                "description": "7-in-1 USB-C hub with HDMI,\n    ethernet, and multiple USB ports",
                "created": "Jan 09 2024 09:12:38 +0000",
            },
            {
                "id": 4,
                "name": "Mechanical Switches",
                "description": "   Cherry MX Blue mechanical    switches for custom keyboards   ",
                "created": "Jan 07 2023 21:05:47 +0900",
            },
            {
                "id": 5,
                "name": "Laptop Stand",
                "description": "Adjustable aluminum laptop \n stand with cooling ventilation",
                "created": "Jan 04 2022 14:18:55 -0500",
            },
        ]
        self._equipments = [
            {
                "id": 1,
                "name": "3D Printer",
                "description": "High-precision FDM 3D printer\nwith dual extruders and auto-leveling",
                "bought": "Jan 15 2026 13:22:45 +0100",
            },
            {
                "id": 2,
                "name": "Soldering Station",
                "description": "Digital temperature-controlled soldering station with hot air gun",
                "bought": "Jan 13 2025 08:15:30 -0800",
            },
            {
                "id": 3,
                "name": "Oscilloscope",
                "description": "4-channel digital storage \n\n    oscilloscope with 100MHz bandwidth",
                "bought": "Jan 11 2024 17:40:12 +0000",
            },
            {
                "id": 4,
                "name": "Power Supply",
                "description": "Adjustable\nbench\npower\nsupply\nwith\ndual\noutput\n0-30V\n0-5A\n",
                "bought": "Jan 09 2023 20:55:28 +0900",
            },
            {
                "id": 5,
                "name": "Logic Analyzer",
                "description": "   16-channel    USB    logic    analyzer    with    protocol    decoding    capabilities",
                "bought": "Jan 06 2022 12:08:17 -0500",
            },
        ]

    async def get_users(self) -> list[dict]:
        """Get the users from the database.

        Returns:
            list[dict]: The users from the database in the following structure:
            [
              {
                "id": 1,
                "name": "John Doe",
                "email": "john.doe@example.com",
                "created": "Jan 14 2026 09:41:05 +0100",
              }
            ]
        """
        return self._users

    async def get_products(self) -> list[dict]:
        """Get the products from the database.

        Returns:
            list[dict]: The products from the database in the following structure:
            [
              {
                "id": 1,
                "name": "Wireless Keyboard",
                "description": "Ergonomic wireless keyboard with backlit keys and USB-C charging",
                "created": "Jan 13 2026 10:30:15 +0100",
              }
            ]
        """
        return self._products

    async def get_equipments(self) -> list[dict]:
        """Get the equipments from the database.

        Returns:
            list[dict]: The equipments from the database in the following structure:
            [
              {
                "id": 1,
                "name": "3D Printer",
                "description": "High-precision FDM 3D printer with dual extruders and auto-leveling",
                "bought": "Jan 15 2026 13:22:45 +0100",
              }
            ]
        """
        return self._equipments

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=20)))
    async def parse_normalize_and_validate_records(self, tasks: list[str]) -> dict[str, list[dict]]:
        """
        Parse, normalize and validate all database records.

        You are given a list of tasks. Each task might apply to several types of records.
        The tasks give you the instructions on how to parse, normalize or validate the records from the database.

        You need to return the parsed, normalized and validated records in the following structure:
        {
          "users": [
            ...
          ],
          "products": [
            ...
          ],
          "equipments": [
            ...
          ]
        }
        """
        ...
