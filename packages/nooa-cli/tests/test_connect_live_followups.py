# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result wording for failures seen in live NVIDIA Build and Inference Hub tests."""

from nooa_cli.commands import _connect_view as view


def test_account_404_says_the_model_is_not_available_for_the_account():
    record = {"error": "NotFoundError", "status_code": 404, "unavailable_for_account": True}
    assert view.check_failure(record) == "This model isn't available for your account (HTTP 404)."
    # A plain 404 still means the route or interface is missing.
    assert view.check_failure({"error": "NotFoundError", "status_code": 404}) == (
        "No route found for this interface or model."
    )
