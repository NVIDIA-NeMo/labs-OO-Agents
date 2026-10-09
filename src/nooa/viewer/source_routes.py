# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authenticated management of explicitly configured trace sources."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import sources

router = APIRouter(prefix="/api/sources")


class AddRunRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=256)


@router.post("/{name}/runs")
def add_source_run(name: str, body: AddRunRequest):
    """Select a run on an installed source and persist it in local configuration."""
    try:
        return sources.add_run(name, body.run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Source or run not found") from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=405, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail="Could not persist source configuration"
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502, detail="Could not validate the remote run catalog"
        ) from exc
