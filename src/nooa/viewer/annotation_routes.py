# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation API endpoints for the trace viewer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import otlp_store

router = APIRouter()


class AnnotationCreate(BaseModel):
    session_id: str
    span_id: str | None = None
    target: str | None = None
    name: str
    score: float | None = None
    label: str | None = None
    comment: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "human"


class AnnotationUpdate(BaseModel):
    score: float | None = None
    label: str | None = None
    comment: str | None = None
    tags: list[str] | None = None
    target: str | None = None
    session_id: str | None = None


@router.get("/api/traces/{session_id:path}/annotations")
def get_trace_annotations(session_id: str):
    return otlp_store.list_annotations(session_id)


@router.post("/api/annotations")
def create_annotation(body: AnnotationCreate):
    try:
        return otlp_store.create_annotation(body.model_dump())
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.patch("/api/annotations/{annotation_id}")
def update_annotation(annotation_id: str, body: AnnotationUpdate):
    updates = body.model_dump(exclude_unset=True)
    updates.pop("session_id", None)
    try:
        return otlp_store.update_annotation(annotation_id, updates)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/api/annotations/{annotation_id}", status_code=204)
def delete_annotation(annotation_id: str):
    try:
        otlp_store.delete_annotation(annotation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/tags")
def get_tags():
    tags = otlp_store.list_tags()
    return {"tags": tags, "total": len(tags)}
