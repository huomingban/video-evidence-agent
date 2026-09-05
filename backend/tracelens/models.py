"""Shared domain and request models."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class EvidenceIn(BaseModel):
    video_id: str = Field(min_length=1, max_length=200)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=10000)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    video_id: str | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, max_length=100)


class DemoSeedIn(BaseModel):
    video_id: str = Field(default="demo-video", min_length=1, max_length=200)


class BilibiliIn(BaseModel):
    bvid: str = Field(min_length=3, max_length=200)


class AuthIn(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


@dataclass(frozen=True)
class Evidence:
    id: int
    video_id: str
    start_seconds: float
    end_seconds: float
    text: str
    source: str = "ASR"
