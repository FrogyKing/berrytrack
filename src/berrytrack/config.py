"""Validated experiment configuration; unknown options are errors."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["berrytrack", "bytetrack", "botsort", "deepsort"] = "berrytrack"
    high_threshold: float = Field(0.5, ge=0, le=1)
    low_threshold: float = Field(0.1, ge=0, le=1)
    match_threshold: float = Field(0.72, gt=0, le=1)
    min_hits: int = Field(2, ge=1)
    max_age: int = Field(30, ge=1)
    memory_age: int = Field(120, ge=1)
    embedding_dim: Literal[32, 64, 128] = 64
    appearance: bool = True
    motion: bool = True
    camera_compensation: bool = True
    context: bool = True
    adaptive: bool = True
    uncertainty_trigger: bool = True
    static_scene: bool = False
    local_rigidity: bool = True
    rigidity_weight: float = Field(0.35, ge=0, le=1)
    rigidity_neighbors: int = Field(6, ge=3, le=16)
    rigidity_gate: float = Field(4.0, gt=1)
    neighbor_k: int = Field(4, ge=1, le=16)
    gate_distance: float = Field(6.0, gt=0)
    ambiguity_margin: float = Field(0.12, ge=0, le=1)
    seed: int = 42


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "berrytrack"
    tracker: TrackerConfig = Field(default_factory=lambda: TrackerConfig())
    image_size: int = Field(640, ge=64)
    detection_threshold: float = Field(0.1, ge=0, le=1)
    seed: int = 42
    split: Literal["train", "val", "test"] = "val"


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return ExperimentConfig.model_validate(yaml.safe_load(handle) or {})
