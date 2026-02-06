"""
Shared state schema for the ML pipeline agent graph.
Uses TypedDict so LangGraph can checkpoint and restore cleanly.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class PipelineStatus(str, Enum):
    PENDING = "pending"
    DATA_VALIDATION = "data_validation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    AWAITING_APPROVAL = "awaiting_approval"
    DEPLOYING = "deploying"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class AgentMessage(TypedDict):
    role: str          # "supervisor" | "data_agent" | "training_agent" | ...
    content: str
    tool_calls: Optional[List[Dict[str, Any]]]
    timestamp: str


class MLPipelineState(TypedDict):
    # ── Identity ───────────────────────────────────────────────
    run_id: str
    model_name: str
    data_uri: str

    # ── Status tracking ────────────────────────────────────────
    status: PipelineStatus
    current_agent: str
    error: Optional[str]
    retry_count: int

    # ── Data validation ────────────────────────────────────────
    data_schema: Optional[Dict[str, Any]]
    row_count: Optional[int]
    validation_passed: bool

    # ── Training ───────────────────────────────────────────────
    training_job_name: Optional[str]
    training_metrics: Optional[Dict[str, float]]
    model_artifact_uri: Optional[str]

    # ── Evaluation ─────────────────────────────────────────────
    eval_scores: Optional[Dict[str, float]]
    eval_passed: bool
    eval_report_uri: Optional[str]

    # ── Deployment ─────────────────────────────────────────────
    endpoint_name: Optional[str]
    deployment_variant: Optional[str]
    auto_approve: bool
    human_approved: Optional[bool]

    # ── Conversation history ───────────────────────────────────
    messages: List[AgentMessage]
