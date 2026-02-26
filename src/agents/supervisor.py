"""
Supervisor agent — routes tasks between specialised sub-agents
based on pipeline state and LLM reasoning.
"""
from __future__ import annotations

import logging
from typing import Literal

try:
    from langchain_aws import ChatBedrock
except ImportError:
    ChatBedrock = object
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from src.agents.state import MLPipelineState, PipelineStatus

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the supervisor of an automated ML pipeline.
Your job is to decide which agent should act next based on the current pipeline state.

Agents available:
- data_agent: validates and profiles datasets
- training_agent: submits and monitors SageMaker training jobs
- evaluation_agent: runs model evaluation and quality gates
- deployment_agent: handles blue/green deployments and rollbacks
- FINISH: pipeline is complete or has failed unrecoverably

Current state:
- Status: {status}
- Model: {model_name}
- Validation passed: {validation_passed}
- Eval passed: {eval_passed}
- Human approved: {human_approved}
- Error: {error}
- Retry count: {retry_count}

Respond with ONLY the agent name or FINISH."""),
    ("human", "Which agent should act next?"),
])


class SupervisorAgent:
    def __init__(self, llm: ChatBedrock):
        self.llm = llm
        self.chain = SUPERVISOR_PROMPT | llm

    @traceable(name="supervisor-route")
    def route(self, state: MLPipelineState) -> Literal[
        "data_agent", "training_agent", "evaluation_agent", "deployment_agent", "FINISH"
    ]:
        # Hard rules first — no need to burn tokens for obvious transitions
        if state["status"] == PipelineStatus.FAILED and state["retry_count"] >= 3:
            logger.warning("Max retries reached, finishing with failure")
            return "FINISH"

        if state["status"] == PipelineStatus.COMPLETE:
            return "FINISH"

        if state["status"] == PipelineStatus.AWAITING_APPROVAL:
            if state.get("human_approved") is True:
                return "deployment_agent"
            elif state.get("human_approved") is False:
                return "FINISH"
            # Still waiting
            return "FINISH"

        # LLM routing for ambiguous states
        response = self.chain.invoke({
            "status": state["status"],
            "model_name": state["model_name"],
            "validation_passed": state["validation_passed"],
            "eval_passed": state["eval_passed"],
            "human_approved": state.get("human_approved"),
            "error": state.get("error"),
            "retry_count": state["retry_count"],
        })

        decision = response.content.strip().lower().replace("-", "_")
        valid = {"data_agent", "training_agent", "evaluation_agent", "deployment_agent", "finish"}
        if decision not in valid:
            logger.warning("Unexpected routing decision '%s', defaulting to FINISH", decision)
            return "FINISH"

        return decision.upper() if decision == "finish" else decision
