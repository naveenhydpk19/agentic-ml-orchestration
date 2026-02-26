"""
Local demo — runs the full agentic ML pipeline with zero AWS dependencies.

Usage:
    python demo_local.py
    python demo_local.py --model-name my-classifier --auto-approve

What it does:
  1. Runs the full LangGraph state machine (supervisor → data → training → eval → deploy)
  2. All AWS calls (SageMaker, Bedrock) are replaced with deterministic mocks
  3. Human-in-the-loop approval simulated via CLI prompt (or --auto-approve)
  4. Full agent trace printed at each step
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal

sys.path.insert(0, str(Path(__file__).parent))


# ── Mock Bedrock LLM ─────────────────────────────────────────────
class MockBedrockLLM:
    """Returns deterministic routing decisions for the supervisor."""

    def invoke(self, prompt: Any) -> Any:
        prompt_str = str(prompt)

        class Response:
            content = "data_agent"

        # Routing logic based on prompt keywords
        if "validation_passed: True" in prompt_str and "training_job_name: None" in prompt_str:
            Response.content = "training_agent"
        elif "training_job_name" in prompt_str and "None" not in prompt_str and "eval_passed: False" in prompt_str:
            Response.content = "evaluation_agent"
        elif "eval_passed: True" in prompt_str and "human_approved: None" in prompt_str:
            Response.content = "deployment_agent"
        elif "COMPLETE" in prompt_str or "ROLLED_BACK" in prompt_str:
            Response.content = "FINISH"
        else:
            Response.content = "data_agent"

        return Response()


# ── Patch agents before importing pipeline ───────────────────────
import src.agents.supervisor as sup_module
import src.agents.training_agent as train_module

# Patch supervisor to use MockBedrockLLM
_original_supervisor_init = None


# ── Mock SageMaker ────────────────────────────────────────────────
_MOCK_JOB_NAME = None


def mock_submit_training_job(model_name, data_uri, instance_type="ml.p3.2xlarge", hyperparameters=None):
    global _MOCK_JOB_NAME
    job_name = f"{model_name}-local-{int(time.time())}"
    _MOCK_JOB_NAME = job_name
    print(f"    [MockSageMaker] Submitted training job: {job_name}")
    print(f"    [MockSageMaker] Instance: {instance_type} | Data: {data_uri}")
    time.sleep(0.3)  # simulate latency
    return {"job_name": job_name, "status": "InProgress"}


def mock_poll_training_job(job_name, poll_interval_seconds=1):
    print(f"    [MockSageMaker] Polling job: {job_name}")
    for i in range(3):
        print(f"    [MockSageMaker] Status: InProgress ({(i+1)*2} min elapsed)...")
        time.sleep(0.2)
    print(f"    [MockSageMaker] Status: Completed ✅")
    return {
        "status": "Completed",
        "metrics": {
            "val_accuracy": 0.943,
            "val_f1": 0.931,
            "train_loss": 0.182,
        },
        "artifact_uri": f"s3://ml-platform-artifacts/models/{job_name}/output/model.tar.gz",
    }


# Tools are overridden in LocalMLPipeline._training_agent directly


# ── Import pipeline after patching ───────────────────────────────
from src.agents.state import MLPipelineState, PipelineStatus
from src.agents.supervisor import SupervisorAgent
from src.agents.training_agent import TrainingAgent


# ── Simple local state machine (no LangGraph needed locally) ─────
class LocalMLPipeline:
    """
    Runs the same agent logic as the LangGraph version,
    but as a simple sequential state machine for local demo.
    """

    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve

    def run(self, model_name: str, data_uri: str) -> MLPipelineState:
        state: MLPipelineState = {
            "run_id": str(uuid.uuid4())[:8],
            "model_name": model_name,
            "data_uri": data_uri,
            "status": PipelineStatus.PENDING,
            "current_agent": "supervisor",
            "error": None,
            "retry_count": 0,
            "data_schema": None,
            "row_count": None,
            "validation_passed": False,
            "training_job_name": None,
            "training_metrics": None,
            "model_artifact_uri": None,
            "eval_scores": None,
            "eval_passed": False,
            "eval_report_uri": None,
            "endpoint_name": None,
            "deployment_variant": None,
            "auto_approve": self.auto_approve,
            "human_approved": None,
            "messages": [],
        }

        print(f"\n  Run ID: {state['run_id']}")
        print(f"  Model:  {model_name}")
        print(f"  Data:   {data_uri}\n")

        state = self._data_agent(state)
        state = self._training_agent(state)
        state = self._evaluation_agent(state)
        state = self._approval_gate(state)
        if state.get("human_approved"):
            state = self._deployment_agent(state)

        return state

    def _data_agent(self, state):
        print("─" * 50)
        print("🔍 [DataAgent] Validating dataset...")
        time.sleep(0.2)
        print(f"  Schema: parquet | Label col: 'target' | Rows: 142,880")
        print(f"  Null check: PASS | Schema drift: NONE")
        print(f"  ✅ Data validation passed")
        return {
            **state,
            "status": PipelineStatus.TRAINING,
            "validation_passed": True,
            "data_schema": {"format": "parquet", "label_col": "target"},
            "row_count": 142880,
        }

    def _training_agent(self, state):
        print("─" * 50)
        print("🏋️  [TrainingAgent] Submitting SageMaker training job...")
        submit_result = mock_submit_training_job(state["model_name"], state["data_uri"])
        job_name = submit_result["job_name"]
        poll_result = mock_poll_training_job(job_name)
        if poll_result["status"] != "Completed":
            return {**state, "status": PipelineStatus.FAILED, "error": "Training failed", "retry_count": state["retry_count"] + 1}
        m = poll_result["metrics"]
        print(f"  val_accuracy: {m.get("val_accuracy", "N/A")} | val_f1: {m.get("val_f1", "N/A")}")
        print(f"  ✅ Training complete — artifact: {poll_result["artifact_uri"]}")
        return {**state, "status": PipelineStatus.EVALUATION, "training_job_name": job_name, "training_metrics": m, "model_artifact_uri": poll_result["artifact_uri"], "current_agent": "evaluation_agent"}

    def _evaluation_agent(self, state):
        print("─" * 50)
        print("📊 [EvaluationAgent] Running Ragas evaluation...")
        time.sleep(0.3)
        scores = {
            "faithfulness": 0.91,
            "answer_relevancy": 0.88,
            "context_precision": 0.84,
            "context_recall": 0.79,
        }
        passed = all(v >= 0.75 for v in scores.values())
        print(f"  faithfulness:      {scores['faithfulness']} {'✅' if scores['faithfulness'] >= 0.75 else '❌'}")
        print(f"  answer_relevancy:  {scores['answer_relevancy']} {'✅' if scores['answer_relevancy'] >= 0.75 else '❌'}")
        print(f"  context_precision: {scores['context_precision']} {'✅' if scores['context_precision'] >= 0.75 else '❌'}")
        print(f"  context_recall:    {scores['context_recall']} {'✅' if scores['context_recall'] >= 0.75 else '❌'}")
        print(f"  {'✅ All thresholds passed — recommending promotion' if passed else '❌ Evaluation failed'}")
        return {
            **state,
            "eval_scores": scores,
            "eval_passed": passed,
            "status": PipelineStatus.AWAITING_APPROVAL if passed else PipelineStatus.FAILED,
        }

    def _approval_gate(self, state):
        print("─" * 50)
        print("🔐 [Supervisor] Deployment approval required...")
        if self.auto_approve:
            print("  --auto-approve flag set → auto-approving ✅")
            approved = True
        else:
            ans = input("  Approve production deployment? [y/N]: ").strip().lower()
            approved = ans in ("y", "yes")
            if not approved:
                print("  ❌ Deployment rejected by operator")

        return {**state, "human_approved": approved}

    def _deployment_agent(self, state):
        print("─" * 50)
        print("🚀 [DeploymentAgent] Starting blue/green deployment...")
        endpoint = f"{state['model_name']}-endpoint"
        time.sleep(0.3)
        print(f"  Endpoint: {endpoint}")
        print(f"  Strategy: blue/green | Traffic: shifting to green variant")
        time.sleep(0.2)
        print(f"  Monitoring for anomalies (10s window)...")
        time.sleep(0.3)
        print(f"  ✅ Deployment complete — no anomalies detected")
        return {
            **state,
            "status": PipelineStatus.COMPLETE,
            "endpoint_name": endpoint,
            "deployment_variant": "green",
        }


def main():
    parser = argparse.ArgumentParser(description="Agentic ML Orchestration — Local Demo")
    parser.add_argument("--model-name", default="domain-classifier-v2")
    parser.add_argument("--data-uri", default="s3://demo-bucket/data/train")
    parser.add_argument("--auto-approve", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Agentic ML Orchestration Engine — Local Demo")
    print("  (No AWS / Bedrock / SageMaker needed)")
    print("=" * 60)

    pipeline = LocalMLPipeline(auto_approve=args.auto_approve)
    final = pipeline.run(args.model_name, args.data_uri)

    print("─" * 50)
    print(f"\n{'✅ Pipeline COMPLETE' if final['status'] == PipelineStatus.COMPLETE else '❌ Pipeline FAILED'}")
    print(f"  Final status:  {final['status']}")
    if final.get("endpoint_name"):
        print(f"  Endpoint:      {final['endpoint_name']}")
    if final.get("training_metrics"):
        print(f"  Best metrics:  {final['training_metrics']}")
    print("\nTo run with real AWS:")
    print("  • Configure AWS credentials + SageMaker role in configs/")
    print("  • Use build_graph() from src/workflows/ml_pipeline.py")


if __name__ == "__main__":
    main()
