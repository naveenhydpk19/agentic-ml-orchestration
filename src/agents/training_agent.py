"""
Training agent — submits SageMaker training jobs, polls status,
collects metrics, and updates pipeline state.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Dict

import boto3
from langchain.tools import tool
from langsmith import traceable

from src.agents.state import MLPipelineState, PipelineStatus

logger = logging.getLogger(__name__)

sm_client = boto3.client("sagemaker")


@tool
def submit_training_job(
    model_name: str,
    data_uri: str,
    instance_type: str = "ml.p3.2xlarge",
    hyperparameters: Dict[str, str] | None = None,
) -> Dict:
    """Submit a SageMaker training job and return the job name."""
    job_name = f"{model_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    hp = hyperparameters or {"epochs": "10", "learning_rate": "2e-5", "batch_size": "32"}

    sm_client.create_training_job(
        TrainingJobName=job_name,
        HyperParameters=hp,
        AlgorithmSpecification={
            "TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310",
            "TrainingInputMode": "File",
        },
        RoleArn="arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole",
        InputDataConfig=[{
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {"S3Uri": data_uri, "S3DataType": "S3Prefix"}},
        }],
        OutputDataConfig={"S3OutputPath": f"s3://ml-platform-artifacts/models/{model_name}"},
        ResourceConfig={"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 50},
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
    )
    logger.info("Submitted training job: %s", job_name)
    return {"job_name": job_name, "status": "InProgress"}


@tool
def poll_training_job(job_name: str, poll_interval_seconds: int = 60) -> Dict:
    """Poll a SageMaker training job until it completes or fails."""
    terminal = {"Completed", "Failed", "Stopped"}
    while True:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        logger.info("Job %s: %s", job_name, status)

        if status in terminal:
            metrics = {
                m["MetricName"]: m["Value"]
                for m in resp.get("FinalMetricDataList", [])
            }
            return {
                "status": status,
                "metrics": metrics,
                "artifact_uri": resp.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
            }
        time.sleep(poll_interval_seconds)


class TrainingAgent:
    def __init__(self):
        self.tools = [submit_training_job, poll_training_job]

    @traceable(name="training-agent-run")
    def run(self, state: MLPipelineState) -> MLPipelineState:
        try:
            submit_result = submit_training_job.invoke({
                "model_name": state["model_name"],
                "data_uri": state["data_uri"],
            })
            job_name = submit_result["job_name"]

            poll_result = poll_training_job.invoke({"job_name": job_name})

            if poll_result["status"] != "Completed":
                return {
                    **state,
                    "status": PipelineStatus.FAILED,
                    "error": f"Training job {job_name} ended with status {poll_result['status']}",
                    "retry_count": state["retry_count"] + 1,
                }

            return {
                **state,
                "status": PipelineStatus.EVALUATION,
                "training_job_name": job_name,
                "training_metrics": poll_result["metrics"],
                "model_artifact_uri": poll_result["artifact_uri"],
                "current_agent": "evaluation_agent",
            }

        except Exception as e:
            logger.exception("Training agent error")
            return {
                **state,
                "status": PipelineStatus.FAILED,
                "error": str(e),
                "retry_count": state["retry_count"] + 1,
            }
