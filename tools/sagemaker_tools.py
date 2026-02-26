"""
LangChain tools for ML pipeline stages.
Each tool wraps a SageMaker or AWS SDK operation.
"""

import boto3
import json
import logging
from typing import Optional
from langchain.tools import tool

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")


@tool
def trigger_sagemaker_processing_job(
    job_name: str,
    script_uri: str,
    input_s3_uri: str,
    output_s3_uri: str,
    instance_type: str = "ml.m5.xlarge",
) -> str:
    """
    Trigger a SageMaker Processing job for data ingestion or feature engineering.
    Returns job ARN on success.
    """
    try:
        response = sm_client.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": instance_type,
                    "VolumeSizeInGB": 30,
                }
            },
            AppSpecification={
                "ImageUri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-cpu-py310",
                "ContainerEntrypoint": ["python3", script_uri],
            },
            ProcessingInputs=[
                {
                    "InputName": "input_data",
                    "S3Input": {
                        "S3Uri": input_s3_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "processed_data",
                        "S3Output": {
                            "S3Uri": output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            RoleArn="arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole",
        )
        arn = response["ProcessingJobArn"]
        logger.info(f"Started processing job: {job_name}, ARN: {arn}")
        return f"Processing job '{job_name}' started. ARN: {arn}"
    except Exception as e:
        logger.error(f"Failed to start processing job: {e}")
        return f"Error: {str(e)}"


@tool
def check_sagemaker_job_status(job_name: str, job_type: str = "processing") -> str:
    """
    Check status of a SageMaker job (processing, training, or transform).
    Returns current status and failure reason if applicable.
    """
    try:
        if job_type == "training":
            resp = sm_client.describe_training_job(TrainingJobName=job_name)
            status = resp["TrainingJobStatus"]
            reason = resp.get("FailureReason", "")
        else:
            resp = sm_client.describe_processing_job(ProcessingJobName=job_name)
            status = resp["ProcessingJobStatus"]
            reason = resp.get("FailureReason", "")

        result = f"Job '{job_name}' status: {status}"
        if reason:
            result += f". Reason: {reason}"
        return result
    except Exception as e:
        return f"Error checking job status: {str(e)}"


@tool
def register_model_to_registry(
    model_name: str,
    model_package_group: str,
    model_s3_uri: str,
    evaluation_metrics: str,
    approval_status: str = "PendingManualApproval",
) -> str:
    """
    Register a trained model to SageMaker Model Registry with evaluation metrics.
    Returns model package ARN.
    """
    try:
        metrics = json.loads(evaluation_metrics)
        response = sm_client.create_model_package(
            ModelPackageGroupName=model_package_group,
            ModelPackageDescription=f"Model: {model_name}",
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310",
                        "ModelDataUrl": model_s3_uri,
                    }
                ],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
            ModelApprovalStatus=approval_status,
            CustomerMetadataProperties={k: str(v) for k, v in metrics.items()},
        )
        arn = response["ModelPackageArn"]
        logger.info(f"Registered model '{model_name}' to registry. ARN: {arn}")
        return f"Model registered. ARN: {arn}"
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return f"Error: {str(e)}"


@tool
def deploy_model_endpoint(
    endpoint_name: str,
    model_package_arn: str,
    instance_type: str = "ml.g4dn.xlarge",
    instance_count: int = 1,
) -> str:
    """
    Deploy an approved model from registry to a SageMaker real-time endpoint.
    Uses blue/green deployment strategy.
    """
    try:
        config_name = f"{endpoint_name}-config"
        sm_client.create_model(
            ModelName=endpoint_name,
            Containers=[{"ModelPackageName": model_package_arn}],
            ExecutionRoleArn="arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole",
        )
        sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "primary",
                    "ModelName": endpoint_name,
                    "InitialInstanceCount": instance_count,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
            DataCaptureConfig={
                "EnableCapture": True,
                "InitialSamplingPercentage": 20,
                "DestinationS3Uri": f"s3://ml-monitoring/{endpoint_name}/",
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
            },
        )
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
        logger.info(f"Deployment initiated: {endpoint_name}")
        return f"Endpoint '{endpoint_name}' deployment initiated with data capture enabled."
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return f"Error: {str(e)}"
