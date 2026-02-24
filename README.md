# Agentic ML Workflow Orchestration Engine

A multi-agent system built on LangGraph that automates end-to-end ML pipeline execution — from data validation and feature engineering through model evaluation and deployment triggers. Designed to operate autonomously with tool-use, memory, reflection, and human-in-the-loop escalation.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Supervisor Agent             │
                    │  (routes tasks, manages state)       │
                    └──────────┬──────────────┬────────────┘
                               │              │
              ┌────────────────┘              └───────────────┐
              ▼                                               ▼
   ┌──────────────────┐                          ┌──────────────────────┐
   │  Data Agent       │                          │  Training Agent       │
   │  - validate       │                          │  - trigger SageMaker  │
   │  - profile        │                          │  - monitor job        │
   │  - transform      │                          │  - collect metrics    │
   └──────────────────┘                          └──────────────────────┘
                                                              │
                                                  ┌───────────▼──────────┐
                                                  │  Evaluation Agent     │
                                                  │  - run Ragas evals    │
                                                  │  - check thresholds   │
                                                  │  - approve/reject     │
                                                  └───────────┬──────────┘
                                                              │
                                                  ┌───────────▼──────────┐
                                                  │  Deployment Agent     │
                                                  │  - blue/green deploy  │
                                                  │  - canary rollout     │
                                                  │  - rollback on fail   │
                                                  └──────────────────────┘
```

## Features

- **LangGraph state machine** — typed state, conditional routing, cycle detection
- **Multi-agent coordination** — supervisor pattern with specialised sub-agents
- **Tool use** — SageMaker, S3, Lambda, Slack notification, approval gate tools
- **Persistent memory** — checkpoint-based state recovery across failures
- **LangSmith observability** — full trace of every agent step and tool call
- **Human-in-the-loop** — escalation to Slack for high-risk deployment decisions
- **Reflection loop** — agents self-critique and retry on low-confidence outputs

## Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph, LangChain |
| LLM Backend | AWS Bedrock (Claude 3 Sonnet) |
| Orchestration | AWS Lambda, EventBridge |
| ML Platform | AWS SageMaker |
| Observability | LangSmith |
| Notifications | Slack SDK |
| State Storage | DynamoDB (checkpoints) |

## Quick Start

```bash
git clone https://github.com/naveenhydpk19/agentic-ml-orchestration
cd agentic-ml-orchestration
pip install -r requirements.txt

# Trigger a full pipeline run
python -m src.workflows.ml_pipeline \
  --model-name my-classifier \
  --data-uri s3://my-bucket/data/train \
  --auto-approve false
```

## Example Run

```
[Supervisor]  Received pipeline request for my-classifier
[DataAgent]   Validating dataset at s3://my-bucket/data/train...
[DataAgent]   Schema OK | 142,880 rows | no nulls in label column
[TrainingAgent] Submitting SageMaker training job: my-classifier-20240301-143022
[TrainingAgent] Job status: InProgress (12 min elapsed)
[TrainingAgent] Job complete — val_accuracy: 0.943, val_f1: 0.931
[EvalAgent]   Running Ragas eval on held-out set...
[EvalAgent]   faithfulness: 0.91 ✅  answer_relevancy: 0.88 ✅
[EvalAgent]   All thresholds passed — recommending promotion
[Supervisor]  Escalating to human approval (production deployment)
[Human]       ✅ Approved via Slack
[DeployAgent] Starting blue/green deployment to sagemaker endpoint
[DeployAgent] Traffic shifted 100% to new variant — monitoring for 10 min
[DeployAgent] No anomalies detected. Deployment complete ✅
```

## License

MIT
