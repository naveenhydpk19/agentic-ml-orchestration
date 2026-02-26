"""
LangSmith integration for agent trace observability.
Tracks token usage, latency, tool calls, and regression testing across prompt versions.
"""

import os
import time
import functools
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from langsmith import Client
from langsmith.run_helpers import traceable
import logging

logger = logging.getLogger(__name__)


@dataclass
class TraceMetrics:
    run_id: str
    agent_name: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls: List[str] = field(default_factory=list)
    stages_completed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def success_rate(self, expected_stages: List[str]) -> float:
        completed = set(self.stages_completed)
        expected = set(expected_stages)
        if not expected:
            return 1.0
        return len(completed & expected) / len(expected)


class LangSmithTracer:
    """
    Wraps agent runs with LangSmith tracing.
    Supports regression testing of agent behavior across prompt versions.
    """

    def __init__(
        self,
        project_name: str = "ml-pipeline-agent",
        api_key: Optional[str] = None,
    ):
        self.project_name = project_name
        self.client = Client(
            api_key=api_key or os.environ.get("LANGCHAIN_API_KEY"),
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        logger.info(f"LangSmith tracing enabled for project: {project_name}")

    def trace_agent_run(self, func: Callable) -> Callable:
        """Decorator to trace agent invocations and collect metrics."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            run_id = f"{func.__name__}_{int(start)}"
            try:
                result = func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                logger.info(f"Agent run '{run_id}' completed in {latency:.1f}ms")
                return result
            except Exception as e:
                logger.error(f"Agent run '{run_id}' failed: {e}")
                raise
        return wrapper

    def run_regression_suite(
        self,
        dataset_name: str,
        agent_callable: Callable,
        prompt_version: str = "latest",
    ) -> Dict[str, Any]:
        """
        Run agent against a LangSmith dataset for regression testing.
        Returns pass/fail metrics per test case.
        """
        logger.info(f"Running regression suite: dataset={dataset_name}, prompt={prompt_version}")
        results = {"passed": 0, "failed": 0, "errors": 0, "cases": []}

        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            examples = list(self.client.list_examples(dataset_id=dataset.id))
            logger.info(f"Loaded {len(examples)} examples from dataset '{dataset_name}'")

            for example in examples:
                case_result = {"id": str(example.id), "status": "unknown"}
                try:
                    output = agent_callable(example.inputs.get("task", ""))
                    expected = example.outputs or {}

                    passed = self._evaluate_output(output, expected)
                    case_result["status"] = "passed" if passed else "failed"
                    if passed:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                except Exception as e:
                    case_result["status"] = "error"
                    case_result["error"] = str(e)
                    results["errors"] += 1

                results["cases"].append(case_result)

        except Exception as e:
            logger.error(f"Regression suite failed: {e}")
            results["suite_error"] = str(e)

        total = results["passed"] + results["failed"] + results["errors"]
        results["pass_rate"] = results["passed"] / total if total > 0 else 0.0
        logger.info(f"Regression complete: {results['passed']}/{total} passed ({results['pass_rate']:.1%})")
        return results

    def _evaluate_output(self, output: Any, expected: Dict[str, Any]) -> bool:
        if not expected:
            return output is not None
        final_stage = expected.get("final_stage")
        if final_stage and isinstance(output, dict):
            return output.get("pipeline_stage") == final_stage
        return True

    def get_token_usage(self, run_name: str, limit: int = 100) -> Dict[str, int]:
        """Fetch aggregate token usage for recent runs of a given name."""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            runs = self.client.list_runs(
                project_name=self.project_name,
                filter=f'eq(name, "{run_name}")',
                limit=limit,
            )
            for run in runs:
                usage = getattr(run, "total_tokens", 0) or 0
                total["total_tokens"] += usage
        except Exception as e:
            logger.warning(f"Could not fetch token usage: {e}")
        return total
