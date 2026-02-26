"""
Multi-agent ML pipeline orchestration using LangGraph.
Implements supervisor pattern with tool-use, memory, and reflection.
"""

from typing import List, Dict, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_aws import ChatBedrock
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, AIMessage, BaseMessage
import operator
import logging

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    pipeline_stage: str
    context: Dict[str, Any]
    iteration: int
    max_iterations: int


class MLPipelineAgent:
    """
    Supervisor agent orchestrating end-to-end ML pipeline execution.
    Stages: data_ingestion → feature_engineering → model_eval → deployment
    """

    STAGE_TRANSITIONS = {
        "data_ingestion": "feature_engineering",
        "feature_engineering": "model_eval",
        "model_eval": "deployment",
        "deployment": END,
    }

    def __init__(
        self,
        tools: List[BaseTool],
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region: str = "us-east-1",
        max_iterations: int = 20,
    ):
        self.tools = tools
        self.max_iterations = max_iterations
        self.tool_executor = ToolExecutor(tools)
        self.llm = ChatBedrock(
            model_id=model_id,
            region_name=region,
        ).bind_tools(tools)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("tool_executor", self._tool_node)
        graph.add_node("reflector", self._reflection_node)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "tools": "tool_executor",
                "reflect": "reflector",
                "end": END,
            },
        )
        graph.add_edge("tool_executor", "supervisor")
        graph.add_edge("reflector", "supervisor")

        return graph.compile()

    def _supervisor_node(self, state: AgentState) -> AgentState:
        logger.info(f"Supervisor: stage={state['pipeline_stage']}, iter={state['iteration']}")
        response = self.llm.invoke(state["messages"])
        state["messages"].append(response)
        state["iteration"] += 1
        return state

    def _tool_node(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        results = []
        for call in tool_calls:
            invocation = ToolInvocation(tool=call["name"], tool_input=call["args"])
            result = self.tool_executor.invoke(invocation)
            results.append(f"Tool '{call['name']}' result: {result}")
            logger.info(f"Executed tool '{call['name']}'")

        state["messages"].append(HumanMessage(content="\n".join(results)))
        return state

    def _reflection_node(self, state: AgentState) -> AgentState:
        """Pause and self-critique before advancing pipeline stage."""
        reflection_prompt = (
            f"You are at stage '{state['pipeline_stage']}'. "
            "Review the steps taken so far. "
            "Are there any issues or missing validations before proceeding? "
            "Be concise and critical."
        )
        response = self.llm.invoke(
            state["messages"] + [HumanMessage(content=reflection_prompt)]
        )
        state["messages"].append(response)
        logger.info(f"Reflection complete at stage: {state['pipeline_stage']}")
        return state

    def _route(self, state: AgentState) -> str:
        if state["iteration"] >= state["max_iterations"]:
            logger.warning("Max iterations reached, forcing end.")
            return "end"

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        if tool_calls:
            return "tools"

        content = getattr(last_message, "content", "").lower()
        if "stage complete" in content or "advance to next stage" in content:
            next_stage = self.STAGE_TRANSITIONS.get(state["pipeline_stage"], END)
            if next_stage == END:
                return "end"
            state["pipeline_stage"] = next_stage
            return "reflect"

        return "end"

    def run(self, task: str, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        initial_state = AgentState(
            messages=[HumanMessage(content=task)],
            pipeline_stage="data_ingestion",
            context=initial_context or {},
            iteration=0,
            max_iterations=self.max_iterations,
        )
        final_state = self.graph.invoke(initial_state)
        logger.info(f"Pipeline completed. Final stage: {final_state['pipeline_stage']}")
        return final_state
