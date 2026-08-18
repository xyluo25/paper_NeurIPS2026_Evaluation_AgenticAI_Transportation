"""
##############################################################
# Created Date: Friday, June 27th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
"""

import ast
import json
import re
from pathlib import Path
from typing import Any, Optional

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from proj_llm import llm_openai
from proj_rag import rag_tool, rag_tool_sim_parameters
from proj_tools import HIL_Tools, usr_defined_tools


llm = llm_openai
MODULE_DIR = Path(__file__).resolve().parent

bot_prefix = (MODULE_DIR / "chat_prompt.txt").read_text(encoding="utf-8")
bot_suffix = (
    "IMPORTANT: If the tool's output is a dict, respond only with that dict as "
    "text literal."
)

HIL_INTERRUPT_CONFIG = {
    tool_name: {
        "allowed_decisions": ["approve", "edit", "reject"],
        "description": (
            "Human approval is required before this RealTwin workflow tool can "
            "run. Review the tool name and arguments before continuing."
        ),
    }
    for tool_name in HIL_Tools
}


def _agent_prompt(role: str) -> str:
    """Return the role-specific prompt shared by each DeepAgents subagent."""

    return f"{role}\n\n{bot_prefix}\n\n{bot_suffix}"


subagents = [
    {
        "name": "osm_agent",
        "description": (
            "Use this agent for OpenStreetMap tasks from plain-language user "
            "requests, including city/place metadata, bounding boxes, browser "
            "wizard downloads, relation ID downloads, named-place OSM downloads, "
            "and OSM network visualization."
        ),
        "system_prompt": _agent_prompt(
            "You are an OSM agent that can assist with OpenStreetMap related tasks."
        ),
        "tools": usr_defined_tools.get("osm_tools", []),
        "model": llm,
        "interrupt_on": HIL_INTERRUPT_CONFIG,
    },
    {
        "name": "sumo_agent",
        "description": (
            "Use this agent for SUMO tasks from plain-language user requests, "
            "including installation checks, detected versions, installation or "
            "setup of a specific SUMO version, and PNG snapshots or visualization "
            "of SUMO .net.xml network files."
        ),
        "system_prompt": _agent_prompt(
            "You are a SUMO agent that can assist with SUMO related tasks."
        ),
        "tools": usr_defined_tools.get("sumo_tools", []),
        "model": llm,
        "interrupt_on": HIL_INTERRUPT_CONFIG,
    },
    {
        "name": "realtwin_agent",
        "description": (
            "Use this agent for RealTwin workflow tasks from plain-language user "
            "requests, including showing current or default settings, editing "
            "settings, saving settings for user editing, input generation, sample "
            "runs, simulation, and calibration."
        ),
        "system_prompt": _agent_prompt(
            "You are a RealTwin agent that can assist with RealTwin related tasks."
        ),
        "tools": usr_defined_tools.get("realtwin_tools", []),
        "model": llm,
        "interrupt_on": HIL_INTERRUPT_CONFIG,
    },
    {
        "name": "rag_agent",
        "description": (
            "Use this agent for natural-language knowledge-base questions about "
            "RealTwin, ORNL ARMS, the development team, listed members, and "
            "suggested values or ranges for lane-changing, car-following, and "
            "behavior simulation parameters."
        ),
        "system_prompt": _agent_prompt(
            "You are a RAG agent that can retrieve RealTwin and traffic "
            "simulation reference information."
        ),
        "tools": [rag_tool, rag_tool_sim_parameters],
        "model": llm,
        "interrupt_on": HIL_INTERRUPT_CONFIG,
    },
]

SupervisorAgent = create_deep_agent(
    model=llm,
    system_prompt=(
        "You are a chatbot supervisor responsible for delegating user requests "
        "to specialized subagents through the task tool.\n"
        "Users normally do not know internal tool or subagent names. Infer the "
        "right subagent from the user's goal, domain terms, file paths, IDs, "
        "place names, requested output, and follow-up context. Do not ask the "
        "user to provide a tool name when the intent and arguments are already "
        "clear.\n"
        "For simple conversation, definitions, or general transportation "
        "simulation questions that do not need tools, answer directly without "
        "delegating to a subagent.\n"
        "For supported Agentic RealTwin workflows, call task with the matching "
        "subagent_type instead of using a generic agent.\n"
        "Use osm_agent for OpenStreetMap tasks. This includes requests to find "
        "place metadata or bounding boxes, open an OSM web/browser wizard, "
        "download OSM/OpenStreetMap data by relation ID, download OSM data for "
        "a named place, and visualize an OSM road network.\n"
        "Use realtwin_agent for RealTwin-related tasks, such as showing, editing, "
        "and saving current or default settings, input generation from YAML, "
        "sample scenario runs, simulation, and calibration.\n"
        "Use rag_agent for retrieval-augmented generation tasks for development "
        "team information and suggested lane-changing, car-following, and behavior "
        "parameters. Include min_gap, acceleration, deceleration, sigma, tau, and "
        "emergencyDecel when parameter ranges are requested.\n"
        "Use sumo_agent for SUMO-related tasks. This includes checking whether "
        "SUMO is installed, reporting detected versions, installing or setting "
        "up a requested SUMO version, and creating images or snapshots from "
        "SUMO .net.xml network files.\n"
        "You may call more than one subagent sequentially when a workflow spans "
        "multiple domains.\n"
        "Approval-required workflow tools are critical. When the current user "
        "request asks for any approval-required tool by name or by intent, never "
        "answer from prior conversation memory, cached results, or earlier tool "
        "outputs. You must call the relevant tool so Human-In-The-Loop review can "
        "interrupt before execution.\n"
        "Call at most one approval-required workflow tool at a time.\n"
        "Whenever a tool returns a dictionary or JSON, reply with only that "
        "literal dict or JSON.\n"
        "Always return the output of any executed tool."
    ),
    subagents=subagents,
    interrupt_on=HIL_INTERRUPT_CONFIG,
    checkpointer=MemorySaver(),
)

HIL_Agent = SupervisorAgent
HIL_TOOL_NAMES = set(HIL_INTERRUPT_CONFIG)
_pending_hil_tool_calls: list[dict[str, Any]] = []


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    """Return a field from a dict-like object or object attribute."""

    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def extract_hil_tool_calls_from_interrupts(interrupts: Any) -> list[dict[str, Any]]:
    """Return approval-required tool calls from DeepAgents interrupt payloads."""

    if not interrupts:
        return []

    interrupt_items = interrupts
    if not isinstance(interrupt_items, (list, tuple)):
        interrupt_items = [interrupt_items]

    tool_calls: list[dict[str, Any]] = []
    for interrupt_item in interrupt_items:
        interrupt_value = _get_value(interrupt_item, "value", interrupt_item)
        if isinstance(interrupt_value, list):
            interrupt_values = interrupt_value
        else:
            interrupt_values = [interrupt_value]

        for value_item in interrupt_values:
            action_requests = _get_value(value_item, "action_requests", [])
            review_configs = _get_value(value_item, "review_configs", [])
            for index, action_request in enumerate(action_requests):
                review_config = {}
                if index < len(review_configs):
                    review_config = review_configs[index]

                action_request_data = _get_value(
                    action_request,
                    "action_request",
                    {},
                )
                tool_name = (
                    _get_value(action_request, "name")
                    or _get_value(action_request, "action")
                    or _get_value(review_config, "action_name")
                    or _get_value(action_request_data, "action")
                )
                if tool_name not in HIL_TOOL_NAMES:
                    continue

                tool_args = _get_value(action_request, "args", None)
                if tool_args is None:
                    tool_args = _get_value(action_request_data, "args", {})

                tool_calls.append(
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": _get_value(action_request, "id", f"interrupt_{index}"),
                        "type": "tool_call",
                        "description": _get_value(action_request, "description", ""),
                        "allowed_decisions": _get_value(
                            review_config,
                            "allowed_decisions",
                            [],
                        ),
                    }
                )

    return tool_calls


def get_pending_hil_tool_calls() -> list[dict[str, Any]]:
    """Return the critical tool calls captured from the latest interrupt."""

    return [dict(tool_call) for tool_call in _pending_hil_tool_calls]


def stream_hil_agent(agent_input: Any, thread: dict) -> tuple[list[Any], bool]:
    """Stream the DeepAgents graph and return messages plus interrupt status."""

    global _pending_hil_tool_calls
    _pending_hil_tool_calls = []

    messages: list[Any] = []
    interrupted = False

    for event in HIL_Agent.stream(agent_input, thread, stream_mode="values"):
        if "messages" in event:
            messages = event["messages"]
        if "__interrupt__" in event:
            interrupted = True
            _pending_hil_tool_calls = extract_hil_tool_calls_from_interrupts(
                event["__interrupt__"]
            )
            break

    try:
        state = HIL_Agent.get_state(thread)
        state_messages = state.values.get("messages", [])
        if state_messages:
            messages = state_messages
    except Exception:
        pass

    if interrupted and not _pending_hil_tool_calls:
        _pending_hil_tool_calls = extract_hil_tool_calls(messages)

    return messages, interrupted


def resume_hil_agent(
    decisions: list[dict[str, Any]], thread: dict
) -> tuple[list[Any], bool]:
    """Resume a paused DeepAgents graph with human review decisions."""

    return stream_hil_agent(Command(resume={"decisions": decisions}), thread)


def extract_hil_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Return approval-required tool calls from the latest AI message."""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            message_tool_calls = [
                tool_call
                for tool_call in message.tool_calls
                if tool_call.get("name") in HIL_TOOL_NAMES
            ]
            if message_tool_calls:
                return message_tool_calls
    if _pending_hil_tool_calls:
        return get_pending_hil_tool_calls()
    return []


def format_tool_selection(tool_calls: list[dict[str, Any]]) -> str:
    """Format pending tool calls for the editable HIL textbox."""

    if len(tool_calls) == 1:
        tool_call = tool_calls[0]
        return (
            f"Tool invoked: {tool_call.get('name')}, "
            f"with arguments: {tool_call.get('args', {})}"
        )

    return "Tools invoked: " + json.dumps(
        [
            {"name": tool_call.get("name"), "args": tool_call.get("args", {})}
            for tool_call in tool_calls
        ],
        ensure_ascii=False,
    )


def parse_tool_selection(tool_selection: str) -> Optional[dict[str, Any]]:
    """Parse an edited HIL textbox value into a DeepAgents edit action."""

    single_match = re.search(
        r"Tool invoked:\s*(?P<name>[^,]+),\s*with arguments:\s*(?P<args>.*)",
        tool_selection,
        re.DOTALL,
    )
    if single_match:
        try:
            edited_args = ast.literal_eval(single_match.group("args").strip())
        except (SyntaxError, ValueError):
            return None

        return {
            "name": single_match.group("name").strip(),
            "args": edited_args,
        }

    multi_match = re.search(
        r"Tools invoked:\s*(?P<tools>\[.*\])", tool_selection, re.DOTALL
    )
    if multi_match:
        try:
            tools = json.loads(multi_match.group("tools"))
        except json.JSONDecodeError:
            return None

        if tools:
            first_tool = tools[0]
            return {
                "name": first_tool.get("name", ""),
                "args": first_tool.get("args", {}),
            }

    return None


def generate_review_decisions(
    tool_confirm: str,
    tool_selection: str,
    original_tool_calls: list[dict[str, Any]],
    user_feedback: str = "",
) -> list[dict[str, Any]]:
    """Convert UI confirmation into DeepAgents HITL decisions."""

    if tool_confirm == "Yes":
        return [{"type": "approve"} for _ in original_tool_calls]

    edited_action = parse_tool_selection(tool_selection)
    original_first_action = None
    if original_tool_calls:
        first_tool_call = original_tool_calls[0]
        original_first_action = {
            "name": first_tool_call.get("name"),
            "args": first_tool_call.get("args", {}),
        }

    if edited_action and edited_action != original_first_action:
        decisions = [{"type": "edit", "edited_action": edited_action}]
        decisions.extend(
            {
                "type": "reject",
                "message": (
                    "Only the first edited tool call was approved by the user. "
                    f"User feedback: {user_feedback}"
                ).strip(),
            }
            for _ in original_tool_calls[1:]
        )
        return decisions

    reject_message = (
        "The user rejected this tool call from the Human-In-The-Loop UI."
    )
    if user_feedback:
        reject_message = f"{reject_message} User feedback: {user_feedback}"

    return [
        {
            "type": "reject",
            "message": reject_message,
        }
        for _ in original_tool_calls
    ]


def generate_verification_message(message: AIMessage) -> AIMessage:
    """Generate a readable verification message from pending tool calls."""

    serialized_tool_calls = json.dumps(
        message.tool_calls,
        indent=2,
    )
    return AIMessage(
        content=(
            "I plan to invoke the following tools. Please review the tool name "
            "and arguments in the Human-In-The-Loop section.\n"
            f"{serialized_tool_calls}"
        ),
        id=message.id,
    )


def catch_tool_calls(inputs: dict, thread: dict) -> Optional[AIMessage]:
    """Stream the app and return the first AI message with HIL tool calls."""

    messages, _ = stream_hil_agent(inputs, thread)
    for message in reversed(messages):
        if isinstance(message, AIMessage) and extract_hil_tool_calls(messages):
            return message
    return None
