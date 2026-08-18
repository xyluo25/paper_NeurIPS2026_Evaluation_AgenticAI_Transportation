"""
##############################################################
# Created Date: Monday, June 30th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
"""

import os
import re
import shutil
import json
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import gradio as gr
import uuid

from loguru import logger

os.chdir(Path(__file__).parent)

path_tmp = Path(__file__).parent / "proj_tmp_gradio"
if path_tmp.exists():
    shutil.rmtree(path_tmp)
path_tmp.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(path_tmp)

warnings.filterwarnings("ignore")
path_log = Path(__file__).parent.parent / "proj_log/real_twin_sim.log"

logger.add(
    "./proj_log/real_twin_sim.log",
    format="{time} {level} {message}",
    level="INFO",
    rotation="10 MB",
)

logger.info("Agentic Real-Twin is starting...")

path_css = [
    "./chat_css.css",
]
gr.set_static_paths(
    paths=[
        Path(__file__).parent / "proj_tmp_output",
        Path(__file__).parent / "assets",
        Path(__file__).parent / "cache",
        Path(__file__).parent / "proj_config",
        Path(__file__).parent / "proj_llm",
        Path(__file__).parent / "proj_tmp_gradio",
    ]
)

CHAIN_THOUGHT = os.environ.get("REALTWIN_CHAIN_THOUGHT", "").lower() in {
    "1",
    "true",
    "yes",
}
MAX_THOUGHTS_CHARS = 12000
MAX_CONVERSATION_MEMORY_TURNS = 12
MAX_CONVERSATION_MESSAGE_CHARS = 1200
CRITICAL_TOOL_INTENTS = {
    "get_place_info": (
        "get_place_info",
        "openstreetmap metadata",
        "openstreetmap place",
        "place metadata",
        "place information",
        "place info",
        "osm metadata",
        "osm place info",
        "osm place metadata",
        "bounding box",
        "city metadata",
    ),
    "get_osm_from_relation_id": (
        "get_osm_from_relation_id",
        "relation id",
        "relation number",
        "download openstreetmap",
        "download osm",
        "download map data",
        "download map",
        "download osm data",
        "retrieve osm data",
        "retrieve openstreetmap",
    ),
    "get_osm_from_web": (
        "get_osm_from_web",
        "browser wizard",
        "manual download",
        "web wizard",
        "osm web wizard",
        "openstreetmap wizard",
    ),
    "vis_osm": (
        "vis_osm",
        "show openstreetmap network",
        "show osm network",
        "visualize openstreetmap",
        "visualize osm",
        "network visualization",
        "osm network visualization",
    ),
    "install_sumo": (
        "install_sumo",
        "set up sumo",
        "install sumo",
        "install sumo version",
        "set up sumo version",
        "make sure sumo version",
    ),
    "sumo_net_snapshot": (
        "sumo_net_snapshot",
        "png image",
        "png snapshot",
        "network snapshot",
        "snapshot using sumo",
        "visualize sumo network",
        ".net.xml",
    ),
    "realtwin_edit_config": (
        "realtwin_edit_config",
        "edit config",
        "update realtwin configuration",
        "change realtwin setting",
    ),
    "realtwin_save_config": (
        "realtwin_save_config",
        "save realtwin configuration",
    ),
    "realtwin_sample_run": (
        "realtwin_sample_run",
        "sample run",
    ),
    "realtwin_inputs_generation": (
        "realtwin_inputs_generation",
        "generate realtwin inputs",
        "inputs generation",
    ),
    "realtwin_simulation": (
        "realtwin_simulation",
        "realtwin simulation",
        "simulation and calibration",
    ),
}
PREWARM_AGENT_STACK = os.environ.get("REALTWIN_PREWARM_AGENT", "1").lower() not in {
    "0",
    "false",
    "no",
}
pending_hil_tool_calls = []
pending_user_message = ""
_agent_functions: dict[str, Any] | None = None
_agent_functions_lock = threading.Lock()
_hil_tools: list[str] | None = None
_mealpy_optimizer_updated = False
_message_classes: tuple[Any, Any] | None = None
_transcriber: Any = None


def ensure_mealpy_optimizer_updated() -> None:
    """Patch Mealpy optimizer once, only before tools can use it."""

    global _mealpy_optimizer_updated
    if _mealpy_optimizer_updated:
        return

    from proj_util import update_mealpy_optimizer

    update_mealpy_optimizer()
    _mealpy_optimizer_updated = True


def get_agent_functions() -> dict[str, Any]:
    """Lazy-load the DeepAgents stack after the Gradio page is available."""

    global _agent_functions
    if _agent_functions is None:
        with _agent_functions_lock:
            if _agent_functions is not None:
                return _agent_functions

            logger.info("Initializing Agentic RealTwin agent stack...")
            ensure_mealpy_optimizer_updated()
            from chat_bot_supervisor_HIL import (
                extract_hil_tool_calls,
                format_tool_selection,
                generate_review_decisions,
                generate_verification_message,
                resume_hil_agent,
                stream_hil_agent,
            )

            _agent_functions = {
                "extract_hil_tool_calls": extract_hil_tool_calls,
                "format_tool_selection": format_tool_selection,
                "generate_review_decisions": generate_review_decisions,
                "generate_verification_message": generate_verification_message,
                "resume_hil_agent": resume_hil_agent,
                "stream_hil_agent": stream_hil_agent,
            }
            logger.info("Agentic RealTwin agent stack is ready.")
    return _agent_functions


def prewarm_agent_stack() -> None:
    """Initialize the DeepAgents stack in the background after app startup."""

    try:
        get_agent_functions()
    except Exception as exc:
        logger.warning(f"Agent stack prewarm failed: {exc}")

    try:
        from proj_rag import prewarm_rag_resources

        prewarm_rag_resources()
    except Exception as exc:
        logger.warning(f"RAG prewarm failed: {exc}")


def get_hil_tools() -> list[str]:
    """Lazy-load critical tool names for HIL display logic."""

    global _hil_tools
    if _hil_tools is None:
        from proj_tools import HIL_Tools

        _hil_tools = list(HIL_Tools)
    return _hil_tools


def get_message_classes() -> tuple[Any, Any]:
    """Lazy-load LangChain message classes used after submit/resume."""

    global _message_classes
    if _message_classes is None:
        from langchain_core.messages import AIMessage, HumanMessage

        _message_classes = (AIMessage, HumanMessage)
    return _message_classes


def get_audio_transcriber() -> Any:
    """Lazy-load the Whisper pipeline only when an audio file is submitted."""

    global _transcriber
    if _transcriber is not None:
        return _transcriber

    from transformers import pipeline

    try:
        _transcriber = pipeline(
            "automatic-speech-recognition",
            model="./proj_llm/whisper-base-en",
        )
    except Exception:
        _transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en",
        )
        _transcriber.save_pretrained(Path(__file__).parent / "proj_llm/whisper-base-en")

    return _transcriber


def transcribe_audio(audio_file: str) -> str:
    """Transcribe an uploaded audio file into text."""

    from gradio.processing_utils import audio_from_file
    import numpy as np

    print(f"  INFO:audio file: {audio_file}")
    sr, y = audio_from_file(audio_file)

    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    max_amplitude = np.max(np.abs(y))
    if max_amplitude > 0:
        y /= max_amplitude

    return get_audio_transcriber()({"sampling_rate": sr, "raw": y})["text"]


def process_result_message(msg: str) -> str:
    """Process the result message to ensure it is a string.

    input is: result["messages"][-1].content

    """

    if not isinstance(msg, str):
        try:
            return json.dumps(msg, ensure_ascii=False)
        except Exception:
            return str(msg)
    return str(msg)


def format_thoughts_message(messages: Any) -> str:
    """Format the latest agent trace without sending unbounded text to Gradio."""

    if isinstance(messages, dict):
        message_items = messages.get("messages", [])
    else:
        message_items = messages or []

    thoughts = []
    for message in message_items:
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            content = f"{content}\nTool calls: {tool_calls}".strip()
        if not content:
            content = str(message)
        if len(content) > 2000:
            content = f"{content[:2000]}\n...[truncated]"

        thoughts.append(f"Thoughts: {content}")

    if not thoughts:
        return "No agent thoughts for this step."

    thoughts_message = "Latest Agentic Thoughts:\n\n" + "\n\n".join(thoughts)
    if len(thoughts_message) <= MAX_THOUGHTS_CHARS:
        return thoughts_message

    half_limit = MAX_THOUGHTS_CHARS // 2
    return (
        thoughts_message[:half_limit]
        + "\n\n...[middle truncated for UI speed]...\n\n"
        + thoughts_message[-half_limit:]
    )


def append_chat_messages(
    chat_history: list,
    user_message: str | None = None,
    assistant_message: str | None = None,
    file_paths: list[str] | None = None,
) -> list:
    """Return Gradio 6 Chatbot messages with optional user, files, and response."""

    chat_messages = list(chat_history or [])

    if user_message:
        chat_messages.append({"role": "user", "content": user_message})

    for file_path in file_paths or []:
        chat_messages.append(
            {"role": "assistant", "content": {"path": Path(file_path).as_posix()}}
        )

    if assistant_message:
        chat_messages.append({"role": "assistant", "content": assistant_message})

    return chat_messages


def reset(chatbot: list, thoughts_state: str, _hil_selection: gr.Text):
    """Reset chat history, thoughts, HIL review, and session memory."""

    thoughts_state = "To be responded..."

    return (
        [],
        thoughts_state,
        gr.update(value=thoughts_state),
        gr.update(visible=False),
        "Show Agentic Thoughts",
        False,
        gr.update(value=None, visible=True, interactive=False),
        gr.update(value={"text": "", "files": []}, visible=True, interactive=False),
        str(uuid.uuid4()),
        [],
    )


def reset_hil_review_on_submit() -> tuple[dict, dict, dict]:
    """Clear stale HIL tool review controls before a new request runs."""

    return (
        gr.update(value=None, visible=True, interactive=False),
        gr.update(value=None, visible=True, interactive=False),
        gr.update(
            value={"text": "", "files": []},
            visible=True,
            interactive=False,
        ),
    )


def process_user_input(
    msg: str,
    chat_history: list,
    thoughts_state: str,
    conversation_thread_id: str,
    conversation_memory: list,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    request_started = time.perf_counter()
    if CHAIN_THOUGHT:
        print("new round...")
        print("msg............")
        print(f"input:{msg}")
    # start a dialogue here:

    msg_res = "I am Agentic Real-Twin AI, how can I help you?"
    if isinstance(msg, str):
        msg_res = msg.strip()

    elif isinstance(msg, dict):
        # load message directly from input text box
        if msg.get("text"):
            msg_res = msg["text"].strip()

        # load message from uploaded file
        if msg.get("files"):
            file_names = msg["files"]
            for file in file_names:
                # check input is audio.wav or audio.mp3
                if file.endswith(".wav") or file.endswith(".mp3"):
                    msg_res = transcribe_audio(file)
                    print(f"  INFO:transcribe: {msg_res}")
                # check input is image.png or image.jpg
                elif file.endswith(".png") or file.endswith(".jpg"):
                    # read the image file and convert to text
                    # image_text = DTSBot.image_to_text(file)
                    msg_res += f" (Image input received at {file})"
                # check input is video.mp4 or video.avi
                elif file.endswith(".mp4") or file.endswith(".avi"):
                    # read the video file and convert to text
                    # video_text = DTSBot.video_to_text(file)
                    pass
                # check input is csv, json, xml, text, xlsx, or xls
                elif (
                    file.endswith(".csv")
                    or file.endswith(".json")
                    or file.endswith(".xml")
                    or file.endswith(".txt")
                    or file.endswith(".xlsx")
                    or file.endswith(".xls")
                ):
                    # read the file and convert to text
                    # file_text = DTSBot.file_to_text(file)
                    print(
                        "Invalid audio file format. Only .wav and .mp3 are supported."
                    )
                else:
                    print(
                        "Invalid file format. Only .wav, .mp3, .png, .jpg, .mp4, .avi, .csv, .json, .xml, .txt, .xlsx, and .xls are supported."
                    )

    if CHAIN_THOUGHT:
        print(f"  INFO:msg_res: {msg_res}")

    if not conversation_thread_id:
        conversation_thread_id = str(uuid.uuid4())
    conversation_memory = list(conversation_memory or [])
    recent_memory = conversation_memory[-MAX_CONVERSATION_MEMORY_TURNS:]
    memory_lines = []
    for turn_index, memory_turn in enumerate(recent_memory, start=1):
        previous_user = str(memory_turn.get("user", "")).strip()
        previous_assistant = str(memory_turn.get("assistant", "")).strip()
        if len(previous_user) > MAX_CONVERSATION_MESSAGE_CHARS:
            previous_user = (
                previous_user[:MAX_CONVERSATION_MESSAGE_CHARS] + "\n...[truncated]"
            )
        if len(previous_assistant) > MAX_CONVERSATION_MESSAGE_CHARS:
            previous_assistant = (
                previous_assistant[:MAX_CONVERSATION_MESSAGE_CHARS]
                + "\n...[truncated]"
            )
        memory_lines.append(
            f"Turn {turn_index}\nUser: {previous_user}\nAssistant: {previous_assistant}"
        )

    msg_lower = msg_res.lower()
    critical_tool_matches = [
        tool_name
        for tool_name, intent_terms in CRITICAL_TOOL_INTENTS.items()
        if any(intent_term in msg_lower for intent_term in intent_terms)
    ]
    is_critical_tool_request = bool(critical_tool_matches)
    conversation_context = "\n\n".join(memory_lines)
    agent_msg_res = msg_res
    followup_cues = [
        " it",
        " its",
        " they",
        " them",
        " their",
        " that",
        " this",
        " those",
        " these",
        " same",
        " previous",
        " last",
        " above",
        " continue",
        " then",
        " next",
        "what about",
        "how about",
        "how many",
    ]
    question_starts = ("what", "who", "where", "when", "which", "how", "why")
    message_words = msg_lower.split()
    is_followup_like_message = bool(conversation_context) and (
        any(cue in f" {msg_lower}" for cue in followup_cues)
        or (len(message_words) <= 8 and msg_lower.startswith(question_starts))
    )
    if is_critical_tool_request:
        agent_msg_res = (
            "Critical approval-required tool request. Do not answer from "
            "conversation memory, cached prior results, or previous tool outputs. "
            "Extract arguments only from the current user request, call the "
            "requested critical tool, and wait for Human-In-The-Loop approval "
            "before any tool result is produced.\n\n"
            f"Matched critical tools: {', '.join(critical_tool_matches)}\n"
            f"Current user request:\n{msg_res}"
        )
    elif conversation_context and is_followup_like_message:
        agent_msg_res = (
            "Conversation memory from previous turns is provided only to resolve "
            "follow-up references. The current user request remains authoritative.\n\n"
            f"{conversation_context}\n\n"
            f"Current user request:\n{msg_res}"
        )

    memory_lower = conversation_context.lower()
    is_arms_followup_request = (
        (
            "ornl arms" in memory_lower
            or "ornl_arms.txt" in memory_lower
            or "applied research for mobility systems" in memory_lower
        )
        and any(
            term in msg_lower
            for term in [
                "who",
                "how many",
                "member",
                "staff",
                "developer",
                "name",
                "list",
                "email",
                "phone",
                "contact",
                "profile",
                "link",
                "they",
                "them",
                "their",
            ]
        )
    )
    is_sim_parameter_request = (
        ("parameter" in msg_lower or "parameters" in msg_lower or "range" in msg_lower)
        and (
            "simulation" in msg_lower
            or "sumo" in msg_lower
            or "car-following" in msg_lower
            or "car following" in msg_lower
            or "lane-changing" in msg_lower
            or "lane changing" in msg_lower
            or "driving behavior" in msg_lower
            or "behavior" in msg_lower
            or any(
                parameter_name in msg_lower
                for parameter_name in [
                    "min_gap",
                    "min gap",
                    "acceleration",
                    "deceleration",
                    "sigma",
                    "tau",
                    "emergencydecel",
                    "emergency decel",
                ]
            )
        )
    )
    is_realtwin_knowledge_request = (
        ("realtwin" in msg_lower or "real-twin" in msg_lower)
        and any(
            term in msg_lower
            for term in [
                "what",
                "who",
                "developer",
                "develops",
                "team",
                "about",
                "purpose",
                "does",
            ]
        )
        and not any(
            workflow_term in msg_lower
            for workflow_term in [
                "config",
                "configuration",
                "setting",
                "save",
                "update",
                "change",
                "input",
                "run",
                "simulate",
                "simulation",
                "calibration",
                "sample",
            ]
        )
    )
    is_knowledge_base_rag_request = (
        is_arms_followup_request
        or is_realtwin_knowledge_request
        or (
            "ornl" in msg_lower
            and "arms" in msg_lower
            and any(
                term in msg_lower
                for term in ["member", "staff", "developer", "team", "how many"]
            )
        )
    )
    if not is_critical_tool_request and (
        is_sim_parameter_request or is_knowledge_base_rag_request
    ):
        try:
            if is_sim_parameter_request:
                from proj_rag import rag_tool_sim_parameters

                tool_name = "rag_tool_sim_parameters"
                direct_bot_response = rag_tool_sim_parameters.invoke(
                    {"question": msg_res}
                )
                direct_bot_response = direct_bot_response.split(
                    "\n\nFull local source text for accuracy:",
                    maxsplit=1,
                )[0]
            else:
                from proj_rag import rag_tool

                tool_name = "rag_tool"
                direct_question = msg_res
                if is_arms_followup_request and (
                    "ornl" not in msg_lower or "arms" not in msg_lower
                ):
                    direct_question = (
                        "This is a follow-up question about ORNL ARMS staff from "
                        f"the previous answer. Current question: {msg_res}"
                    )
                direct_bot_response = rag_tool.invoke({"question": direct_question})

            logger.info(
                "Direct local RAG request completed in "
                f"{time.perf_counter() - request_started:.2f}s; tool={tool_name}"
            )
            updated_conversation_memory = (
                conversation_memory + [{"user": msg_res, "assistant": direct_bot_response}]
            )[-MAX_CONVERSATION_MEMORY_TURNS:]
            thoughts_msg_response = (
                "Latest Agentic Thoughts:\n\n"
                f"Thoughts: Routed directly to `{tool_name}` for a local "
                "knowledge-base answer. No approval-required tool was needed."
            )
            return (
                "",
                append_chat_messages(
                    chat_history,
                    user_message=msg_res,
                    assistant_message=direct_bot_response,
                ),
                thoughts_msg_response,
                thoughts_msg_response,
                gr.update(value=None, visible=True, interactive=False),
                gr.update(value=None, visible=True, interactive=False),
                gr.update(
                    value={"text": "", "files": []},
                    visible=True,
                    interactive=False,
                ),
                conversation_thread_id,
                updated_conversation_memory,
            )
        except Exception as exc:
            logger.warning(f"Direct local RAG path failed; using DeepAgents: {exc}")

    global thread
    thread = {"configurable": {"thread_id": conversation_thread_id}}
    AIMessage, HumanMessage = get_message_classes()
    user_message = [HumanMessage(content=agent_msg_res)]
    agent_functions = get_agent_functions()

    global pending_user_message
    pending_user_message = msg_res

    global bot_response
    bot_response = "Tool approval is required before I continue."
    bot_response_Message = None
    bot_result, is_interrupted = agent_functions["stream_hil_agent"](
        {"messages": user_message}, thread
    )
    hil_tool_calls = agent_functions["extract_hil_tool_calls"](bot_result)
    global pending_hil_tool_calls
    pending_hil_tool_calls = hil_tool_calls if is_interrupted else []

    logger.info(
        "Agent request completed in "
        f"{time.perf_counter() - request_started:.2f}s; "
        f"messages={len(bot_result) if isinstance(bot_result, list) else 'unknown'}; "
        f"interrupted={is_interrupted}"
    )
    if CHAIN_THOUGHT:
        logger.info(format_thoughts_message(bot_result))

    if is_critical_tool_request and not is_interrupted:
        blocked_response = (
            "This request requires a Human-In-The-Loop tool review. I did not "
            "use prior conversation memory or previous tool results, and no tool "
            "output was produced because the approval interrupt was not generated."
        )
        thoughts_msg_response = (
            "Latest Agentic Thoughts:\n\n"
            "Thoughts: Critical tool request detected, but no approval interrupt "
            "was returned. The generated response was blocked to prevent bypassing "
            "Human-In-The-Loop review."
        )
        logger.warning(
            "Blocked critical tool response without HIL interrupt; "
            f"matched_tools={critical_tool_matches}"
        )
        return (
            "",
            append_chat_messages(
                chat_history,
                user_message=msg_res,
                assistant_message=blocked_response,
            ),
            thoughts_msg_response,
            thoughts_msg_response,
            gr.update(value=None, visible=True, interactive=False),
            gr.update(value=None, visible=True, interactive=False),
            gr.update(value={"text": "", "files": []}, visible=True, interactive=False),
            conversation_thread_id,
            conversation_memory,
        )

    try:
        # Get Tool info to update on HIL section
        has_hil_review = False
        tool_name = ""
        hil_selection_C = gr.update(value=None, visible=True, interactive=False)
        hil_confirm_C = gr.update(value=None, visible=True, interactive=False)
        if is_interrupted and hil_tool_calls:
            has_hil_review = True
            tool_name = hil_tool_calls[0]["name"]
            tool_selection_text = agent_functions["format_tool_selection"](
                hil_tool_calls
            )

            hil_selection_C = gr.update(
                value=tool_selection_text,
                visible=True,
                interactive=True,
            )
            hil_confirm_C = gr.update(
                value=None,
                visible=True,
                interactive=True,
            )
    except Exception as e:
        print(f"  ERROR:Failed to process tool calls: {e}")
        has_hil_review = False
        tool_name = ""
        hil_selection_C = gr.update(value=None, visible=True, interactive=False)
        hil_confirm_C = gr.update(value=None, visible=True, interactive=False)

    try:
        # Get the last AIMessage (Not Transfer Message) from the bot result
        for Message in bot_result[::-1]:
            if isinstance(Message, AIMessage):
                content_message = Message.content
                # content_message is not empty and does not contain "transfer"
                if content_message and "transfer" not in content_message.lower():
                    bot_response = Message.content
                    bot_response_Message = Message
                    break

    except Exception:
        print("  ERROR:Failed to process bot response, using last message content.")
        bot_response = process_result_message(bot_result[-1].content)

    if CHAIN_THOUGHT:
        print(f"Dialog response: {bot_response}")

    # extract filename from the response
    regex_lst = [
        re.compile(r"`([^`]+)`"),
        re.compile(r"\((.*?)\)"),
    ]

    filenames = None

    try:
        filenames_extract = [regex.findall(bot_response) for regex in regex_lst]
        # print(f"  INFO:filenames before: {filenames_extract}")
        for fname_lst in filenames_extract:
            # make sure the filename is a list of strings and not empty
            if fname_lst:
                try:
                    if Path(fname_lst[0]).exists():
                        filenames = fname_lst
                        break
                except Exception as e:
                    print(f"  ERROR:Failed to process filename {fname_lst[0]}: {e}")
                    filenames = None

    except AttributeError:
        filenames = None

    # print(f"  INFO:filenames after: {filenames}")
    if filenames:
        chat_history_response = append_chat_messages(
            chat_history,
            user_message=msg_res,
            file_paths=[Path(fname).as_posix() for fname in filenames],
            assistant_message=bot_response,
        )
    else:
        chat_history_response = append_chat_messages(
            chat_history,
            user_message=msg_res,
            assistant_message=bot_response,
        )

    # update the thoughts message from memory from langgraph
    thoughts_msg_response = format_thoughts_message(bot_result)

    try:
        global verification_message
        if bot_response_Message is not None and is_interrupted and hil_tool_calls:
            verification_message = agent_functions["generate_verification_message"](
                bot_response_Message
            )
            if CHAIN_THOUGHT:
                verification_message.pretty_print()
        else:
            verification_message = HumanMessage(
                content="Please confirm the tool and arguments."
            )
    except Exception as e:
        print(f"  ERROR:Failed to generate verification message: {e}")
        verification_message = HumanMessage(
            content="Please confirm the tool and arguments."
        )

    if CHAIN_THOUGHT:
        print("hil_selection_C: ", hil_selection_C)

    # Tool have been called, with hil implemented
    if has_hil_review:
        if tool_name in get_hil_tools():
            chat_history_response = chat_history
            thoughts_msg_response = thoughts_state

            if tool_name in ["realtwin_inputs_generation", "realtwin_simulation"]:
                path_user_input = Path(__file__).parent / "datasets/User_Input"
                # check if the folder exists
                if path_user_input.exists():
                    # check if the folder is empty
                    if not any(path_user_input.iterdir()):
                        pass
                else:
                    path_user_input = Path(__file__).parent / "datasets/example2"

                gr.Warning(
                    message=f"""<h1 style="color: blue;">Please prepare the <b>Control</b> and <b>Traffic</b>
                    data and fill in the <b>Matchup Table</b> in folder: <i>{path_user_input}</i>."</h1>
                            <br>
                            <p style="color: red;">Please confirm the tool and arguments in <b>Human-In-The-Loop</b> Section</p>
                            <br>
                            <p style="color: green;">For more information, please refer to:
                            <a href="https://real-twin.readthedocs.io/en/latest/index.html#">Real-Twin Documentation</a></p>
                            """,
                    duration=None,
                )

            else:
                gr.Info(
                    message=f"""<p style="opacity: 1; background-color: white;">Tool {tool_name} is called,
                    please confirm the tool and arguments from <b>Human-In-Loop</b> Section.</p>
                    """,
                    duration=15,
                )

        return (
            "",
            chat_history_response,
            thoughts_msg_response,
            thoughts_msg_response,
            # show the selection for user
            hil_selection_C,
            # show confirmation radio
            hil_confirm_C,
            gr.update(value={"text": "", "files": []}, visible=True, interactive=False),
            conversation_thread_id,
            conversation_memory,
        )

    # No HIL Needed
    else:
        updated_conversation_memory = (
            conversation_memory + [{"user": msg_res, "assistant": bot_response}]
        )[-MAX_CONVERSATION_MEMORY_TURNS:]
        return (
            "",
            chat_history_response,
            thoughts_msg_response,
            thoughts_msg_response,
            # show the selection for user
            gr.update(value=None, visible=True, interactive=False),
            # show confirmation radio
            gr.update(value=None, visible=True, interactive=False),
            gr.update(value={"text": "", "files": []}, visible=True, interactive=False),
            conversation_thread_id,
            updated_conversation_memory,
        )


def process_hil_input(
    tool_selection,
    tool_confirm,
    chat_history,
    thoughts_state,
    conversation_thread_id,
    conversation_memory,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    request_started = time.perf_counter()
    global pending_hil_tool_calls

    if not tool_confirm:
        return (
            gr.update(),
            gr.update(
                value=None,
                visible=True,
                interactive=True,
            ),
            gr.update(value={"text": "", "files": []}, visible=True, interactive=False),
            chat_history,
            thoughts_state,
            thoughts_state,
            conversation_memory,
        )

    if tool_confirm == "No":
        gr.Info(
            message=(
                "Please provide the corrected parameters or instructions in the "
                "Human-In-The-Loop feedback box, then submit it to continue."
            ),
            duration=8,
        )
        return (
            gr.update(),
            gr.update(
                value="No",
                visible=True,
                interactive=True,
            ),
            gr.update(
                value={"text": "", "files": []},
                placeholder=(
                    "Correct the tool arguments above, or tell the agent what "
                    "to change before continuing..."
                ),
                interactive=True,
                visible=True,
            ),
            chat_history,
            thoughts_state,
            thoughts_state,
            conversation_memory,
        )

    AIMessage, _ = get_message_classes()
    agent_functions = get_agent_functions()

    try:
        thread_config = {"configurable": {"thread_id": conversation_thread_id}}
        decisions = agent_functions["generate_review_decisions"](
            tool_confirm=tool_confirm,
            tool_selection=tool_selection or "",
            original_tool_calls=pending_hil_tool_calls,
        )
        bot_result, is_interrupted = agent_functions["resume_hil_agent"](
            decisions, thread_config
        )
        pending_hil_tool_calls = (
            agent_functions["extract_hil_tool_calls"](bot_result)
            if is_interrupted
            else []
        )
    except Exception as e:
        bot_result = [
            AIMessage(content=f"Error while resuming the approved tool call: {e}")
        ]
        is_interrupted = False
        pending_hil_tool_calls = []

    logger.info(
        "HIL resume completed in "
        f"{time.perf_counter() - request_started:.2f}s; "
        f"messages={len(bot_result) if isinstance(bot_result, list) else 'unknown'}; "
        f"interrupted={is_interrupted}"
    )

    bot_response = "The tool review was processed."
    try:
        for Message in bot_result[::-1]:
            if isinstance(Message, AIMessage):
                content_message = Message.content
                if content_message and "transfer" not in content_message.lower():
                    bot_response = Message.content
                    break
    except Exception:
        print("  ERROR:Failed to process bot response, using last message content.")
        if bot_result:
            bot_response = process_result_message(bot_result[-1].content)

    thoughts_msg_response = format_thoughts_message(bot_result)

    hil_selection_response = gr.update(
        value=None,
        visible=True,
        interactive=False,
    )
    hil_confirm_response = gr.update(
        value=tool_confirm,
        visible=True,
        interactive=False,
    )
    if is_interrupted:
        bot_response = (
            "Another approval-required tool call is pending. Please submit the "
            "current review before continuing."
        )
        hil_tool_calls = agent_functions["extract_hil_tool_calls"](bot_result)
        if hil_tool_calls:
            hil_selection_response = gr.update(
                value=agent_functions["format_tool_selection"](hil_tool_calls),
                visible=True,
                interactive=True,
            )
            hil_confirm_response = gr.update(
                value=None,
                visible=True,
                interactive=True,
            )

    chat_user_message = pending_user_message or ""
    chat_history_response = append_chat_messages(
        chat_history,
        user_message=chat_user_message,
        assistant_message=bot_response,
    )
    updated_conversation_memory = (
        list(conversation_memory or [])
        + [{"user": chat_user_message, "assistant": bot_response}]
    )[-MAX_CONVERSATION_MEMORY_TURNS:]

    return (
        hil_selection_response,
        hil_confirm_response,
        gr.update(value={"text": "", "files": []}, visible=True, interactive=False),
        chat_history_response,
        thoughts_msg_response,
        thoughts_msg_response,
        updated_conversation_memory,
    )


def process_hil_feedback(
    hil_feedback,
    tool_selection,
    chat_history,
    thoughts_state,
    conversation_thread_id,
    conversation_memory,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """Resume a rejected HIL tool call only after user feedback is submitted."""

    request_started = time.perf_counter()
    global pending_hil_tool_calls
    feedback_text = ""
    feedback_files = []
    if isinstance(hil_feedback, str):
        feedback_text = hil_feedback.strip()
    elif isinstance(hil_feedback, dict):
        feedback_text = str(hil_feedback.get("text") or "").strip()
        feedback_files = list(hil_feedback.get("files") or [])

    if feedback_files:
        feedback_text = (
            f"{feedback_text}\nUploaded files: "
            + ", ".join(str(file_path) for file_path in feedback_files)
        ).strip()

    edited_action = None
    try:
        agent_functions = get_agent_functions()
        edited_action = agent_functions["generate_review_decisions"](
            tool_confirm="No",
            tool_selection=tool_selection or "",
            original_tool_calls=pending_hil_tool_calls,
            user_feedback=feedback_text,
        )
    except TypeError:
        agent_functions = get_agent_functions()
        edited_action = agent_functions["generate_review_decisions"](
            tool_confirm="No",
            tool_selection=tool_selection or "",
            original_tool_calls=pending_hil_tool_calls,
        )

    if not pending_hil_tool_calls:
        gr.Warning(
            message="No pending Human-In-The-Loop tool call was found.",
            duration=8,
        )
        return (
            gr.update(),
            gr.update(),
            gr.update(
                value=hil_feedback,
                placeholder="No pending tool call was found. Submit a new request.",
                interactive=True,
                visible=True,
            ),
            chat_history,
            thoughts_state,
            thoughts_state,
            conversation_memory,
        )

    if not feedback_text and edited_action and all(
        decision.get("type") == "reject" for decision in edited_action
    ):
        gr.Warning(
            message=(
                "Please describe what is wrong, or edit the tool arguments before "
                "submitting feedback."
            ),
            duration=8,
        )
        return (
            gr.update(),
            gr.update(
                value="No",
                visible=True,
                interactive=True,
            ),
            gr.update(
                value=hil_feedback,
                placeholder="Describe the correction or edit the tool arguments above.",
                interactive=True,
                visible=True,
            ),
            chat_history,
            thoughts_state,
            thoughts_state,
            conversation_memory,
        )

    AIMessage, _ = get_message_classes()
    try:
        thread_config = {"configurable": {"thread_id": conversation_thread_id}}
        bot_result, is_interrupted = agent_functions["resume_hil_agent"](
            edited_action, thread_config
        )
        pending_hil_tool_calls = (
            agent_functions["extract_hil_tool_calls"](bot_result)
            if is_interrupted
            else []
        )
    except Exception as e:
        bot_result = [
            AIMessage(content=f"Error while resuming with HIL feedback: {e}")
        ]
        is_interrupted = False
        pending_hil_tool_calls = []

    logger.info(
        "HIL feedback resume completed in "
        f"{time.perf_counter() - request_started:.2f}s; "
        f"messages={len(bot_result) if isinstance(bot_result, list) else 'unknown'}; "
        f"interrupted={is_interrupted}"
    )

    bot_response = "The tool review feedback was processed."
    try:
        for Message in bot_result[::-1]:
            if isinstance(Message, AIMessage):
                content_message = Message.content
                if content_message and "transfer" not in content_message.lower():
                    bot_response = Message.content
                    break
    except Exception:
        print("  ERROR:Failed to process bot response, using last message content.")
        if bot_result:
            bot_response = process_result_message(bot_result[-1].content)

    thoughts_msg_response = format_thoughts_message(bot_result)
    hil_selection_response = gr.update(
        value=None,
        visible=True,
        interactive=False,
    )
    hil_confirm_response = gr.update(
        value="No",
        visible=True,
        interactive=False,
    )
    if is_interrupted:
        bot_response = (
            "Another approval-required tool call is pending. Please submit the "
            "current review before continuing."
        )
        hil_tool_calls = agent_functions["extract_hil_tool_calls"](bot_result)
        if hil_tool_calls:
            hil_selection_response = gr.update(
                value=agent_functions["format_tool_selection"](hil_tool_calls),
                visible=True,
                interactive=True,
            )
            hil_confirm_response = gr.update(
                value=None,
                visible=True,
                interactive=True,
            )

    chat_user_message = pending_user_message or ""
    feedback_message = feedback_text or "The user edited the tool arguments."
    chat_history_response = append_chat_messages(
        chat_history,
        user_message=f"{chat_user_message}\n\nHIL feedback: {feedback_message}",
        assistant_message=bot_response,
    )
    updated_conversation_memory = (
        list(conversation_memory or [])
        + [{"user": chat_user_message, "assistant": bot_response}]
    )[-MAX_CONVERSATION_MEMORY_TURNS:]

    return (
        hil_selection_response,
        hil_confirm_response,
        gr.update(value={"text": "", "files": []}, visible=True, interactive=False),
        chat_history_response,
        thoughts_msg_response,
        thoughts_msg_response,
        updated_conversation_memory,
    )


def save_uploaded_file(file):

    print("file type: ", type(file))
    print("file uploaded: ", file.name)
    shutil.copyfile(file.name, "./temp_input/temp_uploaded_file.csv")
    print("file saved as: ", "./temp_input/temp_uploaded_file.csv")
    return gr.UploadButton(
        label="File Uploaded",
        scale=None,
        elem_id="file_layout",
        file_count="single",
        file_types=[".csv"],
    )


# Accept the event argument, even if not used
def update_thoughts_visibility(
    t_button: str,
    thoughts_state: str,
    thoughts_popup_visible: bool,
) -> list[Any]:
    """Show or hide the popup window with the latest agent trace."""

    if t_button == "Show Agentic Thoughts" or not thoughts_popup_visible:
        print("Show thoughts button clicked")
        return [
            "Hide Agentic Thoughts",
            gr.update(visible=True),
            gr.update(value=thoughts_state),
            True,
        ]

    print("Hide thoughts button clicked")
    return [
        "Show Agentic Thoughts",
        gr.update(visible=False),
        gr.update(value=thoughts_state),
        False,
    ]


def close_thoughts_popup(thoughts_state: str) -> list[Any]:
    """Hide the agent-thoughts popup from its close button."""

    return [
        "Show Agentic Thoughts",
        gr.update(visible=False),
        gr.update(value=thoughts_state),
        False,
    ]


with gr.Blocks(
    title="RealTwin AI",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.pink
    ),
    css_paths=path_css,
) as demo:
    # define the GUI framework:
    with gr.Row(variant="panel", elem_id="title_header_layout"):
        gr.Label(
            label="",
            show_label=False,
            container=False,
            value=(
                "Agentic Traffic Intelligence: Augmented Human-In-The-Loop Scenario Generation"
                " for Microscopic Traffic Simulation"
            ),
            elem_id="title_header",
        )

    with gr.Row(visible=True, variant="panel", elem_id="main_layout", equal_height=True):
        with gr.Column(
            visible=True,
            variant="default",
            min_width=280,
            scale=1,
            elem_id="left_layout",
        ):
            with gr.Column(
                visible=True,
                variant="panel",
                scale=2,
                elem_id="input_container",
                min_width=280,
            ):
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    label="Prompt or question",
                    file_count="multiple",
                    placeholder="Enter message or upload file...",
                    show_label=False,
                    sources=["microphone", "upload"],
                    lines=1,
                    scale=1,
                    autofocus=True,
                    # show_copy_button=True,
                    elem_id="input_text_layout",
                    # stop_btn="stop"
                )

                clearBtn = gr.ClearButton(scale=1, elem_id="clear_button")

            gr.Markdown(
                "### Human-In-The-Loop: Tool Selection and Actions",
                elem_id="hil_title",
            )
            # For Human-In-Loop (HIL) confirmation and feedback
            # with gr.Row(variant="panel", elem_id="hil_layout", scale=1):
            with gr.Column(variant="panel", elem_id="hil_layout", scale=1):
                gr.Markdown(
                    (
                        "**Please confirm the tool and arguments. Click Yes or No.**\n\n"
                        "Notes:\n"
                        "1. Select No for Wrong Tool.\n"
                        "2. Select No and edit args for Wrong args."
                    ),
                    elem_id="hil_guidance",
                )
                hil_selection = gr.Textbox(
                    elem_id="hil_selection",
                    value=None,
                    visible=True,
                    interactive=False,
                    label="Tool Selection",
                )
                hil_confirm = gr.Radio(
                    choices=["Yes", "No"],
                    visible=True,
                    elem_id="hil_confirm",
                    interactive=False,
                    label="Human Confirmation",
                )
                hil_feedback = gr.MultimodalTextbox(
                    value={"text": "", "files": []},
                    placeholder=(
                        "Correct the tool arguments above, or tell the agent "
                        "what to change before continuing..."
                    ),
                    interactive=False,
                    label="",
                    show_label=False,
                    sources=[],
                    visible=True,
                    elem_id="hil_feedback",
                )

            gr.Examples(
                label="Question Hints",
                elem_id="example_layout",
                examples=[
                    "Explain what a traffic digital twin is for transportation simulation.",
                    (
                        "What does the ORNL ARMS group do, and how many people "
                        "are listed in the local knowledge base?"
                    ),
                    (
                        "Recommend safe starting ranges for car-following and "
                        "lane-changing behavior parameters in a SUMO simulation."
                    ),
                    "Find OpenStreetMap metadata for Knoxville, TN.",
                    (
                        "Open the browser wizard for manually downloading "
                        "OpenStreetMap data."
                    ),
                    "Download OpenStreetMap data for relation ID 196150.",
                    (
                        "For Knoxville, TN, download the OpenStreetMap network "
                        "and show me a visualization."
                    ),
                    (
                        "Show the network visualization from the already downloaded "
                        "OpenStreetMap folder."
                    ),
                    "Check whether SUMO is installed and report the detected version.",
                    "Install SUMO using the default supported version.",
                    "Make sure SUMO version 1.21.0 is installed.",
                    (
                        "Create a PNG image of the SUMO network file "
                        "datasets/example2/output/SUMO/chatt.net.xml."
                    ),
                    "Show the default RealTwin settings.",
                    "Show the current RealTwin settings.",
                    "Change the RealTwin Traffic setting to VERY Well.",
                    "Save the current RealTwin settings for user editing.",
                    "Run the RealTwin sample scenario with the default configuration.",
                    "Prepare RealTwin input files from path_config.yaml.",
                    (
                        "Run the RealTwin simulation and calibration with the "
                        "current settings."
                    ),
                ],
                inputs=[chat_input],
                examples_per_page=4,
            )

            with gr.Column(variant="panel", elem_id="thoughts_panel"):
                thoughtsBtn = gr.Button(
                    "Show Agentic Thoughts",
                    elem_id="thoughts_button",
                )

        chatbot = gr.Chatbot(
            elem_id="right_layout",
            label="Agentic Response",
            scale=2,
            min_height="60vh",
            layout="bubble",
            buttons=[],
            editable=None,
            resizable=True,
            avatar_images=("assets/user.png", "assets/ai_assistant.png"),
            watermark="Agentic Real-Twin Developed by ORNL ARMS Team",
            placeholder="To be responded...",
        )

    with gr.Column(visible=False, elem_id="thoughts_popup") as thoughtsPopup:
        with gr.Row(elem_id="thoughts_popup_header"):
            gr.Markdown("### Agentic Thoughts", elem_id="thoughts_popup_title")
            thoughtsCloseBtn = gr.Button("Close", elem_id="thoughts_popup_close")
        thoughtsPopupText = gr.Textbox(
            label="",
            value="To be responded...",
            interactive=False,
            lines=20,
            max_lines=20,
            elem_id="thoughts_popup_message",
            buttons=["copy"],
        )

    thoughtsState = gr.State(value="To be responded...")
    thoughtsPopupVisible = gr.State(value=False)
    conversation_thread_id = gr.State(value=str(uuid.uuid4()))
    conversation_memory = gr.State(value=[])

    # Event handlers
    hil_reset_event = chat_input.submit(
        reset_hil_review_on_submit,
        inputs=None,
        outputs=[hil_selection, hil_confirm, hil_feedback],
        show_progress="hidden",
    )

    chat_msg = hil_reset_event.then(
        process_user_input,
        [
            chat_input,
            chatbot,
            thoughtsState,
            conversation_thread_id,
            conversation_memory,
        ],
        [
            chat_input,
            chatbot,
            thoughtsState,
            thoughtsPopupText,
            hil_selection,
            hil_confirm,
            hil_feedback,
            conversation_thread_id,
            conversation_memory,
        ],
        show_progress_on=[chatbot, hil_selection],  # chat_input,
        show_progress="minimal",
    )

    clearBtn.click(
        reset,
        [chatbot, thoughtsState, hil_selection],
        [
            chatbot,
            thoughtsState,
            thoughtsPopupText,
            thoughtsPopup,
            thoughtsBtn,
            thoughtsPopupVisible,
            hil_selection,
            hil_feedback,
            conversation_thread_id,
            conversation_memory,
        ],
        cancels=[hil_reset_event, chat_msg],  # cancel ongoing processing
    )

    thoughtsBtn.click(
        update_thoughts_visibility,
        [thoughtsBtn, thoughtsState, thoughtsPopupVisible],
        [thoughtsBtn, thoughtsPopup, thoughtsPopupText, thoughtsPopupVisible],
        show_progress="hidden",
    )

    thoughtsCloseBtn.click(
        close_thoughts_popup,
        [thoughtsState],
        [thoughtsBtn, thoughtsPopup, thoughtsPopupText, thoughtsPopupVisible],
        show_progress="hidden",
    )

    hil_confirm.change(
        fn=process_hil_input,
        inputs=[
            hil_selection,
            hil_confirm,
            chatbot,
            thoughtsState,
            conversation_thread_id,
            conversation_memory,
        ],
        outputs=[
            hil_selection,
            hil_confirm,
            hil_feedback,
            chatbot,
            thoughtsState,
            thoughtsPopupText,
            conversation_memory,
        ],
        show_progress_on=[hil_confirm],
    )

    hil_feedback.submit(
        fn=process_hil_feedback,
        inputs=[
            hil_feedback,
            hil_selection,
            chatbot,
            thoughtsState,
            conversation_thread_id,
            conversation_memory,
        ],
        outputs=[
            hil_selection,
            hil_confirm,
            hil_feedback,
            chatbot,
            thoughtsState,
            thoughtsPopupText,
            conversation_memory,
        ],
        show_progress_on=[hil_feedback, hil_confirm],
    )

if __name__ == "__main__":
    if PREWARM_AGENT_STACK:
        threading.Thread(target=prewarm_agent_stack, daemon=True).start()

    # Add C: to allowed_paths
    demo.launch(
        inbrowser=True,
        debug=False,
        allowed_paths=[
            str((Path(__file__).parent.parent).resolve()),
            Path(__file__).parent.parent,
        ],
    )
