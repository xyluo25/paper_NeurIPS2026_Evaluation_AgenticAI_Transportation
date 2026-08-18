"""
##############################################################
# Created Date: Friday, June 20th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
"""

import os
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_PERSIST_DIR = PACKAGE_DIR / "chroma_store"
LOCAL_CACHE_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
PERSIST_DIR = Path(
    os.environ.get(
        "REALTWIN_CHROMA_STORE",
        LOCAL_CACHE_ROOT / "Agentic_RealTwin" / "chroma_store",
    )
)
DOCS_FOLDER = PACKAGE_DIR / "rag_datasets"
_retriever: Any = None
_agent_rag: Any = None
_source_text_cache: dict[Path, str] = {}

try:
    RAG_TOP_K = max(1, int(os.environ.get("REALTWIN_RAG_TOP_K", "12")))
except ValueError:
    RAG_TOP_K = 12


def get_proj_llm_object(name: str) -> Any:
    """Return a lazily initialized object from proj_llm."""

    try:
        from .. import proj_llm
    except Exception:
        import sys

        path_llm = Path(__file__).parent.parent
        sys.path.append(str(path_llm))
        import proj_llm

    return getattr(proj_llm, name)


def load_documents(folder_path: str | Path) -> list[Any]:
    """Load local text documents and split them for vector storage."""

    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    for filepath in Path(folder_path).iterdir():
        if filepath.suffix.lower() == ".pdf":
            continue
        if filepath.suffix.lower() != ".txt":
            continue

        print(f"  :Loading file: {filepath.name}")
        loader = TextLoader(str(filepath))
        raw_documents = loader.load()
        splitted_documents = text_splitter.split_documents(raw_documents)
        documents.extend(splitted_documents)
    return documents


def get_retriever() -> Any:
    """Open the RAG vector store on first retrieval request."""

    global _retriever
    if _retriever is not None:
        return _retriever

    import shutil

    from langchain_chroma import Chroma

    embeddings = get_proj_llm_object("embeddings_openai")

    # Copy the committed Chroma store into a runtime cache before opening it.
    # Newer Chroma versions rewrite index files on load, so opening the tracked
    # store directly creates noisy binary diffs during normal app usage.
    if not PERSIST_DIR.exists() or not any(PERSIST_DIR.iterdir()):
        if SOURCE_PERSIST_DIR.exists() and any(SOURCE_PERSIST_DIR.iterdir()):
            if PERSIST_DIR.exists():
                shutil.rmtree(PERSIST_DIR)
            PERSIST_DIR.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SOURCE_PERSIST_DIR, PERSIST_DIR)

    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        vector_store = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
        )
        doc_count = len(vector_store.get().get("ids", []))
    else:
        documents = load_documents(DOCS_FOLDER)
        vector_store = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=str(PERSIST_DIR),
        )
        doc_count = len(documents)

    _retriever = vector_store.as_retriever(
        search_kwargs={"k": max(1, min(RAG_TOP_K, doc_count))}
    )
    return _retriever


@tool
def rag_tool(question: str) -> str:
    """Answer RealTwin and development-team questions from local documents."""

    question_lower = question.lower()
    is_arms_member_question = (
        "ornl" in question_lower
        and "arms" in question_lower
        and any(
            term in question_lower
            for term in ["member", "staff", "developer", "team", "how many"]
        )
    )
    if is_arms_member_question:
        arms_file = DOCS_FOLDER / "ORNL_ARMS.txt"
        arms_text = _source_text_cache.get(arms_file)
        if arms_text is None:
            arms_text = arms_file.read_text(encoding="utf-8")
            _source_text_cache[arms_file] = arms_text

        group_summary = arms_text.split("Current staff members in ARMS group:")[0]
        staff_records = []
        for staff_block in arms_text.split("\n\n"):
            staff_name = re.search(r'^"Name":\s*"([^"]+)"', staff_block, re.MULTILINE)
            if not staff_name:
                continue

            staff_email = re.search(
                r'^"Email":\s*"([^"]*)"', staff_block, re.MULTILINE
            )
            staff_phone = re.search(
                r'^"Phone":\s*"([^"]*)"', staff_block, re.MULTILINE
            )
            staff_profile = re.search(
                r'^"PersonPageLink":\s*"([^"]*)"', staff_block, re.MULTILINE
            )
            staff_records.append(
                {
                    "name": staff_name.group(1),
                    "email": staff_email.group(1) if staff_email else "",
                    "phone": staff_phone.group(1) if staff_phone else "",
                    "profile": staff_profile.group(1) if staff_profile else "",
                }
            )

        include_email = any(term in question_lower for term in ["email", "contact"])
        include_phone = any(term in question_lower for term in ["phone", "contact"])
        include_profile = any(
            term in question_lower for term in ["profile", "page", "link", "url"]
        )
        member_lines = []
        for staff_record in staff_records:
            line_parts = [staff_record["name"]]
            if include_email:
                line_parts.append(f"Email: {staff_record['email']}")
            if include_phone:
                line_parts.append(f"Phone: {staff_record['phone']}")
            if include_profile:
                line_parts.append(f"Profile: {staff_record['profile']}")
            member_lines.append("- " + "; ".join(line_parts))

        return (
            f"{group_summary.strip()}\n\n"
            f"Current staff members listed in {arms_file.name}: "
            f"{len(staff_records)}\n"
            f"{chr(10).join(member_lines)}"
        )

    retrieved_documents = get_retriever().invoke(question)
    return "\n\n".join(
        getattr(document, "page_content", str(document))
        for document in retrieved_documents
    )


@tool
def rag_tool_sim_parameters(question: str) -> str:
    """Answer traffic-simulation parameter recommendation questions."""

    question_lower = question.lower()
    parameter_terms = [
        "parameter",
        "range",
        "recommend",
        "min_gap",
        "min gap",
        "acceleration",
        "deceleration",
        "sigma",
        "tau",
        "emergencydecel",
        "emergency decel",
        "lane-changing",
        "lane changing",
        "car-following",
        "car following",
    ]
    if any(term in question_lower for term in parameter_terms):
        parameters_file = DOCS_FOLDER / "transportation_simulation_parameters_cleaned.txt"
        parameters_text = _source_text_cache.get(parameters_file)
        if parameters_text is None:
            parameters_text = parameters_file.read_text(encoding="utf-8")
            _source_text_cache[parameters_file] = parameters_text

        return (
            "Source: transportation_simulation_parameters_cleaned.txt\n\n"
            "Recommended parameter evidence from the local knowledge base:\n"
            "- min_gap: The source does not label one field exactly as min_gap. "
            "Use the closest model-specific evidence: VISSIM lane-change min "
            "headway 1.5-2 ft, network calibration min headway 0.1-1.0 m "
            "(Arlington), 0.1-0.9 m (Covington), 0.75-1.25 m adjusted "
            "(Charlottesville), priority min headway 5-20 m with min gap time "
            "3-6 s, and Krauss/IDM d_min values of 2.5 m / 2.0 m.\n"
            "- acceleration: Relevant acceleration evidence includes W99 CC7 "
            "0-1.0 m/s^2, W99 CC8 1.0-8.0 m/s^2 from standstill, and W99 "
            "CC9 0.5-3.0 m/s^2 at 80 km/h or 50 mph.\n"
            "- deceleration: Relevant ranges include VISSIM max deceleration "
            "-5.0 to -1.0 m/s^2, accepted deceleration -3.0 to -0.2 m/s^2, "
            "AccDecelOwn -10 to 0 m/s^2, CoopDecel -10 to 0 m/s^2, and "
            "MaxDecelOwn -10 to -0.01 m/s^2. Keep the target simulator's sign "
            "convention when entering values.\n"
            "- sigma: The Krauss calibration result lists sigma = 0.55; the "
            "IDM case is marked n.a. in the source.\n"
            "- tau: Krauss/IDM results list tau = 1.0 s / 0.8 s. Related "
            "headway-time evidence includes W99 CC1 0.85-1.05 s, with broader "
            "W99cc1 bounds of 0.5-3.0 s.\n"
            "- emergencyDecel: The source does not provide a field named "
            "emergencyDecel. Use related safety/emergency evidence only when "
            "mapping to a simulator-specific field: emergency stop distance "
            "6.56-30.0 ft, maximum deceleration own/trailing -20 to -3 ft/s^2, "
            "CoopDecel -10 to 0 m/s^2, and MaxDecelOwn -10 to -0.01 m/s^2.\n\n"
            "Full local source text for accuracy:\n"
            f"{parameters_text.strip()}"
        )

    retrieved_documents = get_retriever().invoke(question)
    return "\n\n".join(
        getattr(document, "page_content", str(document))
        for document in retrieved_documents
    )


def prewarm_rag_resources() -> None:
    """Warm local source caches and the vector retriever without changing results."""

    for source_file_name in [
        "ORNL_ARMS.txt",
        "transportation_simulation_parameters_cleaned.txt",
    ]:
        source_file = DOCS_FOLDER / source_file_name
        if source_file.exists() and source_file not in _source_text_cache:
            _source_text_cache[source_file] = source_file.read_text(encoding="utf-8")

    get_retriever()


def get_agent_rag() -> Any:
    """Create the standalone RAG agent on first direct use."""

    global _agent_rag
    if _agent_rag is not None:
        return _agent_rag

    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import create_react_agent

    _agent_rag = create_react_agent(
        model=get_proj_llm_object("llm_openai"),
        tools=[rag_tool],
        prompt=SystemMessage(
            content="""You are a Reasoning Agent with access to `rag_tool`.
        Follow this loop strictly for every user query:

        1. Thought: Reflect on whether you need to query `rag_tool`.
        2. Action: If you do, invoke `rag_tool["<your question>"]`.
        3. Observation: Record the result returned by `rag_tool`.
        4. (Repeat Thought/Action/Observation as needed.)
        5. Answer: Provide the final response to the user, incorporating any retrieved information.

        Attention, You must call `rag_tool` at least once before giving your Answer.
        Do NOT jump directly to the final Answer without retrieval.
        """
        ),
        name="rag_agent",
    )
    return _agent_rag


class LazyRAGAgent:
    """Proxy that preserves the previous agent_RAG export without eager setup."""

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the lazily initialized RAG agent."""

        return get_agent_rag().invoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Stream from the lazily initialized RAG agent."""

        return get_agent_rag().stream(*args, **kwargs)


agent_RAG = LazyRAGAgent()


if __name__ == "__main__":
    question = "What is ORNL ARMS group doing and how many members in the group?"

    input_data = {"messages": [{"role": "user", "content": f"{question}"}]}
    answer = agent_RAG.invoke(input_data)
    print("Agent's final reply:\n", answer["messages"][-1].content)
