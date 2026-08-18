"""
##############################################################
# Created Date: Saturday, July 5th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
"""

import os
from pathlib import Path
from typing import Any

import yaml

LLM_CONFIG = yaml.load(
    open(Path(__file__).parent.parent / "proj_config" / "llm_config.yaml"),
    Loader=yaml.FullLoader,
)
if LLM_CONFIG["OPENAI_API_TYPE"] == "openai":
    os.environ["OPENAI_API_KEY"] = LLM_CONFIG["OPENAI_KEY"]

_OBJECT_CACHE: dict[str, Any] = {}


def _get_llm_openai() -> Any:
    """Create the OpenAI chat model on first use."""

    from langchain_openai import ChatOpenAI

    model_name = (
        os.environ.get("REALTWIN_OPENAI_MODEL")
        or LLM_CONFIG.get("OPENAI_MODEL")
        or "gpt-4o"
    )

    return ChatOpenAI(
        temperature=0,
        model_name=model_name,
        max_tokens=1024,
        request_timeout=60,
    )


def _get_llm_llama3() -> Any:
    """Create the Ollama chat model on first use."""

    from langchain_ollama import ChatOllama

    return ChatOllama(model="llama3.1")


def _get_embeddings_openai() -> Any:
    """Create OpenAI embeddings on first use."""

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(chunk_size=20)


def _get_embeddings_llama3() -> Any:
    """Create Ollama embeddings on first use."""

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model="llama3.1")


def _get_embeddings_hf() -> Any:
    """Create local HuggingFace embeddings on first use."""

    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import SentenceTransformer

    path_embeddings = Path(__file__).parent / "all-MiniLM-L6-v2"
    try:
        return HuggingFaceEmbeddings(
            model_name=str(path_embeddings),
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"device": "cpu"},
        )
    except Exception:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        model.save_pretrained(path_embeddings)

        return HuggingFaceEmbeddings(
            model_name=str(path_embeddings),
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"device": "cpu"},
        )


_FACTORIES = {
    "llm_openai": _get_llm_openai,
    "llm_llama3": _get_llm_llama3,
    "embeddings_hf": _get_embeddings_hf,
    "embeddings_openai": _get_embeddings_openai,
    "embeddings_llama3": _get_embeddings_llama3,
}


def __getattr__(name: str) -> Any:
    """Return lazily initialized LLM and embedding objects."""

    if name not in _FACTORIES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name not in _OBJECT_CACHE:
        _OBJECT_CACHE[name] = _FACTORIES[name]()
    return _OBJECT_CACHE[name]


__all__ = [
    "llm_openai",
    "llm_llama3",
    "embeddings_hf",
    "embeddings_openai",
    "embeddings_llama3",
]
