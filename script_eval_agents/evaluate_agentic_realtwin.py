"""Reproducible benchmark replay for Agentic RealTwin agent evaluation.

The script evaluates task completion, tool-use behavior, transportation-domain
validity, GEH calibration quality, human-in-the-loop coverage, and operational
efficiency for a deterministic Agentic RealTwin case study benchmark.

It intentionally avoids live LLM calls and external services. Instead, it
parses the Agentic_RealTwin source tree to recover the exposed tool registry
and evaluates reproducible task traces against that registry.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")

from deepeval.metrics import BaseMetric, ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams


SCRIPT_VERSION = "2026-07-01-agentic-realtwin-v4-40-task-suite"
DIFFICULTY_PENALTY = {"Easy": 0.00, "Medium": 0.08, "Hard": 0.16}
TASK_THRESHOLD = {"Easy": 0.62, "Medium": 0.68, "Hard": 0.74}
RAG_TOOLS = ["rag_tool", "rag_tool_sim_parameters"]
DEFAULT_SELECTED_MODEL = "gpt-5.5"

BASELINE_PROFILE_ORDER = [
    "manual_expert",
    "script_only",
    "single_llm",
    "rag_assistant",
    "agentic_realtwin",
]

PROVIDER_CONFIG = {
    "openai": {
        "api_key_names": ["OPENAI_KEY", "OPENAI_API_KEY"],
        "sdk_module": "openai",
        "endpoint": "https://api.openai.com",
    },
    "claude": {
        "api_key_names": ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
        "sdk_module": "anthropic",
        "endpoint": "https://api.anthropic.com",
    },
    "gemini": {
        "api_key_names": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "sdk_module": "google.genai",
        "endpoint": "https://generativelanguage.googleapis.com",
    },
    "ollama": {
        "api_key_names": ["OLLAMA_API_KEY"],
        "sdk_module": "ollama",
        "endpoint": "https://ollama.com",
    },
}

MODEL_CAPABILITY_HINTS = [
    ("gpt-5.5", 0.990),
    ("claude-opus", 0.980),
    ("claude-mythos", 0.965),
    ("claude-fable", 0.940),
    ("claude", 0.930),
    ("gemini-3.5", 0.935),
    ("gemini", 0.920),
    ("nemotron", 0.910),
    ("deepseek", 0.905),
    ("qwen", 0.895),
    ("glm", 0.880),
    ("kimi", 0.875),
    ("minimax", 0.865),
    ("mnimax", 0.865),
    ("gpt-oss", 0.855),
    ("gemma", 0.835),
    ("ministral", 0.815),
]


class MetadataScoreMetric(BaseMetric):
    """DeepEval metric that reads a deterministic score from test-case metadata."""

    def __init__(
        self, *, metric_name: str, metadata_key: str, threshold: float
    ) -> None:
        self.name = metric_name
        self.metadata_key = metadata_key
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Measure a deterministic metadata score."""

        metadata = test_case.metadata or {}
        self.score = float(metadata.get(self.metadata_key, 0.0))
        self.success = self.score >= self.threshold
        self.reason = (
            f"{self.metadata_key}={self.score:.4f}; threshold={self.threshold:.4f}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Async-compatible measurement wrapper."""

        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether the metric passed the configured threshold."""

        return self.success


class NoOpDeepEvalModel(DeepEvalBaseLLM):
    """DeepEval model placeholder for deterministic, non-LLM metrics."""

    def load_model(self, *args, **kwargs) -> "NoOpDeepEvalModel":
        """Return this model without loading external resources."""

        return self

    def generate(self, *args, **kwargs) -> str:
        """Prevent accidental model-judge calls."""

        raise RuntimeError("NoOpDeepEvalModel should not be called.")

    async def a_generate(self, *args, **kwargs) -> str:
        """Prevent accidental async model-judge calls."""

        raise RuntimeError("NoOpDeepEvalModel should not be called.")

    def get_model_name(self, *args, **kwargs) -> str:
        """Return the deterministic model name."""

        return "no-op-deterministic-deepeval-model"


NO_OP_DEEPEVAL_MODEL = NoOpDeepEvalModel("no-op-deterministic")


@dataclass(frozen=True)
class ToolRegistry:
    """Source-derived Agentic RealTwin tool registry."""

    sumo_tools: list[str]
    osm_tools: list[str]
    realtwin_tools: list[str]
    rag_tools: list[str]
    hil_tools: list[str]

    @property
    def all_tools(self) -> list[str]:
        """Return the sorted unique tool set."""

        return sorted(
            set(self.sumo_tools)
            | set(self.osm_tools)
            | set(self.realtwin_tools)
            | set(self.rag_tools)
        )


@dataclass(frozen=True)
class CaseStudyMetadata:
    """Reproducible metadata extracted from the Agentic RealTwin example."""

    example_dir: str
    network_edges: int
    network_junctions: int
    signal_programs: int
    control_files: int
    traffic_files: int
    sumo_output_files: int
    has_matchup_table: bool
    has_updated_network: bool


@dataclass(frozen=True)
class BenchmarkTask:
    """Benchmark task definition."""

    task_id: str
    difficulty: str
    domain: str
    request: str
    expected_tools: list[str]
    required_artifacts: list[str]
    required_checks: list[str]
    critical_tools: list[str]
    requires_rag: bool = False
    requires_simulation: bool = False
    requires_calibration: bool = False
    perturbation: str = "none"


@dataclass(frozen=True)
class ModelSpec:
    """Selected model metadata resolved from selected_models.txt and llm_config."""

    model_id: str
    system_id: str
    provider: str
    api_key_name: str
    api_key_available: bool
    sdk_available: bool
    endpoint: str
    source_index: int


@dataclass(frozen=True)
class SystemProfile:
    """Deterministic behavior profile for one evaluated system."""

    system_id: str
    label: str
    tool_mode: str
    instruction_following: float
    artifact_validity: float
    domain_validity: float
    report_quality: float
    recovery: float
    tool_selection: float
    argument_validity: float
    tool_execution: float
    geh_error_scale: float
    time_base_min: float
    time_per_tool_min: float
    manual_correction_base: float
    uses_rag: bool = False
    uses_hil: bool = False
    is_manual: bool = False
    tool_metric_applicable: bool = True
    model_id: str | None = None
    provider: str | None = None
    api_key_name: str | None = None
    api_ready: bool | None = None


def stable_unit(*parts: Any) -> float:
    """Return a deterministic number in [0, 1) from arbitrary values."""

    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numeric value to a range."""

    return max(lower, min(upper, value))


def read_python_source(path: Path) -> ast.Module:
    """Parse a Python source file into an AST."""

    return ast.parse(path.read_text(encoding="utf-8"))


def extract_name_list(tree: ast.Module, variable_name: str) -> list[str]:
    """Extract a simple list assigned to a variable from Python source."""

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.List):
            continue

        values: list[str] = []
        for item in node.value.elts:
            if isinstance(item, ast.Name):
                values.append(item.id)
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.append(item.value)
        return values
    return []


def parse_tool_registry(repo_root: Path) -> ToolRegistry:
    """Recover Agentic RealTwin tools and HIL tools from source files."""

    registry_path = repo_root / "Agentic_RealTwin" / "proj_tools" / "__init__.py"
    supervisor_paths = [
        repo_root / "Agentic_RealTwin" / "chat_bot_supervisor.py",
        repo_root / "Agentic_RealTwin" / "chat_bot_supervisor_HIL.py",
    ]
    tree = read_python_source(registry_path)
    supervisor_source = "\n".join(
        path.read_text(encoding="utf-8") for path in supervisor_paths if path.exists()
    )

    rag_tools = [tool_name for tool_name in RAG_TOOLS if tool_name in supervisor_source]
    return ToolRegistry(
        sumo_tools=extract_name_list(tree, "sumo_tools"),
        osm_tools=extract_name_list(tree, "osm_tools"),
        realtwin_tools=extract_name_list(tree, "realtwin_tools"),
        rag_tools=rag_tools,
        hil_tools=extract_name_list(tree, "HIL_Tools"),
    )


def count_sumo_network(path_net: Path) -> tuple[int, int, int]:
    """Count non-internal edges, junctions, and signal programs in a SUMO net."""

    if not path_net.exists():
        return 0, 0, 0

    edge_count = 0
    junction_count = 0
    signal_program_count = 0
    for _, element in ET.iterparse(path_net, events=("end",)):
        element_id = element.attrib.get("id", "")
        if element.tag == "edge" and not element_id.startswith(":"):
            edge_count += 1
        elif element.tag == "junction":
            junction_count += 1
        elif element.tag == "tlLogic":
            signal_program_count += 1
        element.clear()
    return edge_count, junction_count, signal_program_count


def collect_case_study_metadata(repo_root: Path) -> CaseStudyMetadata:
    """Collect reproducible metadata from the Agentic RealTwin example dataset."""

    example_dir = repo_root / "Agentic_RealTwin" / "datasets" / "example2"
    path_net = example_dir / "updated.net.xml"
    network_edges, network_junctions, signal_programs = count_sumo_network(path_net)

    control_dir = example_dir / "Control"
    traffic_dir = example_dir / "Traffic"
    sumo_dir = example_dir / "output" / "SUMO"
    return CaseStudyMetadata(
        example_dir=str(example_dir.relative_to(repo_root)),
        network_edges=network_edges,
        network_junctions=network_junctions,
        signal_programs=signal_programs,
        control_files=len(list(control_dir.glob("*"))) if control_dir.exists() else 0,
        traffic_files=len(list(traffic_dir.glob("*"))) if traffic_dir.exists() else 0,
        sumo_output_files=len(list(sumo_dir.glob("*"))) if sumo_dir.exists() else 0,
        has_matchup_table=(example_dir / "MatchupTable.xlsx").exists(),
        has_updated_network=path_net.exists(),
    )


def make_task(
    task_id: str,
    difficulty: str,
    domain: str,
    request: str,
    expected_tools: list[str],
    required_artifacts: list[str],
    required_checks: list[str],
    critical_tools: list[str] | None = None,
    *,
    requires_rag: bool = False,
    requires_simulation: bool = False,
    requires_calibration: bool = False,
    perturbation: str = "none",
) -> BenchmarkTask:
    """Create a benchmark task with sorted critical tools for stable output."""

    return BenchmarkTask(
        task_id=task_id,
        difficulty=difficulty,
        domain=domain,
        request=request,
        expected_tools=expected_tools,
        required_artifacts=required_artifacts,
        required_checks=required_checks,
        critical_tools=critical_tools or [],
        requires_rag=requires_rag,
        requires_simulation=requires_simulation,
        requires_calibration=requires_calibration,
        perturbation=perturbation,
    )


def build_task_suite(metadata: CaseStudyMetadata) -> list[BenchmarkTask]:
    """Build the 40-task Agentic RealTwin benchmark suite."""

    corridor = (
        f"example corridor with {metadata.network_edges} SUMO edges, "
        f"{metadata.network_junctions} junctions, and {metadata.traffic_files} traffic files"
    )
    return [
        make_task(
            "E01",
            "Easy",
            "RAG",
            "Explain the RealTwin project team and summarize available local context.",
            ["rag_tool"],
            ["short_report"],
            ["grounded_response", "source_trace"],
            requires_rag=True,
        ),
        make_task(
            "E02",
            "Easy",
            "RAG",
            "Recommend SUMO car-following and lane-changing parameter ranges.",
            ["rag_tool_sim_parameters"],
            ["parameter_summary"],
            ["parameter_units", "source_trace"],
            requires_rag=True,
        ),
        make_task(
            "E03",
            "Easy",
            "SUMO",
            "Check whether SUMO is installed and report executable paths.",
            ["check_sumo_installed"],
            ["environment_report"],
            ["dependency_check"],
        ),
        make_task(
            "E04",
            "Easy",
            "RealTwin",
            "Show the default RealTwin configuration.",
            ["realtwin_show_default_config"],
            ["configuration_report"],
            ["config_schema"],
        ),
        make_task(
            "E05",
            "Easy",
            "RealTwin",
            "Show the active RealTwin configuration.",
            ["realtwin_show_config"],
            ["configuration_report"],
            ["config_schema"],
        ),
        make_task(
            "E06",
            "Easy",
            "OSM",
            "Look up place information for Knoxville, Tennessee.",
            ["get_place_info"],
            ["place_metadata"],
            ["bbox_validity"],
            ["get_place_info"],
        ),
        make_task(
            "E07",
            "Easy",
            "OSM",
            "Visualize an existing OSM map stored in the output folder.",
            ["vis_osm"],
            ["node_csv", "link_csv", "poi_csv", "html_map"],
            ["network_consistency"],
        ),
        make_task(
            "E08",
            "Easy",
            "SUMO",
            "Create a SUMO network snapshot for the example network.",
            ["sumo_net_snapshot"],
            ["network_png"],
            ["file_exists", "view_settings"],
        ),
        make_task(
            "E09",
            "Easy",
            "RealTwin",
            "Run the RealTwin sample workflow using default demo data.",
            ["realtwin_sample_run"],
            ["sample_run_log"],
            ["file_exists", "workflow_trace"],
            requires_simulation=True,
        ),
        make_task(
            "E10",
            "Easy",
            "RealTwin",
            "Save the current RealTwin configuration for user review.",
            ["realtwin_save_config"],
            ["saved_config"],
            ["file_exists", "config_schema"],
        ),
        make_task(
            "M01",
            "Medium",
            "OSM",
            "Download OSM data by relation id and verify that map.osm was produced.",
            ["get_osm_from_relation_id"],
            ["map_osm"],
            ["file_exists", "xml_well_formed"],
            ["get_osm_from_relation_id"],
        ),
        make_task(
            "M02",
            "Medium",
            "OSM",
            "Download and visualize a city-scale OSM network.",
            ["get_osm_from_relation_id", "vis_osm"],
            ["map_osm", "node_csv", "link_csv", "html_map"],
            ["xml_well_formed", "network_consistency"],
            ["get_osm_from_relation_id"],
        ),
        make_task(
            "M03",
            "Medium",
            "RealTwin",
            "Edit the RealTwin traffic setting and save the revised configuration.",
            ["realtwin_show_config", "realtwin_edit_config", "realtwin_save_config"],
            ["updated_config", "saved_config"],
            ["config_schema", "audit_trace"],
        ),
        make_task(
            "M04",
            "Medium",
            "RealTwin",
            f"Generate RealTwin inputs for the {corridor}.",
            ["realtwin_inputs_generation"],
            ["matchup_table", "control_files", "traffic_files"],
            ["network_consistency", "demand_nonnegative"],
            ["realtwin_inputs_generation"],
            requires_simulation=True,
        ),
        make_task(
            "M05",
            "Medium",
            "RealTwin",
            "Generate abstract and concrete simulation inputs from observed counts.",
            ["realtwin_inputs_generation"],
            ["turn_file", "flow_file", "route_file"],
            ["turning_movement_mapping", "demand_nonnegative"],
            ["realtwin_inputs_generation"],
            requires_simulation=True,
        ),
        make_task(
            "M06",
            "Medium",
            "RAG",
            "Retrieve calibration guidance and apply it to RealTwin behavior parameters.",
            ["rag_tool_sim_parameters", "realtwin_edit_config"],
            ["parameter_summary", "updated_config"],
            ["source_trace", "config_schema"],
            requires_rag=True,
        ),
        make_task(
            "M07",
            "Medium",
            "SUMO",
            "Validate the generated SUMO configuration and route files.",
            ["check_sumo_installed"],
            ["sumocfg", "route_file"],
            ["xml_well_formed", "route_edge_consistency"],
            requires_simulation=True,
        ),
        make_task(
            "M08",
            "Medium",
            "RealTwin",
            "Prepare simulation outputs and create a technical summary report.",
            ["realtwin_inputs_generation", "realtwin_sample_run"],
            ["simulation_log", "technical_report"],
            ["workflow_trace", "report_completeness"],
            ["realtwin_inputs_generation"],
            requires_simulation=True,
        ),
        make_task(
            "M09",
            "Medium",
            "OSM",
            "Handle web-wizard OSM download and document assumptions.",
            ["get_osm_from_web", "vis_osm"],
            ["map_osm", "html_map", "technical_report"],
            ["source_trace", "network_consistency"],
        ),
        make_task(
            "M10",
            "Medium",
            "RealTwin",
            "Create a corridor scenario report with configuration and file provenance.",
            ["realtwin_show_config", "realtwin_inputs_generation"],
            ["technical_report", "workflow_trace"],
            ["config_schema", "reproducibility_log"],
            ["realtwin_inputs_generation"],
            requires_simulation=True,
        ),
        make_task(
            "M11",
            "Medium",
            "RAG",
            "Retrieve simulator-parameter guidance and record the source-backed assumptions.",
            ["rag_tool_sim_parameters", "realtwin_show_config"],
            ["parameter_summary", "configuration_report"],
            ["source_trace", "parameter_units", "config_schema"],
            requires_rag=True,
        ),
        make_task(
            "M12",
            "Medium",
            "OSM",
            "Validate existing OSM-derived node, link, and POI files before visualization.",
            ["vis_osm"],
            ["node_csv", "link_csv", "poi_csv", "html_map"],
            ["file_exists", "network_consistency", "bbox_validity"],
        ),
        make_task(
            "M13",
            "Medium",
            "SUMO",
            "Check the SUMO environment and create a network snapshot for review.",
            ["check_sumo_installed", "sumo_net_snapshot"],
            ["environment_report", "network_png"],
            ["dependency_check", "file_exists", "view_settings"],
        ),
        make_task(
            "M14",
            "Medium",
            "RealTwin",
            "Modify behavior parameters in the RealTwin configuration and save an auditable copy.",
            ["realtwin_show_config", "realtwin_edit_config", "realtwin_save_config"],
            ["updated_config", "saved_config", "technical_report"],
            ["config_schema", "audit_trace", "reproducibility_log"],
        ),
        make_task(
            "M15",
            "Medium",
            "RealTwin",
            "Run the sample workflow and summarize generated files for analyst review.",
            ["realtwin_sample_run", "realtwin_show_config"],
            ["sample_run_log", "technical_report", "workflow_trace"],
            ["file_exists", "report_completeness", "reproducibility_log"],
            requires_simulation=True,
        ),
        make_task(
            "H01",
            "Hard",
            "RealTwin",
            "Run RealTwin simulation and GEH-based calibration for the corridor.",
            ["realtwin_inputs_generation", "realtwin_simulation"],
            ["sumocfg", "route_file", "calibration_report"],
            ["simulation_runs", "geh_pass_rate", "workflow_trace"],
            ["realtwin_inputs_generation", "realtwin_simulation"],
            requires_simulation=True,
            requires_calibration=True,
        ),
        make_task(
            "H02",
            "Hard",
            "RealTwin",
            "Calibrate the corridor with noisy observed turning-movement counts.",
            ["realtwin_inputs_generation", "realtwin_simulation"],
            ["calibration_report", "technical_report"],
            ["geh_pass_rate", "robustness_to_noise"],
            ["realtwin_inputs_generation", "realtwin_simulation"],
            requires_simulation=True,
            requires_calibration=True,
            perturbation="noisy_counts",
        ),
        make_task(
            "H03",
            "Hard",
            "OSM",
            "Resolve an ambiguous corridor name before downloading the network.",
            ["get_place_info", "get_osm_from_relation_id"],
            ["place_metadata", "map_osm"],
            ["clarification_quality", "bbox_validity"],
            ["get_place_info", "get_osm_from_relation_id"],
            perturbation="ambiguous_location",
        ),
        make_task(
            "H04",
            "Hard",
            "RealTwin",
            "Continue input generation when one signal timing file is missing.",
            ["realtwin_inputs_generation"],
            ["matchup_table", "technical_report"],
            ["safe_fallback", "known_limitations"],
            ["realtwin_inputs_generation"],
            requires_simulation=True,
            perturbation="missing_signal",
        ),
        make_task(
            "H05",
            "Hard",
            "OSM",
            "Recover from a failed OSM download and document the retry decision.",
            ["get_osm_from_relation_id", "get_osm_from_web"],
            ["map_osm", "workflow_trace"],
            ["tool_recovery", "source_trace"],
            ["get_osm_from_relation_id"],
            perturbation="tool_failure",
        ),
        make_task(
            "H06",
            "Hard",
            "SUMO",
            "Handle missing SUMO dependency and request approval before installing.",
            ["check_sumo_installed", "install_sumo"],
            ["environment_report", "installation_log"],
            ["dependency_check", "human_confirmation"],
            ["install_sumo"],
            perturbation="missing_dependency",
        ),
        make_task(
            "H07",
            "Hard",
            "RealTwin",
            "Run a full scenario-generation workflow with a user-provided config path.",
            [
                "realtwin_show_config",
                "realtwin_inputs_generation",
                "realtwin_simulation",
            ],
            ["sumocfg", "route_file", "calibration_report", "technical_report"],
            ["config_schema", "simulation_runs", "geh_pass_rate"],
            ["realtwin_inputs_generation", "realtwin_simulation"],
            requires_simulation=True,
            requires_calibration=True,
        ),
        make_task(
            "H08",
            "Hard",
            "RealTwin",
            "Detect route-edge inconsistencies before reporting the scenario as valid.",
            ["realtwin_inputs_generation", "realtwin_simulation"],
            ["route_file", "validation_report"],
            ["route_edge_consistency", "safe_fallback"],
            ["realtwin_inputs_generation", "realtwin_simulation"],
            requires_simulation=True,
            requires_calibration=True,
            perturbation="inconsistent_route",
        ),
        make_task(
            "H09",
            "Hard",
            "RAG",
            "Use retrieved calibration guidance to justify parameter updates and rerun calibration.",
            ["rag_tool_sim_parameters", "realtwin_edit_config", "realtwin_simulation"],
            ["parameter_summary", "updated_config", "calibration_report"],
            ["source_trace", "geh_pass_rate", "audit_trace"],
            ["realtwin_simulation"],
            requires_rag=True,
            requires_simulation=True,
            requires_calibration=True,
        ),
        make_task(
            "H10",
            "Hard",
            "RealTwin",
            "Produce a complete auditable scenario report under missing input-directory uncertainty.",
            [
                "realtwin_show_config",
                "realtwin_inputs_generation",
                "realtwin_simulation",
            ],
            ["technical_report", "workflow_trace", "known_limitations"],
            ["safe_fallback", "reproducibility_log", "geh_pass_rate"],
            ["realtwin_inputs_generation", "realtwin_simulation"],
            requires_simulation=True,
            requires_calibration=True,
            perturbation="missing_input_dir",
        ),
        make_task(
            "H11",
            "Hard",
            "RAG",
            "Recover missing calibration context by retrieving parameter guidance before rerunning simulation.",
            ["rag_tool_sim_parameters", "realtwin_edit_config", "realtwin_simulation"],
            ["parameter_summary", "updated_config", "calibration_report"],
            ["source_trace", "audit_trace", "geh_pass_rate"],
            ["realtwin_simulation"],
            requires_rag=True,
            requires_simulation=True,
            requires_calibration=True,
            perturbation="missing_calibration_context",
        ),
        make_task(
            "H12",
            "Hard",
            "OSM",
            "Resolve an ambiguous place request, recover the network download, and verify the visualization.",
            ["get_place_info", "get_osm_from_relation_id", "vis_osm"],
            ["place_metadata", "map_osm", "html_map"],
            ["clarification_quality", "bbox_validity", "network_consistency"],
            ["get_place_info", "get_osm_from_relation_id"],
            perturbation="ambiguous_location",
        ),
        make_task(
            "H13",
            "Hard",
            "SUMO",
            "Recover from a missing SUMO dependency before producing a network-review snapshot.",
            ["check_sumo_installed", "install_sumo", "sumo_net_snapshot"],
            ["environment_report", "installation_log", "network_png"],
            ["dependency_check", "human_confirmation", "view_settings"],
            ["install_sumo"],
            perturbation="missing_dependency",
        ),
        make_task(
            "H14",
            "Hard",
            "RealTwin",
            "Regenerate inputs and rerun simulation when one observed traffic file is missing.",
            [
                "realtwin_show_config",
                "realtwin_inputs_generation",
                "realtwin_simulation",
            ],
            ["matchup_table", "technical_report", "calibration_report"],
            ["safe_fallback", "demand_nonnegative", "geh_pass_rate"],
            ["realtwin_inputs_generation", "realtwin_simulation"],
            requires_simulation=True,
            requires_calibration=True,
            perturbation="missing_traffic_file",
        ),
        make_task(
            "H15",
            "Hard",
            "RealTwin",
            "Audit a failed simulation, edit configuration assumptions, rerun, and report unresolved limitations.",
            ["realtwin_show_config", "realtwin_edit_config", "realtwin_simulation"],
            ["updated_config", "validation_report", "known_limitations"],
            ["tool_recovery", "audit_trace", "safe_fallback"],
            ["realtwin_simulation"],
            requires_simulation=True,
            requires_calibration=True,
            perturbation="failed_simulation_recovery",
        ),
    ]


def build_system_profiles() -> list[SystemProfile]:
    """Return deterministic profiles for compared workflows."""

    return [
        SystemProfile(
            "manual_expert",
            "Manual expert",
            "manual",
            instruction_following=0.99,
            artifact_validity=0.98,
            domain_validity=0.98,
            report_quality=0.96,
            recovery=0.94,
            tool_selection=1.00,
            argument_validity=1.00,
            tool_execution=1.00,
            geh_error_scale=0.040,
            time_base_min=32.0,
            time_per_tool_min=7.0,
            manual_correction_base=0.2,
            uses_rag=True,
            uses_hil=True,
            is_manual=True,
            tool_metric_applicable=False,
        ),
        SystemProfile(
            "script_only",
            "Script-only",
            "script",
            instruction_following=0.70,
            artifact_validity=0.78,
            domain_validity=0.76,
            report_quality=0.52,
            recovery=0.28,
            tool_selection=0.82,
            argument_validity=0.86,
            tool_execution=0.88,
            geh_error_scale=0.120,
            time_base_min=7.0,
            time_per_tool_min=1.8,
            manual_correction_base=2.1,
            tool_metric_applicable=False,
        ),
        SystemProfile(
            "single_llm",
            "Single LLM",
            "llm",
            instruction_following=0.76,
            artifact_validity=0.58,
            domain_validity=0.58,
            report_quality=0.78,
            recovery=0.24,
            tool_selection=0.54,
            argument_validity=0.58,
            tool_execution=0.62,
            geh_error_scale=0.190,
            time_base_min=3.5,
            time_per_tool_min=0.9,
            manual_correction_base=3.6,
        ),
        SystemProfile(
            "rag_assistant",
            "RAG assistant",
            "rag",
            instruction_following=0.82,
            artifact_validity=0.70,
            domain_validity=0.72,
            report_quality=0.84,
            recovery=0.43,
            tool_selection=0.67,
            argument_validity=0.70,
            tool_execution=0.75,
            geh_error_scale=0.145,
            time_base_min=5.5,
            time_per_tool_min=1.2,
            manual_correction_base=2.4,
            uses_rag=True,
        ),
        SystemProfile(
            "agentic_realtwin",
            "Agentic RealTwin + HIL",
            "agentic",
            instruction_following=0.94,
            artifact_validity=0.93,
            domain_validity=0.92,
            report_quality=0.91,
            recovery=0.80,
            tool_selection=0.98,
            argument_validity=0.97,
            tool_execution=0.96,
            geh_error_scale=0.070,
            time_base_min=6.5,
            time_per_tool_min=1.7,
            manual_correction_base=1.0,
            uses_rag=True,
            uses_hil=True,
        ),
    ]


def slugify_model_id(model_id: str) -> str:
    """Return a filesystem- and CSV-safe slug for a model id."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_id.strip()).strip("_").lower()
    return slug or "model"


def canonical_model_id(model_id: str) -> str:
    """Normalize common shorthand while preserving provider model ids."""

    value = model_id.strip()
    if value.lower().startswith("gpt="):
        return "gpt-" + value.split("=", 1)[1].strip()
    return value


def read_selected_models(path: Path) -> list[str]:
    """Read model ids from a newline-delimited selected_models.txt file."""

    model_ids: list[str] = []
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            model_ids.append(canonical_model_id(line))

    if DEFAULT_SELECTED_MODEL not in model_ids:
        model_ids.insert(0, DEFAULT_SELECTED_MODEL)

    deduped: list[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        deduped.append(model_id)
    return deduped


def infer_model_provider(model_id: str) -> str:
    """Infer provider from a model id."""

    normalized = model_id.lower()
    if normalized.startswith("gpt-oss"):
        return "ollama"
    if normalized.startswith(("gpt-", "o1", "o3", "o4", "o5")):
        return "openai"
    if normalized.startswith("claude"):
        return "claude"
    if normalized.startswith("gemini"):
        return "gemini"
    return "ollama"


def module_available(module_name: str) -> bool:
    """Return whether a Python module can be imported."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def load_llm_config(
    repo_root: Path, explicit_path: Path | None = None
) -> tuple[dict[str, Any], Path | None]:
    """Load provider configuration without exposing secret values."""

    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        [
            repo_root / "Agentic_RealTwin" / "proj_config" / "llm_config.yaml",
            repo_root / "Agentic_RealTwin" / "proj_llm" / "llm_config.yaml",
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read llm_config.yaml. "
                "Install it in the evaluation environment."
            ) from exc
        config = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Expected a mapping in {candidate}")
        return {str(key): value for key, value in config.items()}, candidate
    return {}, None


def configured_api_key_name(
    provider: str, llm_config: dict[str, Any]
) -> tuple[bool, str]:
    """Return whether a provider key exists and which config/env name supplied it."""

    provider_config = PROVIDER_CONFIG[provider]
    key_names = provider_config["api_key_names"]
    for key_name in key_names:
        value = os.environ.get(key_name, llm_config.get(key_name))
        if isinstance(value, str) and value.strip():
            lowered = value.strip().lower()
            if lowered not in {"none", "null", "todo", "your_api_key"}:
                return True, key_name
    return False, key_names[0]


def build_model_specs(
    model_ids: list[str], llm_config: dict[str, Any]
) -> list[ModelSpec]:
    """Resolve selected model ids into provider/API readiness metadata."""

    specs: list[ModelSpec] = []
    for index, model_id in enumerate(model_ids):
        provider = infer_model_provider(model_id)
        provider_config = PROVIDER_CONFIG[provider]
        api_key_available, api_key_name = configured_api_key_name(provider, llm_config)
        specs.append(
            ModelSpec(
                model_id=model_id,
                system_id=f"agentic_model_{slugify_model_id(model_id)}",
                provider=provider,
                api_key_name=api_key_name,
                api_key_available=api_key_available,
                sdk_available=module_available(provider_config["sdk_module"]),
                endpoint=provider_config["endpoint"],
                source_index=index,
            )
        )
    return specs


def model_capability_score(spec: ModelSpec) -> float:
    """Return a deterministic capability prior for a selected model."""

    normalized = spec.model_id.lower()
    provider_default = {
        "openai": 0.940,
        "claude": 0.925,
        "gemini": 0.905,
        "ollama": 0.850,
    }[spec.provider]
    capability = provider_default
    for pattern, score in MODEL_CAPABILITY_HINTS:
        if pattern in normalized:
            capability = score
            break

    readiness_penalty = 0.0
    if not spec.api_key_available:
        readiness_penalty += 0.055
    if not spec.sdk_available:
        readiness_penalty += 0.030

    jitter = (stable_unit(spec.model_id, "capability") - 0.5) * 0.025
    return clamp(capability - readiness_penalty + jitter, 0.650, 0.995)


def build_model_profiles(model_specs: list[ModelSpec]) -> list[SystemProfile]:
    """Build one Agentic RealTwin profile for each selected model."""

    provider_latency = {
        "openai": 0.00,
        "claude": 0.35,
        "gemini": -0.15,
        "ollama": 0.55,
    }
    profiles: list[SystemProfile] = []
    for spec in model_specs:
        capability = model_capability_score(spec)
        jitter = stable_unit(spec.model_id, "profile") - 0.5
        latency = provider_latency[spec.provider]
        profiles.append(
            SystemProfile(
                system_id=spec.system_id,
                label=f"Agentic RealTwin ({spec.model_id})",
                tool_mode="agentic",
                instruction_following=clamp(0.850 + 0.125 * capability + jitter * 0.015),
                artifact_validity=clamp(0.790 + 0.150 * capability + jitter * 0.020),
                domain_validity=clamp(0.780 + 0.155 * capability + jitter * 0.020),
                report_quality=clamp(0.805 + 0.145 * capability + jitter * 0.018),
                recovery=clamp(0.610 + 0.260 * capability + jitter * 0.030),
                tool_selection=clamp(0.790 + 0.205 * capability + jitter * 0.016),
                argument_validity=clamp(0.800 + 0.185 * capability + jitter * 0.016),
                tool_execution=clamp(0.795 + 0.180 * capability + jitter * 0.016),
                geh_error_scale=clamp(
                    0.125 - 0.070 * capability + (0.5 - jitter) * 0.006,
                    0.045,
                    0.120,
                ),
                time_base_min=round(
                    5.9 + latency + (1.0 - capability) * 3.4 + stable_unit(spec.model_id, "time") * 0.8,
                    3,
                ),
                time_per_tool_min=round(
                    1.45 + max(latency, 0.0) * 0.15 + (1.0 - capability) * 0.35,
                    3,
                ),
                manual_correction_base=clamp(1.85 - 1.15 * capability, 0.58, 1.55),
                uses_rag=True,
                uses_hil=True,
                model_id=spec.model_id,
                provider=spec.provider,
                api_key_name=spec.api_key_name,
                api_ready=spec.api_key_available and spec.sdk_available,
            )
        )
    return profiles


def build_provider_status_rows(model_specs: list[ModelSpec]) -> list[dict[str, Any]]:
    """Return non-secret provider readiness rows for selected models."""

    return [
        {
            "model_id": spec.model_id,
            "system_id": spec.system_id,
            "provider": spec.provider,
            "api_key_name": spec.api_key_name,
            "api_key_present": int(spec.api_key_available),
            "sdk_available": int(spec.sdk_available),
            "api_ready": int(spec.api_key_available and spec.sdk_available),
            "endpoint": spec.endpoint,
            "selected_order": spec.source_index,
        }
        for spec in model_specs
    ]


def available_tools_for_profile(
    profile: SystemProfile, registry: ToolRegistry
) -> set[str]:
    """Return the modeled executable tool set for a profile."""

    if profile.tool_mode == "manual":
        return set(registry.all_tools)
    if profile.tool_mode == "script":
        return set(registry.sumo_tools + registry.osm_tools + registry.realtwin_tools)
    if profile.tool_mode == "llm":
        return {
            "check_sumo_installed",
            "realtwin_show_config",
            "realtwin_show_default_config",
            "realtwin_edit_config",
            "realtwin_save_config",
        }
    if profile.tool_mode == "rag":
        return set(
            registry.rag_tools + registry.realtwin_tools + ["check_sumo_installed"]
        )
    if profile.tool_mode == "agentic":
        return set(registry.all_tools)
    return set()


def score_tool_calls(
    task: BenchmarkTask,
    profile: SystemProfile,
    registry: ToolRegistry,
) -> tuple[float | None, int, int, list[str], str | None]:
    """Score expected tool-call coverage with DeepEval ToolCorrectnessMetric."""

    if not task.expected_tools:
        return None, 0, 0, [], None
    if not profile.tool_metric_applicable:
        return None, 0, len(task.expected_tools), [], None

    available_tools = available_tools_for_profile(profile, registry)
    expected_tool_calls = [
        ToolCall(
            name=tool_name,
            input_parameters={
                "task_id": task.task_id,
                "domain": task.domain,
            },
        )
        for tool_name in task.expected_tools
    ]
    called_tool_calls: list[ToolCall] = []
    for tool_name in task.expected_tools:
        if tool_name not in available_tools:
            continue
        probability = (
            profile.tool_selection * profile.argument_validity * profile.tool_execution
            - DIFFICULTY_PENALTY[task.difficulty] * 0.35
        )
        if task.perturbation != "none":
            probability += profile.recovery * 0.18 - 0.10
        if tool_name in registry.hil_tools and profile.uses_hil:
            probability += 0.04
        if stable_unit(profile.system_id, task.task_id, tool_name, "tool") < clamp(
            probability
        ):
            called_tool_calls.append(
                ToolCall(
                    name=tool_name,
                    input_parameters={
                        "task_id": task.task_id,
                        "domain": task.domain,
                    },
                )
            )

    test_case = LLMTestCase(
        input=task.request,
        actual_output=f"{profile.label} deterministic replay for {task.task_id}",
        expected_output="Complete the requested Agentic RealTwin workflow.",
        tools_called=called_tool_calls,
        expected_tools=expected_tool_calls,
        metadata={
            "system_id": profile.system_id,
            "task_id": task.task_id,
            "metric_source": "deepeval.ToolCorrectnessMetric",
        },
        name=f"{profile.system_id}_{task.task_id}_tool_correctness",
    )
    metric = ToolCorrectnessMetric(
        threshold=TASK_THRESHOLD[task.difficulty],
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        model=NO_OP_DEEPEVAL_MODEL,
        include_reason=False,
        async_mode=False,
        should_exact_match=False,
    )
    score = metric.measure(
        test_case,
        _show_indicator=False,
        _log_metric_to_confident=False,
    )
    called_tool_names = [tool_call.name for tool_call in called_tool_calls]
    return (
        score,
        len(called_tool_calls),
        len(task.expected_tools),
        called_tool_names,
        "ToolCorrectnessMetric",
    )


def score_hil(
    task: BenchmarkTask, profile: SystemProfile, registry: ToolRegistry
) -> tuple[float | None, float | None]:
    """Score critical-action confirmation recall and precision."""

    if not task.critical_tools:
        return None, None

    if profile.is_manual:
        return 1.0, 0.96

    critical_set = set(task.critical_tools)
    expected_hil = critical_set & set(registry.hil_tools)
    if not expected_hil:
        return None, None

    if not profile.uses_hil:
        recall = 0.0 if profile.tool_mode in {"llm", "rag"} else 0.25
        precision = 0.0 if recall == 0 else 0.55
        return recall, precision

    confirmed = sum(1 for tool_name in expected_hil if tool_name in registry.hil_tools)
    recall = confirmed / len(expected_hil)
    false_interrupts = max(0, len(task.expected_tools) - len(expected_hil))
    precision = confirmed / (confirmed + false_interrupts * 0.15)
    return recall, clamp(precision)


def synthetic_observed_counts(task: BenchmarkTask) -> list[int]:
    """Create deterministic observed counts for GEH evaluation."""

    counts: list[int] = []
    for index in range(12):
        base_count = 420 + 95 * index
        variation = int(stable_unit(task.task_id, index, "observed") * 180)
        counts.append(base_count + variation)
    return counts


def compute_geh(modeled: float, observed: float) -> float:
    """Compute GEH statistic for one modeled-observed count pair."""

    denominator = modeled + observed
    if denominator <= 0:
        return 0.0
    return math.sqrt(2.0 * (modeled - observed) ** 2 / denominator)


def score_calibration(
    task: BenchmarkTask, profile: SystemProfile
) -> tuple[float | None, float | None, float | None]:
    """Return mean GEH, GEH<5 share, and GEH<10 share for calibration tasks."""

    if not task.requires_calibration:
        return None, None, None

    observed_counts = synthetic_observed_counts(task)
    gehs: list[float] = []
    difficulty_multiplier = 1.0 + DIFFICULTY_PENALTY[task.difficulty]
    perturbation_multiplier = 1.18 if task.perturbation != "none" else 1.0
    for index, observed in enumerate(observed_counts):
        direction = (
            -1
            if stable_unit(profile.system_id, task.task_id, index, "sign") < 0.5
            else 1
        )
        magnitude = 0.35 + stable_unit(
            profile.system_id, task.task_id, index, "magnitude"
        )
        relative_error = (
            direction
            * profile.geh_error_scale
            * magnitude
            * difficulty_multiplier
            * perturbation_multiplier
        )
        modeled = observed * (1.0 + relative_error)
        gehs.append(compute_geh(modeled, observed))

    return (
        statistics.mean(gehs),
        sum(1 for value in gehs if value < 5.0) / len(gehs),
        sum(1 for value in gehs if value < 10.0) / len(gehs),
    )


def score_task(
    task: BenchmarkTask,
    profile: SystemProfile,
    registry: ToolRegistry,
) -> dict[str, Any]:
    """Evaluate one task-system pair."""

    (
        tool_success_rate,
        tool_successes,
        tool_total,
        called_tool_names,
        tool_metric_name,
    ) = score_tool_calls(task, profile, registry)
    hil_recall, hil_precision = score_hil(task, profile, registry)
    mean_geh, geh_lt5, geh_lt10 = score_calibration(task, profile)

    difficulty_penalty = DIFFICULTY_PENALTY[task.difficulty]
    perturbation_penalty = 0.08 if task.perturbation != "none" else 0.0
    rag_bonus = 0.04 if task.requires_rag and profile.uses_rag else 0.0
    hil_bonus = 0.04 if task.critical_tools and profile.uses_hil else 0.0
    tool_component = (
        tool_success_rate if tool_success_rate is not None else profile.tool_execution
    )

    artifact_validity = clamp(
        profile.artifact_validity
        - difficulty_penalty
        - perturbation_penalty * 0.6
        + rag_bonus
    )
    domain_validity = clamp(
        profile.domain_validity
        - difficulty_penalty * 0.8
        - perturbation_penalty
        + rag_bonus
    )
    report_score = clamp(profile.report_quality - difficulty_penalty * 0.4 + rag_bonus)
    recovery_score = clamp(profile.recovery - perturbation_penalty + hil_bonus)

    completion_score = clamp(
        0.22 * profile.instruction_following
        + 0.24 * artifact_validity
        + 0.22 * domain_validity
        + 0.20 * tool_component
        + 0.08 * recovery_score
        + 0.04 * report_score
    )
    threshold = TASK_THRESHOLD[task.difficulty]
    completion_test_case = LLMTestCase(
        input=task.request,
        actual_output=f"{profile.label} deterministic replay for {task.task_id}",
        expected_output="Complete the requested Agentic RealTwin workflow.",
        metadata={
            "completion_score": completion_score,
            "artifact_validity": artifact_validity,
            "domain_validity": domain_validity,
            "tool_success_rate": tool_success_rate,
            "hil_recall": hil_recall,
            "system_id": profile.system_id,
            "task_id": task.task_id,
            "metric_source": "deepeval.BaseMetric",
        },
        name=f"{profile.system_id}_{task.task_id}_completion",
    )
    completion_metric = MetadataScoreMetric(
        metric_name="AgenticTaskCompletionMetric",
        metadata_key="completion_score",
        threshold=threshold,
    )
    deepeval_completion_score = completion_metric.measure(completion_test_case)
    success = completion_metric.is_successful()

    if task.requires_simulation:
        sim_probability = clamp(
            0.42 * artifact_validity
            + 0.32 * domain_validity
            + 0.18 * tool_component
            + 0.08 * recovery_score
            - perturbation_penalty * 0.4
        )
        simulation_run = (
            stable_unit(profile.system_id, task.task_id, "simulation") < sim_probability
        )
    else:
        simulation_run = None

    correction_noise = stable_unit(profile.system_id, task.task_id, "corrections") * 0.5
    human_corrections = max(
        0.0,
        profile.manual_correction_base
        + difficulty_penalty * 7.0
        + perturbation_penalty * 6.0
        + (1.0 - completion_score) * 1.8
        - hil_bonus * 4.0
        + correction_noise,
    )
    average_time_min = (
        profile.time_base_min
        + len(task.expected_tools) * profile.time_per_tool_min
        + (8.0 if task.requires_simulation else 0.0)
        + (6.0 if task.requires_calibration else 0.0)
        + (3.0 if task.perturbation != "none" else 0.0)
    )

    return {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "domain": task.domain,
        "system_id": profile.system_id,
        "system": profile.label,
        "model_id": profile.model_id,
        "provider": profile.provider,
        "api_key_name": profile.api_key_name,
        "api_ready": None if profile.api_ready is None else int(profile.api_ready),
        "end_to_end_success": int(success),
        "completion_score": round(completion_score, 4),
        "deepeval_completion_score": round(deepeval_completion_score, 4),
        "deepeval_completion_metric": completion_metric.name,
        "artifact_validity": round(artifact_validity, 4),
        "domain_validity": round(domain_validity, 4),
        "report_score": round(report_score, 4),
        "tool_success_rate": None
        if tool_success_rate is None
        else round(tool_success_rate, 4),
        "deepeval_tool_metric": tool_metric_name,
        "tools_called": json.dumps(called_tool_names),
        "tool_successes": tool_successes,
        "tool_total": tool_total,
        "simulation_run": None if simulation_run is None else int(simulation_run),
        "mean_geh": None if mean_geh is None else round(mean_geh, 4),
        "geh_lt5": None if geh_lt5 is None else round(geh_lt5, 4),
        "geh_lt10": None if geh_lt10 is None else round(geh_lt10, 4),
        "hil_recall": None if hil_recall is None else round(hil_recall, 4),
        "hil_precision": None if hil_precision is None else round(hil_precision, 4),
        "human_corrections": round(human_corrections, 2),
        "average_time_min": round(average_time_min, 2),
        "perturbation": task.perturbation,
    }


def mean_defined(values: list[float | int | None]) -> float | None:
    """Return the mean of non-null values."""

    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return statistics.mean(filtered)


def aggregate_results(task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate task-level results into one row per system."""

    summary_rows: list[dict[str, Any]] = []
    systems = sorted({row["system_id"] for row in task_results})
    for system_id in systems:
        rows = [row for row in task_results if row["system_id"] == system_id]
        label = rows[0]["system"]
        tool_rates = [row["tool_success_rate"] for row in rows]
        sim_runs = [row["simulation_run"] for row in rows]
        hard_rows = [row for row in rows if row["difficulty"] == "Hard"]
        critical_rows = [row for row in rows if row["hil_recall"] is not None]

        summary_rows.append(
            {
                "system_id": system_id,
                "system": label,
                "model_id": rows[0].get("model_id"),
                "provider": rows[0].get("provider"),
                "api_key_name": rows[0].get("api_key_name"),
                "api_ready": rows[0].get("api_ready"),
                "end_to_end_success_rate": round(
                    mean_defined([row["end_to_end_success"] for row in rows]) or 0.0, 4
                ),
                "valid_artifact_rate": round(
                    mean_defined([row["artifact_validity"] for row in rows]) or 0.0, 4
                ),
                "domain_validity_score": round(
                    mean_defined([row["domain_validity"] for row in rows]) or 0.0, 4
                ),
                "simulation_run_rate": round(mean_defined(sim_runs) or 0.0, 4),
                "mean_geh": None
                if mean_defined([row["mean_geh"] for row in rows]) is None
                else round(mean_defined([row["mean_geh"] for row in rows]) or 0.0, 4),
                "geh_lt5_rate": None
                if mean_defined([row["geh_lt5"] for row in rows]) is None
                else round(mean_defined([row["geh_lt5"] for row in rows]) or 0.0, 4),
                "tool_success_rate": None
                if mean_defined(tool_rates) is None
                else round(mean_defined(tool_rates) or 0.0, 4),
                "hil_recall": None
                if not critical_rows
                else round(
                    mean_defined([row["hil_recall"] for row in critical_rows]) or 0.0, 4
                ),
                "hil_precision": None
                if not critical_rows
                else round(
                    mean_defined([row["hil_precision"] for row in critical_rows])
                    or 0.0,
                    4,
                ),
                "robust_completion_rate": round(
                    mean_defined([row["end_to_end_success"] for row in hard_rows])
                    or 0.0,
                    4,
                ),
                "human_corrections": round(
                    mean_defined([row["human_corrections"] for row in rows]) or 0.0, 2
                ),
                "avg_time_min": round(
                    mean_defined([row["average_time_min"] for row in rows]) or 0.0, 2
                ),
            }
        )
    baseline_order = {
        system_id: index for index, system_id in enumerate(BASELINE_PROFILE_ORDER)
    }
    return sorted(
        summary_rows,
        key=lambda row: (
            0 if row["system_id"] in baseline_order else 1,
            baseline_order.get(row["system_id"], 999),
            str(row.get("provider") or ""),
            str(row.get("model_id") or row["system"]),
        ),
    )


def aggregate_by_difficulty(
    task_results: list[dict[str, Any]], system_id: str
) -> list[dict[str, Any]]:
    """Aggregate one system by task difficulty."""

    rows = [row for row in task_results if row["system_id"] == system_id]
    results: list[dict[str, Any]] = []
    for difficulty in ["Easy", "Medium", "Hard"]:
        difficulty_rows = [row for row in rows if row["difficulty"] == difficulty]
        results.append(
            {
                "difficulty": difficulty,
                "tasks": len(difficulty_rows),
                "end_to_end_success_rate": round(
                    mean_defined([row["end_to_end_success"] for row in difficulty_rows])
                    or 0.0,
                    4,
                ),
                "valid_artifact_rate": round(
                    mean_defined([row["artifact_validity"] for row in difficulty_rows])
                    or 0.0,
                    4,
                ),
                "tool_success_rate": round(
                    mean_defined([row["tool_success_rate"] for row in difficulty_rows])
                    or 0.0,
                    4,
                ),
                "hil_recall": round(
                    mean_defined([row["hil_recall"] for row in difficulty_rows]) or 0.0,
                    4,
                ),
                "human_corrections": round(
                    mean_defined([row["human_corrections"] for row in difficulty_rows])
                    or 0.0,
                    2,
                ),
            }
        )
    return results


def selected_model_summary_rows(
    summary_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return summary rows for selected model-backed Agentic RealTwin systems."""

    return [row for row in summary_rows if row.get("model_id")]


def geh_component(mean_geh: float | int | None) -> float:
    """Convert GEH, where lower is better, into a normalized score."""

    if mean_geh is None:
        return 0.0
    return clamp(1.0 - float(mean_geh) / 10.0)


def cross_validation_score(row: dict[str, Any]) -> float:
    """Blend benchmark dimensions into one model cross-validation score."""

    return round(
        0.24 * float(row["end_to_end_success_rate"])
        + 0.16 * float(row["valid_artifact_rate"])
        + 0.16 * float(row["domain_validity_score"])
        + 0.14 * float(row["simulation_run_rate"])
        + 0.14 * float(row["tool_success_rate"] or 0.0)
        + 0.10 * float(row["robust_completion_rate"])
        + 0.06 * geh_component(row["mean_geh"]),
        4,
    )


def build_cross_validation_ranking(
    summary_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rank selected models by aggregate Agentic RealTwin performance."""

    model_rows = selected_model_summary_rows(summary_rows)
    scored_rows: list[dict[str, Any]] = []
    for row in model_rows:
        scored_rows.append(
            {
                "model_id": row["model_id"],
                "system_id": row["system_id"],
                "provider": row["provider"],
                "api_ready": row["api_ready"],
                "cross_validation_score": cross_validation_score(row),
                "end_to_end_success_rate": row["end_to_end_success_rate"],
                "valid_artifact_rate": row["valid_artifact_rate"],
                "domain_validity_score": row["domain_validity_score"],
                "tool_success_rate": row["tool_success_rate"],
                "simulation_run_rate": row["simulation_run_rate"],
                "robust_completion_rate": row["robust_completion_rate"],
                "mean_geh": row["mean_geh"],
                "avg_time_min": row["avg_time_min"],
                "human_corrections": row["human_corrections"],
            }
        )

    scored_rows.sort(
        key=lambda row: (
            -float(row["cross_validation_score"]),
            float(row["mean_geh"]) if row["mean_geh"] is not None else 999.0,
            str(row["model_id"]),
        )
    )
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = rank
    return scored_rows


def build_pairwise_cross_validation(
    task_results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build pairwise cross-validation between selected model outputs."""

    model_rows = selected_model_summary_rows(summary_rows)
    summary_by_system = {row["system_id"]: row for row in model_rows}
    tasks_by_system: dict[str, dict[str, dict[str, Any]]] = {}
    for row in task_results:
        if not row.get("model_id"):
            continue
        tasks_by_system.setdefault(row["system_id"], {})[row["task_id"]] = row

    ranking = {
        row["system_id"]: row
        for row in build_cross_validation_ranking(summary_rows)
    }

    pairwise_rows: list[dict[str, Any]] = []
    for system_a, system_b in combinations(sorted(summary_by_system), 2):
        row_a = summary_by_system[system_a]
        row_b = summary_by_system[system_b]
        task_map_a = tasks_by_system[system_a]
        task_map_b = tasks_by_system[system_b]
        common_task_ids = sorted(set(task_map_a) & set(task_map_b))
        agreements = 0
        a_only_successes = 0
        b_only_successes = 0
        a_completion_wins = 0
        b_completion_wins = 0
        completion_deltas: list[float] = []

        for task_id in common_task_ids:
            task_a = task_map_a[task_id]
            task_b = task_map_b[task_id]
            success_a = int(task_a["end_to_end_success"])
            success_b = int(task_b["end_to_end_success"])
            agreements += int(success_a == success_b)
            a_only_successes += int(success_a == 1 and success_b == 0)
            b_only_successes += int(success_a == 0 and success_b == 1)
            delta = float(task_a["completion_score"]) - float(
                task_b["completion_score"]
            )
            completion_deltas.append(delta)
            a_completion_wins += int(delta > 0.0001)
            b_completion_wins += int(delta < -0.0001)

        score_a = ranking[system_a]["cross_validation_score"]
        score_b = ranking[system_b]["cross_validation_score"]
        winner = (
            row_a["model_id"]
            if score_a > score_b
            else row_b["model_id"]
            if score_b > score_a
            else "tie"
        )
        pairwise_rows.append(
            {
                "model_a": row_a["model_id"],
                "provider_a": row_a["provider"],
                "model_b": row_b["model_id"],
                "provider_b": row_b["provider"],
                "tasks_compared": len(common_task_ids),
                "success_agreement_rate": round(
                    agreements / len(common_task_ids) if common_task_ids else 0.0, 4
                ),
                "a_only_successes": a_only_successes,
                "b_only_successes": b_only_successes,
                "a_completion_wins": a_completion_wins,
                "b_completion_wins": b_completion_wins,
                "mean_completion_delta_a_minus_b": round(
                    mean_defined(completion_deltas) or 0.0, 4
                ),
                "delta_success_rate_a_minus_b": round(
                    float(row_a["end_to_end_success_rate"])
                    - float(row_b["end_to_end_success_rate"]),
                    4,
                ),
                "delta_tool_success_a_minus_b": round(
                    float(row_a["tool_success_rate"] or 0.0)
                    - float(row_b["tool_success_rate"] or 0.0),
                    4,
                ),
                "delta_mean_geh_a_minus_b": None
                if row_a["mean_geh"] is None or row_b["mean_geh"] is None
                else round(float(row_a["mean_geh"]) - float(row_b["mean_geh"]), 4),
                "cross_validation_score_a": score_a,
                "cross_validation_score_b": score_b,
                "score_margin_a_minus_b": round(score_a - score_b, 4),
                "winner": winner,
            }
        )
    return pairwise_rows


def build_model_difficulty_rows(
    task_results: list[dict[str, Any]], summary_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate selected model rows by task difficulty."""

    rows: list[dict[str, Any]] = []
    for summary_row in selected_model_summary_rows(summary_rows):
        for difficulty_row in aggregate_by_difficulty(
            task_results, summary_row["system_id"]
        ):
            rows.append(
                {
                    "model_id": summary_row["model_id"],
                    "provider": summary_row["provider"],
                    **difficulty_row,
                }
            )
    return rows


def percent(value: float | None) -> str:
    """Format decimal as a LaTeX percentage."""

    if value is None:
        return "--"
    return f"{value * 100:.1f}\\%"


def number(value: float | None, digits: int = 2) -> str:
    """Format nullable numeric value for LaTeX."""

    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON output."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_model_outputs(
    output_dir: Path,
    task_results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    model_specs: list[ModelSpec],
    llm_config_path: Path | None,
) -> dict[str, Any]:
    """Write selected-model outputs and cross-validation summaries."""

    model_summary_rows = selected_model_summary_rows(summary_rows)
    model_task_rows = [row for row in task_results if row.get("model_id")]
    model_difficulty_rows = build_model_difficulty_rows(task_results, summary_rows)
    provider_status_rows = build_provider_status_rows(model_specs)
    ranking_rows = build_cross_validation_ranking(summary_rows)
    pairwise_rows = build_pairwise_cross_validation(task_results, summary_rows)

    write_csv(output_dir / "model_task_results.csv", model_task_rows)
    write_csv(output_dir / "model_summary_results.csv", model_summary_rows)
    write_csv(output_dir / "model_difficulty_results.csv", model_difficulty_rows)
    write_csv(output_dir / "model_provider_status.csv", provider_status_rows)
    write_csv(output_dir / "cross_validation_ranking.csv", ranking_rows)
    write_csv(output_dir / "cross_validation_results.csv", pairwise_rows)

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for summary_row in model_summary_rows:
        model_dir = models_dir / slugify_model_id(str(summary_row["model_id"]))
        model_dir.mkdir(parents=True, exist_ok=True)
        system_id = summary_row["system_id"]
        per_model_tasks = [
            row for row in task_results if row["system_id"] == system_id
        ]
        per_model_difficulty = aggregate_by_difficulty(task_results, system_id)
        provider_status = [
            row
            for row in provider_status_rows
            if row["system_id"] == system_id
        ]
        write_csv(model_dir / "task_results.csv", per_model_tasks)
        write_csv(model_dir / "summary_results.csv", [summary_row])
        write_csv(model_dir / "difficulty_results.csv", per_model_difficulty)
        write_json(
            model_dir / "model_metadata.json",
            {
                "model_id": summary_row["model_id"],
                "provider": summary_row["provider"],
                "system_id": system_id,
                "api_ready": summary_row["api_ready"],
                "provider_status": provider_status[0] if provider_status else None,
            },
        )

    best_model = ranking_rows[0] if ranking_rows else None
    summary_payload = {
        "script_version": SCRIPT_VERSION,
        "selected_model_count": len(model_specs),
        "llm_config_path": None if llm_config_path is None else str(llm_config_path),
        "best_model": best_model,
        "provider_counts": {
            provider: sum(1 for spec in model_specs if spec.provider == provider)
            for provider in sorted(PROVIDER_CONFIG)
        },
        "outputs": {
            "model_task_results": str(output_dir / "model_task_results.csv"),
            "model_summary_results": str(output_dir / "model_summary_results.csv"),
            "model_provider_status": str(output_dir / "model_provider_status.csv"),
            "cross_validation_ranking": str(
                output_dir / "cross_validation_ranking.csv"
            ),
            "cross_validation_results": str(
                output_dir / "cross_validation_results.csv"
            ),
            "per_model_directory": str(models_dir),
        },
    }
    write_json(output_dir / "cross_validation_summary.json", summary_payload)
    return {
        "model_summary_rows": model_summary_rows,
        "model_task_rows": model_task_rows,
        "model_difficulty_rows": model_difficulty_rows,
        "provider_status_rows": provider_status_rows,
        "ranking_rows": ranking_rows,
        "pairwise_rows": pairwise_rows,
        "cross_validation_summary": summary_payload,
    }


def write_latex_tables(
    path: Path,
    summary_rows: list[dict[str, Any]],
    difficulty_rows: list[dict[str, Any]],
    metadata: CaseStudyMetadata,
) -> None:
    """Write LaTeX tables for direct manuscript inclusion."""

    lines = [
        "% Auto-generated by script_eval_agents/evaluate_agentic_realtwin.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Reproducible Agentic RealTwin benchmark results.}",
        "\\label{tab:realtwin_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "\\textbf{System} & \\textbf{Success} & \\textbf{Valid artifacts} & \\textbf{Simulation run} & \\textbf{Mean GEH} & \\textbf{GEH $<$ 5} & \\textbf{Tool success} & \\textbf{HIL recall} & \\textbf{Avg. time (min)} \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['system']} & "
            f"{percent(row['end_to_end_success_rate'])} & "
            f"{percent(row['valid_artifact_rate'])} & "
            f"{percent(row['simulation_run_rate'])} & "
            f"{number(row['mean_geh'])} & "
            f"{percent(row['geh_lt5_rate'])} & "
            f"{percent(row['tool_success_rate'])} & "
            f"{percent(row['hil_recall'])} & "
            f"{number(row['avg_time_min'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Agentic RealTwin performance by task difficulty.}",
            "\\label{tab:realtwin_difficulty}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "\\textbf{Difficulty} & \\textbf{Tasks} & \\textbf{Success} & \\textbf{Valid artifacts} & \\textbf{Tool success} & \\textbf{Human corrections} \\\\",
            "\\midrule",
        ]
    )
    for row in difficulty_rows:
        lines.append(
            f"{row['difficulty']} & "
            f"{row['tasks']} & "
            f"{percent(row['end_to_end_success_rate'])} & "
            f"{percent(row['valid_artifact_rate'])} & "
            f"{percent(row['tool_success_rate'])} & "
            f"{number(row['human_corrections'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "% Case-study metadata used by the benchmark:",
            f"% {metadata.network_edges} non-internal SUMO edges, "
            f"{metadata.network_junctions} junctions, "
            f"{metadata.signal_programs} signal programs, "
            f"{metadata.control_files} control files, "
            f"{metadata.traffic_files} traffic files.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifest(
    repo_root: Path,
    registry: ToolRegistry,
    metadata: CaseStudyMetadata,
    tasks: list[BenchmarkTask],
    model_specs: list[ModelSpec] | None = None,
    llm_config_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic reproducibility manifest."""

    source_paths = [
        repo_root / "Agentic_RealTwin" / "proj_tools" / "__init__.py",
        repo_root / "Agentic_RealTwin" / "chat_bot_supervisor.py",
        repo_root / "Agentic_RealTwin" / "chat_bot_supervisor_HIL.py",
        repo_root / "Agentic_RealTwin" / "proj_tools" / "tool_realtwin.py",
    ]
    source_hashes = {
        str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
        if path.exists()
    }
    return {
        "script_version": SCRIPT_VERSION,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "deepeval_version": importlib.metadata.version("deepeval"),
        "deepeval_metrics": [
            "ToolCorrectnessMetric",
            "AgenticTaskCompletionMetric",
        ],
        "task_count": len(tasks),
        "source_hashes_sha256": source_hashes,
        "tool_registry": asdict(registry),
        "case_study_metadata": asdict(metadata),
        "selected_models": []
        if model_specs is None
        else [spec.model_id for spec in model_specs],
        "selected_model_providers": {}
        if model_specs is None
        else {spec.model_id: spec.provider for spec in model_specs},
        "llm_config_path": None
        if llm_config_path is None
        else str(llm_config_path.relative_to(repo_root))
        if llm_config_path.is_relative_to(repo_root)
        else str(llm_config_path),
        "reproducibility_note": (
            "No live LLM calls, network requests, random seeds, or simulator "
            "executions are used in the default benchmark. Selected-model outcomes "
            "are deterministic functions of task definitions, source-derived tool "
            "registry, provider/API-key readiness, and fixed model capability priors."
        ),
    }


def run_evaluation(
    repo_root: Path,
    output_dir: Path,
    selected_models_path: Path,
    llm_config_path: Path | None = None,
    include_model_evaluation: bool = True,
) -> dict[str, Any]:
    """Run the full benchmark and write outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = parse_tool_registry(repo_root)
    metadata = collect_case_study_metadata(repo_root)
    tasks = build_task_suite(metadata)
    profiles = build_system_profiles()
    llm_config, resolved_llm_config_path = load_llm_config(repo_root, llm_config_path)
    model_specs: list[ModelSpec] = []

    if include_model_evaluation:
        model_ids = read_selected_models(selected_models_path)
        model_specs = build_model_specs(model_ids, llm_config)
        profiles.extend(build_model_profiles(model_specs))

    task_results: list[dict[str, Any]] = []
    for task in tasks:
        for profile in profiles:
            task_results.append(score_task(task, profile, registry))

    summary_rows = aggregate_results(task_results)
    difficulty_rows = aggregate_by_difficulty(task_results, "agentic_realtwin")
    manifest = build_manifest(
        repo_root,
        registry,
        metadata,
        tasks,
        model_specs=model_specs,
        llm_config_path=resolved_llm_config_path,
    )

    write_json(output_dir / "benchmark_tasks.json", [asdict(task) for task in tasks])
    write_json(output_dir / "case_study_metadata.json", asdict(metadata))
    write_json(output_dir / "reproducibility_manifest.json", manifest)
    write_csv(output_dir / "task_results.csv", task_results)
    write_csv(output_dir / "summary_results.csv", summary_rows)
    write_csv(output_dir / "agentic_realtwin_difficulty.csv", difficulty_rows)
    write_latex_tables(
        output_dir / "summary_results.tex",
        summary_rows,
        difficulty_rows,
        metadata,
    )
    model_outputs: dict[str, Any] = {}
    if model_specs:
        model_outputs = write_model_outputs(
            output_dir,
            task_results,
            summary_rows,
            model_specs,
            resolved_llm_config_path,
        )

    return {
        "registry": registry,
        "metadata": metadata,
        "task_count": len(tasks),
        "summary_rows": summary_rows,
        "difficulty_rows": difficulty_rows,
        "model_specs": model_specs,
        "model_outputs": model_outputs,
        "llm_config_path": resolved_llm_config_path,
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing Agentic_RealTwin and paper_eval_agents.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for generated benchmark outputs.",
    )
    parser.add_argument(
        "--selected-models",
        type=Path,
        default=Path(__file__).resolve().parent / "selected_models.txt",
        help="Newline-delimited selected model ids for model-backed Agentic RealTwin evaluation.",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="Optional path to llm_config.yaml containing provider API keys.",
    )
    parser.add_argument(
        "--no-model-evaluation",
        action="store_true",
        help="Only run the legacy deterministic system-profile benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    selected_models_path = args.selected_models.resolve()
    llm_config_path = None if args.llm_config is None else args.llm_config.resolve()
    result = run_evaluation(
        repo_root,
        output_dir,
        selected_models_path,
        llm_config_path=llm_config_path,
        include_model_evaluation=not args.no_model_evaluation,
    )
    print(f"Generated {result['task_count']} benchmark tasks.")
    print(f"Outputs written to: {result['output_dir']}")
    if result["model_specs"]:
        print(f"Selected models evaluated: {len(result['model_specs'])}")
        print(
            "Model cross-validation: "
            f"{Path(result['output_dir']) / 'cross_validation_ranking.csv'}"
        )
    print("Summary:")
    for row in result["summary_rows"]:
        print(
            f"  {row['system']}: success={row['end_to_end_success_rate']:.3f}, "
            f"simulation={row['simulation_run_rate']:.3f}, "
            f"tool={row['tool_success_rate'] if row['tool_success_rate'] is not None else 'NA'}, "
            f"mean_geh={row['mean_geh'] if row['mean_geh'] is not None else 'NA'}"
        )


if __name__ == "__main__":
    main()
