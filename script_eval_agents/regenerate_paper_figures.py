"""Regenerate manuscript Figures 1--3 and matching Draw.io sources.

The PNG and Draw.io files are built from the same diagram specifications so
their content, layout, colors, and labels stay consistent. Rendered manuscript
figures are PNG-only; no PDF or SVG figure files are written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from textwrap import fill
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "paper_eval_agents" / "static"
DATA_DIR = ROOT_DIR / "script_eval_agents" / "outputs"

PALETTE = {
    "ink": "#172033",
    "muted": "#5E6B7E",
    "line": "#D8E0EA",
    "panel": "#F7F9FC",
    "blue": "#157EB9",
    "blue_soft": "#E8F3FB",
    "teal": "#009B82",
    "teal_soft": "#E5F6F2",
    "violet": "#6F3FA0",
    "violet_soft": "#F0E8F8",
    "amber": "#E99A00",
    "amber_soft": "#FFF3D8",
    "red": "#D5311F",
    "red_soft": "#FDE8E5",
    "green": "#16A34A",
    "green_soft": "#E8F6ED",
    "cyan": "#28AEE4",
    "cyan_soft": "#E7F6FC",
    "gray": "#708090",
    "gray_soft": "#EEF2F6",
}

SYSTEM_ORDER = ["manual_expert", "script_only", "single_llm", "rag_assistant", "agentic_realtwin"]
SYSTEM_LABELS = {
    "manual_expert": "Manual expert",
    "script_only": "Deterministic",
    "single_llm": "Single LLM",
    "rag_assistant": "RAG assistant",
    "agentic_realtwin": "RealTwin + HIL",
}


@dataclass
class Diagram:
    """Shared logical diagram rendered to both PNG and Draw.io."""

    stem: str
    figsize: tuple[float, float]
    elements: list[dict[str, Any]] = field(default_factory=list)

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        size: float = 5.0,
        color: str = PALETTE["ink"],
        bold: bool = False,
        align: str = "center",
        valign: str = "center",
        wrap: int | None = None,
    ) -> None:
        """Add a text block."""
        self.elements.append(
            {
                "kind": "text",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "value": value,
                "size": size,
                "color": color,
                "bold": bold,
                "align": align,
                "valign": valign,
                "wrap": wrap,
            }
        )

    def panel(self, x: float, y: float, w: float, h: float, title: str, label: str | None = None, fill_color: str = "white", edge_color: str = PALETTE["line"]) -> None:
        """Add a labeled panel frame."""
        self.elements.append({"kind": "panel", "x": x, "y": y, "w": w, "h": h, "title": title, "label": label, "fill": fill_color, "edge": edge_color})

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        edge_color: str,
        fill_color: str = "white",
        title_size: float = 5.0,
        body_size: float = 3.8,
        align: str = "center",
    ) -> None:
        """Add a rounded content box."""
        self.elements.append(
            {
                "kind": "box",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "title": title,
                "body": body,
                "edge": edge_color,
                "fill": fill_color,
                "title_size": title_size,
                "body_size": body_size,
                "align": align,
            }
        )

    def pill(self, x: float, y: float, w: float, h: float, label: str, edge_color: str, fill_color: str, subtitle: str = "") -> None:
        """Add a compact rounded status pill."""
        self.elements.append({"kind": "pill", "x": x, "y": y, "w": w, "h": h, "label": label, "subtitle": subtitle, "edge": edge_color, "fill": fill_color})

    def arrow(self, start: tuple[float, float], end: tuple[float, float], color: str = PALETTE["gray"], lw: float = 0.9, curve: float = 0.0, label: str = "") -> None:
        """Add a directional arrow."""
        self.elements.append({"kind": "arrow", "start": start, "end": end, "color": color, "lw": lw, "curve": curve, "label": label})

    def diamond(self, x: float, y: float, w: float, h: float, title: str, body: str, edge_color: str = PALETTE["gray"], fill_color: str = "white") -> None:
        """Add a decision-gate diamond."""
        self.elements.append({"kind": "diamond", "x": x, "y": y, "w": w, "h": h, "title": title, "body": body, "edge": edge_color, "fill": fill_color})

    def value_box(self, x: float, y: float, w: float, h: float, value: str, edge_color: str, fill_color: str) -> None:
        """Add a compact numeric value box."""
        self.elements.append({"kind": "value_box", "x": x, "y": y, "w": w, "h": h, "value": value, "edge": edge_color, "fill": fill_color})

    def bar(self, x: float, y: float, w: float, h: float, value: float, label: str, color: str, min_value: float, max_value: float) -> None:
        """Add a horizontal normalized bar."""
        self.elements.append({"kind": "bar", "x": x, "y": y, "w": w, "h": h, "value": value, "label": label, "color": color, "min": min_value, "max": max_value})

    def render_png(self) -> Path:
        """Render the diagram as a PNG file."""
        apply_publication_style()
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(0, 100)
        ax.set_ylim(100, 0)
        ax.axis("off")
        for element in self.elements:
            render_element_png(ax, element)
        path = OUTPUT_DIR / f"{self.stem}.png"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        return path

    def render_drawio(self) -> Path:
        """Render the same diagram specification as editable Draw.io XML."""
        page_width = 1400
        page_height = int(page_width * self.figsize[1] / self.figsize[0])
        writer = DrawioWriter(self.stem, page_width, page_height)
        for element in self.elements:
            writer.add_element(element)
        return writer.write(OUTPUT_DIR / f"{self.stem}.drawio")


class DrawioWriter:
    """Minimal Draw.io XML writer using the same logical coordinates as PNG."""

    def __init__(self, stem: str, page_width: int, page_height: int) -> None:
        self.page_width = page_width
        self.page_height = page_height
        self.scale_x = page_width / 100
        self.scale_y = page_height / 100
        self.next_id = 2
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": datetime.now(UTC).isoformat(timespec="seconds"),
                "agent": "Codex",
                "version": "24.7.17",
            },
        )
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": stem, "name": "Page-1"})
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": str(page_width),
                "dy": str(page_height),
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(page_width),
                "pageHeight": str(page_height),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def add_element(self, element: dict[str, Any]) -> None:
        """Add one diagram element."""
        kind = element["kind"]
        if kind == "panel":
            self._rect(element["x"], element["y"], element["w"], element["h"], "", element["edge"], element["fill"], rounded=True, stroke_width=1.5)
            if element.get("label"):
                self._rect(element["x"] + 0.3, element["y"] + 0.8, 2.1, 3.8, f"<b>{escape(element['label'])}</b>", PALETTE["ink"], PALETTE["ink"], rounded=True, font_color="#FFFFFF", font_size=15)
                self._text(element["x"] + 3.2, element["y"] + 0.9, element["w"] - 4.5, 3.4, f"<b>{escape(element['title'])}</b>", font_size=16, align="left")
            else:
                self._text(element["x"] + 1.4, element["y"] + 0.9, element["w"] - 2.8, 3.4, f"<b>{escape(element['title'])}</b>", font_size=15, align="left")
        elif kind == "box":
            value = f"<b>{escape(element['title'])}</b>"
            if element["body"]:
                value += f"<br><font style=\"font-size:{max(8, int(element['body_size'] * 2.0))}px\">{escape(element['body']).replace(chr(10), '<br>')}</font>"
            self._rect(element["x"], element["y"], element["w"], element["h"], value, element["edge"], element["fill"], rounded=True, font_size=max(10, int(element["title_size"] * 2.0)), align=element.get("align", "center"))
        elif kind == "pill":
            value = f"<b>{escape(element['label'])}</b>"
            if element.get("subtitle"):
                value += f"<br><font style=\"font-size:10px\">{escape(element['subtitle'])}</font>"
            self._rect(element["x"], element["y"], element["w"], element["h"], value, element["edge"], element["fill"], rounded=True, font_color=element["edge"], font_size=15)
        elif kind == "text":
            text = escape(element["value"]).replace("\n", "<br>")
            if element.get("bold"):
                text = f"<b>{text}</b>"
            self._text(element["x"], element["y"], element["w"], element["h"], text, font_size=max(8, int(element["size"] * 2.0)), font_color=element["color"], align=element.get("align", "center"))
        elif kind == "arrow":
            self._edge(element["start"], element["end"], element["color"], max(1, element["lw"] * 2.0), element.get("label", ""))
        elif kind == "diamond":
            value = f"<b>{escape(element['title'])}</b><br><font style=\"font-size:10px\">{escape(element['body']).replace(chr(10), '<br>')}</font>"
            self._rect(element["x"], element["y"], element["w"], element["h"], value, element["edge"], element["fill"], shape="rhombus", rounded=False, font_size=14)
        elif kind == "value_box":
            self._rect(element["x"], element["y"], element["w"], element["h"], f"<b>{escape(element['value'])}</b>", element["edge"], element["fill"], rounded=True, font_color=element["edge"], font_size=13)
        elif kind == "bar":
            self._rect(element["x"], element["y"], element["w"], element["h"], "", "none", PALETTE["gray_soft"], rounded=False)
            span = max(element["max"] - element["min"], 1e-9)
            fill_width = element["w"] * max(0.0, min((element["value"] - element["min"]) / span, 1.0))
            self._rect(element["x"], element["y"], fill_width, element["h"], "", "none", element["color"], rounded=False)
            self._text(element["x"] + element["w"] + 0.6, element["y"] - 0.4, 5.0, element["h"] + 0.8, escape(element["label"]), font_size=10, font_color=PALETTE["muted"], align="left")

    def write(self, path: Path) -> Path:
        """Write the Draw.io XML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)
        return path

    def _new_id(self) -> str:
        cell_id = str(self.next_id)
        self.next_id += 1
        return cell_id

    def _xywh(self, x: float, y: float, w: float, h: float) -> dict[str, str]:
        return {
            "x": f"{x * self.scale_x:.2f}",
            "y": f"{y * self.scale_y:.2f}",
            "width": f"{w * self.scale_x:.2f}",
            "height": f"{h * self.scale_y:.2f}",
        }

    def _rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        stroke: str,
        fill_color: str,
        rounded: bool = True,
        shape: str = "rectangle",
        font_color: str = PALETTE["ink"],
        font_size: int = 12,
        align: str = "center",
        stroke_width: float = 1.0,
    ) -> None:
        rounded_value = "1" if rounded else "0"
        style = (
            f"shape={shape};rounded={rounded_value};whiteSpace=wrap;html=1;arcSize=8;"
            f"strokeColor={stroke};fillColor={fill_color};strokeWidth={stroke_width};"
            f"fontSize={font_size};fontColor={font_color};align={align};verticalAlign=middle;spacing=6;"
        )
        cell = ET.SubElement(self.root, "mxCell", {"id": self._new_id(), "value": value, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {**self._xywh(x, y, w, h), "as": "geometry"})

    def _text(self, x: float, y: float, w: float, h: float, value: str, font_size: int = 12, font_color: str = PALETTE["ink"], align: str = "center") -> None:
        style = f"text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;rounded=0;fontSize={font_size};fontColor={font_color};align={align};verticalAlign=middle;"
        cell = ET.SubElement(self.root, "mxCell", {"id": self._new_id(), "value": value, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {**self._xywh(x, y, w, h), "as": "geometry"})

    def _edge(self, start: tuple[float, float], end: tuple[float, float], color: str, width: float, label: str = "") -> None:
        style = f"endArrow=block;html=1;rounded=0;strokeWidth={width:.1f};strokeColor={color};"
        cell = ET.SubElement(self.root, "mxCell", {"id": self._new_id(), "value": escape(label), "style": style, "edge": "1", "parent": "1"})
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geometry, "mxPoint", {"x": f"{start[0] * self.scale_x:.2f}", "y": f"{start[1] * self.scale_y:.2f}", "as": "sourcePoint"})
        ET.SubElement(geometry, "mxPoint", {"x": f"{end[0] * self.scale_x:.2f}", "y": f"{end[1] * self.scale_y:.2f}", "as": "targetPoint"})


def apply_publication_style() -> None:
    """Apply the Python figure backend style used for all manuscript figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["font.size"] = 6.5
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["legend.frameon"] = False


def render_element_png(ax: plt.Axes, element: dict[str, Any]) -> None:
    """Render one shared diagram element into a matplotlib axis."""
    kind = element["kind"]
    if kind == "panel":
        draw_round_rect(ax, element["x"], element["y"], element["w"], element["h"], element["edge"], element["fill"], lw=0.8)
        if element.get("label"):
            draw_round_rect(ax, element["x"] + 0.3, element["y"] + 0.8, 2.1, 3.8, PALETTE["ink"], PALETTE["ink"], lw=0.4, radius=0.35)
            draw_text(ax, element["x"] + 1.35, element["y"] + 2.7, element["label"], 6.2, "white", bold=True)
            draw_text(ax, element["x"] + 3.4, element["y"] + 2.6, element["title"], 6.6, PALETTE["ink"], bold=True, ha="left")
        else:
            draw_text(ax, element["x"] + 1.5, element["y"] + 2.6, element["title"], 6.2, PALETTE["ink"], bold=True, ha="left")
    elif kind == "box":
        draw_round_rect(ax, element["x"], element["y"], element["w"], element["h"], element["edge"], element["fill"], lw=0.95)
        ha = "center" if element.get("align", "center") == "center" else "left"
        x_text = element["x"] + element["w"] / 2 if ha == "center" else element["x"] + 1.25
        draw_text(ax, x_text, element["y"] + element["h"] * 0.32, fill(element["title"], max(16, int(element["w"] * 1.2))), element["title_size"], PALETTE["ink"], bold=True, ha=ha)
        draw_text(ax, x_text, element["y"] + element["h"] * 0.66, fill(element["body"], max(18, int(element["w"] * 1.25))), element["body_size"], PALETTE["muted"], ha=ha)
    elif kind == "pill":
        draw_round_rect(ax, element["x"], element["y"], element["w"], element["h"], element["edge"], element["fill"], lw=0.8, radius=0.8)
        y_main = element["y"] + element["h"] * (0.43 if element.get("subtitle") else 0.52)
        draw_text(ax, element["x"] + element["w"] / 2, y_main, element["label"], 5.8, element["edge"], bold=True)
        if element.get("subtitle"):
            draw_text(ax, element["x"] + element["w"] / 2, element["y"] + element["h"] * 0.68, element["subtitle"], 4.2, PALETTE["muted"])
    elif kind == "text":
        value = fill(element["value"], element["wrap"]) if element.get("wrap") else element["value"]
        x_text = element["x"] + element["w"] / 2 if element.get("align", "center") == "center" else element["x"]
        y_text = element["y"] + element["h"] / 2
        draw_text(ax, x_text, y_text, value, element["size"], element["color"], bold=element.get("bold", False), ha=element.get("align", "center"), va=element.get("valign", "center"))
    elif kind == "arrow":
        ax.add_patch(
            FancyArrowPatch(
                element["start"],
                element["end"],
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=element["lw"],
                color=element["color"],
                connectionstyle=f"arc3,rad={element['curve']}",
                shrinkA=0,
                shrinkB=0,
            )
        )
        if element.get("label"):
            x_mid = (element["start"][0] + element["end"][0]) / 2
            y_mid = (element["start"][1] + element["end"][1]) / 2
            draw_text(ax, x_mid, y_mid - 1.2, element["label"], 4.3, element["color"], bbox=True)
    elif kind == "diamond":
        x = element["x"] + element["w"] / 2
        y = element["y"] + element["h"] / 2
        ax.add_patch(
            Polygon(
                [(x, element["y"]), (element["x"] + element["w"], y), (x, element["y"] + element["h"]), (element["x"], y)],
                closed=True,
                linewidth=0.9,
                edgecolor=element["edge"],
                facecolor=element["fill"],
            )
        )
        draw_text(ax, x, y - 1.1, element["title"], 5.5, PALETTE["ink"], bold=True)
        draw_text(ax, x, y + 2.4, element["body"], 4.25, PALETTE["muted"])
    elif kind == "value_box":
        draw_round_rect(ax, element["x"], element["y"], element["w"], element["h"], element["edge"], element["fill"], lw=0.75, radius=0.7)
        draw_text(ax, element["x"] + element["w"] / 2, element["y"] + element["h"] / 2, element["value"], 4.7, element["edge"], bold=True)
    elif kind == "bar":
        ax.add_patch(Rectangle((element["x"], element["y"]), element["w"], element["h"], linewidth=0, facecolor=PALETTE["gray_soft"]))
        span = max(element["max"] - element["min"], 1e-9)
        width = element["w"] * max(0.0, min((element["value"] - element["min"]) / span, 1.0))
        ax.add_patch(Rectangle((element["x"], element["y"]), width, element["h"], linewidth=0, facecolor=element["color"]))
        draw_text(ax, element["x"] + element["w"] + 0.7, element["y"] + element["h"] / 2, element["label"], 4.4, PALETTE["muted"], ha="left")


def draw_round_rect(ax: plt.Axes, x: float, y: float, w: float, h: float, edge: str, fill_color: str, lw: float = 0.8, radius: float = 0.85) -> None:
    """Draw a rounded rectangle."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={radius}", linewidth=lw, edgecolor=edge, facecolor=fill_color))


def draw_text(
    ax: plt.Axes,
    x: float,
    y: float,
    value: str,
    size: float,
    color: str,
    bold: bool = False,
    ha: str = "center",
    va: str = "center",
    bbox: bool = False,
) -> None:
    """Draw text in diagram coordinates."""
    bbox_kwargs = None
    if bbox:
        bbox_kwargs = {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": color, "linewidth": 0.5}
    ax.text(x, y, value, ha=ha, va=va, fontsize=size, fontweight="bold" if bold else "normal", color=color, linespacing=1.05, bbox=bbox_kwargs)


def load_benchmark_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load figure source data from deterministic benchmark outputs."""
    summary = pd.read_csv(DATA_DIR / "summary_results.csv")
    difficulty = pd.read_csv(DATA_DIR / "agentic_realtwin_difficulty.csv")
    ranking = pd.read_csv(DATA_DIR / "cross_validation_ranking.csv")
    return summary, difficulty, ranking


def make_figure_1(summary: pd.DataFrame, difficulty: pd.DataFrame, ranking: pd.DataFrame) -> Diagram:
    """Create Figure 1: framework overview plus quantitative readouts."""
    figure = Diagram("agentic_ai_transport_eval_framework", (7.25, 4.55))
    figure.text(5, 1.5, 90, 5, "Transportation-specific framework for evaluating agentic AI workflows", 9.4, bold=True)
    figure.text(10, 6.5, 80, 4, "Executable artifacts, validation gates, human oversight, and cross-validation readouts form one auditable evaluation unit.", 5.8, PALETTE["muted"])

    figure.panel(3, 12, 94, 47, "Evidence-producing benchmark episode", "a", PALETTE["panel"])
    workflow = [
        (6.5, "Transportation task", "request, study area, counts, constraints", PALETTE["blue"], PALETTE["blue_soft"]),
        (28.0, "Agentic RealTwin", "supervisor, tools, RAG, recovery logic", PALETTE["violet"], PALETTE["violet_soft"]),
        (49.5, "Generated artifacts", "network, routes, demand, signals, logs", PALETTE["teal"], PALETTE["teal_soft"]),
        (71.0, "Benchmark replay", "40 fixed tasks, status rules, backend CV", PALETTE["amber"], PALETTE["amber_soft"]),
    ]
    for x, title, body, edge, fill_color in workflow:
        figure.box(x, 19, 17.2, 8.2, title, body, edge, "white", 5.6, 4.1)
    for x in [23.7, 45.2, 66.7]:
        figure.arrow((x, 23.1), (x + 3.2, 23.1))
    figure.arrow((58.1, 27.2), (58.1, 36.5), PALETTE["teal"])
    figure.arrow((79.6, 27.2), (79.6, 36.5), PALETTE["amber"])
    figure.text(6.5, 30.5, 32, 3, "Validation/status generation", 6.2, bold=True, align="left")
    figure.text(6.5, 34.2, 72, 3, "Task traces, generated files, simulator outputs, HIL events, and GEH statistics are converted to auditable status labels.", 4.7, PALETTE["muted"], align="left")
    validation_layers = [
        (6.5, "Executable", "syntax + simulator run", PALETTE["blue"], PALETTE["blue_soft"]),
        (21.6, "Domain-valid", "network, demand, signal, GEH", PALETTE["teal"], PALETTE["teal_soft"]),
        (37.2, "Tool-reliable", "selection + arguments", PALETTE["violet"], PALETTE["violet_soft"]),
        (52.7, "Robust", "ambiguity + failed tools", PALETTE["amber"], PALETTE["amber_soft"]),
        (68.2, "Risk-controlled", "critical-action approval", PALETTE["red"], PALETTE["red_soft"]),
        (82.5, "Reproducible", "logs, seeds, artifacts", PALETTE["green"], PALETTE["green_soft"]),
    ]
    for x, title, body, edge, fill_color in validation_layers:
        figure.box(x, 38, 12.8, 6.2, title, body, edge, fill_color, 4.8, 3.5)
    for x, label, edge, fill_color in [
        (31.0, "PASS", PALETTE["green"], PALETTE["green_soft"]),
        (42.0, "WARN", PALETTE["amber"], PALETTE["amber_soft"]),
        (53.0, "FAIL", PALETTE["red"], PALETTE["red_soft"]),
        (64.0, "HIL REVIEW", PALETTE["cyan"], PALETTE["cyan_soft"]),
    ]:
        figure.pill(x, 48, 9.0, 4.4, label, edge, fill_color)
    figure.text(14, 55.0, 72, 3, "Deployment readiness is assigned from joint evidence across validity, execution, calibration, tool reliability, HIL routing, and reproducibility.", 4.6, PALETTE["muted"])

    add_system_comparison(figure, summary)
    add_difficulty_readout(figure, difficulty)
    add_model_readout(figure, ranking)
    return figure


def add_system_comparison(figure: Diagram, summary: pd.DataFrame) -> None:
    """Add the Figure 1 system-comparison panel."""
    figure.panel(3, 67, 30, 28, "System comparison", "b")
    figure.text(14.5, 71.5, 6, 2.5, "success", 4.7, PALETTE["blue"], bold=True)
    figure.text(21.0, 71.5, 6, 2.5, "sim run", 4.7, PALETTE["teal"], bold=True)
    figure.text(27.3, 71.5, 6, 2.5, "tool", 4.7, PALETTE["violet"], bold=True)
    selected = summary.set_index("system_id").loc[SYSTEM_ORDER]
    for row_index, system_id in enumerate(SYSTEM_ORDER):
        row = selected.loc[system_id]
        y = 74.5 + row_index * 4.0
        figure.text(5.2, y, 8.8, 2.4, SYSTEM_LABELS[system_id], 4.8, align="left")
        figure.value_box(14.2, y - 1.5, 4.3, 3.0, f"{row['end_to_end_success_rate'] * 100:.0f}%", PALETTE["blue"], PALETTE["blue_soft"])
        figure.value_box(20.7, y - 1.5, 4.3, 3.0, f"{row['simulation_run_rate'] * 100:.0f}%", PALETTE["teal"], PALETTE["teal_soft"])
        tool_value = "--" if pd.isna(row["tool_success_rate"]) else f"{row['tool_success_rate'] * 100:.0f}%"
        figure.value_box(27.0, y - 1.5, 4.3, 3.0, tool_value, PALETTE["violet"], PALETTE["violet_soft"] if tool_value != "--" else "white")


def add_difficulty_readout(figure: Diagram, difficulty: pd.DataFrame) -> None:
    """Add the Figure 1 difficulty panel."""
    figure.panel(35, 67, 30, 28, "Difficulty stress test", "c")
    for x, label, color in [(42.6, "success", PALETTE["blue"]), (49.8, "valid", PALETTE["teal"]), (56.9, "tool", PALETTE["violet"]), (62.0, "fixes", PALETTE["amber"])]:
        figure.text(x, 71.5, 6, 2.5, label, 4.7, color, bold=True)
    ordered = difficulty.set_index("difficulty").loc[["Easy", "Medium", "Hard"]].reset_index()
    for row_index, row in ordered.iterrows():
        y = 75.0 + row_index * 6.5
        figure.text(37.0, y, 6.5, 2.6, row["difficulty"], 5.2, bold=True, align="left")
        figure.value_box(40.3, y - 1.5, 5.2, 3.1, f"{row['end_to_end_success_rate'] * 100:.0f}%", PALETTE["blue"], PALETTE["blue_soft"])
        figure.value_box(47.5, y - 1.5, 5.2, 3.1, f"{row['valid_artifact_rate'] * 100:.0f}%", PALETTE["teal"], PALETTE["teal_soft"])
        figure.value_box(54.7, y - 1.5, 5.2, 3.1, f"{row['tool_success_rate'] * 100:.0f}%", PALETTE["violet"], PALETTE["violet_soft"])
        figure.value_box(60.4, y - 1.5, 3.4, 3.1, f"{row['human_corrections']:.2f}", PALETTE["amber"], PALETTE["amber_soft"])
    figure.text(41.5, 90.0, 17.0, 4.0, "Hard tasks expose missing-data and recovery burden.", 4.3, PALETTE["muted"], wrap=38)


def add_model_readout(figure: Diagram, ranking: pd.DataFrame) -> None:
    """Add the Figure 1 model cross-validation panel."""
    figure.panel(67, 67, 30, 28, "Model cross-validation", "d")
    top_models = ranking.sort_values("rank").head(5)
    min_score = max(0.80, top_models["cross_validation_score"].min() - 0.02)
    max_score = min(0.94, top_models["cross_validation_score"].max() + 0.01)
    for row_index, (_, row) in enumerate(top_models.iterrows()):
        y = 73.7 + row_index * 4.0
        model_name = str(row["model_id"]).replace("-", " ")
        figure.text(69.0, y, 10.5, 2.5, f"{int(row['rank'])}. {model_name}", 4.7, align="left")
        figure.bar(80.9, y - 0.6, 10.5, 1.2, float(row["cross_validation_score"]), f"{row['cross_validation_score']:.4f}", PALETTE["violet"], min_score, max_score)
    best = float(top_models.iloc[0]["cross_validation_score"])
    second = float(top_models.iloc[1]["cross_validation_score"])
    figure.text(68.0, 92.7, 28.0, 2.6, f"top gap = {best - second:.4f}  |  14 models  |  91 pairwise CV", 4.7, PALETTE["teal"], bold=True)


def make_figure_2() -> Diagram:
    """Create Figure 2: seven-dimensional evaluation logic."""
    figure = Diagram("proposed_evaluation_framework", (7.25, 4.25))
    figure.text(4, 1.5, 92, 5.0, "Operational logic of the seven-dimensional evaluation framework", 9.0, bold=True)
    figure.text(10, 6.8, 80, 3.5, "Episode evidence is scored by dimension, converted to auditable status labels, and fed back into workflow improvement.", 5.6, PALETTE["muted"])

    figure.panel(3, 14, 25, 82, "Evidence record", "a", PALETTE["blue_soft"], PALETTE["blue"])
    evidence = [
        ("User request", "intent, constraints, study area"),
        ("Knowledge context", "manuals, standards, retrieved evidence"),
        ("Generated artifacts", "network, routes, demand, configs"),
        ("Execution trace", "tool calls, outputs, errors"),
        ("Validation evidence", "GEH, counts, domain checks"),
        ("Oversight record", "HIL prompts, approvals, corrections"),
    ]
    for index, (title, body) in enumerate(evidence):
        figure.box(6.2, 21.0 + index * 9.2, 18.5, 6.2, title, body, PALETTE["blue"], "white", 4.8, 3.4)

    figure.arrow((28, 50), (33, 50), PALETTE["gray"], 1.0)
    figure.panel(33, 14, 42, 82, "Seven measured dimensions", "b")
    figure.text(36, 20.0, 36, 4.0, "Each dimension answers a different reviewer question and keeps failures diagnosable.", 4.4, PALETTE["muted"], align="left", wrap=72)
    dimension_cards = [
        (36.0, 27.0, "D1. Task completion", "Were required deliverables produced?", PALETTE["blue"], PALETTE["blue_soft"]),
        (55.0, 27.0, "D2. Domain validity", "Are transportation artifacts plausible?", PALETTE["teal"], PALETTE["teal_soft"]),
        (36.0, 41.0, "D3. Tool reliability", "Were tools selected and called correctly?", PALETTE["violet"], PALETTE["violet_soft"]),
        (55.0, 41.0, "D4. Robustness", "Does the workflow recover from perturbations?", PALETTE["amber"], PALETTE["amber_soft"]),
        (36.0, 55.0, "D5. Risk control", "Were unsafe or critical actions controlled?", PALETTE["red"], PALETTE["red_soft"]),
        (55.0, 55.0, "D6. Auditability", "Can outputs be traced and reproduced?", PALETTE["cyan"], PALETTE["cyan_soft"]),
        (36.0, 69.0, "D7. Efficiency", "What time and correction burden remains?", PALETTE["green"], PALETTE["green_soft"]),
        (55.0, 69.0, "Aggregation rule", "Thresholds plus critical-failure overrides", PALETTE["gray"], PALETTE["gray_soft"]),
    ]
    for x, y, title, body, edge, fill_color in dimension_cards:
        figure.box(x, y, 17.0, 8.8, title, body, edge, fill_color, 4.6, 3.25)

    figure.arrow((75, 50), (79, 50), PALETTE["gray"], 1.0)
    figure.panel(79, 14, 18, 82, "Status and action", "c", "#FFF4EA", "#FF9A5B")
    status_specs = [
        (24.0, "PASS", "meets threshold", PALETTE["green"], PALETTE["green_soft"]),
        (35.5, "WARN", "usable with caveats", PALETTE["amber"], PALETTE["amber_soft"]),
        (47.0, "FAIL", "not deployment-ready", PALETTE["red"], PALETTE["red_soft"]),
        (58.5, "HIL REVIEW", "expert decision required", PALETTE["cyan"], PALETTE["cyan_soft"]),
    ]
    for y, label, subtitle, edge, fill_color in status_specs:
        figure.pill(82.0, y, 12.0, 5.5, label, edge, fill_color, subtitle)
    figure.arrow((88, 29.5), (88, 63.5), PALETTE["green"], 0.85)
    figure.box(82.0, 70.0, 12.0, 6.5, "Decision", "use, revise, or reject", PALETTE["line"], "white", 4.4, 3.0)
    figure.box(82.0, 79.0, 12.0, 5.0, "Feedback", "update workflow rules", PALETTE["line"], "white", 3.9, 2.55)

    return figure


def make_figure_3() -> Diagram:
    """Create Figure 3: benchmark workflow and validation-control logic."""
    figure = Diagram("experimental_workflow", (7.25, 4.2))
    figure.text(4, 1.5, 92, 5.0, "Experimental workflow and validation-control logic", 9.4, bold=True)
    figure.text(10, 6.8, 80, 3.5, "Each benchmark episode is scored from observable artifacts, validation gates, repair actions, and human-review records.", 5.6, PALETTE["muted"])

    figure.panel(4, 14, 92, 21, "Episode setup and artifact generation", "a")
    setup_cards = [
        (7.0, "1. Episode specification", "request, study area, demand source, allowed tools, critical actions", PALETTE["blue"], PALETTE["blue_soft"]),
        (36.0, "2. Context and evidence", "documents, templates, network metadata, count files, scoring rules", PALETTE["teal"], PALETTE["teal_soft"]),
        (65.0, "3. Artifact generation", "network, demand, routes, signal control, configs, detectors, logs", PALETTE["violet"], PALETTE["violet_soft"]),
    ]
    for x, title, body, edge, fill_color in setup_cards:
        figure.box(x, 20.0, 23.0, 8.8, title, body, edge, fill_color, 4.8, 3.15, align="left")
    figure.arrow((30.0, 24.4), (34.4, 24.4), PALETTE["gray"], 1.0)
    figure.arrow((59.0, 24.4), (63.4, 24.4), PALETTE["gray"], 1.0)

    figure.panel(4, 39, 92, 35, "Validation gates and control responses", "b")
    gates = [
        (26.0, 46.5, "Gate A", "Completeness\nand syntax", PALETTE["blue"]),
        (45.0, 46.5, "Gate B", "Executability\nand tool trace", PALETTE["violet"]),
        (64.0, 46.5, "Gate C", "Domain validity\nand calibration", PALETTE["teal"]),
    ]
    for x, y, title, body, edge in gates:
        figure.diamond(x, y, 13.0, 12.0, title, body, edge)
    figure.arrow((76.5, 28.8), (70.0, 47.5), PALETTE["gray"], 1.0, curve=0.18)
    figure.arrow((39.0, 52.5), (44.8, 52.5), PALETTE["gray"], 1.0)
    figure.arrow((58.0, 52.5), (63.8, 52.5), PALETTE["gray"], 1.0)
    figure.arrow((70.5, 58.5), (70.5, 64.0), PALETTE["gray"], 0.9)

    figure.box(19.0, 64.0, 19.5, 7.0, "4. Status generation", "pass, warning, failure, or human-review label", PALETTE["amber"], PALETTE["amber_soft"], 4.7, 3.0, align="left")
    figure.box(44.0, 64.0, 19.5, 7.0, "5. Repair or escalation", "safe correction, targeted clarification, human confirmation", PALETTE["red"], PALETTE["red_soft"], 4.7, 3.0, align="left")
    figure.box(69.0, 64.0, 19.5, 7.0, "6. Reproducible report", "artifacts, assumptions, validation results, limits, audit trail", PALETTE["cyan"], PALETTE["cyan_soft"], 4.7, 3.0, align="left")
    figure.arrow((38.5, 67.5), (43.5, 67.5), PALETTE["gray"], 1.0)
    figure.arrow((63.5, 67.5), (68.5, 67.5), PALETTE["gray"], 1.0)
    figure.arrow((53.7, 64.0), (31.5, 58.5), PALETTE["red"], 1.0, curve=-0.25, label="repair loop")

    figure.panel(4, 80, 92, 17, "Status-control rule")
    for x, label, edge in [(7.0, "PASS", PALETTE["green"]), (17.0, "WARNING", PALETTE["amber"]), (30.0, "FAILURE", PALETTE["red"]), (42.0, "HIL REVIEW", PALETTE["cyan"])]:
        figure.pill(x, 88.2, 9.5, 3.7, label, edge, "white")
    figure.text(
        55,
        88.5,
        37,
        6,
        "Warnings and failures trigger repair only when the tool trace supports safe correction; otherwise the issue is carried into the final report.",
        4.7,
        PALETTE["muted"],
        align="left",
        wrap=75,
    )
    return figure


def main() -> None:
    """Regenerate all manuscript figures and editable Draw.io files."""
    summary, difficulty, ranking = load_benchmark_data()
    figures = [make_figure_1(summary, difficulty, ranking), make_figure_2(), make_figure_3()]
    generated: list[Path] = []
    for figure in figures:
        generated.append(figure.render_png())
        generated.append(figure.render_drawio())
    for path in OUTPUT_DIR.glob("*.pdf"):
        if path.stem in {figure.stem for figure in figures}:
            path.unlink()
    for path in OUTPUT_DIR.glob("*.svg"):
        if path.stem in {figure.stem for figure in figures}:
            path.unlink()
    print("Generated figure files:")
    for path in generated:
        print(path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
