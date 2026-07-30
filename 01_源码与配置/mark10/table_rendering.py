from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import ROOT


TABLE_DIR = ROOT / "04_汇总表格"
OUTPUT_DIR = ROOT / "05_论文图表" / "TABLE_PNG"

HEADER_COLOR = "#24303A"
HEADER_TEXT_COLOR = "#FFFFFF"
BODY_COLORS = ("#F4F6F7", "#FFFFFF")
GRID_COLOR = "#C5CDD3"
TEXT_COLOR = "#111820"
NOTE_COLOR = "#53606C"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _draw_table(
    data: list[list[str]],
    headers: list[str],
    title_number: str,
    title_text: str,
    output_path: Path,
    *,
    figsize: tuple[float, float],
    column_widths: list[float],
    bbox: tuple[float, float, float, float],
    font_size: float,
    note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    ax.axis("off")
    fig.text(
        0.025,
        0.945,
        title_number,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#05090C",
    )
    fig.text(
        0.205,
        0.945,
        title_text,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#05090C",
    )

    table = ax.table(
        cellText=data,
        colLabels=headers,
        colWidths=column_widths,
        cellLoc="center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.75)
        cell.PAD = 0.12
        text = cell.get_text()
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            text.set_color(HEADER_TEXT_COLOR)
            text.set_fontweight("bold")
        else:
            cell.set_facecolor(BODY_COLORS[(row - 1) % 2])
            text.set_color(TEXT_COLOR)
            if column == 0:
                text.set_ha("left")

    if note:
        fig.text(
            0.025,
            0.045,
            note,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color=NOTE_COLOR,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches=None)
    plt.close(fig)


def render_table_i(source: Path, output: Path) -> None:
    table = pd.read_csv(source)
    expected = ["Symbol", "Definition"]
    if table.columns.tolist() != expected:
        raise ValueError(f"Table I columns must be {expected}")
    symbol_math = {
        "x_i": r"$x_i$",
        "q_i": r"$q_i$",
        "v_i": r"$v_i$",
        "K(x)": r"$K(\mathbf{x})$",
        "W(x)": r"$W(\mathbf{x})$",
        "V(x)": r"$V(\mathbf{x})$",
        "C_w": r"$C_w$",
        "V_avail": r"$V_{\mathrm{avail}}$",
        "B_total": r"$B_{\mathrm{total}}$",
        "J(x)": r"$J(\mathbf{x})$",
    }
    data = [
        [symbol_math.get(str(row.Symbol), str(row.Symbol)), str(row.Definition)]
        for row in table.itertuples(index=False)
    ]
    _draw_table(
        data,
        expected,
        "TABLE I",
        "Main notation",
        output,
        figsize=(9.5, 6.6),
        column_widths=[0.22, 0.78],
        bbox=(0.025, 0.07, 0.95, 0.78),
        font_size=14,
    )


def render_table_ii(source: Path, output: Path) -> None:
    table = pd.read_csv(source)
    expected = ["Model", "MAE down", "RMSE down", "R2 up", "Spearman up"]
    if table.columns.tolist() != expected:
        raise ValueError(f"Table II columns must be {expected}")
    headers = ["Model", "MAE ↓", "RMSE ↓", "R² ↑", "Spearman ↑"]
    data: list[list[str]] = []
    for row in table.itertuples(index=False):
        data.append([str(row[0])] + [f"{float(value):.5f}" for value in row[1:]])

    _draw_table(
        data,
        headers,
        "TABLE II",
        "Workload profiler prediction accuracy",
        output,
        figsize=(10.5, 4.4),
        column_widths=[0.32, 0.17, 0.17, 0.17, 0.17],
        bbox=(0.025, 0.24, 0.95, 0.57),
        font_size=13.5,
        note=(
            "Out-of-pool evaluation (n=500); all workloads are mean-normalized.\n"
            "Prediction target: mean-normalized Data workload derived from observed execution time."
        ),
    )


def main() -> None:
    _configure_style()
    render_table_i(
        TABLE_DIR / "table_i_symbols.csv",
        OUTPUT_DIR / "Table_I_main_notation.png",
    )
    render_table_ii(
        TABLE_DIR / "table_ii_profiler_metrics.csv",
        OUTPUT_DIR / "Table_II_profiler_metrics.png",
    )
    print(f"Generated Table I-II PNG files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
