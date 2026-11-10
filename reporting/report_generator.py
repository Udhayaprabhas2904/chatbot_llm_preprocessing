
"""
Professional LLM Preprocessing Comparison Report Generator.

Features:
- Database and API result field names
- Baseline vs preprocessed comparison
- Model-wise performance summaries
- Successful and failed response counts
- Average latency and response length
- Paired query comparisons
- Failure details
- Baseline vs preprocessed latency chart
- Baseline vs preprocessed response-length chart
- PDF table header wrapping
"""

from pathlib import Path
from collections import defaultdict
from xml.sax.saxutils import escape
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)


# ============================================================
# COLORS
# ============================================================

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2874A6")
LIGHT_BLUE = colors.HexColor("#DCEAF7")
LIGHT_GREY = colors.HexColor("#F2F4F7")
DARK_GREY = colors.HexColor("#475569")
GREEN = colors.HexColor("#198754")
RED = colors.HexColor("#C0392B")
WHITE = colors.white


# ============================================================
# DATA NORMALIZATION
# ============================================================

def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_record(row: dict) -> dict:
    """Convert database/API records into one consistent structure."""

    model = row.get("model") or row.get("model_name") or "unknown"

    condition = (
        row.get("condition")
        or row.get("condition_name")
        or "unknown"
    )

    response = (
        row.get("response")
        if row.get("response") is not None
        else row.get("response_text", "")
    )

    response = str(response or "")

    latency = (
        row.get("processing_time_seconds")
        if row.get("processing_time_seconds") is not None
        else row.get("latency_seconds")
    )

    try:
        latency = float(latency or 0)
    except (TypeError, ValueError):
        latency = 0.0

    response_length = row.get("response_length")

    if response_length is None:
        response_length = len(response)

    try:
        response_length = int(response_length or 0)
    except (TypeError, ValueError):
        response_length = len(response)

    error_message = row.get("error_message")

    success_value = row.get("success")
    if success_value is None:
        success_value = row.get("successful")

    if success_value is not None:
        successful = bool(success_value) and not error_message
    else:
        successful = not error_message and bool(response.strip())

    return {
        "id": row.get("id"),
        "experiment_id": row.get("experiment_id"),
        "query": row.get("query") or row.get("query_text") or "",
        "model": str(model),
        "condition": str(condition).lower(),
        "response": response,
        "response_length": response_length,
        "processing_time_seconds": latency,
        "preprocessing_seconds": _safe_float(row.get("preprocessing_seconds")),
        "generation_seconds": _safe_float(row.get("generation_seconds", latency)),
        "end_to_end_seconds": _safe_float(row.get("end_to_end_seconds")),
        "error_message": str(error_message or ""),
        "successful": successful,
        "preprocessing_technique": (
            row.get("preprocessing_technique") or "none"
        ),
    }


# ============================================================
# FORMATTING HELPERS
# ============================================================

def make_paragraph(text, style):
    """Create a safely escaped ReportLab paragraph."""

    return Paragraph(
        escape(str(text)).replace("\n", "<br/>"),
        style,
    )


def average(values):
    values = list(values)

    if not values:
        return None

    return sum(values) / len(values)


def format_seconds(value):
    if value is None:
        return "N/A"

    return f"{value:.3f}"


def format_characters(value):
    if value is None:
        return "N/A"

    return f"{value:.1f}"


# ============================================================
# TABLE HELPERS
# ============================================================

def create_table(data, widths, header=True):
    """Create a styled table with wrapped cells and headers."""

    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "ReportTableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=WHITE,
        alignment=0,
        wordWrap="CJK",
    )

    cell_style = ParagraphStyle(
        "ReportTableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9,
        textColor=colors.black,
        wordWrap="CJK",
    )

    formatted_data = []

    for row_index, row in enumerate(data):
        formatted_row = []

        for value in row:
            if isinstance(value, Paragraph):
                formatted_row.append(value)
            elif header and row_index == 0:
                formatted_row.append(
                    make_paragraph(value, header_style)
                )
            else:
                formatted_row.append(
                    make_paragraph(value, cell_style)
                )

        formatted_data.append(formatted_row)

    table = Table(
        formatted_data,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )

    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                WHITE,
                LIGHT_GREY,
            ]),
        ])

    table.setStyle(TableStyle(commands))

    return table


# ============================================================
# MODEL SUMMARY
# ============================================================

def build_model_summary(records):
    """Summarize results by model and condition."""

    grouped = defaultdict(list)

    for record in records:
        grouped[
            (record["model"], record["condition"])
        ].append(record)

    data = [[
        "Model",
        "Condition",
        "Total",
        "Successful",
        "Failed",
        "Avg latency (s)",
        "Avg response chars",
    ]]

    for (model, condition), items in sorted(grouped.items()):

        successful = [
            item for item in items
            if item["successful"]
        ]

        avg_latency = average(
            item["processing_time_seconds"]
            for item in successful
        )

        avg_length = average(
            item["response_length"]
            for item in successful
        )

        data.append([
            model,
            condition,
            str(len(items)),
            str(len(successful)),
            str(len(items) - len(successful)),
            format_seconds(avg_latency),
            format_characters(avg_length),
        ])

    return data


# ============================================================
# BASELINE VS PREPROCESSED COMPARISON
# ============================================================

def build_comparison_summary(records):
    """Compare baseline and preprocessed records by model and query."""

    grouped = defaultdict(dict)

    for record in records:

        if not record["successful"]:
            continue

        key = (
            record["model"],
            record["query"],
        )

        grouped[key][record["condition"]] = record

    model_pairs = defaultdict(list)

    for (model, query), conditions in grouped.items():

        baseline = conditions.get("baseline")
        preprocessed = conditions.get("preprocessed")

        if baseline is None or preprocessed is None:
            continue

        model_pairs[model].append({
            "baseline_latency":
                baseline["processing_time_seconds"],

            "preprocessed_latency":
                preprocessed["processing_time_seconds"],

            "baseline_length":
                baseline["response_length"],

            "preprocessed_length":
                preprocessed["response_length"],
        })

    data = [[
        "Model",
        "Paired queries",
        "Baseline latency (s)",
        "Preprocessed latency (s)",
        "Latency change (s)",
        "Baseline chars",
        "Preprocessed chars",
        "Length change",
    ]]

    for model, pairs in sorted(model_pairs.items()):

        baseline_latency = average(
            pair["baseline_latency"]
            for pair in pairs
        )

        preprocessed_latency = average(
            pair["preprocessed_latency"]
            for pair in pairs
        )

        baseline_length = average(
            pair["baseline_length"]
            for pair in pairs
        )

        preprocessed_length = average(
            pair["preprocessed_length"]
            for pair in pairs
        )

        latency_delta = preprocessed_latency - baseline_latency
        length_delta = preprocessed_length - baseline_length

        data.append([
            model,
            str(len(pairs)),
            format_seconds(baseline_latency),
            format_seconds(preprocessed_latency),
            f"{latency_delta:+.3f}",
            format_characters(baseline_length),
            format_characters(preprocessed_length),
            f"{length_delta:+.1f}",
        ])

    paired_count = sum(
        len(items) for items in model_pairs.values()
    )

    return data, len(grouped), paired_count


# ============================================================
# FAILURE DETAILS
# ============================================================

def build_failure_table(records):
    """Create a table containing failed model requests."""

    failures = [
        record for record in records
        if not record["successful"]
    ]

    query_style = ParagraphStyle(
        "FailureQuery",
        fontSize=7,
        leading=9,
    )

    error_style = ParagraphStyle(
        "FailureError",
        fontSize=7,
        leading=9,
    )

    data = [[
        "Model",
        "Condition",
        "Query",
        "Error",
    ]]

    for record in failures:

        data.append([
            record["model"],
            record["condition"],

            make_paragraph(
                record["query"][:180],
                query_style,
            ),

            make_paragraph(
                record["error_message"][:500]
                or "Empty response without an error message.",
                error_style,
            ),
        ])

    return data


# ============================================================
# CHART GENERATION
# ============================================================

def create_comparison_charts(records, output_dir):
    """
    Generate baseline vs preprocessed charts.

    Returns:
        List of generated chart paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_records = [
        record for record in records
        if record["successful"]
    ]

    models = sorted({
        record["model"]
        for record in successful_records
    })

    if not models:
        return []

    conditions = ["baseline", "preprocessed"]

    latency_values = {
        condition: []
        for condition in conditions
    }

    length_values = {
        condition: []
        for condition in conditions
    }

    for model in models:

        for condition in conditions:

            matching = [
                record for record in successful_records
                if record["model"] == model
                and record["condition"] == condition
            ]

            latency_values[condition].append(
                average(
                    record["processing_time_seconds"]
                    for record in matching
                )
            )

            length_values[condition].append(
                average(
                    record["response_length"]
                    for record in matching
                )
            )

    chart_paths = []

    # --------------------------------------------------------
    # CHART 1: LATENCY COMPARISON
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x_positions = list(range(len(models)))
    bar_width = 0.35

    baseline_latency = [
        value if value is not None else 0
        for value in latency_values["baseline"]
    ]

    preprocessed_latency = [
        value if value is not None else 0
        for value in latency_values["preprocessed"]
    ]

    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        baseline_latency,
        bar_width,
        label="Baseline",
    )

    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        preprocessed_latency,
        bar_width,
        label="Preprocessed",
    )

    ax.set_title("Baseline vs Preprocessed: Average Latency")
    ax.set_xlabel("Model")
    ax.set_ylabel("Latency (seconds)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    latency_chart_path = output_dir / "latency_comparison.png"
    fig.savefig(latency_chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    chart_paths.append(latency_chart_path)

    # --------------------------------------------------------
    # CHART 2: RESPONSE LENGTH COMPARISON
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 4.5))

    baseline_length = [
        value if value is not None else 0
        for value in length_values["baseline"]
    ]

    preprocessed_length = [
        value if value is not None else 0
        for value in length_values["preprocessed"]
    ]

    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        baseline_length,
        bar_width,
        label="Baseline",
    )

    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        preprocessed_length,
        bar_width,
        label="Preprocessed",
    )

    ax.set_title("Baseline vs Preprocessed: Average Response Length")
    ax.set_xlabel("Model")
    ax.set_ylabel("Response characters")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    length_chart_path = output_dir / "response_length_comparison.png"
    fig.savefig(length_chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    chart_paths.append(length_chart_path)

    return chart_paths


# ============================================================
# PDF GENERATOR
# ============================================================

def build_pdf_report(
    rows: list[dict],
    experiment_id: str | None,
    output_dir: Path,
) -> Path:
    """Generate a professional LLM comparison PDF report."""

    if not rows:
        raise ValueError(
            "Cannot generate report: no result records found."
        )

    records = [
        normalize_record(row)
        for row in rows
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_id = "".join(
        character
        for character in (experiment_id or "all-results")
        if character.isalnum() or character in "-_"
    ) or "all-results"

    report_path = (
        output_dir
        / f"experiment_report_{safe_id}.pdf"
    )

    chart_dir = output_dir / "charts"

    chart_paths = create_comparison_charts(
        records,
        chart_dir,
    )

    document = SimpleDocTemplate(
        str(report_path),
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=DARK_GREY,
        alignment=TA_CENTER,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        textColor=NAVY,
        fontSize=13,
        spaceBefore=10,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=DARK_GREY,
    ))

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(Paragraph(
        "LLM PREPROCESSING COMPARISON REPORT",
        ParagraphStyle(
            "MainTitle",
            parent=styles["Title"],
            fontSize=21,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
    ))

    story.append(Paragraph(
        "Baseline vs Preprocessed Model Performance Analysis",
        styles["ReportSubtitle"],
    ))

    story.append(Spacer(1, 5 * mm))

    # --------------------------------------------------------
    # EXPERIMENT INFORMATION
    # --------------------------------------------------------

    successful_count = sum(
        record["successful"]
        for record in records
    )

    failed_count = len(records) - successful_count

    models = sorted({
        record["model"]
        for record in records
    })

    queries = sorted({
        record["query"]
        for record in records
    })

    info_data = [
        ["Experiment ID", experiment_id or "All experiments"],
        ["Generated at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Total records", str(len(records))],
        ["Models tested", str(len(models))],
        ["Queries tested", str(len(queries))],
        ["Successful responses", str(successful_count)],
        ["Failed responses", str(failed_count)],
    ]

    info_table = Table(
        info_data,
        colWidths=[45 * mm, 120 * mm],
    )

    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(info_table)
    story.append(Spacer(1, 5 * mm))

    # --------------------------------------------------------
    # MODEL-WISE SUMMARY
    # --------------------------------------------------------

    story.append(Paragraph(
        "1. Model-wise Performance Summary",
        styles["SectionHeading"],
    ))

    summary_data = build_model_summary(records)

    story.append(create_table(
        summary_data,
        [
            35 * mm,
            30 * mm,
            20 * mm,
            25 * mm,
            20 * mm,
            35 * mm,
            40 * mm,
        ],
    ))

    story.append(Spacer(1, 6 * mm))

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    story.append(Paragraph(
        "2. Baseline vs Preprocessed Comparison",
        styles["SectionHeading"],
    ))

    comparison_data, _, paired_count = (
        build_comparison_summary(records)
    )

    if len(comparison_data) > 1:

        story.append(create_table(
            comparison_data,
            [
                29 * mm,
                22 * mm,
                29 * mm,
                31 * mm,
                25 * mm,
                28 * mm,
                32 * mm,
                25 * mm,
            ],
        ))

        story.append(Spacer(1, 3 * mm))

        story.append(Paragraph(
            f"Successfully paired model-query comparisons: {paired_count}. "
            "Negative latency change means preprocessing was faster. "
            "Positive latency change means preprocessing was slower. "
            "Negative response-length change means the response was shorter.",
            styles["SmallNote"],
        ))

    else:

        story.append(Paragraph(
            "No complete successful baseline/preprocessed pairs were "
            "available. Check the experiment conditions and failed requests.",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 5 * mm))

    # --------------------------------------------------------
    # TIMING BREAKDOWN (preprocessing, generation, end-to-end)
    # --------------------------------------------------------

    story.append(Paragraph(
        "3. Timing Breakdown (Preprocessing vs Model Generation)",
        styles["SectionHeading"],
    ))

    timing_groups = defaultdict(list)
    for record in records:
        if record["successful"]:
            timing_groups[(record["model"], record["condition"])].append(record)

    timing_data = [[
        "Model", "Condition", "Preprocessing (s)",
        "Generation (s)", "End-to-end (s)"
    ]]

    for (model, condition), items in sorted(timing_groups.items()):
        timing_data.append([
            model,
            condition,
            format_seconds(average(
                item["preprocessing_seconds"] for item in items
                if item["preprocessing_seconds"] is not None
            )),
            format_seconds(average(
                item["generation_seconds"] for item in items
                if item["generation_seconds"] is not None
            )),
            format_seconds(average(
                item["end_to_end_seconds"] for item in items
                if item["end_to_end_seconds"] is not None
            )),
        ])

    if len(timing_data) > 1:
        story.append(create_table(
            timing_data,
            [35 * mm, 35 * mm, 45 * mm, 45 * mm, 45 * mm],
        ))
        story.append(Paragraph(
            "Preprocessing time is the time spent transforming the input. "
            "Generation time is model inference. End-to-end time includes "
            "both preprocessing and inference. These timing fields are "
            "shown only when they were saved in the experiment database.",
            styles["SmallNote"],
        ))
    else:
        story.append(Paragraph(
            "Timing breakdown metrics were not available in the database.",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 5 * mm))

    # --------------------------------------------------------
    # COMPARISON CHARTS
    # --------------------------------------------------------

    story.append(Paragraph(
        "4. Performance Comparison Charts",
        styles["SectionHeading"],
    ))

    if chart_paths:

        chart_width = 250 * mm
        chart_height = 105 * mm

        for chart_path in chart_paths:

            chart_image = Image(
                str(chart_path),
                width=chart_width,
                height=chart_height,
            )

            story.append(
                KeepTogether([
                    chart_image,
                    Spacer(1, 4 * mm),
                ])
            )

    else:

        story.append(Paragraph(
            "Charts could not be generated because there were no "
            "successful response records.",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 5 * mm))

    # --------------------------------------------------------
    # FAILURE ANALYSIS
    # --------------------------------------------------------

    story.append(Paragraph(
        "5. Failed Requests and Error Details",
        styles["SectionHeading"],
    ))

    failure_data = build_failure_table(records)

    if len(failure_data) > 1:

        story.append(create_table(
            failure_data,
            [
                30 * mm,
                30 * mm,
                75 * mm,
                115 * mm,
            ],
        ))

    else:

        story.append(Paragraph(
            "No failed requests were recorded.",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 6 * mm))

    # --------------------------------------------------------
    # INTERPRETATION NOTES
    # --------------------------------------------------------

    story.append(Paragraph(
        "6. Interpretation and Evaluation Notes",
        styles["SectionHeading"],
    ))

    notes = [
        "Latency values are averaged over successful responses only.",
        "Failed requests are reported separately.",
        "Baseline and preprocessed comparisons are paired using the "
        "same model and original query.",
        "Response length measures the number of characters returned; "
        "it does not establish answer correctness.",
        "A shorter response or lower latency does not automatically "
        "mean better response quality.",
        "Quality evaluation requires a documented rubric or validated "
        "benchmark. No factual accuracy score is inferred by this report.",
    ]

    for note in notes:
        story.append(Paragraph(
            f"&bull; {escape(note)}",
            styles["SmallNote"],
        ))
        story.append(Spacer(1, 2))

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    document.build(story)

    return report_path
