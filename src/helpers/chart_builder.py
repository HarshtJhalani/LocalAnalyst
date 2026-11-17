
# ============================================================
# FILE: src/helpers/chart_builder.py
# PURPOSE: Pick and draw the chart for a query result
#          (06_charts_summary.ipynb, steps 3-5)
#
# The chart is chosen by rules from the result's shape, never by the
# model: 1 row + 1 number -> stat tile; time + number -> line (gaps for
# missing periods); time + 2-3 numbers -> small multiples (never a
# second y-axis); category + number -> horizontal bars, largest first;
# 2 categories -> grouped bars; 2 numbers -> scatter; 1 number ->
# histogram; otherwise table only. A list of rows (the question asks for
# rows / records / a table, or the result has more than
# MAX_CHART_COLUMNS columns) stays a table unless a chart is asked for.
# A chart type named in the question (pie, donut, column, bar, line,
# area, stacked, scatter, histogram) is drawn when the result fits it
# (a pie: 2-6 positive parts); otherwise the shaped chart with a note.
# Colours are the validated reference palette, light and dark selected
# separately.
#
#   choose_chart()   result frame -> spec dict
#   build_figure()   spec -> Plotly figure (None for table / none)
# ============================================================

import math
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    BAR_GAP,
    BAR_SLOT_PX,
    CHART_WIDTH,
    MAX_BARS,
    MAX_CHART_COLUMNS,
    MAX_SERIES,
    MAX_SMALL_MULTIPLES,
    MIN_HISTOGRAM_ROWS,
    MIN_SCATTER_ROWS,
)


# ============================================================
# 1. PALETTE
# ============================================================


THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink_secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}


FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


# ============================================================
# 2. READING THE RESULT'S SHAPE
# ============================================================


TEMPORAL_VALUE = re.compile(r"^\d{4}(-\d{2}(-\d{2}( \d{2}:\d{2}(:\d{2})?)?)?)?$")


TEMPORAL_NAME = re.compile(r"(date|month|year|week|day|quarter|period|time)",
                           re.IGNORECASE)


def is_temporal(series):
    """ISO-like dates / months, or a year column named like one."""

    values = series.dropna().astype(str)

    if values.empty or not values.str.match(TEMPORAL_VALUE).all():
        return False

    # Plain 4-digit numbers are only years when the name says so.
    if values.str.fullmatch(r"\d{4}").all():
        return bool(TEMPORAL_NAME.search(str(series.name)))

    return True


def column_roles(frame):
    """{"temporal": [...], "numeric": [...], "category": [...]}"""

    roles = {"temporal": [], "numeric": [], "category": []}

    for column in frame.columns:
        series = frame[column]

        if is_temporal(series):
            roles["temporal"].append(column)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            roles["numeric"].append(column)
        else:
            roles["category"].append(column)

    return roles


# ============================================================
# 3. CHOOSING THE CHART
# ============================================================


# The question asks for rows themselves, or for a chart.
ROW_REQUEST = re.compile(r"\b(rows?|records?|entries|tabular|as a table|in a table|raw data)\b",
                         re.IGNORECASE)
CHART_REQUEST = re.compile(r"\b(chart|plot|graph|visuali[sz]e|trend|pie|donut|doughnut|"
                           r"histogram|scatter)\b", re.IGNORECASE)

# A chart type named in the question, checked in this order (a "stacked
# bar chart" is a stacked chart, not a plain bar chart).
REQUESTED_TYPES = [
    ("donut", r"\b(donut|doughnut)\b"),
    ("pie", r"\bpie\b"),
    ("stacked_bar", r"\bstacked\b"),
    ("area", r"\barea\s+(chart|graph|plot)s?\b|\bas an area\b"),
    ("column", r"\b(column|vertical bar)\s+(chart|graph|plot)s?\b|\bas columns\b"),
    ("histogram", r"\bhistogram"),
    ("scatter", r"\bscatter"),
    ("line", r"\bline\s+(chart|graph|plot)s?\b|\b(as|to|into) (a )?lines?\b"),
    ("bar", r"\bbar\s+(chart|graph|plot)s?\b|\b(as|to|into) (a )?bars?\b|\bbar plot\b"),
]
MAX_PIE_SLICES = 6        # more parts read better as bars


def requested_chart(*texts):
    """The chart type the user named ('pie', 'line', ...), or None."""

    text = " ".join(t for t in texts if t)

    for chart_type, pattern in REQUESTED_TYPES:
        if re.search(pattern, text, re.IGNORECASE):
            return chart_type

    return None


def choose_chart(frame, question="", original_question=""):
    """
    Pick a chart: a type named in the question when the result can carry
    it, otherwise the rules on the result's shape.

    original_question is the user's own words; a follow-up rewritten into
    a standalone question can lose "as a pie chart".

    Returns:
        spec dict: {"type", "x", "y", "color", "title", "note"}
        type: stat | line | area | multiples | bar | column | grouped_bar |
              stacked_bar | pie | donut | scatter | histogram | table | none
    """

    spec = {"type": "table", "x": None, "y": None, "color": None,
            "title": question.strip().rstrip("?") or None, "note": None}

    if frame is None or frame.empty:
        return {**spec, "type": "none"}

    roles = column_roles(frame)
    numeric = roles["numeric"]

    # Single headline number.
    if len(frame) == 1 and len(numeric) == 1:
        return {**spec, "type": "stat", "y": numeric[0]}

    asked = requested_chart(question, original_question)
    words = f"{question} {original_question}"
    shaped = shape_chart(frame, roles, spec, words,
                         chart_asked=bool(asked or CHART_REQUEST.search(words)))

    return apply_request(frame, roles, shaped, asked) if asked else shaped


def apply_request(frame, roles, shaped, asked):
    """The requested type when the result fits it, else the shaped chart + a note."""

    temporal, numeric, category = roles["temporal"], roles["numeric"], roles["category"]
    rows = len(frame)
    labels = category or temporal               # what names each bar / slice / point
    base = {**shaped, "color": None}

    def fallback(reason):
        note = f"{reason} Shown as {shaped['type'].replace('_', ' ')} instead."
        return {**shaped, "note": note if shaped["type"] not in ("table", "none")
                else f"{reason} Shown as a table."}

    if asked == shaped["type"]:
        return shaped

    if asked in ("pie", "donut"):
        if len(labels) != 1 or not numeric:
            return fallback("A pie chart needs one name column and one number column.")
        values = frame[numeric[0]]
        if not 2 <= rows <= MAX_PIE_SLICES:
            return fallback(f"A pie chart stays readable with 2-{MAX_PIE_SLICES} parts; "
                            f"this result has {rows}.")
        if values.isna().any() or (values < 0).any() or values.sum() <= 0:
            return fallback("A pie chart needs positive values that add up to a whole.")
        return {**base, "type": asked, "x": labels[0], "y": numeric[0]}

    if asked in ("bar", "column"):
        if not (labels and numeric) or rows > MAX_BARS:
            return fallback(f"A {asked} chart needs a name column and a number column "
                            f"(up to {MAX_BARS} rows).")
        x = temporal[0] if temporal and not category else labels[0]
        if asked == "bar":
            return {**base, "type": "bar", "x": numeric[0], "y": x}
        return {**base, "type": "column", "x": x, "y": numeric[0]}

    if asked == "stacked_bar":
        if len(category) >= 2 and numeric and frame[category[1]].nunique() <= MAX_SERIES:
            return {**base, "type": "stacked_bar", "x": numeric[0], "y": category[0],
                    "color": category[1]}
        if temporal and category and numeric and frame[category[0]].nunique() <= MAX_SERIES:
            return {**base, "type": "stacked_bar", "x": numeric[0], "y": temporal[0],
                    "color": category[0]}
        return fallback("A stacked chart needs two name columns (or a date and a name) "
                        "and a number.")

    if asked in ("line", "area"):
        if not (temporal and numeric):
            return fallback(f"A {asked} chart needs a date or period column.")
        if shaped["type"] == "line" and shaped["color"] and asked == "area":
            return {**shaped, "note": "Area charts show one series; drawn as lines."}
        if shaped["type"] in ("line", "multiples") and asked == "line":
            return shaped
        if frame.duplicated([temporal[0]]).any():
            return fallback(f"Several rows per {temporal[0]}.")
        return {**base, "type": asked, "x": temporal[0], "y": numeric[0]}

    if asked == "scatter":
        if len(numeric) >= 2 and rows >= 3:
            return {**base, "type": "scatter", "x": numeric[0], "y": numeric[1]}
        return fallback("A scatter plot needs two number columns.")

    if asked == "histogram":
        if numeric and rows >= 5:
            return {**base, "type": "histogram", "x": numeric[0]}
        return fallback("A histogram needs one number column with at least 5 rows.")

    return shaped


def shape_chart(frame, roles, spec, question="", chart_asked=False):
    """The chart the result's shape suggests (the rules in the file header)."""

    temporal, numeric, category = roles["temporal"], roles["numeric"], roles["category"]
    rows = len(frame)

    # A list of rows is read as a table, unless a chart is asked for.
    if not chart_asked:
        if ROW_REQUEST.search(question):
            return spec

        if len(frame.columns) > MAX_CHART_COLUMNS:
            return {**spec, "note": f"{len(frame.columns)} columns: a row listing, "
                                    "shown as a table."}

    # Change over time.
    if temporal and numeric:
        x = temporal[0]

        # One line per category needs one value per period and category;
        # with several (a second category column) there is no single
        # line to draw, so the result stays a table.
        if category and frame.duplicated([x, category[0]]).any():
            return {**spec, "note": f"Several rows per {x} and {category[0]}; "
                                    "shown as a table."}

        if category and frame[category[0]].nunique() <= MAX_SERIES:
            return {**spec, "type": "line", "x": x, "y": numeric[0],
                    "color": category[0]}

        if category:
            return {**spec, "note": f"Too many {category[0]} values to chart."}

        if len(numeric) == 1:
            return {**spec, "type": "line", "x": x, "y": numeric[0]}

        shown = numeric[:MAX_SMALL_MULTIPLES]
        note = (f"Showing {len(shown)} of {len(numeric)} measures."
                if len(numeric) > len(shown) else None)
        return {**spec, "type": "multiples", "x": x, "y": shown, "note": note}

    # Magnitude by category.
    if category and numeric:
        if len(category) >= 2:
            second = category[1]

            if frame[second].nunique() <= MAX_SERIES:
                return {**spec, "type": "grouped_bar", "x": numeric[0],
                        "y": category[0], "color": second}

            return {**spec, "note": f"Too many {second} values to chart."}

        note = None

        if len(numeric) > 1:
            note = f"Chart shows {numeric[0]}; all columns are in the table."

        if rows > MAX_BARS:
            note = f"Chart shows the top {MAX_BARS} of {rows} rows."

        return {**spec, "type": "bar", "x": numeric[0], "y": category[0],
                "note": note}

    # Two measures against each other.
    if len(numeric) >= 2 and rows >= MIN_SCATTER_ROWS:
        return {**spec, "type": "scatter", "x": numeric[0], "y": numeric[1]}

    # Distribution of one measure.
    if len(numeric) == 1 and rows >= MIN_HISTOGRAM_ROWS:
        return {**spec, "type": "histogram", "x": numeric[0]}

    return spec


# ============================================================
# 4. DRAWING HELPERS
# ============================================================


def format_number(value):
    """Labels: thousands separators, at most 2 decimals."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""

    value = float(value)

    return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"


def readable(name):
    """total_sales -> Total sales"""

    return str(name).replace("_", " ").strip().capitalize()


def tick_format(values):
    """Compact ticks (10k) once an axis reaches 10,000."""

    numbers = pd.to_numeric(pd.Series(values), errors="coerce").abs()
    return "~s" if numbers.max() >= 10_000 else ",~r"


def base_layout(figure, theme, title, height):
    """Surface, ink, hairline grid and hover for every chart."""

    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left",
               "font": {"size": 16, "color": theme["ink"]}},
        width=CHART_WIDTH, height=height,
        margin={"l": 16, "r": 48, "t": 84, "b": 40},
        paper_bgcolor=theme["surface"], plot_bgcolor=theme["surface"],
        font={"family": FONT_FAMILY, "size": 12, "color": theme["ink_secondary"]},
        hoverlabel={"bgcolor": theme["surface"], "bordercolor": theme["axis"],
                    "font": {"color": theme["ink"], "family": FONT_FAMILY}},
        # Legend in one row between the title and the plot, never
        # competing with the x-axis title.
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0,
                "font": {"color": theme["ink_secondary"]}},
        barcornerradius=4,
    )
    figure.update_xaxes(gridcolor=theme["grid"], gridwidth=1,
                        linecolor=theme["axis"], zeroline=False,
                        tickfont={"color": theme["muted"]}, automargin=True,
                        title_font={"color": theme["ink_secondary"]})
    figure.update_yaxes(gridcolor=theme["grid"], gridwidth=1,
                        linecolor=theme["axis"], zeroline=False,
                        tickfont={"color": theme["muted"]}, automargin=True,
                        title_font={"color": theme["ink_secondary"]})

    return figure


def series_colors(values, theme):
    """Fixed colour per category name (sorted), never by rank."""

    names = sorted(pd.Series(values).dropna().astype(str).unique())
    return {name: theme["series"][index] for index, name in enumerate(names)}


def line_trace(x, y, color, theme, name=None, show_end_label=True):
    """2 px line, 8 px markers with a surface ring, value at the end."""

    text = None

    if show_end_label and y.notna().any():
        last = y.last_valid_index()
        text = [format_number(value) if index == last else ""
                for index, value in y.items()]

    return go.Scatter(
        x=x, y=y, name=name, connectgaps=False,
        mode="lines+markers+text" if show_end_label else "lines+markers",
        line={"color": color, "width": 2, "shape": "linear"},
        marker={"size": 8, "color": color,
                "line": {"width": 2, "color": theme["surface"]}},
        text=text, textposition="middle right",
        textfont={"color": theme["ink_secondary"]}, cliponaxis=False,
        hovertemplate=(f"{name} · " if name else "") + "%{x}: %{y:,}<extra></extra>",
    )


# ============================================================
# 5. CHART TYPES
# ============================================================


def draw_stat(frame, spec, theme):
    """Hero number; a text column in the row names what it is about."""

    value = frame[spec["y"]].iloc[0]
    labels = [str(frame[column].iloc[0]) for column in frame.columns
              if column != spec["y"] and pd.notna(frame[column].iloc[0])]
    label = " · ".join(labels + [readable(spec["y"])])

    figure = go.Figure(go.Indicator(
        mode="number", value=None if pd.isna(value) else float(value),
        number={"valueformat": ",.2~f",
                "font": {"size": 56, "color": theme["ink"]}},
        title={"text": label,
               "font": {"size": 14, "color": theme["ink_secondary"]}},
    ))
    return base_layout(figure, theme, spec["title"], 220)


def draw_bar(frame, spec, theme):
    data = (frame.dropna(subset=[spec["x"]])
                 .sort_values(spec["x"], ascending=False)
                 .head(MAX_BARS))
    figure = go.Figure(go.Bar(
        x=data[spec["x"]], y=data[spec["y"]].astype(str), orientation="h",
        marker={"color": theme["series"][0], "line": {"width": 0}},
        text=[format_number(value) for value in data[spec["x"]]],
        textposition="outside", cliponaxis=False,
        textfont={"color": theme["ink_secondary"]},
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    figure.update_layout(bargap=BAR_GAP)
    figure.update_yaxes(autorange="reversed", showgrid=False)
    figure.update_xaxes(title=readable(spec["x"]),
                        tickformat=tick_format(data[spec["x"]]))
    height = 110 + BAR_SLOT_PX * len(data)
    return base_layout(figure, theme, spec["title"], height)


def draw_grouped_bar(frame, spec, theme):
    colors = series_colors(frame[spec["color"]], theme)
    figure = go.Figure()

    for name, color in colors.items():
        part = frame[frame[spec["color"]].astype(str) == name]
        figure.add_trace(go.Bar(
            x=part[spec["x"]], y=part[spec["y"]].astype(str), name=name,
            orientation="h", marker={"color": color, "line": {"width": 0}},
            hovertemplate=f"{name} · %{{y}}: %{{x:,}}<extra></extra>",
        ))

    # bargroupgap leaves a visible surface gap between bars of a group.
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.2)
    figure.update_yaxes(autorange="reversed", showgrid=False)
    figure.update_xaxes(title=readable(spec["x"]),
                        tickformat=tick_format(frame[spec["x"]]))
    height = 130 + BAR_SLOT_PX * frame[spec["y"]].nunique() * max(1, len(colors) // 2)
    return base_layout(figure, theme, spec["title"], min(height, 900))


def draw_line(frame, spec, theme):
    data = frame.sort_values(spec["x"])
    figure = go.Figure()

    if spec["color"]:
        colors = series_colors(data[spec["color"]], theme)
        all_x = sorted(data[spec["x"]].astype(str).unique())

        for name, color in colors.items():
            # Every series on every x; a missing period becomes a gap in
            # the line instead of a straight segment across it.
            part = (data[data[spec["color"]].astype(str) == name]
                    .assign(**{spec["x"]: lambda d: d[spec["x"]].astype(str)})
                    .set_index(spec["x"])[spec["y"]]
                    .reindex(all_x))
            figure.add_trace(line_trace(pd.Series(all_x), part.reset_index(drop=True),
                                        color, theme, name,
                                        show_end_label=len(colors) <= 4))
    else:
        figure.add_trace(line_trace(data[spec["x"]].astype(str),
                                    data[spec["y"]], theme["series"][0], theme))
        figure.update_layout(showlegend=False)

    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(type="category", showgrid=False, showspikes=True,
                        spikecolor=theme["axis"], spikethickness=1,
                        spikedash="solid")
    figure.update_yaxes(title=readable(spec["y"]),
                        tickformat=tick_format(data[spec["y"]]))
    return base_layout(figure, theme, spec["title"], 380)


def draw_multiples(frame, spec, theme):
    """One panel per measure, each with its own y-axis (never overlaid)."""

    data = frame.sort_values(spec["x"])
    measures = spec["y"]
    figure = make_subplots(rows=len(measures), cols=1, shared_xaxes=True,
                           vertical_spacing=0.08,
                           subplot_titles=[readable(m) for m in measures])

    for row, measure in enumerate(measures, start=1):
        figure.add_trace(line_trace(data[spec["x"]].astype(str), data[measure],
                                    theme["series"][0], theme, readable(measure)),
                         row=row, col=1)
        figure.update_yaxes(tickformat=tick_format(data[measure]), row=row, col=1)

    figure.update_layout(showlegend=False, hovermode="x unified")
    figure.update_xaxes(type="category", showgrid=False)
    figure.update_annotations(font={"color": theme["ink_secondary"], "size": 12},
                              x=0, xanchor="left")
    return base_layout(figure, theme, spec["title"], 160 + 200 * len(measures))


def draw_scatter(frame, spec, theme):
    figure = go.Figure(go.Scatter(
        x=frame[spec["x"]], y=frame[spec["y"]], mode="markers",
        marker={"size": 9, "color": theme["series"][0],
                "line": {"width": 2, "color": theme["surface"]}},
        hovertemplate=f"{readable(spec['x'])}: %{{x:,}}<br>"
                      f"{readable(spec['y'])}: %{{y:,}}<extra></extra>",
    ))
    figure.update_xaxes(title=readable(spec["x"]))
    figure.update_yaxes(title=readable(spec["y"]))
    return base_layout(figure, theme, spec["title"], 420)


def draw_histogram(frame, spec, theme):
    figure = go.Figure(go.Histogram(
        x=frame[spec["x"]], marker={"color": theme["series"][0],
                                    "line": {"width": 0}},
        hovertemplate="%{x}: %{y} rows<extra></extra>",
    ))
    figure.update_layout(bargap=0.06)
    figure.update_xaxes(title=readable(spec["x"]), showgrid=False)
    figure.update_yaxes(title="Rows")
    return base_layout(figure, theme, spec["title"], 380)


def draw_pie(frame, spec, theme):
    """
    Pie or donut, drawn only when asked for (2-6 parts). Slices keep the
    name-ordered palette colour, largest first from 12 o'clock, a 2 px
    surface gap between them, names and shares as text outside.
    """

    data = frame.sort_values(spec["y"], ascending=False)
    names = data[spec["x"]].astype(str)
    colors = series_colors(names, theme)
    donut = spec["type"] == "donut"

    figure = go.Figure(go.Pie(
        labels=names, values=data[spec["y"]], sort=False, direction="clockwise",
        rotation=0, hole=0.55 if donut else 0,
        marker={"colors": [colors[name] for name in names],
                "line": {"color": theme["surface"], "width": 2}},
        textinfo="label+percent", textposition="outside",
        textfont={"color": theme["ink_secondary"]},
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))

    if donut:
        figure.add_annotation(text=f"{format_number(data[spec['y']].sum())}<br>"
                                   f"<span style='font-size:12px'>{readable(spec['y'])}</span>",
                              showarrow=False, font={"size": 22, "color": theme["ink"]})

    figure = base_layout(figure, theme, spec["title"], 420)
    figure.update_layout(margin={"l": 40, "r": 40, "t": 84, "b": 40})
    return figure


def draw_column(frame, spec, theme):
    """Vertical bars: time keeps its order, names go largest first."""

    data = frame.dropna(subset=[spec["y"]])
    ordered = is_temporal(data[spec["x"]])
    data = (data.sort_values(spec["x"]) if ordered
            else data.sort_values(spec["y"], ascending=False)).head(MAX_BARS)

    figure = go.Figure(go.Bar(
        x=data[spec["x"]].astype(str), y=data[spec["y"]],
        marker={"color": theme["series"][0], "line": {"width": 0}},
        text=[format_number(value) for value in data[spec["y"]]],
        textposition="outside", cliponaxis=False,
        textfont={"color": theme["ink_secondary"]},
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    figure.update_layout(bargap=BAR_GAP)
    figure.update_xaxes(type="category", showgrid=False)
    figure.update_yaxes(title=readable(spec["y"]), tickformat=tick_format(data[spec["y"]]))
    return base_layout(figure, theme, spec["title"], 400)


def fill_color(hex_color, alpha):
    """#rrggbb -> rgba(r, g, b, alpha) for an area fill."""

    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def draw_area(frame, spec, theme):
    """One series over time: the line plus a light fill down to zero."""

    data = frame.sort_values(spec["x"])
    trace = line_trace(data[spec["x"]].astype(str), data[spec["y"]],
                       theme["series"][0], theme)
    trace.update(fill="tozeroy", fillcolor=fill_color(theme["series"][0], 0.18))

    figure = go.Figure(trace)
    figure.update_layout(showlegend=False, hovermode="x unified")
    figure.update_xaxes(type="category", showgrid=False)
    figure.update_yaxes(title=readable(spec["y"]), tickformat=tick_format(data[spec["y"]]),
                        rangemode="tozero")
    return base_layout(figure, theme, spec["title"], 380)


def draw_stacked_bar(frame, spec, theme):
    """Horizontal stacked bars; a 2 px surface gap between segments."""

    colors = series_colors(frame[spec["color"]], theme)
    order = (frame.groupby(frame[spec["y"]].astype(str))[spec["x"]].sum()
                  .sort_values(ascending=False).index.tolist())
    figure = go.Figure()

    for name, color in colors.items():
        part = frame[frame[spec["color"]].astype(str) == name]
        figure.add_trace(go.Bar(
            x=part[spec["x"]], y=part[spec["y"]].astype(str), name=name,
            orientation="h",
            marker={"color": color, "line": {"width": 2, "color": theme["surface"]}},
            hovertemplate=f"{name} · %{{y}}: %{{x:,}}<extra></extra>",
        ))

    figure.update_layout(barmode="stack", bargap=BAR_GAP)
    figure.update_yaxes(categoryorder="array", categoryarray=order,
                        autorange="reversed", showgrid=False)
    figure.update_xaxes(title=readable(spec["x"]), tickformat=tick_format(frame[spec["x"]]))
    height = 130 + BAR_SLOT_PX * len(order)
    return base_layout(figure, theme, spec["title"], min(height, 900))


DRAWERS = {"stat": draw_stat, "bar": draw_bar, "grouped_bar": draw_grouped_bar,
           "line": draw_line, "multiples": draw_multiples,
           "scatter": draw_scatter, "histogram": draw_histogram,
           "pie": draw_pie, "donut": draw_pie, "column": draw_column,
           "area": draw_area, "stacked_bar": draw_stacked_bar}


def build_figure(frame, spec, mode="light"):
    """Plotly figure for a spec, or None for table / none."""

    drawer = DRAWERS.get(spec["type"])
    return drawer(frame, spec, THEMES[mode]) if drawer else None
