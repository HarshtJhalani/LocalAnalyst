
# ============================================================
# FILE: src/ui/streamlit_app.py
# PURPOSE: Streamlit front end for the Text-to-SQL Data Analyst
#
# Run from the project root (the API must be running):
#   streamlit run src/ui/streamlit_app.py
#
# The UI talks to the FastAPI backend only (API_URL), never to the
# helper modules, so it runs the same in its own container.
#
#   Sidebar   API status, upload, dataset picker, chats
#   Ask       chat: question -> summary, chart, table, SQL, downloads
#   Data      cleaning report, schema the model sees, column profile
# ============================================================

import json
import os

import httpx
import pandas as pd
import plotly.io as pio
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 600           # the API's 240 s question budget + one 300 s CPU model call + margin
CHAT_LIST_HEIGHT = 360         # px; longer chat lists scroll inside the sidebar

EXAMPLE_QUESTIONS = [
    "How many rows are there?",
    "What are the totals per category?",
    "Show the monthly trend.",
]


# ============================================================
# 1. API CALLS
# ============================================================

def api(method, path, **kwargs):
    """
    Call the backend.

    Returns:
        (ok, data, error): data is the JSON body (or bytes for files);
        error is a readable message when ok is False.
    """

    try:
        response = httpx.request(method, f"{API_URL}{path}",
                                 timeout=REQUEST_TIMEOUT, **kwargs)

    except httpx.HTTPError as exc:
        return False, None, f"The API is not reachable at {API_URL} ({type(exc).__name__})."

    if response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
    else:
        data = response.content

    if response.is_success:
        return True, data, None

    if response.status_code == 405:
        return False, data, ("The running API does not have this action; it is probably an "
                             "older version. Restart it (run_api.bat) and try again.")

    detail = data.get("detail") if isinstance(data, dict) else None

    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("error") or json.dumps(detail)

    return False, data, detail or f"HTTP {response.status_code}"


@st.cache_data(ttl=15, show_spinner=False)
def cached_health():
    return api("GET", "/")


@st.cache_data(ttl=120, show_spinner=False)
def cached_details(dataset_id):
    return api("GET", f"/datasets/{dataset_id}")


def short_date(timestamp):
    """'2026-11-16T19:02:56...' -> '16 Nov 19:02' (text unchanged if unparsable)."""

    try:
        return pd.Timestamp(timestamp).strftime("%d %b %H:%M")
    except (ValueError, TypeError):
        return str(timestamp or "")


def theme_mode():
    """'dark' or 'light', following the viewer's Streamlit theme."""

    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


# ============================================================
# 2. STATE
# ============================================================

def init_state():
    defaults = {
        "dataset_id": None,
        "chat_id": None,
        "answers": {},        # (chat_id, position) -> full /ask response
        "downloads": {},      # (chat_id, position, format) -> bytes
        "notice": None,       # (kind, text) shown once after a rerun
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def select_dataset(dataset_id):
    st.session_state.dataset_id = dataset_id
    st.session_state.chat_id = None


def notify(kind, text):
    st.session_state.notice = (kind, text)


# ============================================================
# 3. SIDEBAR
# ============================================================

def sidebar_status():
    ok, health, error = cached_health()

    if ok:
        st.sidebar.success(f"Connected · **{health['model']}** ({health['profile']} profile)")
    else:
        st.sidebar.error(error)

    return ok


def sidebar_upload():
    with st.sidebar.form("upload", clear_on_submit=True):
        uploaded = st.file_uploader(
            "Upload a data file",
            type=["csv", "tsv", "xlsx", "xlsm", "json", "jsonl", "parquet"],
            help="It is cleaned automatically: header row, column names, types, "
                 "missing-value placeholders, duplicates and category spellings.")
        submitted = st.form_submit_button("Upload and clean", width="stretch")

    if submitted and uploaded is not None:
        with st.spinner(f"Cleaning {uploaded.name} ..."):
            ok, data, error = api("POST", "/datasets",
                                  files={"file": (uploaded.name, uploaded.getvalue())})

        if ok:
            select_dataset(data["dataset_id"])
            tables = ", ".join(f"{t['table_name']} ({t['rows']} rows)" for t in data["tables"])
            notify("success", f"{data['file_name']} is ready: {tables}. "
                              "See the Data tab for what was cleaned.")
            st.rerun()

        st.sidebar.error(error)

    elif submitted:
        st.sidebar.warning("Choose a file first.")


def dataset_label(dataset):
    rows = sum(table["rows"] for table in dataset["tables"])
    state = " · ⚠ failed" if dataset["status"] == "failed" else f" · {rows:,} rows"
    return f"{dataset['file_name']} · {short_date(dataset['created_at'])}{state}"


def sidebar_datasets():
    ok, datasets, error = api("GET", "/datasets")

    if not ok:
        st.sidebar.error(error)
        return []

    # Failed uploads are listed too, so they can be inspected and deleted.
    ready = [d for d in datasets if d["status"] in {"completed", "failed"}]

    if not ready:
        st.sidebar.info("Upload a file to start.")
        return []

    labels = {d["dataset_id"]: dataset_label(d) for d in ready}
    ids = list(labels)

    if st.session_state.dataset_id not in ids:
        select_dataset(ids[0])

    chosen = st.sidebar.selectbox("Dataset", ids, index=ids.index(st.session_state.dataset_id),
                                  format_func=labels.get)

    if chosen != st.session_state.dataset_id:
        select_dataset(chosen)
        st.rerun()

    return ready


def sidebar_chats(dataset_id):
    st.sidebar.divider()
    st.sidebar.subheader("Chats")

    if st.sidebar.button("➕ New chat", width="stretch"):
        st.session_state.chat_id = None
        st.rerun()

    ok, chats, error = api("GET", f"/datasets/{dataset_id}/chats")

    if not ok:
        st.sidebar.error(error)
        return

    if chats:
        sidebar_chat_list(chats)

    if chats:
        sidebar_delete_chats(dataset_id, chats)


def sidebar_chat_list(chats):
    """
    The chats in a collapsible section; a long list scrolls inside it
    so the delete controls below stay in reach.
    """

    with st.sidebar.expander("Chat history", expanded=True):
        st.caption(f"{len(chats)} chat(s), most recent first")
        box = st.container(height=CHAT_LIST_HEIGHT) if len(chats) > 8 else st.container()

        with box:
            for chat in chats:
                current = chat["chat_id"] == st.session_state.chat_id
                title = chat["title"] if len(chat["title"]) <= 32 else chat["title"][:31] + "…"
                label = (f"{'▶ ' if current else ''}{title} ({chat['turns']}) · "
                         f"{short_date(chat.get('updated_at'))}")

                if st.button(label, key=f"chat-{chat['chat_id']}", width="stretch",
                             type="primary" if current else "secondary"):
                    st.session_state.chat_id = chat["chat_id"]
                    st.rerun()


def sidebar_delete_chats(dataset_id, chats):
    """Pick one, several or all chats and delete them in one call."""

    names = {chat["chat_id"]: f"{chat['title'][:40]} · {short_date(chat.get('updated_at'))}"
             for chat in chats}

    # Widget state is kept per dataset, and the remembered selection is
    # cleaned before the widget is drawn: a chat deleted elsewhere (or one
    # from another dataset) must never reach the picker.
    keys = {name: f"{name}-{dataset_id}"
            for name in ("delete-all-chats", "delete-chat-ids", "confirm-chats")}
    remembered = st.session_state.get(keys["delete-chat-ids"])

    if remembered is None:
        current = st.session_state.chat_id
        st.session_state[keys["delete-chat-ids"]] = [current] if current in names else []
    else:
        st.session_state[keys["delete-chat-ids"]] = [i for i in remembered if i in names]

    with st.sidebar.expander("Delete chats"):
        st.caption("Permanently removes the chosen chats and their answers.")
        select_all = st.checkbox(f"All chats ({len(chats)})", key=keys["delete-all-chats"])

        if select_all:
            chosen = list(names)
        else:
            chosen = st.multiselect("Chats to delete", options=list(names),
                                    format_func=lambda i: names.get(i, f"Chat {i}"),
                                    key=keys["delete-chat-ids"], placeholder="Choose chats")

        confirm = st.checkbox(f"Yes, delete {len(chosen)} chat(s)",
                              key=keys["confirm-chats"], disabled=not chosen)

        if st.button("Delete chats", type="primary", width="stretch",
                     disabled=not (chosen and confirm)):
            params = {"all": "true"} if select_all else {"chat_ids": chosen}
            ok, data, error = api("DELETE", f"/datasets/{dataset_id}/chats", params=params)

            if ok:
                if st.session_state.chat_id in data["deleted_chats"]:
                    st.session_state.chat_id = None
                for key in keys.values():
                    st.session_state.pop(key, None)
                notify("info", f"Deleted {data['count']} chat(s).")
                st.rerun()

            st.error(error)


def sidebar_delete_dataset(dataset_id):
    with st.sidebar.expander("Delete this dataset"):
        st.caption("Permanently removes the file, its tables and every chat about it.")
        confirm = st.checkbox("Yes, delete this dataset", key="confirm-dataset")

        if st.button("Delete dataset", type="primary", width="stretch",
                     disabled=not confirm):
            ok, _, error = api("DELETE", f"/datasets/{dataset_id}")

            if ok:
                cached_details.clear()
                select_dataset(None)
                notify("info", "Dataset deleted.")
                st.rerun()

            st.error(error)


# ============================================================
# 4. ANSWERS
# ============================================================

def result_frame(payload):
    return pd.DataFrame(payload["rows"], columns=payload["columns"])


def download_buttons(dataset_id, chat_id, position):
    """
    CSV and Excel of the full result. The files are fetched only after
    the user asks for them (the API re-runs the SQL), then cached.
    """

    formats = (("csv", "⬇ CSV"), ("xlsx", "⬇ Excel"))
    keys = [(chat_id, position, file_format) for file_format, _ in formats]

    if not all(key in st.session_state.downloads for key in keys):
        if st.button("Prepare downloads (CSV / Excel)",
                     key=f"prep-{chat_id}-{position}"):
            with st.spinner("Preparing files ..."):
                for (file_format, _), key in zip(formats, keys):
                    ok, data, _ = api("GET", f"/datasets/{dataset_id}/chats/{chat_id}/"
                                             f"turns/{position}/download",
                                      params={"format": file_format})
                    st.session_state.downloads[key] = data if ok else None
            st.rerun()
        return

    columns = st.columns(2)

    for column, (file_format, label), key in zip(columns, formats, keys):
        data = st.session_state.downloads[key]

        if data:
            column.download_button(label, data, width="stretch",
                                   file_name=f"result_chat{chat_id}_turn{position}.{file_format}",
                                   key=f"dl-{chat_id}-{position}-{file_format}")


def render_summary(summary, caveats):
    if summary:
        st.markdown(f"##### {summary}")

    if caveats:
        st.warning("\n".join(f"- {caveat}" for caveat in caveats), icon="⚠️")


def render_figure(figures, chat_id, position):
    """Draw the chart for the viewer's theme; returns it (None when absent)."""

    figure = (figures or {}).get(theme_mode())

    if figure:
        chart = pio.from_json(json.dumps(figure))
        chart.update_layout(width=None)
        st.plotly_chart(chart, theme=None, width="stretch", key=f"chart-{chat_id}-{position}")

    return figure


def render_details(answer):
    """How the answer was produced: interpretation, SQL, attempts."""

    with st.expander("How this was answered"):
        if answer.get("uses_history"):
            st.markdown(f"**Interpreted as:** {answer['standalone_question']}")

        if answer.get("plan"):
            st.markdown(f"**Plan:** {answer['plan']}")

        if answer.get("sql"):
            st.code(answer["sql"], language="sql")

        attempts = answer.get("attempts")

        # A fresh answer lists every attempt; a stored turn only counts them.
        if isinstance(attempts, list) and len(attempts) > 1:
            st.markdown(f"**Self-correction:** {len(attempts)} attempts")
            st.dataframe(pd.DataFrame(attempts), hide_index=True, width="stretch")
        elif isinstance(attempts, int) and attempts > 1:
            st.markdown(f"**Self-correction:** {attempts} attempts")

        if answer.get("seconds") is not None:
            st.caption(f"{answer['seconds']} s")


def stored_turn_as_answer(turn):
    """A chat-history turn in the shape of an /ask response."""

    return {**turn, "chart": turn.get("chart") or {},
            "result": {**turn["preview"], "row_count": turn["row_count"],
                       "truncated": False}}


STATUS_MESSAGES = {
    "refused": (st.info, "This question cannot be answered from the data."),
    "failed": (st.error, "No working query could be produced."),
    "blocked": (st.warning, "The question was not sent."),
    "clarify": (st.info, "Please ask a question about the data."),
}

NO_DETAILS = {"blocked", "clarify"}         # nothing was run: no "How this was answered"


def render_schema_answer(dataset_id, answer):
    """
    "What columns are there?": the sentence, the column table and the
    suggested questions as buttons that ask them in this chat.
    """

    render_summary(answer.get("summary"), None)
    result = answer["result"]
    st.dataframe(result_frame(result), hide_index=True, width="stretch")

    if result["row_count"] > len(result["rows"]):
        st.caption(f"Showing {len(result['rows'])} of {result['row_count']} columns; "
                   "download for all of them.")

    download_buttons(dataset_id, answer["chat_id"], answer["position"])

    suggestions = answer.get("suggestions") or []

    if suggestions:
        st.markdown("**Try asking**")
        columns = st.columns(min(len(suggestions), 3))

        for number, suggestion in enumerate(suggestions):
            if columns[number % len(columns)].button(
                    suggestion, width="stretch",
                    key=f"suggest-{answer['chat_id']}-{answer['position']}-{number}"):
                st.session_state.pending_question = suggestion
                st.rerun()


def render_answer(dataset_id, answer):
    """Summary, caveats, chart, table, downloads and how it was answered."""

    status = answer["status"]

    if status == "schema":
        render_schema_answer(dataset_id, answer)
        return

    if status in STATUS_MESSAGES:
        show, default = STATUS_MESSAGES[status]
        show(answer.get("message") or default)

        if status not in NO_DETAILS:
            render_details(answer)
        return

    render_summary(answer.get("summary"), answer.get("caveats"))
    figure = render_figure(answer.get("figures"), answer["chat_id"], answer["position"])

    if answer["chart"].get("note"):
        st.caption(answer["chart"]["note"])

    result = answer["result"]
    shown = len(result["rows"])

    if result["columns"]:
        with st.expander(f"Table · {result['row_count']} row(s)", expanded=figure is None):
            st.dataframe(result_frame(result), hide_index=True, width="stretch")

            if result["truncated"] or result["row_count"] > shown:
                st.caption(f"Showing the first {shown:,} rows; download for all rows.")

    if status == "answered":
        download_buttons(dataset_id, answer["chat_id"], answer["position"])

    render_details(answer)


# ============================================================
# 5. ASK TAB
# ============================================================

def render_chat(dataset_id, chat_id):
    """Redraw every turn of a chat; False when the chat is gone."""

    ok, chat, error = api("GET", f"/datasets/{dataset_id}/chats/{chat_id}")

    if not ok:
        st.session_state.chat_id = None
        st.warning(error)
        return False

    for turn in chat["turns"]:
        with st.chat_message("user"):
            st.markdown(turn["question"])

        with st.chat_message("assistant"):
            # The fresh response has every row and attempt; otherwise the
            # stored turn (20-row preview).
            answer = (st.session_state.answers.get((chat_id, turn["position"]))
                      or stored_turn_as_answer(turn))
            render_answer(dataset_id, answer)

    return True


def render_intro():
    st.markdown("#### Ask a question about your data")
    st.caption("Plain English works: totals, averages, rankings, trends over time, "
               "filters. Follow-up questions can build on earlier answers.")

    example_columns = st.columns(len(EXAMPLE_QUESTIONS))

    for column, example in zip(example_columns, EXAMPLE_QUESTIONS):
        if column.button(example, width="stretch"):
            st.session_state.pending_question = example
            st.rerun()


def ask_tab(dataset_id):
    chat_id = st.session_state.chat_id

    # The input sits inside a tab, where Streamlit draws it in place:
    # the conversation goes into a container created *before* it, so
    # messages stay above the input box.
    conversation = st.container()
    question = st.chat_input("Ask about this dataset ...")
    question = question or st.session_state.pop("pending_question", None)

    with conversation:
        # Step 1: redraw the chat, or the intro for a new one.
        if chat_id is None:
            render_intro()
        elif not render_chat(dataset_id, chat_id):
            return

        if not question:
            return

        # Step 2: the new question.
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Writing and checking the SQL ..."):
                ok, data, error = api("POST", f"/datasets/{dataset_id}/ask",
                                      json={"question": question, "chat_id": chat_id})

        # A clarify reply to the first message starts no chat: show it once.
        if ok and data["chat_id"] is None:
            notify("info", data.get("message") or "Please ask a question about the data.")
            st.rerun()

        if ok:
            st.session_state.chat_id = data["chat_id"]
            st.session_state.answers[(data["chat_id"], data["position"])] = data
            st.rerun()

        # A blocked question is stored as a turn only in an existing chat.
        detail = data.get("detail") if isinstance(data, dict) else None

        if isinstance(detail, dict) and detail.get("chat_id"):
            notify("warning", error)
            st.rerun()

        st.warning(error)


# ============================================================
# 6. DATA TAB
# ============================================================

def data_tab(dataset_id):
    ok, details, error = cached_details(dataset_id)

    if not ok:
        st.error(error)
        return

    status = details["status"]
    st.markdown(f"#### {status.get('file_name')}")
    st.caption(f"Dataset {dataset_id} · uploaded {status.get('created_at')}")

    for report in details["cleaning_report"]:
        table = details["profile"]["tables"][report["table_name"]]

        with st.container(border=True):
            flag_count = len(report["flags"])
            badge = (f":orange-badge[⚠ {flag_count} to review]" if flag_count
                     else ":green-badge[✓ nothing flagged]")
            st.markdown(f"**Table `{report['table_name']}`** &nbsp; {badge}")
            metrics = st.columns(3)
            metrics[0].metric("Rows after cleaning", report["rows_after"],
                              delta=report["rows_after"] - report["rows_before"],
                              delta_color="off")
            metrics[1].metric("Columns", report["columns_after"])
            metrics[2].metric("Flags to review", len(report["flags"]))

            left, right = st.columns(2)

            with left:
                st.markdown("**Cleaned automatically**")
                for action in report["actions"]:
                    prefix = f"`{action['column']}` " if action["column"] else ""
                    st.markdown(f"- {prefix}{action['detail']}")

            with right:
                st.markdown("**Flagged, left unchanged**")
                for flag in report["flags"] or [{"column": None, "detail": "Nothing flagged."}]:
                    prefix = f"`{flag['column']}` " if flag["column"] else ""
                    st.markdown(f"- {prefix}{flag['detail']}")

            columns = pd.DataFrame([
                {"column": name, "type": column["kind"], "missing": column["missing"],
                 "unique": column["unique"],
                 "range / top values": (f"{column.get('min')} – {column.get('max')}"
                                        if "min" in column else
                                        ", ".join(str(v) for v, _ in column.get("top_values", [])[:3]))}
                for name, column in table["columns"].items()
            ])
            st.dataframe(columns, hide_index=True, width="stretch")

    with st.expander("Schema the model sees"):
        st.code(details["schema_text"], language="text")


# ============================================================
# 7. PAGE
# ============================================================

def main():
    st.set_page_config(page_title="Text-to-SQL Data Analyst", page_icon="📊", layout="wide")
    init_state()

    st.sidebar.title("📊 Data Analyst")

    if not sidebar_status():
        st.title("Text-to-SQL Data Analyst")
        st.error(f"Start the API first: `uvicorn src.main.main:app` (expected at {API_URL}).")
        return

    sidebar_upload()
    datasets = sidebar_datasets()
    dataset_id = st.session_state.dataset_id

    # The notice slot exists on every run, shown or not: if it came and
    # went, everything below would shift by one element, and Streamlit
    # would keep the faded previous page under the new one while a
    # question is being answered.
    notice_slot = st.empty()

    if st.session_state.notice:
        kind, text = st.session_state.notice
        getattr(notice_slot, kind)(text)
        st.session_state.notice = None

    if not datasets or dataset_id is None:
        st.title("Text-to-SQL Data Analyst")
        st.markdown("Ask questions about a spreadsheet in plain English. A local model "
                    "writes the SQL, guardrails check it, and every answer comes with a "
                    "table, a chart and a short summary.")
        steps = st.columns(3)
        steps[0].info("**1 · Upload**\n\nCSV, Excel, JSON or Parquet, in the sidebar.")
        steps[1].info("**2 · Review**\n\nSee what was cleaned and flagged in the Data tab.")
        steps[2].info("**3 · Ask**\n\n" + "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS))
        return

    current = next(d for d in datasets if d["dataset_id"] == dataset_id)

    if current["status"] == "failed":
        st.title(current["file_name"])
        st.error(f"This upload could not be prepared: {current.get('error') or 'unknown error'}")
        st.caption("Delete it from the sidebar and try the file again.")
        sidebar_delete_dataset(dataset_id)
        return

    sidebar_chats(dataset_id)
    sidebar_delete_dataset(dataset_id)

    ask, data = st.tabs(["💬 Ask", "🧹 Data"])

    with ask:
        ask_tab(dataset_id)

    with data:
        data_tab(dataset_id)


main()
