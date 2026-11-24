
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


import sys
from pathlib import Path

# Streamlit runs this file from src/ui/: make the project's config importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import API_PORT          # noqa: E402  (also loads .env)

API_URL = os.getenv("API_URL", f"http://127.0.0.1:{API_PORT}").rstrip("/")
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


@st.cache_data(ttl=300, show_spinner=False)
def cached_preview(dataset_id):
    return api("GET", f"/datasets/{dataset_id}/preview")


@st.cache_data(ttl=300, show_spinner=False)
def cached_review(dataset_id):
    return api("GET", f"/datasets/{dataset_id}/review")


def clear_dataset_caches():
    """After the data was cleaned again, every cached view of it is stale."""

    for cached in (cached_details, cached_preview, cached_review):
        cached.clear()

    # Downloaded cleaned tables are stale too.
    for key in [key for key in st.session_state.downloads if key[0] == "table"]:
        del st.session_state.downloads[key]


def table_downloads(dataset_id, table_name):
    """CSV and Excel of a whole cleaned table, fetched when asked for."""

    formats = (("csv", "⬇ Cleaned CSV"), ("xlsx", "⬇ Cleaned Excel"))
    keys = [("table", dataset_id, table_name, file_format) for file_format, _ in formats]

    if not all(key in st.session_state.downloads for key in keys):
        if st.button("Download the cleaned data (CSV / Excel)",
                     key=f"prep-table-{dataset_id}-{table_name}"):
            with st.spinner("Preparing files ..."):
                for (file_format, _), key in zip(formats, keys):
                    ok, data, _ = api("GET", f"/datasets/{dataset_id}/tables/"
                                             f"{table_name}/download",
                                      params={"format": file_format})
                    st.session_state.downloads[key] = data if ok else None
            st.rerun()
        return

    for column, (file_format, label), key in zip(st.columns(2), formats, keys):
        data = st.session_state.downloads[key]

        if data:
            column.download_button(label, data, width="stretch",
                                   file_name=f"{table_name}_cleaned.{file_format}",
                                   key=f"dl-table-{dataset_id}-{table_name}-{file_format}")


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
                              "Review the cleaning decisions below, then ask away.")
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

        # The technical reason a failed answer gives only in plain words above.
        if answer.get("status") == "failed":
            error = answer.get("error") or (
                attempts[-1].get("error") if isinstance(attempts, list) and attempts else None)
            if error:
                st.markdown(f"**Last error:** `{error}`")

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
    "confirm": (st.warning, "Confirm the change to the data."),
    "cleaned": (st.success, "The data was cleaned again."),
}

# nothing was run: no "How this was answered"
NO_DETAILS = {"blocked", "clarify", "confirm", "cleaned"}


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

    suggestion_buttons(answer)


def suggestion_buttons(answer):
    """Clickable questions offered with an answer; a click asks it."""

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
        suggestion_buttons(answer)

        if status not in NO_DETAILS:
            render_details(answer)
        return

    render_summary(answer.get("summary"), answer.get("caveats"))
    figure = render_figure(answer.get("figures"), answer["chat_id"], answer["position"])

    if answer["chart"].get("note"):
        st.caption(answer["chart"]["note"])

    result = answer["result"]
    shown = len(result["rows"])

    sections = (answer.get("chart") or {}).get("sections")

    # "first 5 and last 5 rows": one table per part.
    if sections and result["columns"]:
        frame, start = result_frame(result), 0

        for section in sections:
            st.markdown(f"**{section['title']}**")
            st.dataframe(frame.iloc[start:start + section["rows"]], hide_index=True,
                         width="stretch")
            start += section["rows"]

    elif result["columns"]:
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

    data_overview(dataset_id)

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


def preview_tables(data):
    """First rows and description of every table."""

    for table in data["tables"]:
        st.markdown(f"**{table['table_name']}**: first {len(table['rows'])} rows")
        st.dataframe(pd.DataFrame(table["rows"], columns=table["columns"]),
                     hide_index=True, width="stretch")
        st.markdown(table["description"])


def starter_buttons(questions, place):
    """Clickable question ideas; a click asks the question."""

    if not questions:
        return

    for column, question in zip(st.columns(len(questions)), questions):
        if column.button(question, width="stretch", key=f"starter|{place}|{question}"):
            st.session_state.pending_question = question
            st.rerun()


def preview_suggestions(data):
    return data.get("suggestions") or [
        q for table in data["tables"] for q in table["suggestions"]][:3]


def render_preview(dataset_id):
    """
    The first look in a new chat: rows, description. Returns the starter
    questions, or [] when the preview is unavailable.
    """

    ok, data, _ = cached_preview(dataset_id)

    if not ok:
        return []

    with st.chat_message("assistant"):
        preview_tables(data)

    return preview_suggestions(data)


def data_overview(dataset_id):
    """The same first look at the top of a running chat, folded away."""

    ok, data, _ = cached_preview(dataset_id)

    if not ok:
        return

    with st.expander("About this data: first rows, description and question ideas"):
        preview_tables(data)
        st.markdown("**Question ideas**")
        starter_buttons(preview_suggestions(data), "chat")


def option_index(item):
    """Radio position of the option chosen last time, else the suggestion."""

    values = [option["value"] for option in item["options"]]
    chosen = item.get("chosen")
    current = "merge" if isinstance(chosen, dict) else chosen

    return values.index(current if current in values else item["suggested"])


def review_choice(item, key):
    """Widgets for one review item; returns the option (or a spelling map)."""

    labels = {option["value"]: option["label"] for option in item["options"]}
    suggested = item["suggested"]

    st.markdown(f"**{item['title']}** · `{item['column']}`"
                + (f" · table `{item['table']}`" if item["table"] else ""))
    st.caption(item["detail"])

    if item.get("guesses"):
        st.dataframe(pd.DataFrame(item["guesses"], columns=["as typed", "read as"]),
                     hide_index=True, height=min(35 * len(item["guesses"]) + 38, 220))

    value = st.radio(
        "Choice", list(labels), index=option_index(item), key=key, horizontal=True,
        label_visibility="collapsed",
        format_func=lambda v: labels[v] + ("  (suggested)" if v == suggested else ""))
    st.caption(f"Why suggested: {item['reason']}")

    # Look-alike values: pick the spelling each one becomes.
    if value == "merge" and item.get("groups"):
        previous = item["chosen"] if isinstance(item.get("chosen"), dict) else {}
        targets = sorted(set(item["groups"]) | set(item["groups"].values()))
        mapping = {}

        for variant, target in item["groups"].items():
            current = previous.get(variant, target)
            mapping[variant] = st.selectbox(
                f"{variant!r} becomes", targets, key=f"{key}|{variant}",
                index=targets.index(current) if current in targets else 0)

        return mapping

    return value


def render_review(dataset_id, items, key_prefix="review"):
    """The review form; applies, applies the suggestions, or skips."""

    choices = {}

    with st.form(f"{key_prefix}_{dataset_id}"):
        for number, item in enumerate(items):
            with st.container(border=True):
                value = review_choice(item, f"{key_prefix}|{dataset_id}|{item['id']}")
                choices.setdefault(item["table"], {}).setdefault(
                    item["column"], {})[item["issue"]] = value

        left, right = st.columns(2)
        apply = left.form_submit_button("Apply and continue", type="primary",
                                        width="stretch")
        skip = right.form_submit_button("Skip: keep the automatic cleaning",
                                        width="stretch")

    if not (apply or skip):
        return

    body = {"skip": True} if skip else {"choices": choices}

    with st.spinner("Cleaning the file again with your choices ..."):
        ok, _, error = api("POST", f"/datasets/{dataset_id}/review", json=body)

    if ok:
        clear_dataset_caches()
        notify("success", "Kept the automatic cleaning." if skip else
               "Applied your choices; the data below reflects them.")
        st.rerun()

    st.error(error)


def review_step(dataset_id):
    """
    Before the first question of a dataset that was never reviewed: the
    decisions the cleaning left open. Returns True while it is shown.
    """

    ok, data, _ = cached_review(dataset_id)

    if not ok or data["review_state"] != "pending" or not data["items"]:
        return False

    with st.chat_message("assistant"):
        st.markdown("#### Before you ask: a few cleaning decisions")
        st.markdown(
            f"The file was cleaned automatically (names, types, spaces, duplicates, "
            f"date formats). **{len(data['items'])} things need your call**; each "
            f"has a suggested option already selected. The original values are "
            f"always kept in the `_raw` tables.")
        render_review(dataset_id, data["items"])

    return True


def render_intro(dataset_id):
    if review_step(dataset_id):
        return

    suggestions = render_preview(dataset_id)

    st.markdown("#### Ask a question about your data")
    st.caption("Plain English works: totals, averages, rankings, trends over time, "
               "filters, and charts (bar, line, pie ...). Follow-up questions can "
               "build on earlier answers.")

    starter_buttons(suggestions or EXAMPLE_QUESTIONS, "intro")


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
            render_intro(dataset_id)
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
            # A cleaning change from the chat: every cached view of the data is stale.
            if data["status"] == "cleaned":
                clear_dataset_caches()

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
            # Flags the user's review choices dealt with are no longer open.
            done = lambda flag: flag.get("resolved") or flag.get("decided")
            open_flags = [flag for flag in report["flags"] if not done(flag)]
            decided = [flag for flag in report["flags"] if done(flag)]
            badge = (f":orange-badge[⚠ {len(open_flags)} still flagged]" if open_flags
                     else ":green-badge[✓ nothing left to review]")
            st.markdown(f"**Table `{report['table_name']}`** &nbsp; {badge}")
            metrics = st.columns(3)
            metrics[0].metric("Rows after cleaning", report["rows_after"],
                              delta=report["rows_after"] - report["rows_before"],
                              delta_color="off")
            metrics[1].metric("Columns", report["columns_after"])
            metrics[2].metric("Still flagged", len(open_flags))

            left, right = st.columns(2)

            with left:
                st.markdown("**Cleaned** (automatically, or by your choice)")
                for action in report["actions"]:
                    prefix = f"`{action['column']}` " if action["column"] else ""
                    st.markdown(f"- {prefix}{action['detail']}")

            with right:
                st.markdown("**Still flagged**")
                for flag in open_flags or [{"column": None, "detail": "Nothing left to review."}]:
                    prefix = f"`{flag['column']}` " if flag["column"] else ""
                    st.markdown(f"- {prefix}{flag['detail']}")

                if decided:
                    st.caption("Decided by you: " + ", ".join(
                        f"{flag['column']} ({flag['issue'].replace('_', ' ')})"
                        for flag in decided) + ". See the list on the left.")

            columns = pd.DataFrame([
                {"column": name, "type": column["kind"], "missing": column["missing"],
                 "unique": column["unique"],
                 "range / top values": (f"{column.get('min')} – {column.get('max')}"
                                        if "min" in column else
                                        ", ".join(str(v) for v, _ in column.get("top_values", [])[:3]))}
                for name, column in table["columns"].items()
            ])
            st.dataframe(columns, hide_index=True, width="stretch")
            table_downloads(dataset_id, report["table_name"])

    ok, review, _ = cached_review(dataset_id)

    if ok and review["items"]:
        with st.expander(f"Change cleaning decisions ({len(review['items'])})"):
            st.caption("Cleans the original file again with these choices. "
                       "Answers already in chats are not re-run.")
            render_review(dataset_id, review["items"], key_prefix="data_review")

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
