
# ============================================================
# FILE: src/helpers/sql_generator.py
# PURPOSE: Turn a question into one SQLite query with the model
#          (04_text_to_sql.ipynb; context argument from 07_memory.ipynb)
#
# Prompt: fixed rules (system), few-shot pairs on a small made-up
# `orders` table, then the dataset's schema text, optional conversation
# context and the question. Output JSON: plan (written first so the
# model names its columns and filters), can_answer, sql.
#
#   build_messages()                schema + question (+ context) -> messages
#   estimate_tokens()               messages -> rough prompt size
#   generate_sql_from_messages()    messages -> {"plan", "can_answer", "sql", ...}
# ============================================================

import json
import logging

from config import PROMPT_CHARS_PER_TOKEN
from src.helpers.model_client import chat_json


logger = logging.getLogger(__name__)


# ============================================================
# 1. PROMPT
# ============================================================
# Tuned on data/input_data/dev_questions.json and measured on the
# held-out set in data/input_data/eval/ (see project_architecture.txt).

SYSTEM_PROMPT = """You are a data analyst who answers questions by writing one SQLite query.

Rules:
1. Write exactly one read-only query that starts with SELECT or WITH. If the question asks to change, delete or add data, set can_answer to false and say in plan that only reading data is possible.
2. Use only the tables and columns in the schema. Never invent columns.
3. Text comparisons are case-sensitive: copy category values exactly as listed in the schema.
4. Dates are TEXT 'YYYY-MM-DD'. Compare them as text, group them with strftime('%Y-%m', col) or strftime('%Y', col).
   "in 2023" means col >= '2023-01-01' AND col < '2024-01-01'; "after 2023" means col >= '2024-01-01'; "before 2023" means col < '2023-01-01'.
5. Boolean columns store 1 for true and 0 for false.
6. Give every computed column a short snake_case alias, e.g. SUM(total_sales) AS total_sales.
7. Group only by the columns the question asks about, and select them so the result is readable. When two columns have similar names (rep, rep_2), use the one without a number unless the question names the other.
8. For "top N", "most" or "highest" use ORDER BY ... DESC LIMIT N (LIMIT 1 for a single winner).
9. Use ROUND(value, 2) for averages and ratios.
10. If the question needs a value that no column holds (for example cost, profit, margin, rating, age), set can_answer to false, say which column is missing in plan, and leave sql empty. Never invent a formula from other columns to stand in for it.

Fill "plan" first: one or two sentences naming the table, columns, filters and grouping you will use."""


SQL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "can_answer": {"type": "boolean"},
        "sql": {"type": "string"},
    },
    "required": ["plan", "can_answer", "sql"],
}


EXAMPLE_SCHEMA = """TABLE orders (500 rows)
  order_id TEXT | unique per row | e.g. 'A001', 'A002', 'A003'
  order_date TEXT | date 'YYYY-MM-DD', 2023-01-01 to 2023-12-31
  store TEXT | values: 'Central', 'East', 'West'
  category TEXT | values: 'Books', 'Games', 'Toys'
  quantity INTEGER | range 1 to 20 | 4 missing
  amount REAL | currency amount | range 5.5 to 980.0
  returned INTEGER | boolean stored as 1 = true, 0 = false"""


FEW_SHOT_EXAMPLES = [
    ("How many orders came from the East store?",
     {"plan": "Count rows of orders where store is 'East'.",
      "can_answer": True,
      "sql": "SELECT COUNT(*) AS orders FROM orders WHERE store = 'East'"}),

    ("What is the total amount per category, highest first?",
     {"plan": "Sum amount from orders grouped by category, sorted descending.",
      "can_answer": True,
      "sql": "SELECT category, SUM(amount) AS total_amount FROM orders "
             "GROUP BY category ORDER BY total_amount DESC"}),

    ("Show the monthly number of returned orders.",
     {"plan": "Filter orders to returned = 1, group by month of order_date and count.",
      "can_answer": True,
      "sql": "SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS returned_orders "
             "FROM orders WHERE returned = 1 GROUP BY month ORDER BY month"}),

    ("Which store has the highest median order amount?",
     {"plan": "Median of amount per store with median_of(json_group_array(amount)), keep the top one.",
      "can_answer": True,
      "sql": "SELECT store, median_of(json_group_array(amount)) AS median_amount "
             "FROM orders GROUP BY store ORDER BY median_amount DESC LIMIT 1"}),

    ("What was the average quantity of Toys orders in March 2023?",
     {"plan": "Average quantity of orders with category 'Toys' and order_date in 2023-03.",
      "can_answer": True,
      "sql": "SELECT ROUND(AVG(quantity), 2) AS average_quantity FROM orders "
             "WHERE category = 'Toys' AND order_date >= '2023-03-01' "
             "AND order_date < '2023-04-01'"}),

    ("How many orders were placed after 2022?",
     {"plan": "After 2022 means from 2023-01-01 onwards (2022 itself excluded), so count orders with order_date >= '2023-01-01'.",
      "can_answer": True,
      "sql": "SELECT COUNT(*) AS orders FROM orders WHERE order_date >= '2023-01-01'"}),

    ("Which customer spent the most?",
     {"plan": "The orders table has no customer column, so this cannot be answered.",
      "can_answer": False,
      "sql": ""}),

    ("What is the profit for each category?",
     {"plan": "Profit needs a cost column and orders has none; amount alone is revenue, so this cannot be answered.",
      "can_answer": False,
      "sql": ""}),
]


# ============================================================
# 2. MESSAGES
# ============================================================

def user_message(schema_text, question, context=None):
    """Schema, optional conversation context, then the question."""

    if context:
        return f"Schema:\n{schema_text}\n\n{context}\n\nQuestion: {question}"

    return f"Schema:\n{schema_text}\n\nQuestion: {question}"


SAME_EXAMPLE_SCHEMA = "(the orders table shown above)"


def build_messages(schema_text, question, context=None):
    """
    System rules + few-shot pairs + the real schema, context and question.

    The example schema is written out in the first pair only; repeating
    it in all of them cost about 800 tokens of a 4096-token context.
    """

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for position, (example_question, example_answer) in enumerate(FEW_SHOT_EXAMPLES):
        example_schema = EXAMPLE_SCHEMA if position == 0 else SAME_EXAMPLE_SCHEMA
        messages.append({"role": "user",
                         "content": user_message(example_schema, example_question)})
        messages.append({"role": "assistant", "content": json.dumps(example_answer)})

    messages.append({"role": "user",
                     "content": user_message(schema_text, question, context)})

    return messages


def estimate_tokens(messages):
    """Rough prompt size in tokens (a few extra per message for its role)."""

    characters = sum(len(message["content"]) for message in messages)

    return characters // PROMPT_CHARS_PER_TOKEN + 4 * len(messages)


# ============================================================
# 3. GENERATING SQL
# ============================================================

def clean_sql(sql):
    """Strip whitespace, code fences and a trailing semicolon."""

    sql = sql.strip()

    if sql.startswith("```"):
        sql = sql.strip("`")
        sql = sql[3:] if sql.lower().startswith("sql") else sql

    return sql.strip().rstrip(";").strip()


def generate_sql_from_messages(settings, messages):
    """
    One model call on a prepared conversation.

    Returns:
        {"plan", "can_answer", "sql", "seconds", "prompt_tokens",
         "output_tokens"}
    """

    answer, stats = chat_json(settings, messages, SQL_ANSWER_SCHEMA)

    return {"plan": answer["plan"].strip(),
            "can_answer": bool(answer["can_answer"]),
            "sql": clean_sql(answer["sql"]), **stats}


def generate_sql(settings, schema_text, question, context=None):
    """Ask the model for a query (single attempt, no checks)."""

    return generate_sql_from_messages(
        settings, build_messages(schema_text, question, context))
