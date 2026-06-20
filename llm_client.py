from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from database import run_select_query, validate_select_sql
from paths import DATABASE_PATH


CATEGORY_ALIASES = {
    "income": "Income",
    "housing": "Housing",
    "rent": "Housing",
    "mortgage": "Housing",
    "groceries": "Groceries",
    "grocery": "Groceries",
    "dining": "Dining",
    "restaurant": "Dining",
    "restaurants": "Dining",
    "coffee": "Coffee",
    "transportation": "Transportation",
    "transport": "Transportation",
    "transit": "Transportation",
    "utilities": "Utilities",
    "utility": "Utilities",
    "subscriptions": "Subscriptions",
    "subscription": "Subscriptions",
    "shopping": "Shopping",
    "healthcare": "Healthcare",
    "health": "Healthcare",
    "education": "Education",
    "childcare": "Childcare",
    "insurance": "Insurance",
    "travel": "Travel",
    "save": "Savings/Investments",
    "invest": "Savings/Investments",
    "savings": "Savings/Investments",
    "investments": "Savings/Investments",
    "investment": "Savings/Investments",
}
BANK_ALIASES = {"rbc": "RBC", "td": "TD", "scotiabank": "Scotiabank", "scotia": "Scotiabank"}
PROFILE_ALIASES = {"student": "Student", "professional": "Professional", "family": "Family"}
MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


SCHEMA_CONTEXT = """
SQLite table: transactions
Columns:
- date TEXT in YYYY-MM-DD format
- merchant TEXT
- description TEXT
- amount REAL, income is positive and expenses are negative
- category TEXT
- profile TEXT
- bank TEXT
- currency TEXT
Use ABS(amount) when reporting spending.
Use amount > 0 for income and amount < 0 for expenses.
Only generate SELECT statements.
"""


@dataclass
class LLMSettings:
    provider: str = "Heuristic"
    ollama_model: str = ""
    hf_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    temperature: float = 0.1


def get_hf_token() -> str | None:
    try:
        import streamlit as st

        token = st.secrets.get("HF_TOKEN")
        if token:
            return str(token)
    except Exception:
        pass
    return os.getenv("HF_TOKEN")


def clean_sql_response(response: str) -> str:
    """Extract the first SELECT statement from an LLM response."""
    if not response:
        return ""

    text = str(response).replace("\x00", " ").strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    match = re.search(r"\bselect\b.*", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(0)

    for statement in [part.strip() for part in text.split(";") if part.strip()]:
        if statement.lower().startswith("select"):
            return statement + ";"
    return text.rstrip(";") + ";"


def build_prompt(question: str, selected_profile: str | None = None) -> str:
    profile_rule = ""
    if selected_profile and selected_profile != "All":
        profile_rule = f"Always filter with profile = '{selected_profile}'."
    return f"""
You convert banking questions into one SQLite SELECT query.
Return only SQL. No markdown. No explanation.
{profile_rule}

{SCHEMA_CONTEXT}

Question: {question}
SQL:
""".strip()


def call_ollama(prompt: str, model: str, temperature: float = 0.1) -> str:
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def call_hugging_face(prompt: str, model: str, temperature: float = 0.1) -> str:
    import requests

    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not configured. Add it to Streamlit Secrets or the environment.")
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "inputs": prompt,
            "parameters": {"temperature": temperature, "max_new_tokens": 220, "return_full_text": False},
        },
        timeout=90,
    )
    response.raise_for_status()
    payload: Any = response.json()
    if isinstance(payload, list) and payload:
        return payload[0].get("generated_text", "")
    if isinstance(payload, dict):
        return payload.get("generated_text", "")
    return str(payload)


def month_bounds(keyword: str) -> tuple[str, str]:
    today = date.today()
    if keyword == "last month":
        year = today.year
        month = today.month - 1
        if month == 0:
            year -= 1
            month = 12
    else:
        year = today.year
        month = today.month
    start = date(year, month, 1)
    end = date(year + int(month == 12), 1 if month == 12 else month + 1, 1)
    return start.isoformat(), end.isoformat()


def detect_category(q: str) -> str | None:
    for alias, category in CATEGORY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return category
    return None


def detect_bank(q: str) -> str | None:
    for alias, bank in BANK_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return bank
    return None


def detect_profile(q: str) -> str | None:
    for alias, profile in PROFILE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return profile
    return None


def detect_year(q: str) -> str | None:
    match = re.search(r"\b(2024|2025|2026)\b", q)
    return match.group(1) if match else None


def detect_month(q: str) -> str | None:
    for month_name, month_number in MONTHS.items():
        if re.search(rf"\b{month_name}\b", q):
            return month_number
    return None


def detect_merchant(q: str) -> str | None:
    patterns = [
        r"\bshow\s+(?:my\s+)?([a-z0-9&.' +\-]+?)\s+transactions",
        r"\b(?:with|at|from)\s+([a-z0-9&.' +\-]+?)(?:\s+in\s+|\s+on\s+|\s+during\s+|\s+for\s+|$|\?)",
    ]
    blocked = set(CATEGORY_ALIASES) | set(BANK_ALIASES) | set(PROFILE_ALIASES)
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            merchant = match.group(1).strip(" ?.")
            if merchant and merchant not in blocked:
                return merchant
    return None


def is_income_question(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in ["income", "receive", "received", "salary", "payroll", "deposit"])


def is_spending_question(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in ["spend", "spent", "spending", "expense", "expenses"])


def build_where(filters: list[str]) -> str:
    return " WHERE " + " AND ".join(filters) if filters else ""


def distinct_values(column: str) -> list[str]:
    if column not in {"profile", "bank", "category"}:
        return []
    result = run_select_query(f"SELECT DISTINCT {column} FROM transactions ORDER BY {column};", DATABASE_PATH)
    return [str(value) for value in result[column].dropna().tolist()]


def remove_condition(sql: str, expression: str) -> str:
    expr = expression.strip()
    patterns = [
        rf"\s+AND\s+{expr}",
        rf"\s+WHERE\s+{expr}\s+AND\s+",
        rf"\s+WHERE\s+{expr}",
    ]
    cleaned = sql
    cleaned = re.sub(patterns[0], "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(patterns[1], " WHERE ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(patterns[2], " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def add_required_condition(sql: str, condition: str) -> str:
    if re.search(re.escape(condition), sql, flags=re.IGNORECASE):
        return sql
    base = sql.rstrip(";")
    anchor = re.search(r"\b(group by|order by|limit)\b", base, flags=re.IGNORECASE)
    if re.search(r"\bwhere\b", base, flags=re.IGNORECASE):
        return re.sub(r"\bwhere\b", f"WHERE {condition} AND ", base, count=1, flags=re.IGNORECASE) + ";"
    if anchor:
        idx = anchor.start()
        return base[:idx] + f" WHERE {condition} " + base[idx:] + ";"
    return base + f" WHERE {condition};"


def replace_filter_value(sql: str, column: str, value: str) -> str:
    safe_value = value.replace("'", "''")
    return re.sub(
        rf"\b{column}\s*=\s*'[^']*'",
        f"{column} = '{safe_value}'",
        sql,
        flags=re.IGNORECASE,
    )


def remove_filter_value(sql: str, column: str, value: str) -> str:
    return remove_condition(sql, rf"{column}\s*=\s*'{re.escape(value)}'")


def remove_amount_filters(sql: str) -> str:
    return remove_condition(sql, r"amount\s*[<>]=?\s*0")


def zero_like_result(result: pd.DataFrame) -> bool:
    if result.empty:
        return True
    total_columns = [col for col in result.columns if col.lower().startswith(("total_", "sum"))]
    if not total_columns:
        return False
    values = pd.to_numeric(result[total_columns].stack(), errors="coerce").fillna(0)
    return bool((values == 0).all())


def sanitize_llm_sql(question: str, sql: str, selected_profile: str | None = None) -> tuple[str, bool]:
    cleaned = validate_select_sql(clean_sql_response(sql)) + ";"
    adjusted = False
    q = question.lower()
    mentioned = {
        "profile": detect_profile(q),
        "bank": detect_bank(q),
        "category": detect_category(q),
    }

    if is_income_question(question):
        cleaned = remove_amount_filters(cleaned)
        cleaned = add_required_condition(cleaned, "amount > 0")
        adjusted = True
    elif is_spending_question(question):
        cleaned = remove_amount_filters(cleaned)
        cleaned = add_required_condition(cleaned, "amount < 0")
        adjusted = True

    for column in ["profile", "bank", "category"]:
        valid_values = distinct_values(column)
        valid_lookup = {value.lower(): value for value in valid_values}
        matches = list(re.finditer(rf"\b{column}\s*=\s*'([^']*)'", cleaned, flags=re.IGNORECASE))
        for match in matches:
            raw_value = match.group(1)
            valid_value = valid_lookup.get(raw_value.lower())
            allow_filter = bool(mentioned[column])
            if column == "profile" and selected_profile and selected_profile != "All":
                allow_filter = True
                valid_value = selected_profile
            if column in {"bank", "category"} and is_income_question(question) and not mentioned[column]:
                allow_filter = False

            if not allow_filter:
                cleaned = remove_filter_value(cleaned, column, raw_value)
                adjusted = True
            elif valid_value:
                if raw_value != valid_value:
                    cleaned = replace_filter_value(cleaned, column, valid_value)
                    adjusted = True
            else:
                cleaned = remove_filter_value(cleaned, column, raw_value)
                adjusted = True

    if selected_profile and selected_profile != "All":
        cleaned = add_profile_filter(cleaned, selected_profile)
    elif mentioned["profile"]:
        cleaned = add_required_condition(cleaned, f"profile = '{mentioned['profile']}'")
        adjusted = True

    if mentioned["bank"] and not re.search(r"\bbank\s*=", cleaned, flags=re.IGNORECASE):
        cleaned = add_required_condition(cleaned, f"bank = '{mentioned['bank']}'")
        adjusted = True

    if (
        mentioned["category"]
        and mentioned["category"] != "Income"
        and not is_income_question(question)
        and not re.search(r"\bcategory\s*=", cleaned, flags=re.IGNORECASE)
    ):
        cleaned = add_required_condition(cleaned, f"category = '{mentioned['category']}'")
        adjusted = True

    return validate_select_sql(cleaned) + ";", adjusted


def add_profile_filter(sql: str, selected_profile: str | None) -> str:
    if not selected_profile or selected_profile == "All" or re.search(r"\bprofile\s*=", sql, flags=re.IGNORECASE):
        return sql
    base = sql.rstrip(";")
    if re.search(r"\bwhere\b", base, flags=re.IGNORECASE):
        return re.sub(r"\bwhere\b", f"WHERE profile = '{selected_profile}' AND ", base, count=1, flags=re.IGNORECASE) + ";"
    order_match = re.search(r"\b(group by|order by|limit)\b", base, flags=re.IGNORECASE)
    if order_match:
        idx = order_match.start()
        return base[:idx] + f" WHERE profile = '{selected_profile}' " + base[idx:] + ";"
    return base + f" WHERE profile = '{selected_profile}';"


def heuristic_sql(question: str, selected_profile: str | None = None) -> str:
    q = question.lower()
    category = detect_category(q)
    bank = detect_bank(q)
    profile = detect_profile(q)
    year = detect_year(q)
    month = detect_month(q)
    merchant = "starbucks" if "starbucks" in q else detect_merchant(q)
    asking_income = is_income_question(question)

    filters = ["amount > 0" if asking_income else "amount < 0"]
    if category and category != "Income":
        filters.append(f"category = '{category}'")
    if bank:
        filters.append(f"bank = '{bank}'")
    if profile:
        filters.append(f"profile = '{profile}'")
    if year:
        filters.append(f"strftime('%Y', date) = '{year}'")
    if month:
        filters.append(f"strftime('%m', date) = '{month}'")
    if merchant:
        safe_merchant = merchant.replace("'", "''").lower()
        filters.append(f"LOWER(merchant) LIKE '%{safe_merchant}%'")
    where_clause = build_where(filters)

    if "recurring" in q:
        sql = """
        SELECT merchant, category, ROUND(AVG(ABS(amount)), 2) AS average_amount, COUNT(*) AS transaction_count
        FROM transactions
        WHERE amount < 0
        GROUP BY merchant, category
        HAVING COUNT(*) >= 6
        ORDER BY transaction_count DESC, average_amount DESC
        LIMIT 20;
        """
    elif ("most frequent" in q or "frequent" in q) and ("subscription" in q or "subscriptions" in q):
        limit = 3 if "top 3" in q or "top three" in q else 1
        sql = f"""
        SELECT merchant, category, COUNT(*) AS transaction_count, ROUND(SUM(ABS(amount)), 2) AS total_spent
        FROM transactions
        {where_clause}
        GROUP BY merchant, category
        ORDER BY transaction_count DESC, total_spent DESC
        LIMIT {limit};
        """
    elif merchant and "transaction" in q:
        sql = f"""
        SELECT date, merchant, description, amount, category, profile, bank
        FROM transactions
        {where_clause}
        ORDER BY date DESC
        LIMIT 50;
        """
    elif "compare" in q and "january" in q and "february" in q:
        sql = """
        SELECT strftime('%Y-%m', date) AS month, category, ROUND(SUM(ABS(amount)), 2) AS total_spent
        FROM transactions
        WHERE amount < 0 AND strftime('%m', date) IN ('01', '02')
        GROUP BY month, category
        ORDER BY month, total_spent DESC;
        """
    elif "largest" in q and "category" in q:
        sql = """
        SELECT category, ROUND(SUM(ABS(amount)), 2) AS total_spent
        FROM transactions
        WHERE amount < 0
        GROUP BY category
        ORDER BY total_spent DESC
        LIMIT 1;
        """
    elif "which profile" in q and ("spent the most" in q or "spending" in q):
        sql = """
        SELECT profile, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        WHERE amount < 0
        GROUP BY profile
        ORDER BY total_spent DESC
        LIMIT 1;
        """
    elif "compare" in q and ("student" in q or "professional" in q or "family" in q):
        sql = """
        SELECT profile, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        WHERE amount < 0
        GROUP BY profile
        ORDER BY total_spent DESC;
        """
    elif "top merchant" in q or "top merchants" in q:
        sql = f"""
        SELECT merchant, category, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        {where_clause}
        GROUP BY merchant, category
        ORDER BY total_spent DESC
        LIMIT 10;
        """
    elif "top categor" in q or ("spending by category" in q and not category):
        sql = f"""
        SELECT category, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        {where_clause}
        GROUP BY category
        ORDER BY total_spent DESC
        LIMIT 10;
        """
    elif "average" in q or "avg" in q:
        sql = f"""
        SELECT ROUND(AVG(ABS(amount)), 2) AS average_transaction_amount, COUNT(*) AS transaction_count
        FROM transactions
        {where_clause};
        """
    elif "how many" in q or "count" in q or "number of transactions" in q:
        count_filters = [item for item in filters if not item.startswith("amount ")]
        sql = f"""
        SELECT COUNT(*) AS transaction_count
        FROM transactions
        {build_where(count_filters)};
        """
    elif asking_income:
        sql = f"""
        SELECT ROUND(SUM(amount), 2) AS total_income, COUNT(*) AS income_transactions
        FROM transactions
        {where_clause};
        """
    elif "by bank" in q and not bank:
        sql = """
        SELECT bank, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        WHERE amount < 0
        GROUP BY bank
        ORDER BY total_spent DESC;
        """
    elif "by profile" in q and not profile:
        sql = """
        SELECT profile, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        WHERE amount < 0
        GROUP BY profile
        ORDER BY total_spent DESC;
        """
    elif "by month" in q or "monthly" in q:
        sql = f"""
        SELECT strftime('%Y-%m', date) AS month, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        {where_clause}
        GROUP BY month
        ORDER BY month;
        """
    elif "by year" in q or "yearly" in q:
        sql = f"""
        SELECT strftime('%Y', date) AS year, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        {where_clause}
        GROUP BY year
        ORDER BY year;
        """
    elif "last month" in q and category:
        start, end = month_bounds("last month")
        sql = f"""
        SELECT ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        WHERE amount < 0 AND category = '{category}' AND date >= '{start}' AND date < '{end}';
        """
    elif category or bank or profile or year or month or merchant:
        sql = f"""
        SELECT ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        {where_clause};
        """
    else:
        sql = """
        SELECT category, ROUND(SUM(ABS(amount)), 2) AS total_spent, COUNT(*) AS transaction_count
        FROM transactions
        WHERE amount < 0
        GROUP BY category
        ORDER BY total_spent DESC
        LIMIT 10;
        """

    return clean_sql_response(add_profile_filter(sql, selected_profile))


def generate_sql(question: str, settings: LLMSettings, selected_profile: str | None = None) -> str:
    prompt = build_prompt(question, selected_profile)
    try:
        if settings.provider == "Ollama":
            response = call_ollama(prompt, settings.ollama_model, settings.temperature)
            return clean_sql_response(add_profile_filter(response, selected_profile))
        if settings.provider == "Hugging Face":
            response = call_hugging_face(prompt, settings.hf_model, settings.temperature)
            return clean_sql_response(add_profile_filter(response, selected_profile))
    except Exception:
        return heuristic_sql(question, selected_profile)
    return heuristic_sql(question, selected_profile)


def guarded_sql(question: str, settings: LLMSettings, selected_profile: str | None = None) -> tuple[str, bool]:
    raw_sql = generate_sql(question, settings, selected_profile)
    if settings.provider == "Heuristic":
        return validate_select_sql(raw_sql) + ";", False
    try:
        return sanitize_llm_sql(question, raw_sql, selected_profile)
    except Exception:
        return heuristic_sql(question, selected_profile), True


def dataframe_to_answer(question: str, sql: str, result: pd.DataFrame) -> str:
    if result.empty:
        return "I did not find matching transactions for that question."

    q = question.lower()
    first = result.iloc[0]
    if "transaction_count" in result.columns and len(result) == 1 and result.columns.tolist() == ["transaction_count"]:
        return f"You have {int(first['transaction_count']):,} matching transaction(s)."
    if "largest" in q and {"category", "total_spent"}.issubset(result.columns):
        return f"The largest expense category is {first['category']} with ${float(first['total_spent']):,.2f}."
    if ("most frequent" in q or "frequent" in q) and {"merchant", "transaction_count"}.issubset(result.columns):
        if len(result) == 1:
            return f"The most frequent matching subscription is {first['merchant']} with {int(first['transaction_count']):,} transaction(s)."
        top_items = ", ".join(
            f"{row['merchant']} ({int(row['transaction_count']):,})" for _, row in result.head(3).iterrows()
        )
        return f"The top matching subscriptions by frequency are: {top_items}."
    if "total_spent" in result.columns and len(result) == 1:
        total = float(first["total_spent"] or 0)
        count = int(first["transaction_count"]) if "transaction_count" in result.columns and pd.notna(first["transaction_count"]) else None
        suffix = f" across {count:,} transaction(s)" if count is not None else ""
        return f"Total spending was ${total:,.2f}{suffix}."
    if "average_transaction_amount" in result.columns:
        avg = float(first["average_transaction_amount"] or 0)
        count = int(first["transaction_count"] or 0)
        return f"The average transaction amount was ${avg:,.2f} across {count:,} matching transaction(s)."
    if "income" in q and "total_income" in result.columns:
        count_text = ""
        if "income_transactions" in result.columns:
            count_text = f" across {int(first['income_transactions'] or 0):,} income transaction(s)"
        return f"Total income received was ${float(first['total_income'] or 0):,.2f}{count_text}."
    if result.shape == (1, 1):
        return f"Result: {result.iloc[0, 0]}"
    if "merchant" in result.columns and "transactions" in q:
        return f"I found {len(result):,} matching merchant transaction(s)."
    if "month" in result.columns and "total_spent" in result.columns:
        total = float(result["total_spent"].fillna(0).sum())
        return f"The monthly results total ${total:,.2f} across {len(result):,} row(s)."
    if "year" in result.columns and "total_spent" in result.columns:
        total = float(result["total_spent"].fillna(0).sum())
        return f"The yearly results total ${total:,.2f} across {len(result):,} row(s)."
    if {"merchant", "total_spent"}.issubset(result.columns):
        return f"The top merchant is {first['merchant']} with ${float(first['total_spent']):,.2f} in spending."
    if {"category", "total_spent"}.issubset(result.columns):
        total = float(result["total_spent"].fillna(0).sum())
        return f"The table summarizes ${total:,.2f} in spending. The largest row is {first['category']} at ${float(first['total_spent']):,.2f}."
    return f"I found {len(result):,} matching row(s). The table below contains the details."


def answer_question(question: str, settings: LLMSettings, selected_profile: str | None = None) -> tuple[str, str, pd.DataFrame]:
    answer, sql, result, _ = answer_question_with_guardrails(question, settings, selected_profile)
    return answer, sql, result


def answer_question_with_guardrails(
    question: str, settings: LLMSettings, selected_profile: str | None = None
) -> tuple[str, str, pd.DataFrame, bool]:
    sql, adjusted = guarded_sql(question, settings, selected_profile)
    result = run_select_query(sql, DATABASE_PATH)
    if settings.provider != "Heuristic" and zero_like_result(result):
        sql = heuristic_sql(question, selected_profile)
        result = run_select_query(sql, DATABASE_PATH)
        adjusted = True
    return dataframe_to_answer(question, sql, result), sql, result, adjusted
