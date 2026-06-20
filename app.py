from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import requests

from data_utils import REQUIRED_COLUMNS, expense_view, load_dataset, normalize_transactions, predict_categories
from database import load_dataframe, run_select_query, save_transactions
from generate_demo_dataset import generate_demo_transactions
from generate_synthetic_dataset import BANKS, PROFILES, YEARS, generate_transactions
from llm_client import LLMSettings, answer_question_with_guardrails, detect_category, detect_merchant, detect_month, detect_year
from paths import DATASET_PATH, DEMO_DATASET_PATH


st.set_page_config(page_title="Spending Analytics", page_icon=":credit_card:", layout="wide")


DATASET_LABELS = {
    "original": "Original grading dataset",
    "demo": "Demo dataset",
    "uploaded": "Uploaded CSV dataset",
}

PROFILE_EXAMPLES = {
    "Student": [
        "How much did I spend on coffee?",
        "How much did I spend on education?",
        "Show my Starbucks transactions.",
        "What did I spend on transportation in 2026?",
        "How much income did I receive?",
        "What are my subscriptions?",
    ],
    "Professional": [
        "How much did I spend on dining?",
        "What did I spend on transportation in 2026?",
        "How much did I save or invest?",
        "What are my top merchants?",
        "How much income did I receive in 2026?",
        "What are my recurring payments?",
    ],
    "Family": [
        "How much did we spend on groceries?",
        "How much did we spend on childcare?",
        "What did we spend on insurance?",
        "What are our recurring payments?",
        "What are our top categories?",
        "How much income did we receive in 2026?",
    ],
}

GLOBAL_EXAMPLES = [
    "What are the top expense categories?",
    "Compare Student, Professional, and Family spending.",
    "How much income did all profiles receive?",
    "Which profile spent the most?",
    "What are the top merchants?",
    "Compare January and February spending.",
]

HF_MODEL_OPTIONS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-2b-it",
]


def set_active_dataset(dataset_key: str) -> None:
    st.session_state["active_dataset"] = dataset_key


def get_active_dataset_label() -> str:
    dataset_key = st.session_state.get("active_dataset", "original")
    return DATASET_LABELS.get(dataset_key, DATASET_LABELS["original"])


def render_active_dataset_banner() -> None:
    st.info(f"Active dataset: {get_active_dataset_label()}")


def chatbot_context_text(selected_profile: str) -> str:
    profile_context = "All profiles" if selected_profile == "All" else f"{selected_profile} profile"
    return f"Context: {get_active_dataset_label()}, {profile_context}."


def is_income_question(question: str) -> bool:
    q = question.lower()
    return "income" in q or "receive" in q or "received" in q


def is_transaction_listing_question(question: str) -> bool:
    q = question.lower()
    return "show" in q and "transaction" in q


def get_ollama_models() -> list[str]:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    models = payload.get("models", [])
    names = [model.get("name") for model in models if model.get("name")]
    return sorted(names)


def has_hf_token() -> bool:
    try:
        token = st.secrets.get("HF_TOKEN")
        if token:
            return True
    except Exception:
        pass
    import os

    return bool(os.getenv("HF_TOKEN"))


def mode_description_text(provider: str, ollama_model: str = "", hf_model: str = "") -> str:
    if provider == "Ollama":
        return (
            "Mode: Ollama Local LLM\n\n"
            f"Model: {ollama_model}\n\n"
            "Questions are converted into SQLite queries using a locally running language model."
        )
    if provider == "Hugging Face":
        return (
            "Mode: Hugging Face Cloud LLM\n\n"
            f"Model: {hf_model}\n\n"
            "Questions are converted into SQLite queries using a cloud-hosted language model."
        )
    return (
        "Mode: Heuristic SQL Generation\n\n"
        "Questions are converted into SQLite queries using deterministic rules."
    )


def sql_literal(value: str) -> str:
    return value.replace("'", "''")


def get_example_groups(active_profiles: list[str], selected_profile: str) -> dict[str, list[str]]:
    if selected_profile != "All":
        return {selected_profile: PROFILE_EXAMPLES.get(selected_profile, GLOBAL_EXAMPLES)}

    example_groups = {profile: PROFILE_EXAMPLES[profile] for profile in active_profiles if profile in PROFILE_EXAMPLES}
    if set(PROFILES).issubset(set(active_profiles)):
        example_groups["Global"] = GLOBAL_EXAMPLES
    if not example_groups:
        example_groups["Global"] = GLOBAL_EXAMPLES
    return example_groups


def profile_context_options(active_profiles: list[str]) -> tuple[list[str], str, bool]:
    if len(active_profiles) == 1:
        return active_profiles, active_profiles[0], True
    return ["All"] + active_profiles, "All", False


def submit_chatbot_question() -> None:
    question = st.session_state.get("chatbot_question_text", "").strip()
    if question:
        st.session_state["chatbot_pending_question"] = question


def reset_chatbot_profile_context(active_profiles: list[str]) -> None:
    options, default_profile, _ = profile_context_options(active_profiles)
    current = st.session_state.get("chatbot_profile_context")
    if current not in options:
        st.session_state["chatbot_profile_context"] = default_profile


def build_transaction_detail_outputs(question: str, selected_profile: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None, str]:
    q = question.lower()
    asking_income = is_income_question(question)
    listing_transactions = is_transaction_listing_question(question)
    category = detect_category(q)
    merchant = "starbucks" if "starbucks" in q else detect_merchant(q)
    year = detect_year(q)
    month = detect_month(q)

    if not category and not merchant and not asking_income:
        return None, None, None, "View monthly summary"

    filters = ["amount > 0" if asking_income else "amount < 0"]
    label_parts = []
    if selected_profile != "All":
        filters.append(f"profile = '{sql_literal(selected_profile)}'")
    if category and category != "Income" and not asking_income:
        filters.append(f"category = '{sql_literal(category)}'")
        label_parts.append(category)
    if merchant and not asking_income:
        filters.append(f"LOWER(merchant) LIKE '%{sql_literal(merchant.lower())}%'")
        label_parts.append(merchant.title())
    if year:
        filters.append(f"strftime('%Y', date) = '{year}'")
    if month:
        filters.append(f"strftime('%m', date) = '{month}'")

    where_clause = " AND ".join(filters)
    if asking_income:
        monthly_value_expr = "ROUND(SUM(amount), 2) AS total_income"
        monthly_title = "View monthly income summary"
    elif listing_transactions:
        monthly_value_expr = "ROUND(SUM(ABS(amount)), 2) AS total_amount"
        monthly_title = "View monthly transaction summary"
    else:
        monthly_value_expr = "ROUND(SUM(ABS(amount)), 2) AS total_spent"
        monthly_title = "View monthly spending summary"

    details_sql = f"""
    SELECT date, merchant, category, amount, bank, profile
    FROM transactions
    WHERE {where_clause}
    ORDER BY date DESC, merchant;
    """
    monthly_sql = f"""
    SELECT strftime('%Y-%m', date) AS month,
           {monthly_value_expr},
           COUNT(*) AS transaction_count
    FROM transactions
    WHERE {where_clause}
    GROUP BY month
    ORDER BY month;
    """
    label = "Income" if asking_income else (" / ".join(label_parts) if label_parts else "Expense")
    return run_select_query(details_sql), run_select_query(monthly_sql), label, monthly_title


def transparent_spending_answer(details_df: pd.DataFrame | None, label: str | None, selected_profile: str, question: str = "") -> str | None:
    if details_df is None or details_df.empty or not label:
        return None
    asking_income = "income" in question.lower() or "receive" in question.lower() or "received" in question.lower()
    total = details_df["amount"].abs().sum()
    count = len(details_df)
    profile_context = "selected context" if selected_profile == "All" else f"{selected_profile} profile"
    if asking_income:
        return (
            f"Income received: ${total:,.2f} across {count:,} income transaction(s).\n\n"
            f"This total is the sum of all matching income transactions in the {profile_context}."
        )
    descriptor = f"{label} expense" if label != "Expense" else "expense"
    return (
        f"{label} spending total: ${total:,.2f} across {count:,} transaction(s).\n\n"
        f"This total is the sum of all matching {descriptor} transactions in the {profile_context}."
    )


def ensure_seed_data() -> pd.DataFrame:
    st.session_state.setdefault("active_dataset", "original")
    if DATASET_PATH.exists():
        data = load_dataset(DATASET_PATH)
    else:
        data = generate_transactions()
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(DATASET_PATH, index=False)

    if load_dataframe().empty:
        save_transactions(data, mode="replace")
    return data


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    profiles = ["All"] + sorted(df["profile"].dropna().unique().tolist())
    banks = ["All"] + sorted(df["bank"].dropna().unique().tolist())
    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())

    profile = st.sidebar.selectbox("Profile", profiles)
    bank = st.sidebar.selectbox("Bank", banks)
    category = st.sidebar.selectbox("Category", categories)

    filtered = df.copy()
    if profile != "All":
        filtered = filtered[filtered["profile"] == profile]
    if bank != "All":
        filtered = filtered[filtered["bank"] == bank]
    if category != "All":
        filtered = filtered[filtered["category"] == category]
    return filtered


def render_upload_page() -> None:
    st.title("📤 Upload Transactions")
    st.caption("🧾 Upload CSV transactions with date, merchant, description, amount, category, profile, bank, and currency columns.")

    uploaded = st.file_uploader("CSV file", type=["csv"])
    mode = st.radio("Database write mode", ["replace", "append"], horizontal=True)
    use_model = st.checkbox("Predict categories with spending_classifier.pkl when available", value=False)

    if uploaded and st.button("Load into SQLite", type="primary"):
        try:
            df = pd.read_csv(uploaded)
            df = predict_categories(df) if use_model else normalize_transactions(df)
            count = save_transactions(df, mode=mode)
            set_active_dataset("uploaded")
            reset_chatbot_profile_context(sorted(df["profile"].dropna().unique().tolist()))
            st.success(f"Loaded {count:,} transactions into SQLite.")
            st.dataframe(df.head(50), use_container_width=True)
            st.rerun()
        except ValueError as exc:
            st.error(
                "The uploaded CSV is missing required columns. "
                f"Please include: {', '.join(REQUIRED_COLUMNS)}. Details: {exc}"
            )
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    if st.button("Reset database from synthetic dataset"):
        data = ensure_seed_data()
        count = save_transactions(data, mode="replace")
        set_active_dataset("original")
        reset_chatbot_profile_context(sorted(data["profile"].dropna().unique().tolist()))
        st.success(f"Database reset with {count:,} synthetic transactions.")
        st.rerun()


def render_dashboard(df: pd.DataFrame) -> None:
    st.title("📊 Spending Dashboard")
    st.caption("💡 Explore spending, income, cash flow, and transaction patterns for the active SQLite dataset.")
    render_active_dataset_banner()
    view = expense_view(df)
    filtered = sidebar_filters(view)

    total_spent = filtered["expense_amount"].sum()
    total_income = filtered["income_amount"].sum()
    net_cashflow = filtered["amount"].sum()
    transaction_count = len(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spent", f"${total_spent:,.2f}")
    c2.metric("Income", f"${total_income:,.2f}")
    c3.metric("Net cash flow", f"${net_cashflow:,.2f}")
    c4.metric("Transactions", f"{transaction_count:,}")

    left, right = st.columns(2)
    by_category = (
        filtered[filtered["expense_amount"] > 0]
        .groupby("category", as_index=False)["expense_amount"]
        .sum()
        .sort_values("expense_amount", ascending=False)
    )
    by_month = filtered.groupby("month", as_index=False).agg(spending=("expense_amount", "sum"), income=("income_amount", "sum"))

    left.plotly_chart(px.bar(by_category, x="category", y="expense_amount", title="Spending by Category"), use_container_width=True)
    right.plotly_chart(px.line(by_month, x="month", y=["spending", "income"], title="Monthly Cash Flow"), use_container_width=True)

    st.subheader("Transactions")
    st.dataframe(filtered.sort_values("date", ascending=False), use_container_width=True)


def render_chatbot(df: pd.DataFrame) -> None:
    st.title("🤖 LLM Chatbot")
    render_active_dataset_banner()
    st.caption("💬 Ask natural-language questions about the SQLite transaction database.")

    active_profiles = sorted(df["profile"].dropna().unique().tolist())
    reset_chatbot_profile_context(active_profiles)
    profile_options, default_profile, disable_profile_select = profile_context_options(active_profiles)
    if disable_profile_select:
        selected_profile = default_profile
        st.session_state["chatbot_profile_context"] = default_profile
        st.selectbox("Profile context", profile_options, index=0, disabled=True, key="single_profile_context_display")
        st.caption(f"👤 Only one profile is available in the active dataset: {selected_profile}.")
    else:
        selected_profile = st.selectbox("Profile context", profile_options, key="chatbot_profile_context")

    provider = st.radio("LLM mode", ["Heuristic", "Ollama", "Hugging Face"], horizontal=True)
    ollama_models = get_ollama_models() if provider == "Ollama" else []
    if provider == "Ollama":
        if ollama_models:
            ollama_model = st.selectbox("Ollama model", ollama_models)
        else:
            st.warning("Ollama is not running or no local models were found.")
            ollama_model = st.selectbox("Ollama model", ["No local Ollama models found"], disabled=True)
    else:
        ollama_model = ""

    if provider == "Hugging Face":
        hf_model = st.selectbox("Hugging Face model", HF_MODEL_OPTIONS)
        if has_hf_token():
            st.success("HF_TOKEN detected. Hugging Face cloud mode is ready.")
        else:
            st.warning(
                "HF_TOKEN is not configured.\n\n"
                "For Streamlit Cloud: add HF_TOKEN in App Settings -> Secrets.\n\n"
                "For local testing: set HF_TOKEN as an environment variable.\n\n"
                "Heuristic and Ollama modes do not require HF_TOKEN."
            )
    else:
        hf_model = HF_MODEL_OPTIONS[0]

    st.info(mode_description_text(provider, ollama_model, hf_model))
    st.caption("The LLM generates SQL. SQLite stores and retrieves the transaction data.")

    example_groups = get_example_groups(active_profiles, selected_profile)

    examples = next(iter(example_groups.values()))
    st.caption("🗃️ You are asking questions against the currently loaded SQLite dataset.")
    st.text_input(
        "Question",
        placeholder=examples[0],
        key="chatbot_question_text",
        on_change=submit_chatbot_question,
    )
    if st.button("Ask"):
        submit_chatbot_question()
    with st.expander("Example questions"):
        for group_name, group_examples in example_groups.items():
            st.markdown(f"**{group_name}**")
            st.write("\n".join(f"- {item}" for item in group_examples))

    question = st.session_state.pop("chatbot_pending_question", "").strip()
    if question:
        if provider == "Ollama" and not ollama_models:
            st.error("Ollama mode is selected, but no local Ollama models are available.")
            return
        settings = LLMSettings(provider=provider, ollama_model=ollama_model, hf_model=hf_model)
        try:
            answer, sql, result, sql_adjusted = answer_question_with_guardrails(question, settings, selected_profile)
            details_df, monthly_df, detail_label, monthly_title = build_transaction_detail_outputs(question, selected_profile)
            transparent_answer = transparent_spending_answer(details_df, detail_label, selected_profile, question)
            st.write(chatbot_context_text(selected_profile))
            st.write(transparent_answer or answer)
            if sql_adjusted:
                st.caption("LLM SQL was adjusted or replaced because it contained unsupported filters.")
            if details_df is not None and not details_df.empty and "transaction" in question.lower():
                st.subheader("Transaction Details")
                st.dataframe(details_df, use_container_width=True, hide_index=True)
            st.code(sql, language="sql")
            st.subheader("Aggregated Result")
            st.dataframe(result, use_container_width=True)
            if details_df is not None and not details_df.empty:
                with st.expander("View transactions used for this answer"):
                    st.dataframe(details_df, use_container_width=True, hide_index=True)
            if monthly_df is not None and not monthly_df.empty:
                with st.expander(monthly_title):
                    st.dataframe(monthly_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Chatbot query failed: {exc}")


def render_database_page(df: pd.DataFrame) -> None:
    st.title("🗄️ Database")
    render_active_dataset_banner()
    st.caption("🔎 Current SQLite transaction table.")

    if df.empty:
        st.info("The database does not contain any transactions yet.")
        return

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(data):,}")
    c2.metric("Date range", f"{data['date'].min().date()} to {data['date'].max().date()}")
    c3.metric("Profiles", f"{data['profile'].nunique():,}")
    c4.metric("Banks", f"{data['bank'].nunique():,}")

    left, middle, right = st.columns(3)
    left.write("Profiles")
    left.dataframe(pd.DataFrame({"profile": sorted(data["profile"].dropna().unique())}), use_container_width=True, hide_index=True)
    middle.write("Banks")
    middle.dataframe(pd.DataFrame({"bank": sorted(data["bank"].dropna().unique())}), use_container_width=True, hide_index=True)
    right.write("Categories")
    right.dataframe(pd.DataFrame({"category": sorted(data["category"].dropna().unique())}), use_container_width=True, hide_index=True)

    export_df = df.drop(columns=["id"], errors="ignore")
    st.download_button(
        "Download transactions CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="transactions_export.csv",
        mime="text/csv",
    )
    st.subheader("Full Transaction Table")
    st.dataframe(data.sort_values("date", ascending=False), use_container_width=True)


def render_demo_summary(df: pd.DataFrame) -> None:
    view = expense_view(df)
    total_income = view["income_amount"].sum()
    total_spending = view["expense_amount"].sum()
    net_cash_flow = total_income - total_spending

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(view):,}")
    c2.metric("Profiles", ", ".join(sorted(view["profile"].unique())))
    c3.metric("Banks", ", ".join(sorted(view["bank"].unique())))
    c4.metric("Years", ", ".join(map(str, sorted(pd.to_datetime(view["date"]).dt.year.unique()))))

    c5, c6 = st.columns(2)
    c5.metric("Total income", f"${total_income:,.2f}")
    c6.metric("Total spending", f"${total_spending:,.2f}")
    st.metric("Net cash flow", f"${net_cash_flow:,.2f}")
    if total_income > 0 and total_spending > total_income * 1.2:
        st.warning("Demo spending is more than 120% of income. Consider regenerating with a different seed.")

    st.subheader("Breakdowns")
    left, right = st.columns(2)
    category_counts = view["category"].value_counts().rename_axis("category").reset_index(name="transactions")
    profile_counts = view["profile"].value_counts().rename_axis("profile").reset_index(name="transactions")
    left.dataframe(category_counts, use_container_width=True, hide_index=True)
    right.dataframe(profile_counts, use_container_width=True, hide_index=True)

    st.subheader("Demo Visualizations")
    expense_rows = view[view["expense_amount"] > 0]
    by_category = expense_rows.groupby("category", as_index=False)["expense_amount"].sum().sort_values("expense_amount", ascending=False)
    by_profile = expense_rows.groupby("profile", as_index=False)["expense_amount"].sum().sort_values("expense_amount", ascending=False)
    by_bank = expense_rows.groupby("bank", as_index=False)["expense_amount"].sum().sort_values("expense_amount", ascending=False)
    by_month = view.groupby("month", as_index=False).agg(spending=("expense_amount", "sum"), income=("income_amount", "sum"))

    chart_left, chart_right = st.columns(2)
    chart_left.plotly_chart(px.bar(by_category, x="category", y="expense_amount", title="Spending by Category"), use_container_width=True)
    chart_right.plotly_chart(px.bar(by_profile, x="profile", y="expense_amount", title="Spending by Profile"), use_container_width=True)
    chart_left.plotly_chart(px.bar(by_bank, x="bank", y="expense_amount", title="Spending by Bank"), use_container_width=True)
    chart_right.plotly_chart(px.line(by_month, x="month", y=["spending", "income"], title="Monthly Cash Flow"), use_container_width=True)


def render_dataset_generator_page() -> None:
    st.title("🎲 Dataset Generator")
    render_active_dataset_banner()
    st.caption("🧪 Create randomized demo data without changing the deterministic grading dataset.")

    st.subheader("Demo Controls")
    selected_profiles = st.multiselect("Profile Selection", PROFILES, default=PROFILES)
    selected_banks = st.multiselect("Bank Selection", BANKS, default=BANKS)
    selected_years = st.multiselect("Years", YEARS, default=YEARS)
    transactions_per_profile = st.slider("Transactions Per Profile", min_value=100, max_value=5000, value=500, step=100)
    seed_text = st.text_input("Random Seed", placeholder="Blank means fully random")

    seed = None
    if seed_text.strip():
        try:
            seed = int(seed_text)
        except ValueError:
            st.error("Random Seed must be blank or a whole number.")
            return

    if not selected_profiles or not selected_banks or not selected_years:
        st.warning("Select at least one profile, bank, and year.")
        return

    if st.button("Generate Demo Dataset", type="primary"):
        demo_df = generate_demo_transactions(
            profiles=selected_profiles,
            banks=selected_banks,
            years=selected_years,
            transactions_per_profile=transactions_per_profile,
            seed=seed,
        )
        DEMO_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        demo_df.to_csv(DEMO_DATASET_PATH, index=False)
        st.session_state["demo_dataset_preview"] = demo_df
        set_active_dataset(st.session_state.get("active_dataset", "original"))
        st.success(f"Generated {len(demo_df):,} demo transactions and saved {DEMO_DATASET_PATH.name}.")

    if DEMO_DATASET_PATH.exists() and "demo_dataset_preview" not in st.session_state:
        st.session_state["demo_dataset_preview"] = normalize_transactions(pd.read_csv(DEMO_DATASET_PATH))

    demo_preview = st.session_state.get("demo_dataset_preview")
    if demo_preview is not None:
        render_demo_summary(demo_preview)
        st.download_button(
            "Download demo_transactions.csv",
            data=normalize_transactions(demo_preview).to_csv(index=False).encode("utf-8"),
            file_name="demo_transactions.csv",
            mime="text/csv",
        )

    col1, col2 = st.columns(2)
    if col1.button("Load Demo Dataset Into SQLite"):
        if not DEMO_DATASET_PATH.exists():
            st.error("Generate a demo dataset first.")
        else:
            demo_df = normalize_transactions(pd.read_csv(DEMO_DATASET_PATH))
            count = save_transactions(demo_df, mode="replace")
            set_active_dataset("demo")
            reset_chatbot_profile_context(sorted(demo_df["profile"].dropna().unique().tolist()))
            st.success(f"Loaded {count:,} demo transactions into SQLite. Open Dashboard or Chatbot to use the demo data.")
            st.rerun()

    if col2.button("Restore Original Dataset"):
        original_df = load_dataset(DATASET_PATH)
        count = save_transactions(original_df, mode="replace")
        set_active_dataset("original")
        reset_chatbot_profile_context(sorted(original_df["profile"].dropna().unique().tolist()))
        st.success(f"Restored {count:,} deterministic grading transactions into SQLite.")
        st.rerun()


def main() -> None:
    ensure_seed_data()
    db_df = load_dataframe()
    nav_options = {
        "📊 Dashboard": "Dashboard",
        "📤 Upload": "Upload",
        "🤖 Chatbot": "Chatbot",
        "🗄️ Database": "Database",
        "🎲 Dataset Generator": "Dataset Generator",
    }
    page_label = st.sidebar.radio("🧭 Navigation", list(nav_options.keys()))
    page = nav_options[page_label]
    if page == "Upload":
        render_upload_page()
    elif page == "Chatbot":
        render_chatbot(db_df)
    elif page == "Database":
        render_database_page(db_df)
    elif page == "Dataset Generator":
        render_dataset_generator_page()
    else:
        render_dashboard(db_df)


if __name__ == "__main__":
    main()
