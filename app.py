from __future__ import annotations

import html
from textwrap import dedent

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

PLOTLY_CONFIG = {"displayModeBar": False}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #081020;
            --panel: rgba(12, 28, 54, 0.78);
            --panel-strong: rgba(13, 39, 78, 0.92);
            --line: rgba(45, 211, 255, 0.28);
            --line-strong: rgba(45, 211, 255, 0.62);
            --cyan: #22d3ee;
            --blue: #3b82f6;
            --violet: #7c3aed;
            --green: #10b981;
            --amber: #f59e0b;
            --text: #e5f2ff;
            --muted: #9fb3c8;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(34, 211, 238, 0.12), transparent 28%),
                radial-gradient(circle at 82% 10%, rgba(124, 58, 237, 0.13), transparent 30%),
                linear-gradient(135deg, #050914 0%, #081020 52%, #03101d 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(3, 10, 24, 0.98), rgba(8, 18, 36, 0.96));
            border-right: 1px solid rgba(45, 211, 255, 0.2);
        }

        section[data-testid="stSidebar"] .stRadio > label,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--text) !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label {
            border: 1px solid rgba(45, 211, 255, 0.12);
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.35rem;
            background: rgba(12, 28, 54, 0.45);
            transition: 160ms ease;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            border-color: var(--line-strong);
            box-shadow: 0 0 18px rgba(34, 211, 238, 0.16);
            transform: translateX(2px);
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        .finance-hero {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.7rem 1.9rem;
            margin-bottom: 1rem;
            background:
                radial-gradient(circle at 80% 12%, rgba(34, 211, 238, 0.24), transparent 16%),
                radial-gradient(circle at 92% 52%, rgba(37, 99, 235, 0.28), transparent 22%),
                radial-gradient(circle at 76% 85%, rgba(124, 58, 237, 0.16), transparent 24%),
                linear-gradient(135deg, rgba(12, 28, 54, 0.95), rgba(5, 15, 31, 0.84)),
                linear-gradient(90deg, rgba(34, 211, 238, 0.07), transparent 44%);
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.28), 0 0 34px rgba(34, 211, 238, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            position: relative;
            overflow: hidden;
            min-height: 178px;
        }

        .finance-hero::after {
            content: "";
            position: absolute;
            inset: auto -60px -90px auto;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            border: 1px solid rgba(34, 211, 238, 0.18);
            box-shadow: 0 0 70px rgba(34, 211, 238, 0.2);
        }

        .finance-kicker {
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .finance-hero h1 {
            color: var(--text);
            font-size: clamp(2rem, 4vw, 3.05rem);
            line-height: 1.05;
            margin: 0;
            letter-spacing: 0;
            max-width: 68%;
        }

        .finance-hero p {
            color: var(--muted);
            margin: 0.55rem 0 0;
            font-size: 1rem;
            max-width: 68%;
        }

        .hero-art {
            position: absolute;
            right: 1.35rem;
            top: 1.2rem;
            width: 34%;
            min-width: 230px;
            height: calc(100% - 2.4rem);
            pointer-events: none;
        }

        .wallet-icon {
            position: absolute;
            right: 5.1rem;
            top: 0.25rem;
            width: 116px;
            height: 96px;
            border: 1px solid rgba(34, 211, 238, 0.42);
            border-radius: 24px;
            display: grid;
            place-items: center;
            font-size: 3rem;
            background:
                radial-gradient(circle at 50% 50%, rgba(34, 211, 238, 0.25), transparent 48%),
                linear-gradient(145deg, rgba(37, 99, 235, 0.42), rgba(3, 10, 24, 0.58));
            box-shadow: 0 0 34px rgba(34, 211, 238, 0.34), inset 0 0 18px rgba(34, 211, 238, 0.16);
        }

        .chart-chip {
            position: absolute;
            width: 54px;
            height: 46px;
            border: 1px solid rgba(34, 211, 238, 0.32);
            border-radius: 14px;
            display: grid;
            place-items: center;
            color: var(--cyan);
            background: rgba(3, 10, 24, 0.52);
            box-shadow: 0 0 20px rgba(37, 99, 235, 0.22);
            font-size: 1.35rem;
        }

        .chart-chip.one { right: 0.3rem; top: 0.15rem; }
        .chart-chip.two { right: 1.8rem; bottom: 0.35rem; }
        .chart-chip.three { right: 10.3rem; bottom: 0.8rem; }

        .dataset-banner {
            border: 1px solid rgba(34, 211, 238, 0.35);
            border-radius: 14px;
            padding: 0.82rem 1rem;
            margin: 0.75rem 0 1.1rem;
            color: #b9ecff;
            background: linear-gradient(90deg, rgba(14, 116, 144, 0.22), rgba(37, 99, 235, 0.12));
        }

        .fintech-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 0.85rem;
            margin: 1rem 0;
        }

        .kpi-card {
            min-height: 146px;
            border: 1px solid rgba(45, 211, 255, 0.24);
            border-radius: 20px;
            padding: 0.95rem;
            background:
                radial-gradient(circle at 18% 18%, rgba(34, 211, 238, 0.13), transparent 24%),
                linear-gradient(145deg, rgba(12, 28, 54, 0.86), rgba(5, 15, 31, 0.78));
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
            overflow: hidden;
        }

        .kpi-card:hover {
            transform: translateY(-3px);
            border-color: rgba(34, 211, 238, 0.78);
            box-shadow: 0 22px 56px rgba(0, 0, 0, 0.35), 0 0 28px rgba(34, 211, 238, 0.22);
        }

        .kpi-top {
            display: flex;
            align-items: flex-start;
            gap: 0.62rem;
            margin-bottom: 0.8rem;
            min-width: 0;
        }

        .kpi-icon {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            font-size: 1.42rem;
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.9), rgba(37, 99, 235, 0.72));
            box-shadow: 0 0 28px rgba(34, 211, 238, 0.24);
            flex: 0 0 48px;
        }

        .kpi-icon-0 { background: linear-gradient(135deg, #10b981, #22d3ee); box-shadow: 0 0 30px rgba(16, 185, 129, 0.28); }
        .kpi-icon-1 { background: linear-gradient(135deg, #0ea5e9, #2563eb); box-shadow: 0 0 30px rgba(37, 99, 235, 0.32); }
        .kpi-icon-2 { background: linear-gradient(135deg, #7c3aed, #22d3ee); box-shadow: 0 0 30px rgba(124, 58, 237, 0.3); }
        .kpi-icon-3 { background: linear-gradient(135deg, #f59e0b, #22d3ee); box-shadow: 0 0 30px rgba(245, 158, 11, 0.24); }

        .kpi-label {
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 700;
        }

        .kpi-value {
            color: #f8fbff;
            font-size: clamp(1.18rem, 1.42vw, 1.55rem);
            font-weight: 900;
            line-height: 1.06;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: clip;
            max-width: 100%;
        }

        .kpi-note {
            color: var(--green);
            font-size: 0.82rem;
            margin-top: 0.65rem;
        }

        .fintech-panel {
            border: 1px solid rgba(45, 211, 255, 0.22);
            border-radius: 18px;
            padding: 1rem;
            background: linear-gradient(145deg, rgba(12, 28, 54, 0.78), rgba(5, 15, 31, 0.72));
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.22);
            min-height: 100%;
        }

        .fintech-panel:hover {
            border-color: rgba(34, 211, 238, 0.38);
            box-shadow: 0 18px 52px rgba(0, 0, 0, 0.28), 0 0 24px rgba(34, 211, 238, 0.1);
        }

        .panel-title {
            color: #f8fbff;
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.9rem;
        }

        .merchant-row,
        .transaction-row,
        .insight-row {
            display: grid;
            gap: 0.8rem;
            align-items: center;
            padding: 0.55rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }

        .merchant-row {
            grid-template-columns: minmax(120px, 1fr) minmax(120px, 1.2fr) auto;
        }

        .transaction-row {
            grid-template-columns: minmax(130px, 1.2fr) minmax(86px, 0.7fr) auto;
        }

        .insight-row {
            grid-template-columns: 24px 1fr;
            color: var(--muted);
        }

        .insight-check {
            color: var(--green);
            text-shadow: 0 0 14px rgba(16, 185, 129, 0.55);
            font-weight: 900;
        }

        .insight-accent {
            color: var(--cyan);
            font-weight: 850;
        }

        .lightbulb-orb {
            float: right;
            width: 82px;
            height: 82px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            margin: 0.1rem 0 0.5rem 0.8rem;
            color: var(--cyan);
            font-size: 2.25rem;
            background: radial-gradient(circle, rgba(34, 211, 238, 0.2), rgba(37, 99, 235, 0.06) 58%, transparent 70%);
            box-shadow: 0 0 34px rgba(34, 211, 238, 0.22);
        }

        .row-name {
            color: var(--text);
            font-weight: 700;
        }

        .row-meta {
            color: var(--muted);
            font-size: 0.78rem;
        }

        .row-amount-positive {
            color: var(--green);
            font-weight: 800;
        }

        .row-amount-negative {
            color: #fb7185;
            font-weight: 800;
        }

        .bar-track {
            height: 8px;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.18);
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--blue), var(--cyan));
            box-shadow: 0 0 14px rgba(34, 211, 238, 0.5);
        }

        .ai-card {
            border: 1px solid rgba(45, 211, 255, 0.28);
            border-radius: 20px;
            padding: 1.2rem;
            margin: 0.95rem 0;
            background:
                radial-gradient(circle at 10% 15%, rgba(34, 211, 238, 0.13), transparent 30%),
                linear-gradient(145deg, rgba(12, 28, 54, 0.86), rgba(5, 15, 31, 0.75));
            box-shadow: 0 18px 52px rgba(0, 0, 0, 0.28);
        }

        .ai-card-title {
            color: #f8fbff;
            font-size: 1.2rem;
            font-weight: 850;
            margin-bottom: 0.7rem;
        }

        .ai-status-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .ai-status-pill {
            border: 1px solid rgba(45, 211, 255, 0.2);
            border-radius: 14px;
            padding: 0.8rem;
            background: rgba(3, 10, 24, 0.5);
        }

        .ai-status-label {
            color: var(--muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        .ai-status-value {
            color: #f8fbff;
            font-weight: 800;
            margin-top: 0.25rem;
            overflow-wrap: anywhere;
        }

        .ai-response {
            border: 1px solid rgba(16, 185, 129, 0.28);
            border-radius: 18px;
            padding: 1rem;
            margin: 1rem 0;
            background: linear-gradient(145deg, rgba(6, 78, 59, 0.2), rgba(5, 15, 31, 0.74));
            box-shadow: 0 0 24px rgba(16, 185, 129, 0.1);
        }

        .ai-response strong {
            color: #a7f3d0;
        }

        .pipeline-panel {
            border: 1px solid rgba(45, 211, 255, 0.2);
            border-radius: 18px;
            padding: 1.05rem 1.2rem;
            margin: 1rem 0;
            background:
                radial-gradient(circle at 90% 50%, rgba(37, 99, 235, 0.2), transparent 28%),
                linear-gradient(90deg, rgba(12, 28, 54, 0.78), rgba(5, 15, 31, 0.66));
            box-shadow: 0 0 34px rgba(34, 211, 238, 0.14), 0 18px 48px rgba(0, 0, 0, 0.24);
        }

        .pipeline-title {
            color: var(--cyan);
            font-size: 1rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }

        .pipeline-strip {
            display: grid;
            grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr;
            gap: 0.55rem;
            align-items: center;
        }

        .pipeline-step {
            text-align: center;
            color: var(--muted);
            font-size: 0.8rem;
        }

        .pipeline-step span {
            display: block;
            color: var(--cyan);
            font-size: 1.55rem;
            margin-bottom: 0.35rem;
            text-shadow: 0 0 18px rgba(34, 211, 238, 0.38);
        }

        .pipeline-arrow {
            color: var(--cyan);
            font-size: 1.35rem;
            text-shadow: 0 0 16px rgba(34, 211, 238, 0.45);
        }

        @media (max-width: 900px) {
            .fintech-grid,
            .ai-status-grid,
            .pipeline-strip {
                grid-template-columns: 1fr;
            }

            .finance-hero h1,
            .finance-hero p {
                max-width: 100%;
            }

            .hero-art {
                display: none;
            }

            .merchant-row,
            .transaction-row {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 1180px) {
            .finance-hero h1,
            .finance-hero p {
                max-width: 74%;
            }

            .hero-art {
                opacity: 0.78;
                transform: scale(0.88);
                transform-origin: right center;
            }
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(45, 211, 255, 0.22);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            background: linear-gradient(145deg, rgba(12, 28, 54, 0.85), rgba(5, 15, 31, 0.78));
            box-shadow: 0 14px 38px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 700;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f8fbff;
        }

        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"] {
            border-radius: 16px;
        }

        div[data-testid="stPlotlyChart"],
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(45, 211, 255, 0.16);
            background: rgba(5, 15, 31, 0.45);
            box-shadow: 0 14px 44px rgba(0, 0, 0, 0.18);
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="primary"] {
            border-radius: 12px !important;
            border: 1px solid rgba(45, 211, 255, 0.38) !important;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.95), rgba(124, 58, 237, 0.9)) !important;
            color: white !important;
            box-shadow: 0 0 22px rgba(37, 99, 235, 0.25);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--cyan) !important;
            box-shadow: 0 0 26px rgba(34, 211, 238, 0.32);
            transform: translateY(-1px);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
            border-radius: 12px !important;
            border-color: rgba(45, 211, 255, 0.18) !important;
            background-color: rgba(3, 10, 24, 0.68) !important;
        }

        .stAlert {
            border-radius: 14px;
        }

        code, pre {
            border-radius: 14px !important;
        }

        h2, h3 {
            color: var(--text);
            letter-spacing: 0;
        }

        .sidebar-brand {
            border: 1px solid rgba(45, 211, 255, 0.24);
            border-radius: 16px;
            padding: 1rem;
            margin: 0.4rem 0 1rem;
            background: linear-gradient(145deg, rgba(12, 28, 54, 0.86), rgba(5, 15, 31, 0.76));
        }

        .sidebar-brand-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0;
        }

        .sidebar-brand-subtitle {
            color: var(--cyan);
            font-size: 0.82rem;
            margin: 0.2rem 0 0;
        }

        .sidebar-tagline {
            border: 1px solid rgba(45, 211, 255, 0.2);
            border-radius: 16px;
            padding: 0.95rem;
            margin-top: 1.35rem;
            color: var(--muted);
            background: linear-gradient(145deg, rgba(12, 28, 54, 0.64), rgba(5, 15, 31, 0.55));
        }

        .sidebar-tagline strong {
            color: var(--cyan);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(icon: str, title: str, subtitle: str, kicker: str = "Spending Analytics") -> None:
    heading = f"{icon} {title}".strip()
    st.markdown(
        dedent(f"""
        <div class="finance-hero">
            <div class="finance-kicker">{kicker}</div>
            <h1>{heading}</h1>
            <p>{subtitle}</p>
            <div class="hero-art">
                <div class="wallet-icon">💳</div>
                <div class="chart-chip one">◔</div>
                <div class="chart-chip two">↗</div>
                <div class="chart-chip three">▥</div>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def money(value: float) -> str:
    return f"${value:,.2f}"


def signed_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def render_kpi_cards(total_income: float, total_spent: float, net_cashflow: float, transaction_count: int) -> None:
    cards = [
        ("💰", "Total Income", money(total_income), "Income transactions in the selected context"),
        ("💳", "Total Spending", money(total_spent), "Expense transactions in the selected context"),
        ("📈", "Net Cash Flow", signed_money(net_cashflow), "Income minus spending"),
        ("🧾", "Transactions", f"{transaction_count:,}", "Rows currently included by filters"),
    ]
    html_cards = ['<div class="fintech-grid">']
    for index, (icon, label, value, note) in enumerate(cards):
        html_cards.append(
            dedent(f"""
            <div class="kpi-card">
                <div class="kpi-top">
                    <div class="kpi-icon kpi-icon-{index}">{icon}</div>
                    <div>
                        <div class="kpi-label">{html.escape(label)}</div>
                        <div class="kpi-value">{html.escape(value)}</div>
                    </div>
                </div>
                <div class="kpi-note">↗ {html.escape(note)}</div>
            </div>
            """).strip()
        )
    html_cards.append("</div>")
    st.markdown("".join(html_cards), unsafe_allow_html=True)


def render_top_merchants_panel(expense_rows: pd.DataFrame) -> None:
    if expense_rows.empty:
        st.markdown('<div class="fintech-panel"><div class="panel-title">Top Merchants</div>No expense transactions found.</div>', unsafe_allow_html=True)
        return
    top_merchants = (
        expense_rows.groupby("merchant", as_index=False)["expense_amount"]
        .sum()
        .sort_values("expense_amount", ascending=False)
        .head(5)
    )
    max_amount = max(float(top_merchants["expense_amount"].max()), 1.0)
    rows = ['<div class="fintech-panel"><div class="panel-title">🏪 Top Merchants</div>']
    for _, row in top_merchants.iterrows():
        amount = float(row["expense_amount"])
        width = max(6, min(100, amount / max_amount * 100))
        rows.append(
            dedent(f"""
            <div class="merchant-row">
                <div class="row-name">{html.escape(str(row["merchant"]))}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
                <div class="row-amount-negative">{money(amount)}</div>
            </div>
            """).strip()
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_recent_transactions_panel(filtered: pd.DataFrame) -> None:
    if filtered.empty:
        st.markdown('<div class="fintech-panel"><div class="panel-title">Recent Transactions</div>No transactions found.</div>', unsafe_allow_html=True)
        return
    recent = filtered.sort_values("date", ascending=False).head(6)
    rows = ['<div class="fintech-panel"><div class="panel-title">🧾 Recent Transactions</div>']
    for _, row in recent.iterrows():
        amount = float(row["amount"])
        amount_class = "row-amount-positive" if amount > 0 else "row-amount-negative"
        rows.append(
            dedent(f"""
            <div class="transaction-row">
                <div>
                    <div class="row-name">{html.escape(str(row["merchant"]))}</div>
                    <div class="row-meta">{html.escape(str(row["category"]))} · {html.escape(str(row["bank"]))}</div>
                </div>
                <div class="row-meta">{html.escape(str(row["date"]))}</div>
                <div class="{amount_class}">{signed_money(amount)}</div>
            </div>
            """).strip()
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_quick_insights_panel(filtered: pd.DataFrame, by_category: pd.DataFrame, net_cashflow: float) -> None:
    top_category = "No expenses"
    if not by_category.empty:
        top_category = str(by_category.iloc[0]["category"])
    profile_count = filtered["profile"].nunique() if "profile" in filtered else 0
    bank_count = filtered["bank"].nunique() if "bank" in filtered else 0
    insights = [
        ("Top spending category", top_category),
        ("Net cash flow", signed_money(net_cashflow)),
        ("Profiles represented", f"{profile_count:,}"),
        ("Banks represented", f"{bank_count:,}"),
    ]
    rows = ['<div class="fintech-panel"><div class="lightbulb-orb">💡</div><div class="panel-title">💡 Quick Insights</div>']
    for label, value in insights:
        rows.append(
            f'<div class="insight-row"><div class="insight-check">✓</div><div>{html.escape(label)}: <span class="insight-accent">{html.escape(value)}</span></div></div>'
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_pipeline_strip() -> None:
    steps = [
        ("🗄️", "Data Generation"),
        ("⚙️", "Preprocessing"),
        ("🧠", "Machine Learning"),
        ("📊", "Analytics"),
        ("🤖", "AI Chatbot"),
        ("🎯", "Insights"),
    ]
    content = ['<div class="pipeline-panel"><div class="pipeline-title">Our Process Pipeline</div><div class="pipeline-strip">']
    for index, (icon, label) in enumerate(steps):
        content.append(f'<div class="pipeline-step"><span>{icon}</span>{html.escape(label)}</div>')
        if index < len(steps) - 1:
            content.append('<div class="pipeline-arrow">→</div>')
    content.append("</div></div>")
    st.markdown("".join(content), unsafe_allow_html=True)


def render_ai_status_card(provider: str, model: str, selected_profile: str) -> None:
    display_model = model or "Deterministic rules"
    st.markdown(
        dedent(f"""
        <div class="ai-card">
            <div class="ai-card-title">🤖 AI Financial Assistant</div>
            <div class="ai-status-grid">
                <div class="ai-status-pill">
                    <div class="ai-status-label">Current mode</div>
                    <div class="ai-status-value">{html.escape(provider)}</div>
                </div>
                <div class="ai-status-pill">
                    <div class="ai-status-label">Current model</div>
                    <div class="ai-status-value">{html.escape(display_model)}</div>
                </div>
                <div class="ai-status-pill">
                    <div class="ai-status-label">Dataset / Profile</div>
                    <div class="ai-status-value">{html.escape(get_active_dataset_label())} · {html.escape(selected_profile)}</div>
                </div>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_ai_response(answer: str, selected_profile: str) -> None:
    st.markdown(
        dedent(f"""
        <div class="ai-response">
            <strong>{html.escape(chatbot_context_text(selected_profile))}</strong><br><br>
            {html.escape(answer).replace(chr(10), "<br>")}
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,15,31,0.35)",
        font_color="#dbeafe",
        title_font_color="#f8fbff",
        colorway=["#22d3ee", "#3b82f6", "#7c3aed", "#10b981", "#f59e0b", "#ef4444", "#14b8a6"],
        margin=dict(l=20, r=20, t=58, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.18)")
    return fig


def set_active_dataset(dataset_key: str) -> None:
    st.session_state["active_dataset"] = dataset_key


def get_active_dataset_label() -> str:
    dataset_key = st.session_state.get("active_dataset", "original")
    return DATASET_LABELS.get(dataset_key, DATASET_LABELS["original"])


def render_active_dataset_banner() -> None:
    st.markdown(
        f'<div class="dataset-banner">🗃️ <strong>Active dataset:</strong> {get_active_dataset_label()}</div>',
        unsafe_allow_html=True,
    )


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


def app_detect_category_group(category: str | None) -> list[str] | None:
    if category == "Food":
        return ["Groceries", "Dining", "Coffee"]
    if category:
        return [category]
    return None


def app_category_condition(categories: list[str]) -> str:
    values = ", ".join(f"'{sql_literal(category)}'" for category in categories)
    return f"category IN ({values})" if len(categories) > 1 else f"category = {values}"


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
    category_group = app_detect_category_group(category)
    merchant = "starbucks" if "starbucks" in q else detect_merchant(q)
    year = detect_year(q)
    month = detect_month(q)

    if not category and not merchant and not asking_income:
        return None, None, None, "View monthly summary"

    filters = ["amount > 0" if asking_income else "amount < 0"]
    label_parts = []
    if selected_profile != "All":
        filters.append(f"profile = '{sql_literal(selected_profile)}'")
    if category_group and "Income" not in category_group and not asking_income:
        filters.append(app_category_condition(category_group))
        label_parts.append("Food" if category == "Food" else category_group[0])
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
    st.sidebar.header("🔎 Filters")
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
    render_page_header(
        "📤",
        "Upload Transactions",
        "Load clean transaction CSV files into SQLite, with optional model-based category prediction.",
    )

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
    render_page_header(
        "",
        "Spending Analytics",
        "Analyze spending, income, cash flow, and transaction behavior using machine learning and natural language queries.",
        kicker="AI-Powered Financial Insights",
    )
    render_active_dataset_banner()
    view = expense_view(df)
    filtered = sidebar_filters(view)

    total_spent = filtered["expense_amount"].sum()
    total_income = filtered["income_amount"].sum()
    net_cashflow = filtered["amount"].sum()
    transaction_count = len(filtered)

    by_category = (
        filtered[filtered["expense_amount"] > 0]
        .groupby("category", as_index=False)["expense_amount"]
        .sum()
        .sort_values("expense_amount", ascending=False)
    )
    by_month = filtered.groupby("month", as_index=False).agg(spending=("expense_amount", "sum"), income=("income_amount", "sum"))
    expense_rows = filtered[filtered["expense_amount"] > 0]

    render_kpi_cards(total_income, total_spent, net_cashflow, transaction_count)

    left, right = st.columns(2)
    left.plotly_chart(
        style_chart(px.bar(by_category, x="category", y="expense_amount", title="Spending by Category")),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    right.plotly_chart(
        style_chart(px.line(by_month, x="month", y=["spending", "income"], title="Monthly Cash Flow")),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    lower_left, lower_right = st.columns(2)
    with lower_left:
        render_top_merchants_panel(expense_rows)
    with lower_right:
        render_recent_transactions_panel(filtered)

    render_quick_insights_panel(filtered, by_category, net_cashflow)
    render_pipeline_strip()

    with st.expander("🧾 View full filtered transaction table"):
        st.dataframe(filtered.sort_values("date", ascending=False), use_container_width=True)


def render_chatbot(df: pd.DataFrame) -> None:
    render_page_header(
        "🤖",
        "LLM Chatbot",
        "Ask natural-language questions and inspect the safe SQLite queries used to answer them.",
        kicker="AI Financial Assistant",
    )
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

    selected_model = ollama_model if provider == "Ollama" else (hf_model if provider == "Hugging Face" else "")
    render_ai_status_card(provider, selected_model, selected_profile)
    st.caption(mode_description_text(provider, ollama_model, hf_model).replace("\n\n", " "))
    st.caption("The LLM generates SQL. SQLite stores and retrieves the transaction data.")

    example_groups = get_example_groups(active_profiles, selected_profile)

    examples = next(iter(example_groups.values()))
    st.markdown('<div class="ai-card-title">💬 Ask the assistant</div>', unsafe_allow_html=True)
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
            render_ai_response(transparent_answer or answer, selected_profile)
            if sql_adjusted:
                st.caption("LLM SQL was adjusted or replaced because it contained unsupported filters.")
            if details_df is not None and not details_df.empty and "transaction" in question.lower():
                st.subheader("Transaction Details")
                st.dataframe(details_df, use_container_width=True, hide_index=True)
            with st.expander("Generated SQL"):
                st.code(sql, language="sql")
            st.subheader("📌 Aggregated Result")
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
    render_page_header(
        "🗄️",
        "Database",
        "Inspect the active SQLite transaction table, summaries, and exportable records.",
    )
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
    st.subheader("🧾 Full Transaction Table")
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

    st.subheader("📌 Breakdowns")
    left, right = st.columns(2)
    category_counts = view["category"].value_counts().rename_axis("category").reset_index(name="transactions")
    profile_counts = view["profile"].value_counts().rename_axis("profile").reset_index(name="transactions")
    left.dataframe(category_counts, use_container_width=True, hide_index=True)
    right.dataframe(profile_counts, use_container_width=True, hide_index=True)

    st.subheader("📊 Demo Visualizations")
    expense_rows = view[view["expense_amount"] > 0]
    by_category = expense_rows.groupby("category", as_index=False)["expense_amount"].sum().sort_values("expense_amount", ascending=False)
    by_profile = expense_rows.groupby("profile", as_index=False)["expense_amount"].sum().sort_values("expense_amount", ascending=False)
    by_bank = expense_rows.groupby("bank", as_index=False)["expense_amount"].sum().sort_values("expense_amount", ascending=False)
    by_month = view.groupby("month", as_index=False).agg(spending=("expense_amount", "sum"), income=("income_amount", "sum"))

    chart_left, chart_right = st.columns(2)
    chart_left.plotly_chart(
        style_chart(px.bar(by_category, x="category", y="expense_amount", title="Spending by Category")),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    chart_right.plotly_chart(
        style_chart(px.bar(by_profile, x="profile", y="expense_amount", title="Spending by Profile")),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    chart_left.plotly_chart(
        style_chart(px.bar(by_bank, x="bank", y="expense_amount", title="Spending by Bank")),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    chart_right.plotly_chart(
        style_chart(px.line(by_month, x="month", y=["spending", "income"], title="Monthly Cash Flow")),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def render_dataset_generator_page() -> None:
    render_page_header(
        "🎲",
        "Dataset Generator",
        "Create randomized demo data without changing the deterministic grading dataset.",
    )
    render_active_dataset_banner()

    st.subheader("🧪 Demo Controls")
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
    apply_theme()
    ensure_seed_data()
    db_df = load_dataframe()
    st.sidebar.markdown(
        dedent("""
        <div class="sidebar-brand">
            <p class="sidebar-brand-title">💳 Spending Analytics</p>
            <p class="sidebar-brand-subtitle">AI-powered financial insights</p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )
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
    st.sidebar.markdown(
        '<div class="sidebar-tagline">✨ <strong>Transforming transactions</strong><br>into intelligent financial insights.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
