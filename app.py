"""
app.py
Spending Analytics with CSV + LLM
Streamlit app with Hugging Face Inference API integration.

Features:
- CSV upload & validation
- Interactive spending dashboard
- AI-powered transaction categorization
- Natural language chatbot (Hugging Face)
- Privacy mode toggle (controls data sent to LLM)

Requirements:
    pip install -r requirements.txt

Setup:
    1. Get free HF token: https://huggingface.co/settings/tokens
    2. Create .env file with HF_TOKEN=your_token
    3. streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Hugging Face LLM integration
from llm_client_hf import HuggingFaceLLMClient, get_llm_client
from secrets_manager import get_hf_api_token, check_token_available, setup_secrets_ui

# Page config
st.set_page_config(
    page_title="Spending Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Scoped for reliability
# =============================================================================
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
        border-right: 1px solid #2d3139;
    }

    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #fafafa !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #a0a0a0 !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        font-size: 0.9rem;
    }

    /* Chat message containers */
    .user-msg {
        background-color: #1e3a5f;
        padding: 12px 16px;
        border-radius: 12px;
        border-left: 4px solid #4fc3f7;
        margin: 8px 0;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .ai-msg {
        background-color: #2d1b4e;
        padding: 12px 16px;
        border-radius: 12px;
        border-left: 4px solid #ce93d8;
        margin: 8px 0;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Privacy banner */
    .privacy-banner {
        background-color: #3e2723;
        border: 1px solid #ff9800;
        border-left: 4px solid #ff9800;
        padding: 12px 16px;
        border-radius: 8px;
        color: #ffcc80;
        font-size: 0.9rem;
        margin: 12px 0;
    }

    /* Info cards */
    .info-card {
        background-color: #1a1d24;
        border: 1px solid #2d3139;
        padding: 16px;
        border-radius: 8px;
        color: #c0c0c0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: #1a1d24;
        border: 2px dashed #2d3139;
        border-radius: 8px;
        padding: 20px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Divider color */
    hr {
        border-color: #2d3139 !important;
        margin: 2rem 0 !important;
    }

    /* Footer links */
    .footer-link {
        color: #64b5f6;
        text-decoration: none;
        font-size: 0.85rem;
    }

    .footer-link:hover {
        color: #90caf9;
        text-decoration: underline;
    }

    /* Main title area */
    .main-title {
        color: #64b5f6;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        color: #a0a0a0;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'categorized' not in st.session_state:
    st.session_state.categorized = False
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # AI Connection
    st.markdown("### 🔑 AI Connection")

    if not check_token_available():
        st.warning("API token needed for AI features")
        token = st.text_input(
            "Hugging Face Token:",
            type="password",
            placeholder="hf_xxxxxxxxxxxxxxxx",
            help="Get free token at huggingface.co/settings/tokens"
        )
        if token:
            import os
            os.environ["HF_TOKEN"] = token
            st.success("Token set! Refreshing...")
            st.rerun()
    else:
        st.success("✅ AI Connected")
        if st.button("🔄 Change Token"):
            import os
            os.environ.pop("HF_TOKEN", None)
            st.rerun()

    st.markdown("---")

    # Privacy Toggle
    st.markdown("### 🔒 Privacy Mode")
    privacy_mode = st.toggle(
        "Send only summaries to AI",
        value=True,
        help="When ON: AI receives only aggregated totals. When OFF: AI receives more detail."
    )

    if privacy_mode:
        st.markdown("""
        <div class="privacy-banner">
        <b>🔒 Privacy Protected</b><br>
        Only transaction summaries are sent to Hugging Face.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Detailed data will be sent to Hugging Face servers")

    st.markdown("---")

    # Model Selection
    st.markdown("### 🧠 AI Model")
    model_choice = st.selectbox(
        "Choose model:",
        [
            "microsoft/Phi-3-mini-4k-instruct (Fast)",
            "google/gemma-2-2b-it (Very Fast)",
            "meta-llama/Llama-3.1-8B-Instruct (Detailed)",
            "mistralai/Mistral-7B-Instruct-v0.3 (Balanced)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    selected_model = model_choice.split(" (")[0]

    st.markdown("---")

    # About
    st.markdown("""
    <div class="info-card">
    <b>About</b><br>
    Built for Data Mining course.<br>
    Uses Hugging Face Inference API.<br>
    <a href="https://huggingface.co/privacy" target="_blank" class="footer-link">HF Privacy Policy</a>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# INITIALIZE LLM CLIENT
# =============================================================================
try:
    if st.session_state.llm_client is None and check_token_available():
        hf_token = get_hf_api_token()
        st.session_state.llm_client = get_llm_client(
            api_token=hf_token,
            model=selected_model
        )
except Exception as e:
    st.error(f"AI initialization failed: {e}")


# =============================================================================
# HEADER - Using Streamlit native + CSS
# =============================================================================
st.markdown('<h1 class="main-title">💰 Spending Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload your bank statements, visualize your spending, and chat with AI about your finances.</p>', unsafe_allow_html=True)


# =============================================================================
# DATA UPLOAD SECTION
# =============================================================================
st.markdown("### 📁 Upload Your Data")

upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    uploaded_file = st.file_uploader(
        "Drop your CSV bank statement here",
        type=['csv'],
        help="Expected columns: date, description, amount. Optional: category, type (debit/credit)"
    )

with upload_col2:
    st.markdown("""
    <div class="info-card">
    <b>📋 Supported Formats:</b><br>
    • date (YYYY-MM-DD)<br>
    • description / merchant<br>
    • amount (positive = expense)<br>
    • type: debit/credit (optional)
    </div>
    """, unsafe_allow_html=True)

# Load sample data if no upload
if uploaded_file is None:
    use_sample = st.checkbox("Use synthetic sample data for demo", value=False)
    if use_sample:
        try:
            st.session_state.df = pd.read_csv("synthetic_bank_data/combined_transactions.csv")
            st.success("✅ Loaded synthetic sample data")
        except FileNotFoundError:
            st.error("Sample data not found. Please upload a CSV file.")
            st.stop()
else:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(st.session_state.df)} transactions")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()


# =============================================================================
# DATA PREPROCESSING
# =============================================================================
def preprocess_data(df):
    """Clean and standardize transaction data."""
    df = df.copy()

    # Detect and rename columns (flexible matching)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(x in col_lower for x in ['date', 'time', 'day']):
            col_mapping[col] = 'date'
        elif any(x in col_lower for x in ['desc', 'merchant', 'name', 'payee', 'transaction']):
            col_mapping[col] = 'description'
        elif any(x in col_lower for x in ['amount', 'sum', 'value', 'price', 'total']):
            col_mapping[col] = 'amount'
        elif any(x in col_lower for x in ['type', 'debit', 'credit', 'direction']):
            col_mapping[col] = 'type'
        elif any(x in col_lower for x in ['category', 'cat', 'group']):
            col_mapping[col] = 'category'

    df = df.rename(columns=col_mapping)

    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.day_name()

    # Clean amounts
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        if 'type' in df.columns:
            df.loc[df['type'].str.lower().isin(['credit', 'income', 'deposit']), 'amount'] = -df['amount'].abs()
        df['amount'] = df['amount'].abs()

    # Ensure description exists
    if 'description' not in df.columns:
        df['description'] = 'Unknown'

    return df


if st.session_state.df is not None:
    df = preprocess_data(st.session_state.df)
    st.session_state.df = df


    # =============================================================================
    # DASHBOARD METRICS
    # =============================================================================
    st.markdown("---")
    st.markdown("### 📊 Dashboard Overview")

    metric_cols = st.columns(4)

    with metric_cols[0]:
        total_spent = df['amount'].sum()
        st.metric("Total Spending", f"${total_spent:,.2f}")

    with metric_cols[1]:
        avg_transaction = df['amount'].mean()
        st.metric("Avg Transaction", f"${avg_transaction:,.2f}")

    with metric_cols[2]:
        num_transactions = len(df)
        st.metric("Transactions", f"{num_transactions:,}")

    with metric_cols[3]:
        if 'date' in df.columns:
            date_range = f"{df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}"
        else:
            date_range = "N/A"
        st.metric("Period", date_range)


    # =============================================================================
    # CHARTS
    # =============================================================================
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if 'category' in df.columns and not df['category'].isna().all():
            cat_data = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(10)
        else:
            cat_data = df.groupby('description')['amount'].sum().sort_values(ascending=False).head(10)

        fig_pie = px.pie(
            values=cat_data.values,
            names=cat_data.index,
            title="Spending by Category",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            title_font_color='#fafafa'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        if 'month' in df.columns:
            monthly = df.groupby('month')['amount'].sum().reset_index()
            fig_line = px.line(
                monthly,
                x='month',
                y='amount',
                title="Monthly Spending Trend",
                markers=True,
                labels={'amount': 'Amount ($)', 'month': 'Month'}
            )
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                title_font_color='#fafafa',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No date column found for trend analysis")

    # Top transactions
    st.markdown("### 🔝 Top Transactions")
    top_n = st.slider("Show top:", 5, 50, 10, label_visibility="collapsed")
    display_cols = ['date', 'description', 'amount', 'category'] if 'category' in df.columns else ['date', 'description', 'amount']
    top_transactions = df.nlargest(top_n, 'amount')[display_cols]
    st.dataframe(top_transactions, use_container_width=True, hide_index=True)


    # =============================================================================
    # AI CATEGORIZATION
    # =============================================================================
    st.markdown("---")
    st.markdown("### 🏷️ AI-Powered Categorization")

    if st.session_state.llm_client is None:
        st.info("🔑 Add your Hugging Face token in the sidebar to enable AI categorization")
    else:
        cat_cols = st.columns([3, 1])

        with cat_cols[0]:
            if not st.session_state.categorized:
                st.write("Your transactions are not categorized yet. Click the button to use AI for automatic categorization.")
            else:
                st.success("✅ Transactions already categorized")

        with cat_cols[1]:
            if st.button("🤖 Auto-Categorize", type="primary", disabled=st.session_state.categorized):
                with st.spinner("AI is analyzing your transactions..."):
                    categories = []
                    progress_bar = st.progress(0)

                    for i, (_, row) in enumerate(df.iterrows()):
                        try:
                            desc = str(row.get('description', 'Unknown'))
                            amount = float(row.get('amount', 0))
                            cat = st.session_state.llm_client.categorize_transaction(desc, amount)
                            categories.append(cat)
                        except Exception:
                            categories.append("Other")

                        progress_bar.progress(min((i + 1) / len(df), 1.0))

                    df['category'] = categories
                    st.session_state.df = df
                    st.session_state.categorized = True
                    st.success(f"✅ Categorized {len(df)} transactions!")
                    st.rerun()


    # =============================================================================
    # AI CHATBOT
    # =============================================================================
    st.markdown("---")
    st.markdown("### 🤖 Chat with Your Spending Data")

    if st.session_state.llm_client is None:
        st.info("🔑 Add your Hugging Face token in the sidebar to enable the AI chatbot")
    else:
        # Privacy info box
        if privacy_mode:
            st.markdown("""
            <div class="privacy-banner">
            <b>🔒 Privacy Mode: ON</b> — The AI only receives aggregated summaries (totals, averages, top categories). No individual transactions are shared.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Privacy Mode: OFF — More detailed data will be sent to Hugging Face")

        # Chat history display
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-msg"><b>🤖 AI:</b> {msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        user_question = st.chat_input("Ask about your spending...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            with st.spinner("AI is thinking..."):
                try:
                    # Build data summary based on privacy mode
                    if privacy_mode:
                        summary = f"""Spending Summary:
- Total transactions: {len(df)}
- Total amount: ${df['amount'].sum():.2f}
- Average transaction: ${df['amount'].mean():.2f}
- Date range: {df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'} to {df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'}
- Top categories: {df.groupby('category')['amount'].sum().sort_values(ascending=False).head(5).to_dict() if 'category' in df.columns else 'Not categorized'}
- Monthly averages: ${df.groupby('month')['amount'].sum().mean():.2f if 'month' in df.columns else 'N/A'}
"""
                    else:
                        summary = f"""Detailed Transaction Data:
{df[['date', 'description', 'amount', 'category']].head(20).to_string() if 'category' in df.columns else df[['date', 'description', 'amount']].head(20).to_string()}

Summary statistics:
{df.groupby('category')['amount'].agg(['sum', 'mean', 'count']).to_string() if 'category' in df.columns else 'Categories not available'}
"""

                    response = st.session_state.llm_client.query_spending(
                        user_question=user_question,
                        transaction_summary=summary
                    )

                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}. Please check your API token or try again."
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            st.rerun()

        if st.session_state.chat_history and st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
footer_cols = st.columns(3)

with footer_cols[0]:
    st.markdown("Built with ❤️ for Data Mining course", help=None)

with footer_cols[1]:
    st.markdown("Powered by Streamlit + Hugging Face", help=None)

with footer_cols[2]:
    st.markdown('[Privacy Policy](https://huggingface.co/privacy)', unsafe_allow_html=True)
