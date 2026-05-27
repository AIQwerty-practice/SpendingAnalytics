# 💰 Spending Analytics with CSV + LLM

An interactive Streamlit application that helps users analyze their spending habits through automated transaction categorization, interactive dashboards, and a natural-language AI chatbot. Upload your bank statements (CSV) and get instant insights powered by AI.

**🌐 Works online** — no local installations needed thanks to Hugging Face Inference API integration.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **📊 Interactive Dashboard** | Visualize spending trends, category breakdowns, and monthly summaries with dynamic charts |
| **🏷️ AI Auto-Categorization** | Automatically classifies transactions into categories (Food, Transport, Entertainment, etc.) using AI |
| **🤖 AI Chatbot** | Ask natural language questions about your spending ("How much did I spend on groceries last month?") |
| **📁 CSV Upload** | Upload your own bank statements; the app reads and classifies the data automatically |
| **🔒 Privacy Mode Toggle** | Control exactly how much data is shared with the AI — aggregated summaries only, or detailed breakdowns |

---

## 📋 Requirements

- **Python 3.9+**
- **Hugging Face API token** (free — get yours at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/spending-analytics.git
cd spending-analytics
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```text
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
requests>=2.31.0
python-dateutil>=2.8.0
scikit-learn>=1.3.0
huggingface_hub>=0.23.0
python-dotenv>=1.0.0
toml>=0.10.2
```

### 4. Set Up Your API Token

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` and add your token:
```
HF_TOKEN=hf_your_token_here
```

> **Get a free token:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📂 Project Structure

```
spending-analytics/
├── app.py                  # Main Streamlit application
├── llm_client_hf.py        # Hugging Face Inference API client
├── secrets_manager.py      # Secure API key handling
├── requirements.txt        # Python dependencies
├── .env.example            # Template for local API keys
├── .gitignore             # Prevents committing secrets
├── README.md              # This file
├── synthetic_bank_data/    # Sample data folder
│   └── combined_transactions.csv
└── assets/
    └── demo_screenshot.png
```

---

## 🖥️ Usage

### 1. Upload Your Data
- Drag and drop your bank statement CSV file
- Or use the included **synthetic sample data** for testing

### 2. Explore the Dashboard
- View spending by category (pie charts, bar charts)
- Track monthly trends over time
- Identify top merchants and recurring payments

### 3. AI Auto-Categorize
- Click **"Auto-Categorize"** to have the AI classify all your transactions
- Categories include: Food, Transport, Entertainment, Shopping, Utilities, Health, Education, Travel, Income, Other

### 4. Chat with Your Data
- Open the chatbot panel
- Ask questions like:
  - *"What was my biggest expense this month?"*
  - *"Compare my food spending between January and February"*
  - *"Show me all transactions over $100"*

---

## 🔒 Privacy & Data Handling

### Privacy Mode (Built-In Toggle)

The app includes a **Privacy Mode** switch in the sidebar that controls exactly what data is sent to the AI:

| Mode | What AI Receives | Best For |
|------|------------------|----------|
| **🔒 ON (Default)** | Only aggregated summaries (totals, averages, top categories) | Real bank statements |
| **⚠️ OFF** | Detailed breakdowns (top 20 transactions, full statistics) | Better AI answers, synthetic data |

### Data Flow

```
Your CSV → Streamlit App (your browser) → AI Summary → Hugging Face API
```

- **Raw CSV files NEVER leave your device**
- Only the text summary you approve is sent to Hugging Face
- Data is encrypted in transit (HTTPS)
- Subject to [Hugging Face's Privacy Policy](https://huggingface.co/privacy)

### Security Best Practices

- ✅ **Never commit `.env` or `.streamlit/secrets.toml` to GitHub**
- ✅ Use `.gitignore` to protect sensitive files
- ✅ Rotate your HF token regularly in [settings](https://huggingface.co/settings/tokens)
- ✅ Use Privacy Mode ON when working with real financial data

---

## ☁️ Deploy to Streamlit Community Cloud (FREE)

### Step 1: Push to GitHub

Make sure your repo includes:
- `app.py`, `llm_client_hf.py`, `secrets_manager.py`
- `requirements.txt`
- `.env.example` (template only — **no real token!**)
- `.gitignore` (must exclude `.env` and secrets)

### Step 2: Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository

### Step 3: Add Your Secret

1. In Streamlit Cloud, go to **Settings → Secrets**
2. Add:
   ```toml
   HF_TOKEN = "hf_your_actual_token"
   ```
3. Click **Save**

### Step 4: Deploy

Your app will be live at `https://your-app-name.streamlit.app`

Anyone with the link can use it immediately — **no installations required**.

---

## 🧠 AI Models Available

Choose your model in the app sidebar:

| Model | Speed | Best For |
|-------|-------|----------|
| **microsoft/Phi-3-mini-4k-instruct** | Fast | General Q&A, categorization *(default)* |
| **google/gemma-2-2b-it** | Very Fast | Simple queries, summaries |
| **meta-llama/Llama-3.1-8B-Instruct** | Medium | Complex reasoning, detailed analysis |
| **mistralai/Mistral-7B-Instruct-v0.3** | Medium | Balanced performance |

All models run on Hugging Face's free Inference API tier.

---

## 🎓 Course Context

This project was built for a **Data Mining course** and covers the full pipeline:

1. **Data Generation** — Synthetic bank transaction dataset creation
2. **Exploratory Data Analysis (EDA)** — Understanding spending patterns
3. **Preprocessing** — Cleaning, normalization, feature engineering
4. **Classification** — AI-powered categorization of transactions
5. **Visualization** — Interactive dashboards with Streamlit + Plotly
6. **LLM Integration** — Natural language interface via Hugging Face API
7. **Deployment** — Cloud hosting with privacy controls

---

## 🤝 Contributing

This is a course project. Feel free to fork and extend with:
- Additional bank statement formats (PDF, Excel parsing)
- More sophisticated ML categorization models
- Multi-language support for the chatbot
- Plaid / Open Banking API integration for live data

---

## 📄 License

MIT License — Created for educational purposes.

---

## 🙋 FAQ

**Q: Do I need to install Ollama or any local LLM?**  
A: **No.** The app uses Hugging Face's cloud API. Everything works online.

**Q: Is my financial data safe?**  
A: With **Privacy Mode ON** (default), only aggregated summaries are sent to the AI. Your raw CSV never leaves your device. For maximum privacy, run locally with a self-hosted LLM (see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)).

**Q: What CSV format is supported?**  
A: The app auto-detects common columns: `date`, `description`/`merchant`, `amount`, `type` (debit/credit), and `category`. Flexible column name matching is built in.

**Q: Can I use a different LLM provider?**  
A: Yes! Modify `llm_client_hf.py` to point to OpenAI, Anthropic, or any OpenAI-compatible API endpoint.

**Q: Is the Hugging Face API really free?**  
A: Yes, there is a generous free tier for testing and small projects. Rate limits apply (~1-2 requests/second). For production use, consider Hugging Face Pro ($9/month) or dedicated inference endpoints.

**Q: Why is the AI slow sometimes?**  
A: Free-tier models have "cold starts" — the first request may take 10-30 seconds as the model loads. Subsequent requests are faster.

---

**Built with ❤️ using Python, Streamlit, Plotly, and Hugging Face**
