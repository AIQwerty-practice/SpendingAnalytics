💰 Spending Analytics with CSV + LLM
An interactive Streamlit application that helps users analyze their spending habits through automated transaction categorization, interactive dashboards, and a natural-language AI chatbot. Upload your bank statements (CSV) and get instant insights powered by a local LLM.
🚀 Features
Table
Feature	Description
📊 Interactive Dashboard	Visualize spending trends, category breakdowns, and monthly summaries with dynamic charts
🏷️ Auto-Categorization	Automatically classifies transactions into categories (Food, Transport, Entertainment, etc.)
🤖 AI Chatbot	Ask natural language questions about your spending ("How much did I spend on groceries last month?")
📁 CSV Upload	Upload your own bank statements; the app reads and classifies the data automatically
🔒 Privacy-First	Runs entirely locally with Ollama — your financial data never leaves your machine
📋 Requirements
Local Setup (Recommended for Development)
Python 3.9+
Ollama installed locally (ollama.com)
Phi-3 model pulled via Ollama:
bash
Copy
ollama pull phi3:mini
For Deployment (No Local Installations Needed)
See the Deployment Options section below for cloud-hosted alternatives that eliminate the need for users to install Ollama or Python.
🛠️ Installation
1. Clone the Repository
bash
Copy
git clone https://github.com/YOUR_USERNAME/spending-analytics.git
cd spending-analytics
2. Create a Virtual Environment
bash
Copy
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
3. Install Dependencies
bash
Copy
pip install -r requirements.txt
requirements.txt:
plain
Copy
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
requests>=2.31.0
python-dateutil>=2.8.0
scikit-learn>=1.3.0
4. Start Ollama (Local LLM)
Ensure Ollama is running in the background:
bash
Copy
ollama serve
5. Run the App
bash
Copy
streamlit run app.py
The app will open at http://localhost:8501
📂 Project Structure
plain
Copy
spending-analytics/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── synthetic_bank_data/    # Sample data folder
│   └── combined_transactions.csv
├── utils/
│   ├── data_loader.py      # CSV reading & validation
│   ├── categorizer.py      # Transaction classification logic
│   ├── llm_client.py       # Ollama/LLM integration
│   └── visualizations.py   # Chart & dashboard helpers
└── assets/
    └── demo_screenshot.png
🖥️ Usage
1. Upload Your Data
Drag and drop your bank statement CSV file
Or use the included synthetic dataset for testing
2. Explore the Dashboard
View spending by category (pie charts, bar charts)
Track monthly trends over time
Identify top merchants and recurring payments
3. Chat with Your Data
Open the chatbot panel
Ask questions like:
"What was my biggest expense this month?"
"Compare my food spending between January and February"
"Show me all transactions over $100"
☁️ Deployment Options
To let users run the app without installing Ollama or Python, consider these free hosting alternatives:
Option 1: Streamlit Community Cloud (Easiest)
Deploy the Streamlit frontend for free, with the LLM hosted separately.
Cost: Free for public repos
Steps: Push to GitHub → Connect at share.streamlit.io → Deploy in one click
Note: You'll need to point the app to a cloud-hosted LLM API instead of local Ollama 
Option 2: Oracle Cloud Free Tier + Ollama
Host Ollama on a free cloud VM so users only access the web app.
Cost: Always free (4 OCPUs + 24GB RAM)
Capacity: Can run Phi-3 Mini, Llama 3.1 8B, or Mistral 7B (quantized)
Performance: ~5–8 tokens/second on ARM instances
Guide: See this tutorial for setup instructions 
Option 3: Hugging Face Inference API (Simplest LLM Hosting)
Replace Ollama with a free Hugging Face API endpoint.
Cost: Free tier available
Pros: No VM management needed
Cons: Requires internet connection; rate limits apply
Option 4: Google Cloud Run (Serverless)
Deploy Ollama in a serverless container that scales to zero.
Cost: Free tier includes $300 credits; pay only when used
Best for: Low-traffic demos or class presentations
Note: Requires GPU quota approval (can take a few days) 
Recommended Architecture for Your Course Project
plain
Copy
┌─────────────────┐      ┌──────────────────┐
│  Streamlit App  │──────▶│  Cloud LLM API   │
│  (Free Hosting) │      │  (OCI/HuggingFace)│
└─────────────────┘      └──────────────────┘
        │
        ▼
┌─────────────────┐
│  User CSV Upload │
└─────────────────┘
This way, anyone with the link can use your app immediately — no setup required.
🧠 LLM Integration Details
The app uses Ollama with Phi-3 Mini by default. The LLM handles:
Natural language to query translation (e.g., "food spending last month" → filtered DataFrame)
Insight generation (summarizing unusual spending patterns)
Category suggestions for uncategorized transactions
Switching to Cloud LLM
In utils/llm_client.py, change the endpoint:
Python
Copy
# Local Ollama (default)
OLLAMA_URL = "http://localhost:11434"

# Cloud-hosted Ollama (e.g., Oracle Cloud)
# OLLAMA_URL = "http://YOUR_VM_IP:11434"

# Hugging Face Inference API
# OLLAMA_URL = "https://api-inference.huggingface.co/models/YOUR_MODEL"
🎓 Course Context
This project was built for a Data Mining course and covers the full pipeline:
Data Generation — Synthetic bank transaction dataset creation
Exploratory Data Analysis (EDA) — Understanding spending patterns
Preprocessing — Cleaning, normalization, feature engineering
Classification — Rule-based + ML categorization of transactions
Visualization — Interactive dashboards with Streamlit
LLM Integration — Natural language interface for data querying
🤝 Contributing
This is a course project. Feel free to fork and extend with:
Additional bank statement formats (PDF, Excel)
More sophisticated ML categorization models
Multi-language support for the chatbot
📄 License
MIT License — Created for educational purposes.
🙋 FAQ
Q: Do users need to install Ollama?
A: Only for local mode. Use the cloud deployment options for zero-install access.
Q: What CSV format is supported?
A: The app expects columns like: date, description, amount, type (debit/credit). It auto-detects common bank statement formats.
Q: Can I use my own LLM instead of Phi-3?
A: Yes! Any model available on Ollama (Llama 3, Mistral, Gemma) or any OpenAI-compatible API will work.
