# 🚀 Deploy Your Spending Analytics App Online with Hugging Face

This guide walks you through switching from local Ollama to Hugging Face's free Inference API so anyone can use your app without installing anything.

---

## 📦 What You're Getting

| File | Purpose |
|------|---------|
| `llm_client_hf.py` | Drop-in replacement for your Ollama client |
| `secrets_manager.py` | Securely handles API keys across environments |
| `app_hf_integration.py` | Example showing how to integrate into your app |
| `requirements_hf.txt` | Updated dependencies with Hugging Face libraries |
| `.env.example` | Template for local development API keys |
| `secrets.toml.example` | Template for Streamlit Cloud secrets |

---

## 🔑 Step 1: Get Your Free Hugging Face API Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a free account (if you don't have one)
3. Click **"New token"**
4. Give it a name (e.g., "spending-analytics")
5. Select **"Read"** role
6. Copy the token (starts with `hf_`)

**Free tier limits (as of 2026):**
- Generous monthly credits for testing
- Rate limits apply (~1-2 requests/second)
- Perfect for course demos and small projects

---

## 🖥️ Step 2: Local Development Setup

### Option A: Using .env file (easiest for local)

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and paste your token:
   ```
   HF_TOKEN=hf_your_actual_token_here
   ```

3. Add `.env` to your `.gitignore` (NEVER commit this file!):
   ```
   .env
   .streamlit/secrets.toml
   __pycache__/
   *.pyc
   ```

### Option B: Using environment variable

```bash
# Windows
set HF_TOKEN=hf_your_token

# macOS/Linux
export HF_TOKEN=hf_your_token
```

---

## 🔄 Step 3: Update Your App Code

### Minimal changes to your existing `app.py`:

**1. Replace Ollama imports:**
```python
# OLD (Ollama)
# import ollama

# NEW (Hugging Face)
from llm_client_hf import get_llm_client
from secrets_manager import get_hf_api_token, check_token_available, setup_secrets_ui
```

**2. Add API key check at the top:**
```python
if not check_token_available():
    setup_secrets_ui()
    st.stop()
```

**3. Replace Ollama initialization:**
```python
# OLD
# response = ollama.chat(model="phi3", messages=[...])

# NEW
hf_token = get_hf_api_token()
llm_client = get_llm_client(api_token=hf_token)
response = llm_client.chat(messages=[...])
answer = response["content"]
```

**4. Update requirements:**
Replace your `requirements.txt` with `requirements_hf.txt` (or add `huggingface_hub>=0.23.0` to your existing one).

---

## ☁️ Step 4: Deploy to Streamlit Community Cloud (FREE)

### 4.1 Push to GitHub

Make sure your repo has:
- `app.py` (your updated app)
- `llm_client_hf.py`
- `secrets_manager.py`
- `requirements.txt` (with huggingface_hub)
- `.gitignore` (with .env and secrets.toml)
- Your data files

**DO NOT include:**
- `.env` file
- `.streamlit/secrets.toml`
- Any file with your actual API token

### 4.2 Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. In **Advanced settings**, add your secret:
   - Key: `HF_TOKEN`
   - Value: `hf_your_actual_token`
6. Click **Deploy**

Your app will be live at `https://your-app-name.streamlit.app`

---

## 🧪 Testing Your Integration

Run this quick test to verify everything works:

```python
# test_hf_connection.py
from llm_client_hf import get_llm_client
from secrets_manager import get_hf_api_token

try:
    token = get_hf_api_token()
    client = get_llm_client(api_token=token)

    response = client.chat([
        {"role": "user", "content": "Say 'Hello from Hugging Face!'"}
    ])

    print("✅ Connection successful!")
    print(f"Response: {response['content']}")

except Exception as e:
    print(f"❌ Error: {e}")
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "API token not found" | Check your `.env` file or Streamlit secrets |
| "Model loading" / timeout | The free tier has cold starts. Wait 10-30 seconds and retry |
| "Rate limit exceeded" | Wait a minute and try again. Free tier has limits |
| "Model not found" | The model ID changed. Check [huggingface.co/models](https://huggingface.co/models) |
| Responses are slow | Use a smaller model like `microsoft/Phi-3-mini-4k-instruct` |
| Want faster responses | Consider Hugging Face Pro ($9/month) or dedicated endpoints |

---

## 🎓 Recommended Models for Your Project

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Fast | General Q&A, categorization |
| `google/gemma-2-2b-it` | 2B | Very Fast | Simple queries, summaries |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Medium | Complex reasoning |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Medium | Detailed analysis |

Default in `llm_client_hf.py` is **Phi-3-mini** — best balance of speed and quality for your use case.

---

## 🔒 Security Best Practices

1. **Never commit API tokens to GitHub**
2. **Use Streamlit secrets for production**
3. **Rotate tokens regularly** in HF settings
4. **Monitor usage** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. **Set up rate limiting** in your app to prevent abuse

---

## 📚 Next Steps

- **Monitor usage**: Check your HF dashboard for API call counts
- **Upgrade if needed**: Pro plan ($9/month) removes rate limits
- **Try other models**: Swap model IDs in `llm_client_hf.py`
- **Add caching**: Use `@st.cache_data` to reduce API calls

---

**Your app is now ready for the world! 🌍**
