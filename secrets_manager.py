"""
secrets_manager.py
Handles API keys securely for Streamlit Cloud deployment.

For local development: Uses .env file or environment variables
For Streamlit Cloud: Uses st.secrets (secrets.toml)
"""

import os
import streamlit as st
from pathlib import Path


def get_hf_api_token() -> str:
    """
    Retrieve Hugging Face API token from multiple sources in order of priority:
    1. Streamlit secrets (for Cloud deployment)
    2. Environment variable HF_TOKEN
    3. Environment variable HUGGINGFACE_API_KEY
    4. Local .env file (for development)

    Returns:
        The API token string

    Raises:
        ValueError: If no token is found anywhere
    """

    # 1. Check Streamlit secrets (for Streamlit Cloud)
    try:
        token = st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACE_API_KEY")
        if token:
            return token
    except Exception:
        pass  # Not running on Streamlit Cloud or secrets not configured

    # 2. Check environment variables
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if token:
        return token

    # 3. Check local .env file (development only)
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_API_KEY="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if token:
                        return token

    # 4. Check .streamlit/secrets.toml (local Streamlit)
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        try:
            import toml
            secrets = toml.load(secrets_path)
            token = secrets.get("HF_TOKEN") or secrets.get("HUGGINGFACE_API_KEY")
            if token:
                return token
        except ImportError:
            pass

    # If we get here, no token was found
    raise ValueError(
        "Hugging Face API token not found!\n\n"
        "Please set it using ONE of these methods:\n"
        "1. Streamlit Cloud: Add HF_TOKEN in Settings > Secrets\n"
        "2. Local .env file: Create a .env file with HF_TOKEN=your_token\n"
        "3. Environment variable: export HF_TOKEN=your_token\n"
        "4. Local Streamlit: Create .streamlit/secrets.toml with HF_TOKEN = \"your_token\"\n\n"
        "Get your free token at: https://huggingface.co/settings/tokens"
    )


def setup_secrets_ui():
    """
    Streamlit UI component for users to input their API token
    if not already configured. Call this in your app if token is missing.
    """
    st.warning("🔑 Hugging Face API token required")

    st.markdown("""
    To use the AI chatbot, you need a free Hugging Face API token.

    **Steps:**
    1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    2. Create a free account (if needed) and generate a token
    3. Paste it below
    """)

    token = st.text_input(
        "Enter your HF API token:",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
        help="Your token will not be stored permanently"
    )

    if token:
        os.environ["HF_TOKEN"] = token
        st.success("Token set for this session! Refresh the page.")
        st.rerun()

    st.info("💡 For permanent setup, add HF_TOKEN to your Streamlit secrets or .env file")

    return token


def check_token_available() -> bool:
    """Check if API token is available without raising error."""
    try:
        get_hf_api_token()
        return True
    except ValueError:
        return False
