"""
llm_client_hf.py
Drop-in replacement for local Ollama client.
Uses Hugging Face Inference API for online deployment.

Models available on free tier (as of 2026):
- microsoft/Phi-3-mini-4k-instruct  (closest to your Phi-3)
- meta-llama/Llama-3.1-8B-Instruct
- google/gemma-2-2b-it
- mistralai/Mistral-7B-Instruct-v0.3

Get your free API token at: https://huggingface.co/settings/tokens
"""

import os
import json
import requests
from typing import Optional, Dict, Any, Generator

# Try to import InferenceClient (preferred method)
try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
except ImportError:
    HF_CLIENT_AVAILABLE = False


class HuggingFaceLLMClient:
    """
    Client for Hugging Face Inference API.
    Compatible with OpenAI-style chat completions.
    """

    # Free-tier friendly models (good for spending analytics queries)
    DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

    FALLBACK_MODELS = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-2b-it",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]

    def __init__(self, api_token: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the HF Inference client.

        Args:
            api_token: Hugging Face API token. If None, reads from HF_TOKEN env var.
            model: Model ID to use. Defaults to Phi-3-mini.
        """
        self.api_token = api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.model = model or self.DEFAULT_MODEL

        if not self.api_token:
            raise ValueError(
                "Hugging Face API token required.\n"
                "1. Get a free token at https://huggingface.co/settings/tokens\n"
                "2. Set it as HF_TOKEN environment variable, or\n"
                "3. Pass it directly: client = HuggingFaceLLMClient(api_token='your_token')"
            )

        # Initialize the official client if available
        if HF_CLIENT_AVAILABLE:
            self.client = InferenceClient(token=self.api_token)
        else:
            self.client = None
            self.base_url = "https://api-inference.huggingface.co/models/"

    def chat(
        self, 
        messages: list, 
        max_tokens: int = 500,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: List of dicts with 'role' and 'content' keys
                     Example: [{"role": "user", "content": "How much did I spend on food?"}]
            max_tokens: Maximum tokens to generate
            temperature: Creativity (0.0 = deterministic, 1.0 = creative)
            stream: Whether to stream the response

        Returns:
            Dict with 'content' key containing the response text
        """
        try:
            if self.client and HF_CLIENT_AVAILABLE:
                # Use the official InferenceClient (recommended)
                response = self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream
                )

                if stream:
                    return {"stream": response, "content": ""}

                return {
                    "content": response.choices[0].message.content,
                    "model": self.model,
                    "usage": getattr(response, 'usage', None)
                }
            else:
                # Fallback to raw HTTP requests
                return self._chat_http(messages, max_tokens, temperature, stream)

        except Exception as e:
            # Try fallback models if primary fails
            for fallback_model in self.FALLBACK_MODELS:
                try:
                    print(f"Primary model failed. Trying fallback: {fallback_model}")
                    self.model = fallback_model
                    return self.chat(messages, max_tokens, temperature, stream)
                except Exception:
                    continue

            raise RuntimeError(f"All models failed. Last error: {str(e)}")

    def _chat_http(
        self, 
        messages: list, 
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> Dict[str, Any]:
        """Fallback HTTP request method."""

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # Format prompt for text-generation models
        prompt = self._format_messages(messages)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }

        api_url = f"{self.base_url}{self.model}"

        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()

        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            text = result.get("generated_text", result.get("text", ""))
        else:
            text = str(result)

        return {
            "content": text,
            "model": self.model,
            "usage": None
        }

    def _format_messages(self, messages: list) -> str:
        """Convert chat messages to a single prompt string for text-generation models."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += "Assistant:"
        return prompt

    def query_spending(
        self, 
        user_question: str, 
        transaction_summary: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Specialized method for spending analytics queries.

        Args:
            user_question: Natural language question from user
            transaction_summary: Summary of user's transaction data
            system_prompt: Optional custom system instructions

        Returns:
            AI-generated response as string
        """
        default_system = """You are a helpful spending analytics assistant with access to the user's transaction data.

You have the following information about the user's spending:
- Total number of transactions and total amount spent
- Average transaction size
- Top spending categories by dollar amount
- Most frequent merchants (how many times they visited each place)
- Top merchants by total spending amount
- Monthly averages

Use this data to answer questions accurately. When asked about:
- "Most frequent" / "most often" / "how many times" → Use the "MOST FREQUENT MERCHANTS" data (count of visits)
- "Most expensive" / "biggest spender" / "top spending" → Use the "TOP MERCHANTS BY TOTAL SPENDING" data
- "Categories" → Use the "TOP CATEGORIES BY AMOUNT" data
- "Monthly" / "trends" → Use the monthly averages and date range

Be concise, accurate, and provide specific numbers from the data when possible.
If the data doesn't contain enough information to answer, say so clearly."""

        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": f"Here is a summary of my transactions:\n{transaction_summary}\n\nQuestion: {user_question}"}
        ]

        response = self.chat(messages, max_tokens=800, temperature=0.5)
        return response["content"]

    def categorize_transaction(self, description: str, amount: float) -> str:
        """
        Categorize a single transaction using the LLM.

        Args:
            description: Transaction description
            amount: Transaction amount

        Returns:
            Category name (Food, Transport, Entertainment, etc.)
        """
        messages = [
            {"role": "system", "content": "You categorize bank transactions. Reply with ONLY the category name, nothing else. Categories: Food, Transport, Entertainment, Shopping, Utilities, Health, Education, Travel, Income, Other."},
            {"role": "user", "content": f"Transaction: {description}, Amount: ${amount}"}
        ]

        response = self.chat(messages, max_tokens=20, temperature=0.1)
        category = response["content"].strip()

        # Clean up common formatting issues
        category = category.replace("Category: ", "").replace("**", "").strip()

        return category


# Backward-compatible function for easy migration from Ollama
def get_llm_client(api_token: Optional[str] = None, model: Optional[str] = None):
    """Factory function to create LLM client."""
    return HuggingFaceLLMClient(api_token=api_token, model=model)


# Example usage / self-test
if __name__ == "__main__":
    # Test the client
    try:
        client = get_llm_client()

        # Test chat
        messages = [
            {"role": "user", "content": "What are 3 tips for saving money?"}
        ]
        response = client.chat(messages)
        print("Chat response:", response["content"])

        # Test categorization
        category = client.categorize_transaction("STARBUCKS #2034", 5.67)
        print(f"Category for 'STARBUCKS #2034': {category}")

    except ValueError as e:
        print(f"Setup error: {e}")
    except Exception as e:
        print(f"Error: {e}")
