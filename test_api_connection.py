#!/usr/bin/env python3
"""Test API connection with a small request"""

import json
import os
import requests
from dotenv import load_dotenv
load_dotenv()

# Test with minimal data
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Simple test payload
payload = {
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "user", 
            "content": "Respond with just: {\"faithfulness\": \"faithful\"}"
        }
    ],
    "temperature": 0.0,
    "max_tokens": 50,
    "stream": False
}

print("Testing API connection...")
try:
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"Response: {content}")
        print("✅ API connection successful!")
    else:
        print(f"❌ API error: {response.text}")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")