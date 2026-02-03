# Agentic Delivery Intelligence (LangChain + LangGraph)

This PoC mirrors your earlier Streamlit app with LangChain+LangGraph orchestration.

## Quickstart
1) Set Groq API key  
   - Environment (recommended):
     ```bash
     export GROQ_API_KEY="your_api_key_here"
     ```
   - Or Streamlit secrets: create `.streamlit/secrets.toml`
     ```toml
     GROQ_API_KEY = "your_api_key_here"
     ```
2) Install
```bash
pip install -r requirements.txt