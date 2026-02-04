import streamlit as st
from PIL import Image
import os
from loaders import load_delivery_data
from models import get_underutilized_employees
from qa import ask_delivery_bot
from llm import get_llm


st.set_page_config(page_title="Delivery Intelligence AI", layout="wide")
st.title("📦 Delivery Intelligence Platform")
# logo_path = "compunnel_logo.jpg"

# if os.path.exists(logo_path):
#     logo = Image.open(logo_path)
#     st.image(logo, width=180)
# else:
#     st.warning("Compunnel logo not found")
# Sidebar
st.sidebar.header("📂 Upload Delivery Data")
st.sidebar.image("compunnel_logo.jpg", width=180)
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

df = None
if uploaded_file:
    try:
        df = load_delivery_data(uploaded_file)
        st.sidebar.success("Data loaded successfully")
    except Exception as e:
        st.sidebar.error(str(e))

# Preview
if df is not None:
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

# Executive Summary
st.subheader("🧠 Executive AI Summary")

if st.button("Run AI Analysis"):
    if df is None:
        st.warning("Please upload data first.")
    else:
        try:
            llm = get_llm()
            prompt = (
                "Analyze the delivery dataset and provide:\n"
                "- Utilization risks\n"
                "- Delivery bottlenecks\n"
                "- Cost concerns\n"
                "- Actionable recommendations\n\n"
                f"{df.head(25).to_string()}"
            )
            summary = llm.invoke(prompt)
            st.success("Analysis completed")
            st.write(summary)
        except Exception as e:
            st.error(f"LLM Error: {e}")

# Underutilized Employees
st.subheader("📉 Underutilized Employees")

if df is not None:
    st.dataframe(get_underutilized_employees(df))

# Ask Bot
st.subheader("🤖 Ask Delivery Intelligence Bot")

question = st.text_input(
    "Ask a question (e.g. Who are underutilized employees?)"
)

if st.button("Get Answer"):
    if df is None:
        st.warning("Please upload data first.")
    else:
        st.write(ask_delivery_bot(df, question))


# # --- BEGIN robust import shim (fixes Windows path/import issues) ---
# import sys, os
# from pathlib import Path
# _THIS_FILE = Path(__file__).resolve()
# _PROJECT_ROOT = _THIS_FILE.parent
# if str(_PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PROJECT_ROOT))
# # --- END robust import shim ---

# import traceback
# import streamlit as st
# import pandas as pd

# # IMPORTANT: set page config FIRST, before ANY other st.* calls
# st.set_page_config(page_title="🧠 Agentic AI – Project & Delivery (LangGraph)", layout="wide")

# # Try imports (render errors to page; do NOT call page_config again)
# loaders_import_error = None
# graph_qa_import_error = None
# try:
#     from loaders import parse_excel, parse_csv_txt, parse_pdf, normalize_and_validate
# except Exception:
#     loaders_import_error = traceback.format_exc()
# try:
#     from graph import build_graph
#     from qa import route_and_summarize
# except Exception:
#     graph_qa_import_error = traceback.format_exc()

# # Expose Groq key from secrets if present
# if "GROQ_API_KEY" in st.secrets:
#     os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# # Header
# col1, col2 = st.columns([1, 6])
# with col1:
#     try:
#         st.image("compunnel_logo.jpg", width=120)
#     except Exception:
#         st.caption(" ")  # keep spacing if logo missing
# with col2:
#     st.title("🧠 Agentic AI – Project & Delivery Intelligence (LangGraph)")
#     st.caption("Enterprise Project, Cost, Risk & HR Intelligence Platform")

# # Surface import errors (stop early)
# if loaders_import_error:
#     st.error("Import failure in **src.loaders** (traceback below).")
#     st.code(loaders_import_error)
#     st.stop()
# if graph_qa_import_error:
#     st.error("Import failure in **src.graph / src.qa** (traceback below).")
#     st.code(graph_qa_import_error)
#     st.stop()

# # Sidebar: Upload + Controls
# st.sidebar.header("📂 Upload Data")
# uploaded_file = st.sidebar.file_uploader(
#     "Upload Delivery Data (Excel / CSV / TXT / PDF)",
#     type=["xlsx", "csv", "txt", "pdf"]
# )
# st.sidebar.header("⚙️ Controls")
# use_llm = st.sidebar.checkbox("Generate Executive AI Summary (LLM)")

# # LLM Health Check
# with st.sidebar.expander("🧪 LLM Health Check", expanded=True):
#     import importlib, time
#     py = sys.executable
#     has_key = bool(os.getenv("GROQ_API_KEY"))
#     st.write("Python:", py)
#     st.write("Has GROQ_API_KEY?", has_key)

#     def ver(mod):
#         try:
#             m = importlib.import_module(mod)
#             return getattr(m, "__version__", "ok")
#         except Exception as e:
#             return f"import failed: {e}"

#     st.write("groq:", ver("groq"))

#     # Direct Groq SDK test
#     ok, msg, elapsed = False, "", None
#     try:
#         from groq import Groq
#         if has_key:
#             t0 = time.time()
#             client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#             r = client.chat.completions.create(
#                 model="llama-3.1-8b-instant",
#                 messages=[{"role": "user", "content": "healthcheck: reply 'OK'"}],
#                 temperature=0,
#             )
#             elapsed = time.time() - t0
#             ok = True
#             msg = r.choices[0].message.content
#         else:
#             msg = "No key; set via .streamlit/secrets.toml or environment."
#     except Exception as e:
#         msg = f"{type(e).__name__}: {e}"
#     st.write("Groq SDK:", "✅" if ok else "❌", msg)
#     if elapsed is not None:
#         st.write(f"Latency: {elapsed:.2f}s")

# # If no file yet, draw guidance and stop
# if uploaded_file is None:
#     st.info("Please upload a file to proceed. Supported: XLSX / CSV / TXT / PDF.")
#     st.stop()

# # Robust loader
# def load_uploaded_data(file) -> pd.DataFrame:
#     file_bytes = file.getvalue()
#     ext = file.name.lower().split(".")[-1]
#     try:
#         if ext == "xlsx":
#             df = parse_excel(file_bytes)
#         elif ext in ["csv", "txt"]:
#             df = parse_csv_txt(file_bytes)
#         elif ext == "pdf":
#             df = parse_pdf(file_bytes)
#         else:
#             st.error("Unsupported file format")
#             st.stop()
#     except Exception as e:
#         st.error("❌ Failed to parse uploaded file")
#         st.code(str(e))
#         st.stop()

#     try:
#         df = normalize_and_validate(df)
#     except Exception as e:
#         st.error("❌ Required columns missing or invalid data")
#         st.code(str(e))
#         st.stop()

#     st.success("✅ Uploaded data validated successfully")
#     with st.expander("Detected columns", expanded=False):
#         st.write(list(df.columns))
#     return df

# # Load data
# data = load_uploaded_data(uploaded_file)

# # Build graph
# workflow = build_graph()

# # Run AI Analysis
# if st.button("🚀 Run AI Analysis", key="btn_run_ai_analysis"):
#     with st.spinner("Running LangGraph multi-agent analysis..."):
#         result = workflow.invoke({"raw_df": data, "use_llm": use_llm})

#         st.subheader("📉 Underutilized Employees")
#         st.dataframe(result.get("low_util"), use_container_width=True)

#         st.subheader("🚨 Delivery Risk Projects (Jira)")
#         st.dataframe(result.get("risk_projects"), use_container_width=True)

#         st.subheader("💰 Loss-Making / Margin Risk Projects")
#         st.dataframe(result.get("loss_projects"), use_container_width=True)

#         st.subheader("⚠️ HR Risk Indicators")
#         st.dataframe(result.get("hr_risks"), use_container_width=True)

#         if use_llm and result.get("llm_summary"):
#             st.subheader("🧠 Executive AI Summary")
#             st.success(result["llm_summary"])
#         elif use_llm and not result.get("llm_summary"):
#             st.info("LLM call skipped or returned empty. Check Health Check in sidebar.")

# # QA block (unique keys + tuple support)
# st.markdown("---")
# st.subheader("🤖 Ask Delivery Intelligence Bot")

# # Unique key on text input to avoid DuplicateWidgetID
# user_q = st.text_input(
#     "Ask about utilization, Jira risk, HR issues, cost overrun, margin, etc.",
#     key="qa_user_question",
# )

# # Unique key on button as well
# if st.button("🧠 Get Answer", key="qa_get_answer_btn"):
#     if not user_q.strip():
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Analyzing..."):
#             from src.models import (
#                 utilization_model,
#                 delivery_risk_model,
#                 cost_margin_model,
#                 hr_health_model,
#             )
#             util_df = utilization_model(data)
#             risk_df = delivery_risk_model(data)
#             cost_df = cost_margin_model(data)
#             hr_df = hr_health_model(data)

#             # route_and_summarize now returns (answer_text, table_df)
#             answer, table_df = route_and_summarize(
#                 user_q, util_df, risk_df, cost_df, hr_df
#             )

#             # Render markdown so any embedded table snippet shows correctly
#             st.markdown(answer)

#             # If a real table was returned (list intent), show it and allow CSV download
#             if table_df is not None and not table_df.empty:
#                 st.dataframe(table_df, use_container_width=True)
#                 csv_bytes = table_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     "⬇️ Download CSV",
#                     data=csv_bytes,
#                     file_name="qa_result.csv",
#                     mime="text/csv",
#                     key="qa_download_csv_btn",
#                 )





