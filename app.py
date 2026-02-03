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

# # Import loaders with clear diagnostics if anything fails
# try:
#     from src.loaders import parse_excel, parse_csv_txt, parse_pdf, normalize_and_validate
# except Exception:
#     st.set_page_config(page_title="Agentic AI – Loader Import Error", layout="wide")
#     st.error("Import failure in **src.loaders** (see traceback below).")
#     st.code(traceback.format_exc())
#     st.stop()

# # Other imports (graph, qa)
# try:
#     from src.graph import build_graph
#     from src.qa import route_and_summarize
# except Exception:
#     st.set_page_config(page_title="Agentic AI – App Import Error", layout="wide")
#     st.error("Import failure in **src.graph / src.qa** (see traceback below).")
#     st.code(traceback.format_exc())
#     st.stop()

# # If Streamlit secrets has the key, expose it so ChatGroq can read it.
# if "GROQ_API_KEY" in st.secrets:
#     os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# # # --- LLM Health Check (temporary debug UI) ---
# # with st.sidebar.expander("🧪 LLM Health Check", expanded=True):
# #     import sys, os, importlib, time
# #     py = sys.executable
# #     has_key = bool(os.getenv("GROQ_API_KEY"))
# #     st.write("Python:", py)
# #     st.write("Has GROQ_API_KEY?", has_key)

# #     def ver(mod):
# #         try:
# #             m = importlib.import_module(mod)
# #             return getattr(m, "__version__", "ok")
# #         except Exception as e:
# #             return f"import failed: {e}"

# #     st.write("groq:", ver("groq"))

# #     # Direct Groq SDK test
# #     ok, msg, elapsed = False, "", None
# #     try:
# #         from groq import Groq
# #         if has_key:
# #             t0 = time.time()
# #             client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# #             r = client.chat.completions.create(
# #                 model="llama-3.1-8b-instant",
# #                 messages=[{"role":"user","content":"healthcheck: reply 'OK'"}],
# #                 temperature=0,
# #             )
# #             elapsed = time.time() - t0
# #             ok = True
# #             msg = r.choices[0].message.content
# #         else:
# #             msg = "No key; set via .streamlit/secrets.toml or environment."
# #     except Exception as e:
# #         msg = f"{type(e).__name__}: {e}"
# #     st.write("Groq SDK:", "✅" if ok else "❌", msg)
# #     if elapsed is not None:
# #         st.write(f"Latency: {elapsed:.2f}s")

# def _has_groq_key() -> bool:
#     return bool(os.getenv("GROQ_API_KEY"))

# st.set_page_config(page_title="🧠 Agentic AI – Project & Delivery (LangGraph)", layout="wide")

# # # --- LLM Health Check (temporary debug UI) ---
# # with st.sidebar.expander("🧪 LLM Health Check", expanded=True):
# #     import sys, os, importlib
# #     py = sys.executable
# #     has_key = bool(os.getenv("GROQ_API_KEY"))
# #     key_mask = None
# #     if has_key:
# #         k = os.getenv("GROQ_API_KEY")
# #         key_mask = k[:6] + "…" + k[-4:] if len(k) >= 10 else "set (masked)"
# #     st.write("Python:", py)
# #     st.write("Has GROQ_API_KEY?", has_key)
# #     if key_mask: st.write("Key (masked):", key_mask)

# #     def _ver(mod):
# #         try:
# #             m = importlib.import_module(mod)
# #             return getattr(m, "__version__", "ok (no __version__)")
# #         except Exception as e:
# #             return f"import failed: {e}"

# #     st.write("langchain:", _ver("langchain"))
# #     st.write("langchain_groq:", _ver("langchain_groq"))
# #     st.write("groq:", _ver("groq"))

# #     # Direct Groq SDK test
# #     groq_ok, groq_msg = False, ""
# #     try:
# #         from groq import Groq
# #         if not has_key:
# #             groq_msg = "No GROQ_API_KEY in env."
# #         else:
# #             client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# #             r = client.chat.completions.create(
# #                 model="llama-3.1-8b-instant",
# #                 messages=[{"role":"user","content":"healthcheck: reply 'OK'"}],
# #             )
# #             groq_ok = True
# #             groq_msg = r.choices[0].message.content
# #     except Exception as e:
# #         groq_msg = f"Groq SDK error: {e}"
# #     st.write("Groq SDK:", "✅" if groq_ok else "❌", groq_msg)

# #     # LangChain ChatGroq test
# #     lc_ok, lc_msg = False, ""
# #     try:
# #         from langchain_groq import ChatGroq
# #         from langchain_core.prompts import ChatPromptTemplate
# #         if not has_key:
# #             lc_msg = "No GROQ_API_KEY in env."
# #         else:
# #             llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
# #             prompt = ChatPromptTemplate.from_messages([("system","You are helpful."),("user","Reply: OK")])
# #             out = (prompt | llm).invoke({})
# #             lc_ok = True
# #             lc_msg = getattr(out, "content", str(out))
# #     except Exception as e:
# #         lc_msg = f"LangChain error: {e}"
# #     st.write("LangChain ChatGroq:", "✅" if lc_ok else "❌", lc_msg)


# # ----- Header (logo made safe) -----
# col1, col2 = st.columns([1, 6])
# with col1:
#     try:
#         st.image("compunnel_logo.jpg", width=120)
#     except Exception:
#         st.caption(" ")  # keep layout spacing if logo missing
# with col2:
#     st.title("🧠 Agentic AI – Project & Delivery Intelligence (LangGraph)")
#     st.caption("Enterprise Project, Cost, Risk & HR Intelligence Platform")

# # ----- Sidebar -----
# st.sidebar.header("📂 Upload Data")
# uploaded_file = st.sidebar.file_uploader(
#     "Upload Delivery Data (Excel / CSV / TXT / PDF)",
#     type=["xlsx", "csv", "txt", "pdf"]
# )
# st.sidebar.header("⚙️ Controls")
# use_llm = st.sidebar.checkbox("Generate Executive AI Summary (LLM)")

# # ----- Help text if no file yet -----
# if uploaded_file is None:
#     st.info("Please upload a file to proceed. Supported: XLSX / CSV / TXT / PDF.")
#     st.stop()

# # ----- Robust loader (no crash on parse errors) -----
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

# # --- LOAD DATA ---
# data = load_uploaded_data(uploaded_file)

# # --- BUILD GRAPH ---
# workflow = build_graph()

# # --- Actions ---
# if st.button("🚀 Run AI Analysis"):
#     with st.spinner("Running LangGraph multi-agent analysis..."):
#         result = workflow.invoke({"raw_df": data, "use_llm": use_llm and _has_groq_key()})

#         st.subheader("📉 Underutilized Employees")
#         st.dataframe(result.get("low_util"), use_container_width=True)

#         st.subheader("🚨 Delivery Risk Projects (Jira)")
#         st.dataframe(result.get("risk_projects"), use_container_width=True)

#         st.subheader("💰 Loss-Making / Margin Risk Projects")
#         st.dataframe(result.get("loss_projects"), use_container_width=True)

#         st.subheader("⚠️ HR Risk Indicators")
#         st.dataframe(result.get("hr_risks"), use_container_width=True)

#         if use_llm:
#             if result.get("llm_summary"):
#                 st.subheader("🧠 Executive AI Summary")
#                 st.success(result["llm_summary"])
#             elif not _has_groq_key():
#                 st.info("GROQ_API_KEY not found. Skipped Executive AI Summary.")
#             else:
#                 st.info("LLM call skipped or returned empty.")

# st.markdown("---")
# st.subheader("🤖 Ask Delivery Intelligence Bot")
# user_q = st.text_input("Ask about utilization, Jira risk, HR issues, cost overrun, margin, etc.")
# if st.button("🧠 Get Answer"):
#     if not user_q.strip():
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Analyzing..."):
#             from src.models import utilization_model, delivery_risk_model, cost_margin_model, hr_health_model
#             util_df = utilization_model(data)
#             risk_df = delivery_risk_model(data)
#             cost_df = cost_margin_model(data)
#             hr_df = hr_health_model(data)
#             answer = route_and_summarize(user_q, util_df, risk_df, cost_df, hr_df)
#             st.success(answer)


# # --- BEGIN robust import shim (fixes Windows path/import issues) ---
# import sys, os
# from pathlib import Path
# _THIS_FILE = Path(__file__).resolve()
# _PROJECT_ROOT = _THIS_FILE.parent
# if str(_PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PROJECT_ROOT))
# # --- END robust import shim ---

# # ---- Import core libs first ----
# import traceback
# import streamlit as st
# import pandas as pd

# # ---- IMPORTANT: set page config FIRST, before ANY other st.* calls ----
# st.set_page_config(page_title="🧠 Agentic AI – Project & Delivery (LangGraph)", layout="wide")

# # ---- Try imports that render to the page on failure (no second page_config) ----
# loaders_import_error = None
# graph_qa_import_error = None
# try:
#     from src.loaders import parse_excel, parse_csv_txt, parse_pdf, normalize_and_validate
# except Exception:
#     loaders_import_error = traceback.format_exc()

# try:
#     from src.graph import build_graph
#     from src.qa import route_and_summarize
# except Exception:
#     graph_qa_import_error = traceback.format_exc()

# # If Streamlit secrets has the key, expose it so Groq SDK can read it.
# if "GROQ_API_KEY" in st.secrets:
#     os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# def _has_groq_key() -> bool:
#     return bool(os.getenv("GROQ_API_KEY"))

# # ---- Header (logo made safe) ----
# col1, col2 = st.columns([1, 6])
# with col1:
#     try:
#         st.image("compunnel_logo.jpg", width=120)
#     except Exception:
#         st.caption(" ")  # keep layout spacing if logo missing
# with col2:
#     st.title("🧠 Agentic AI – Project & Delivery Intelligence (LangGraph)")
#     st.caption("Enterprise Project, Cost, Risk & HR Intelligence Platform")

# # ---- Surface import errors (no second page_config) ----
# if loaders_import_error:
#     st.error("Import failure in **src.loaders** (see traceback below).")
#     st.code(loaders_import_error)
#     st.stop()

# if graph_qa_import_error:
#     st.error("Import failure in **src.graph / src.qa** (see traceback below).")
#     st.code(graph_qa_import_error)
#     st.stop()

# # ---- Sidebar: Upload + Controls ----
# st.sidebar.header("📂 Upload Data")
# uploaded_file = st.sidebar.file_uploader(
#     "Upload Delivery Data (Excel / CSV / TXT / PDF)",
#     type=["xlsx", "csv", "txt", "pdf"]
# )
# st.sidebar.header("⚙️ Controls")
# use_llm = st.sidebar.checkbox("Generate Executive AI Summary (LLM)")

# # ---- LLM Health Check (now it's AFTER page_config) ----
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

# # ---- Stop early if no file yet (after we’ve drawn the header UI) ----
# if uploaded_file is None:
#     st.info("Please upload a file to proceed. Supported: XLSX / CSV / TXT / PDF.")
#     st.stop()

# # ---- Robust loader (no crash on parse errors) ----
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

# # --- LOAD DATA ---
# data = load_uploaded_data(uploaded_file)

# # --- BUILD GRAPH ---
# workflow = build_graph()

# # --- Actions ---
# if st.button("🚀 Run AI Analysis"):
#     with st.spinner("Running LangGraph multi-agent analysis..."):
#         # we pass use_llm as-is; src/llm.py now uses Groq SDK directly
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

# st.markdown("---")
# st.subheader("🤖 Ask Delivery Intelligence Bot")
# user_q = st.text_input("Ask about utilization, Jira risk, HR issues, cost overrun, margin, etc.")
# if st.button("🧠 Get Answer"):
#     if not user_q.strip():
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Analyzing..."):
#             from src.models import utilization_model, delivery_risk_model, cost_margin_model, hr_health_model
#             util_df = utilization_model(data)
#             risk_df = delivery_risk_model(data)
#             cost_df = cost_margin_model(data)
#             hr_df = hr_health_model(data)

#             answer = route_and_summarize(user_q, util_df, risk_df, cost_df, hr_df)
#             st.success(answer)


# # --- existing code above ---
# st.subheader("🤖 Ask Delivery Intelligence Bot")
# user_q = st.text_input("Ask about utilization, Jira risk, HR issues, cost overrun, margin, etc.")
# if st.button("🧠 Get Answer"):
#     if not user_q.strip():
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Analyzing..."):
#             from src.models import utilization_model, delivery_risk_model, cost_margin_model, hr_health_model
#             util_df = utilization_model(data)
#             risk_df = delivery_risk_model(data)
#             cost_df = cost_margin_model(data)
#             hr_df = hr_health_model(data)

#             # ⬇️ changed line: unpack answer and optional table
#             answer, table_df = route_and_summarize(user_q, util_df, risk_df, cost_df, hr_df)

#             # Render the response (markdown to support tables)
#             st.markdown(answer)

#             # If we got a real table, show it and allow download
#             if table_df is not None and not table_df.empty:
#                 st.dataframe(table_df, use_container_width=True)
#                 csv_bytes = table_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     "⬇️ Download CSV",
#                     data=csv_bytes,
#                     file_name="qa_result.csv",
#                     mime="text/csv",
#                 )




# --- BEGIN robust import shim (fixes Windows path/import issues) ---
import sys, os
from pathlib import Path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# --- END robust import shim ---

import traceback
import streamlit as st
import pandas as pd

# IMPORTANT: set page config FIRST, before ANY other st.* calls
st.set_page_config(page_title="🧠 Agentic AI – Project & Delivery (LangGraph)", layout="wide")

# Try imports (render errors to page; do NOT call page_config again)
loaders_import_error = None
graph_qa_import_error = None
try:
    from src.loaders import parse_excel, parse_csv_txt, parse_pdf, normalize_and_validate
except Exception:
    loaders_import_error = traceback.format_exc()
try:
    from src.graph import build_graph
    from src.qa import route_and_summarize
except Exception:
    graph_qa_import_error = traceback.format_exc()

# Expose Groq key from secrets if present
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Header
col1, col2 = st.columns([1, 6])
with col1:
    try:
        st.image("compunnel_logo.jpg", width=120)
    except Exception:
        st.caption(" ")  # keep spacing if logo missing
with col2:
    st.title("🧠 Agentic AI – Project & Delivery Intelligence (LangGraph)")
    st.caption("Enterprise Project, Cost, Risk & HR Intelligence Platform")

# Surface import errors (stop early)
if loaders_import_error:
    st.error("Import failure in **src.loaders** (traceback below).")
    st.code(loaders_import_error)
    st.stop()
if graph_qa_import_error:
    st.error("Import failure in **src.graph / src.qa** (traceback below).")
    st.code(graph_qa_import_error)
    st.stop()

# Sidebar: Upload + Controls
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Delivery Data (Excel / CSV / TXT / PDF)",
    type=["xlsx", "csv", "txt", "pdf"]
)
st.sidebar.header("⚙️ Controls")
use_llm = st.sidebar.checkbox("Generate Executive AI Summary (LLM)")

# LLM Health Check
with st.sidebar.expander("🧪 LLM Health Check", expanded=True):
    import importlib, time
    py = sys.executable
    has_key = bool(os.getenv("GROQ_API_KEY"))
    st.write("Python:", py)
    st.write("Has GROQ_API_KEY?", has_key)

    def ver(mod):
        try:
            m = importlib.import_module(mod)
            return getattr(m, "__version__", "ok")
        except Exception as e:
            return f"import failed: {e}"

    st.write("groq:", ver("groq"))

    # Direct Groq SDK test
    ok, msg, elapsed = False, "", None
    try:
        from groq import Groq
        if has_key:
            t0 = time.time()
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            r = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "healthcheck: reply 'OK'"}],
                temperature=0,
            )
            elapsed = time.time() - t0
            ok = True
            msg = r.choices[0].message.content
        else:
            msg = "No key; set via .streamlit/secrets.toml or environment."
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
    st.write("Groq SDK:", "✅" if ok else "❌", msg)
    if elapsed is not None:
        st.write(f"Latency: {elapsed:.2f}s")

# If no file yet, draw guidance and stop
if uploaded_file is None:
    st.info("Please upload a file to proceed. Supported: XLSX / CSV / TXT / PDF.")
    st.stop()

# Robust loader
def load_uploaded_data(file) -> pd.DataFrame:
    file_bytes = file.getvalue()
    ext = file.name.lower().split(".")[-1]
    try:
        if ext == "xlsx":
            df = parse_excel(file_bytes)
        elif ext in ["csv", "txt"]:
            df = parse_csv_txt(file_bytes)
        elif ext == "pdf":
            df = parse_pdf(file_bytes)
        else:
            st.error("Unsupported file format")
            st.stop()
    except Exception as e:
        st.error("❌ Failed to parse uploaded file")
        st.code(str(e))
        st.stop()

    try:
        df = normalize_and_validate(df)
    except Exception as e:
        st.error("❌ Required columns missing or invalid data")
        st.code(str(e))
        st.stop()

    st.success("✅ Uploaded data validated successfully")
    with st.expander("Detected columns", expanded=False):
        st.write(list(df.columns))
    return df

# Load data
data = load_uploaded_data(uploaded_file)

# Build graph
workflow = build_graph()

# Run AI Analysis
if st.button("🚀 Run AI Analysis", key="btn_run_ai_analysis"):
    with st.spinner("Running LangGraph multi-agent analysis..."):
        result = workflow.invoke({"raw_df": data, "use_llm": use_llm})

        st.subheader("📉 Underutilized Employees")
        st.dataframe(result.get("low_util"), use_container_width=True)

        st.subheader("🚨 Delivery Risk Projects (Jira)")
        st.dataframe(result.get("risk_projects"), use_container_width=True)

        st.subheader("💰 Loss-Making / Margin Risk Projects")
        st.dataframe(result.get("loss_projects"), use_container_width=True)

        st.subheader("⚠️ HR Risk Indicators")
        st.dataframe(result.get("hr_risks"), use_container_width=True)

        if use_llm and result.get("llm_summary"):
            st.subheader("🧠 Executive AI Summary")
            st.success(result["llm_summary"])
        elif use_llm and not result.get("llm_summary"):
            st.info("LLM call skipped or returned empty. Check Health Check in sidebar.")

# QA block (unique keys + tuple support)
st.markdown("---")
st.subheader("🤖 Ask Delivery Intelligence Bot")

# Unique key on text input to avoid DuplicateWidgetID
user_q = st.text_input(
    "Ask about utilization, Jira risk, HR issues, cost overrun, margin, etc.",
    key="qa_user_question",
)

# Unique key on button as well
if st.button("🧠 Get Answer", key="qa_get_answer_btn"):
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing..."):
            from src.models import (
                utilization_model,
                delivery_risk_model,
                cost_margin_model,
                hr_health_model,
            )
            util_df = utilization_model(data)
            risk_df = delivery_risk_model(data)
            cost_df = cost_margin_model(data)
            hr_df = hr_health_model(data)

            # route_and_summarize now returns (answer_text, table_df)
            answer, table_df = route_and_summarize(
                user_q, util_df, risk_df, cost_df, hr_df
            )

            # Render markdown so any embedded table snippet shows correctly
            st.markdown(answer)

            # If a real table was returned (list intent), show it and allow CSV download
            if table_df is not None and not table_df.empty:
                st.dataframe(table_df, use_container_width=True)
                csv_bytes = table_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_bytes,
                    file_name="qa_result.csv",
                    mime="text/csv",
                    key="qa_download_csv_btn",
                )