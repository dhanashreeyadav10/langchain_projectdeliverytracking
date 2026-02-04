
# from __future__ import annotations
import pandas as pd


def load_delivery_data(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format. Upload CSV or Excel.")


import io, re
import pandas as pd

# NOTE:
# We import heavy/optional libs *inside* functions so that a failed import
# does NOT break module import (which causes "cannot import name ..." masking).

REQUIRED_COLS = [
    "employee_id", "employee_name", "department", "designation",
    "employment_type", "location", "experience_years", "cost_per_hour",
    "manager_id", "project_id", "project_name", "client_name",
    "project_type", "start_date", "end_date", "planned_hours",
    "billing_rate", "work_date", "hours_logged", "billable",
    "task_type", "jira_ticket", "ticket_status", "priority",
    "story_points", "attendance_pct", "leave_days", "performance_rating"
]

NUMERIC_COLS = [
    "hours_logged", "cost_per_hour", "billing_rate",
    "attendance_pct", "leave_days", "performance_rating",
    "experience_years", "story_points", "planned_hours"
]

def parse_excel(file_bytes: bytes) -> pd.DataFrame:
    # Lazy import openpyxl via pandas
    try:
        return pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        raise RuntimeError(
            "Excel parsing failed. Ensure 'openpyxl' is installed and the file is a valid .xlsx."
        ) from e

def parse_csv_txt(file_bytes: bytes) -> pd.DataFrame:
    # Lazy import chardet
    try:
        import chardet
    except Exception as e:
        raise RuntimeError("Missing dependency: chardet") from e

    encoding = chardet.detect(file_bytes).get("encoding", "utf-8")
    text = file_bytes.decode(encoding, errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("File has no data rows")
    sample = lines[1]
    delimiter = None
    for d in [",", ";", "\t", "|"]:
        if sample.count(d) >= 5:
            delimiter = d
            break
    if delimiter is None:
        raise ValueError("Unable to detect delimiter")
    header = lines[0].split(delimiter)
    if len(header) < len(REQUIRED_COLS):
        # Auto-repair header
        header_line = delimiter.join(REQUIRED_COLS)
        csv_text = "\n".join([header_line] + lines[1:])
    else:
        csv_text = "\n".join(lines)
    df = pd.read_csv(io.StringIO(csv_text), sep=delimiter, engine="python")
    return df

def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    # Lazy import pdfplumber
    try:
        import pdfplumber
    except Exception as e:
        raise RuntimeError("Missing dependency: pdfplumber") from e

    rows = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                # Assume first row is header; normalize length
                for r in table[1:]:
                    if len(r) >= len(REQUIRED_COLS):
                        rows.append(r[:len(REQUIRED_COLS)])
            else:
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    parts = re.split(r"[,\\s]+", line.strip())
                    if len(parts) >= len(REQUIRED_COLS):
                        rows.append(parts[:len(REQUIRED_COLS)])
    if not rows:
        raise ValueError("No usable tabular data found in PDF")
    return pd.DataFrame(rows, columns=REQUIRED_COLS)

def normalize_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    # Coerce numerics
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalize billable to [0,1]
    if df["billable"].dtype == object:
        df["billable"] = (
            df["billable"].astype(str).str.strip().str.lower()
            .map({"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0})
            .fillna(0)
        )
    else:
        df["billable"] = (df["billable"] > 0).astype(int)
    return df



