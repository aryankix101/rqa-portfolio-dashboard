# -*- coding: utf-8 -*-
#%%
"""
IBKR Flex Pipeline: download -> parse -> performance -> outputs

Requirements:
  pip install requests pandas numpy

Usage:
  # PowerShell / Windows
  $env:IBKR_FLEX_TOKEN = "paste-your-flex-token"
  python ibkr_flex_pipeline.py

Notes:
 - Configure QUERY_IDS below with your actual Flex "Query ID"s.
 - The script expects your Flex queries to output CSV (not XML).
 - If your query is a "multi-section" Statement, it will parse sections like:
     CNAV (Change in NAV), POST (Positions), EQUT, etc.
 - If your query is a single-table CSV, it will return one DataFrame.

Outputs (in ./flex_outputs):
  - raw_*.csv             raw payload saved for audit
  - sections/<CODE>.csv   parsed sections (CNAV, POST, etc.)
"""

import os
import io
import csv
import gzip
import time
import json
import math
import uuid
import errno
import shutil
import typing
import requests
import pandas as pd
import numpy as np
import datetime as datetime
from xml.etree import ElementTree as ET
from rqa_packages.RQA_Imports import *

set_cwd_as_filepath(file=__file__)


#Set Paths
OUTPUT_DIR = "./flex_outputs"
SECTIONS_DIR = os.path.join(OUTPUT_DIR, "sections")

# Flex endpoints (correct method suffixes)
SEND_URL = "https://www.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
GET_URL  = "https://www.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement"

# ---------- EXCEPTIONS ----------
class FlexError(RuntimeError):
    pass

# ---------- UTIL ----------
def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y.%m.%d")

def _parse_xml(text: str) -> ET.Element:
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        raise FlexError("Non-XML response:\n" + text[:600])

def _extract_xml_status(root: ET.Element) -> typing.Tuple[str, str, str]:
    status = (root.findtext(".//Status") or root.findtext(".//status") or "").strip()
    code   = (root.findtext(".//ErrorCode") or root.findtext(".//errorCode") or "").strip()
    msg    = (root.findtext(".//ErrorMessage") or root.findtext(".//errorMessage") or "").strip()
    return (status, code, msg)

# ---------- FLEX DOWNLOAD ----------
def run_flex(query_id: str,
             token: typing.Optional[str] = None,
             version: str = "3",
             max_polls: int = 10,
             poll_wait_secs: int = 3,
             session: typing.Optional[requests.Session] = None) -> bytes:
    """
    Step 1: SendRequest (q = query_id) -> ReferenceCode
    Step 2: Poll GetStatement (q = reference_code) -> CSV/XML (bytes, maybe gzipped)
    """
    token = token or os.getenv("IBKR_FLEX_TOKEN")
    if not token:
        raise FlexError("Flex token missing. Set IBKR_FLEX_TOKEN or pass token=...")

    sess = session or requests.Session()
    headers = {"User-Agent": "RQA-FlexPipeline/1.0"}

    # Step 1
    r1 = sess.get(SEND_URL, params={"t": token, "q": query_id, "v": version}, timeout=30, headers=headers)
    if r1.status_code != 200:
        raise FlexError(f"SendRequest HTTP {r1.status_code}.\n{r1.text[:600]}")
    root = _parse_xml(r1.text)
    status, code, msg = _extract_xml_status(root)
    if status.lower() == "fail":
        raise FlexError(f"SendRequest failed: {code} - {msg or 'Unknown error'}")

    ref = root.findtext(".//ReferenceCode") or root.findtext(".//referenceCode")
    if not ref:
        raise FlexError("No ReferenceCode in SendRequest response:\n" + r1.text[:600])

    # Step 2 (poll)
    last_err = None
    for attempt in range(1, max_polls + 1):
        # First quick wait is shorter
        time.sleep(poll_wait_secs if attempt > 1 else 2)
        r2 = sess.get(GET_URL, params={"t": token, "q": ref, "v": version}, timeout=60, headers=headers)
        if r2.status_code != 200:
            last_err = f"GetStatement HTTP {r2.status_code}.\n{r2.text[:600]}"
            continue

        # XML? Could be status or actual report in XML
        if r2.text.lstrip().startswith("<"):
            try:
                root2 = _parse_xml(r2.text)
                status2, code2, msg2 = _extract_xml_status(root2)
                if status2 and status2.lower() not in {"success", "ready"}:
                    last_err = f"GetStatement status: {status2} {code2} - {msg2}"
                    continue
                # else: valid XML payload; return bytes
                return r2.content
            except FlexError:
                pass  # not valid XML => treat as data below

        # Gzip?
        if r2.headers.get("Content-Encoding", "").lower() == "gzip" or r2.content[:2] == b"\x1f\x8b":
            try:
                return gzip.decompress(r2.content)
            except Exception:
                return r2.content

        # Plain CSV/bytes
        return r2.content

    raise FlexError(last_err or "Report not ready after polling")

# ---------- FLEX CSV PARSERS ----------
def is_multisection_flex_csv(sample_text: str) -> bool:
    # Most multi-section statements start with lines like:
    # "HEADER","CNAV",...  then "DATA","CNAV",...
    return ('"HEADER",' in sample_text) and ('"DATA",' in sample_text)

def parse_multisection_flex_csv_bytes(raw: bytes) -> typing.Dict[str, pd.DataFrame]:
    """
    Parse the "HEADER"/"DATA" multi-section CSV into {code -> DataFrame}.
    """
    text = raw.decode("utf-8", errors="replace")
    sections: typing.Dict[str, typing.List[typing.List[str]]] = {}
    headers: typing.Dict[str, typing.List[str]] = {}
    counts: typing.Dict[str, int] = {}
    current_code: typing.Optional[str] = None

    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if not row:
            continue
        tag = row[0].strip().upper()
        if tag == "HEADER":
            code = row[1].strip().upper() if len(row) > 1 else "UNKNOWN"
            hdr = row[2:]
            headers[code] = hdr
            sections.setdefault(code, [])
            counts[code] = 0
            current_code = code
        elif tag == "DATA":
            code = row[1].strip().upper() if len(row) > 1 else current_code
            if not code or code not in headers:
                # DATA before HEADER? skip
                continue
            values = row[2:]
            # pad/truncate to header length for safety
            if len(values) < len(headers[code]):
                values = values + [""] * (len(headers[code]) - len(values))
            elif len(values) > len(headers[code]):
                values = values[: len(headers[code])]
            sections[code].append(values)
            counts[code] = counts.get(code, 0) + 1
        else:
            # banner or other line; ignore
            continue

    # Convert to DataFrames
    dfs: typing.Dict[str, pd.DataFrame] = {}
    for code, rows in sections.items():
        df = pd.DataFrame(rows, columns=headers[code])
        # normalize column names a bit
        df.columns = [c.strip() for c in df.columns]
        # drop fully empty columns
        df = df.dropna(axis=1, how="all")
        dfs[code] = df

    return dfs

def parse_plain_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Fallback for single-table CSV queries.
    """
    text = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))

# ---------- CALCULATIONS ----------
def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")

#%%
# ---------- MAIN ----------
def main(token, query_Id_dict, split_sections:bool=False):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SECTIONS_DIR)

    # token = os.getenv("IBKR_FLEX_TOKEN")
    if not token:
        print("WARNING: IBKR_FLEX_TOKEN not set. You can pass token=... in run_flex, or set env var.")
    print("Starting Flex pipeline...")

    all_sections: typing.Dict[str, pd.DataFrame] = {}

    if not query_Id_dict:
        print("No QUERY_IDS configured. Please set your Flex Query IDs in QUERY_IDS at top of file.")
        return

    with requests.Session() as sess:
        for label, qid in query_Id_dict.items():
            print(f"Downloading Flex query '{label}' (ID={qid}) ...")
            raw = run_flex(query_id=qid, token=token, session=sess)

            # Save raw for audit
            raw_name = f"raw_{label}_{now_ts()}.csv"
            raw_path = os.path.join(OUTPUT_DIR, raw_name)
            with open(raw_path, "wb") as f:
                f.write(raw)
            print("  Saved raw to", raw_path)

            # Parse
            text_head = raw[:2000].decode("utf-8", errors="replace")
            if split_sections:
                if is_multisection_flex_csv(text_head):
                    dfs = parse_multisection_flex_csv_bytes(raw)
                    print(f"  Parsed {len(dfs)} sections: {', '.join(sorted(dfs.keys()))}")
                    # Merge into all_sections (last write wins)
                    for code, df in dfs.items():
                        all_sections[code] = df
                        out_fp = os.path.join(SECTIONS_DIR, f"{code}_{now_ts()}.csv")
                        df.to_csv(out_fp, index=False)
                else:
                    # Single table
                    df = parse_plain_csv_bytes(raw)
                    code = label.upper()
                    all_sections[code] = df
                    out_fp = os.path.join(SECTIONS_DIR, f"{code}.csv")
                    df.to_csv(out_fp, index=False)
                    print(f"  Parsed single table -> saved {code}.csv")

    print("Done.")

#%%
if __name__ == "__main__":
    #Run DCM SMA Flex
    dcm_query_id_dict = {   
        "DCM_SMA_Attr":  "1260018",      
        "DCM_SMA_All":  "1279840",
    }
    main(token = "571416577420763856366313",query_Id_dict=dcm_query_id_dict)

    #Run RQA Strategy Flex
    rqa_query_id_dict = {
        "RQA_SMA_All":  "1285067", 
    }
    main(token = "158159079699491281111623",query_Id_dict=rqa_query_id_dict)

# %%
