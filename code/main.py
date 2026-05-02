"""
main.py

  The entry point. Reads support_tickets/support_tickets.csv, runs every row
  through the TriageAgent pipeline, and writes results to support_tickets/output.csv.

  Also supports a --sample flag to run against
  support_tickets/sample_support_tickets.csv and compare predictions to
  expected outputs.

HOW TO RUN:
  python main.py # Full run (produces output.csv)

  python main.py --limit 10 # Process only the first N rows (quick smoke test)

  python main.py --verbose # Verbose mode (print each result as it's processed)
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Ensure local package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from agent import TriageAgent

# Paths
ROOT          = Path(__file__).parent.parent
TICKETS_DIR   = ROOT / "support_tickets"
INPUT_CSV     = TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV    = TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV    = TICKETS_DIR / "output_3.csv"

OUTPUT_COLS = [
    "issue", "subject", "company",
    "status", "product_area", "response", "justification", "request_type",
]

EXPECTED_COLS = ["status", "product_area", "request_type"] 


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_row(agent: TriageAgent, row: pd.Series, verbose: bool = False) -> dict:
    """
    Call agent.triage() for a single DataFrame row.
    Returns a dictionary with all output fields, safe to concat into a DataFrame.
    """
    issue = str(row.get("issue", "") or "").strip()
    subject = str(row.get("subject", "") or "").strip()
    company = str(row.get("company", "") or "").strip()

    if not issue:
        return {
            "status": "escalated",
            "product_area": "Unknown",
            "response": "No issue text provided.",
            "justification": "Empty ticket — escalated by default.",
            "request_type": "invalid",
        }

    result = agent.triage(issue=issue, subject=subject, company=company)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Issue   : {issue[:120]}")
        print(f"  Domain  : {company}")
        print(f"  Status  : {result.get('status')}")
        print(f"  Area    : {result.get('product_area')}")
        print(f"  Type    : {result.get('request_type')}")
        print(f"  Response: {result.get('response', '')[:200]}")

    return result


def run(df: pd.DataFrame, agent: TriageAgent, verbose: bool, resume_set: set) -> pd.DataFrame:
    """
    Process every row in df, return a new DataFrame with output columns added.
    resume_set: set of row indices already processed (skipped when --resume).
    """
    records = []

    for idx, row in df.iterrows():
        if idx in resume_set:
            print(f"  [skip row {idx}] already processed")
            records.append(None)
            continue

        print(f"  [row {idx+1}/{len(df)}] processing …", end=" ", flush=True)
        t0 = time.time()

        try:
            result = process_row(agent, row, verbose=verbose)
        except Exception as exc:
            print(f"ERROR: {exc}")
            result = {
                "status": "escalated",
                "product_area":  "Error",
                "response": "An internal error occurred. Please contact support.",
                "justification": f"Agent error: {exc}",
                "request_type": "product_issue",
            }

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) → {result.get('status')} / {result.get('request_type')}")
        records.append(result)

    # Build output DataFrame
    out_rows = []
    for (idx, row), result in zip(df.iterrows(), records):
        base = {
            "issue":   row.get("issue", ""),
            "subject": row.get("subject", ""),
            "company": row.get("company", ""),
        }
        if result:
            base.update(result)
        out_rows.append(base)

    return pd.DataFrame(out_rows, columns=OUTPUT_COLS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Support triage agent")
    parser.add_argument("--limit",   type=int, default=None, help="Process only first N rows")
    parser.add_argument("--verbose", action="store_true", help="Print each result")
    args = parser.parse_args()

    input_path = INPUT_CSV
    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path, encoding="utf-8")
    df.columns = df.columns.str.lower()
    print(f"[main] Loaded {len(df)} rows from {input_path.name}")

    if args.limit:
        df = df.head(args.limit)
        print(f"[main] Limited to first {args.limit} rows")

    resume_set: set = set()

    # Init agent
    print("[main] Initialising TriageAgent …")
    agent = TriageAgent()

    # Run
    print(f"\n[main] Processing tickets …\n")
    out_df = run(df, agent, verbose=args.verbose, resume_set=resume_set)

    # Save output
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[main] Output written to: {OUTPUT_CSV.resolve()}")

    # Summary counts
    print("\n[main] Summary:")
    if "status" in out_df.columns:
        print(out_df["status"].value_counts().to_string())
    if "request_type" in out_df.columns:
        print(out_df["request_type"].value_counts().to_string())


if __name__ == "__main__":
    main()