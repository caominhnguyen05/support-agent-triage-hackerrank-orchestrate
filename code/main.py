"""
main.py

  The entry point. Reads support_tickets/support_tickets.csv, runs every row
  through the TriageAgent pipeline, and writes results to support_tickets/output.csv.

HOW TO RUN:
  python main.py                 # Full run on support_tickets.csv  → output.csv
  python main.py --test          # Run on test_input.csv            → test_output.csv
  python main.py --limit 10      # Process only the first N rows (quick smoke test)
  python main.py --verbose       # Verbose mode (print each result as it's processed)
  python main.py --test --verbose --limit 5   # Combine flags freely
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from agent import TriageAgent

# Paths
ROOT        = Path(__file__).parent.parent
TICKETS_DIR = ROOT / "support_tickets"

INPUT_CSV  = TICKETS_DIR / "support_tickets.csv"
TEST_CSV   = TICKETS_DIR / "test_input.csv"      # generated from sample_support_tickets.csv
OUTPUT_CSV = TICKETS_DIR / "output.csv"
TEST_OUTPUT_CSV = TICKETS_DIR / "test_output.csv"

OUTPUT_COLS = [
    "issue", "subject", "company",
    "status", "product_area", "response", "justification", "request_type",
]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_row(agent: TriageAgent, row: pd.Series, verbose: bool = False) -> dict:
    """
    Call agent.triage() for a single DataFrame row.
    Returns a dictionary with all output fields, safe to concat into a DataFrame.
    """
    issue   = str(row.get("issue",   "") or "").strip()
    subject = str(row.get("subject", "") or "").strip()
    company = str(row.get("company", "") or "").strip()

    if not issue:
        return {
            "status":        "escalated",
            "product_area":  "Unknown",
            "response":      "No issue text provided.",
            "justification": "Empty ticket — escalated by default.",
            "request_type":  "invalid",
        }

    result = agent.triage(issue=issue, subject=subject, company=company)

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Issue   : {issue[:120]}")
        print(f"  Domain  : {company or '(inferred)'}")
        print(f"  Status  : {result.get('status')}")
        print(f"  Area    : {result.get('product_area')}")
        print(f"  Type    : {result.get('request_type')}")
        print(f"  Response: {result.get('response', '')[:200]}")

    return result


def run(df: pd.DataFrame, agent: TriageAgent, verbose: bool) -> pd.DataFrame:
    """
    Process every row in df, return a new DataFrame with output columns added.
    """
    records = []

    for idx, row in df.iterrows():
        print(f"  [row {idx + 1}/{len(df)}] processing …", end=" ", flush=True)
        t0 = time.time()

        try:
            result = process_row(agent, row, verbose=verbose)
        except Exception as exc:
            print(f"ERROR: {exc}")
            result = {
                "status":        "escalated",
                "product_area":  "Error",
                "response":      "An internal error occurred. Please contact support.",
                "justification": f"Agent error: {exc}",
                "request_type":  "product_issue",
            }

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) → {result.get('status')} / {result.get('request_type')}")
        records.append(result)

    out_rows = []
    for (_, row), result in zip(df.iterrows(), records):
        base = {
            "issue":   row.get("issue",   ""),
            "subject": row.get("subject", ""),
            "company": row.get("company", ""),
        }
        base.update(result)
        out_rows.append(base)

    return pd.DataFrame(out_rows, columns=OUTPUT_COLS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Support triage agent")
    parser.add_argument("--test",    action="store_true", help="Run on test_input.csv instead of support_tickets.csv")
    parser.add_argument("--limit",  type=int, default=None, help="Process only first N rows")
    parser.add_argument("--verbose", action="store_true",  help="Print each result as it is processed")
    args = parser.parse_args()

    if args.test:
        input_path  = TEST_CSV
        output_path = TEST_OUTPUT_CSV
        print("[main] TEST MODE — using test_input.csv → test_output.csv")
    else:
        input_path  = INPUT_CSV
        output_path = OUTPUT_CSV

    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path, encoding="utf-8")
    df.columns = df.columns.str.lower()
    print(f"[main] Loaded {len(df)} rows from {input_path.name}")

    if args.limit:
        df = df.head(args.limit)
        print(f"[main] Limited to first {args.limit} rows")

    print("[main] Initialising TriageAgent …")
    agent = TriageAgent()

    print(f"\n[main] Processing tickets …\n")
    out_df = run(df, agent, verbose=args.verbose)

    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[main] Output written to: {output_path.resolve()}")

    print("\n[main] Summary:")
    if "status" in out_df.columns:
        print(out_df["status"].value_counts().to_string())
    if "request_type" in out_df.columns:
        print(out_df["request_type"].value_counts().to_string())


if __name__ == "__main__":
    main()