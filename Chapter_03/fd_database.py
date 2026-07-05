"""
fd_database.py
Builds and maintains fd_master_database.csv -- a mock FD records table that
stands in for the NBFC's real SQL database. This is what the agent's
validate_fd_reference tool will actually query against in a later chapter,
instead of just regex-checking the format.

The 20 seed rows are NOT invented from scratch: FD_No, Customer_Name, and
Mobile_Number are extracted from real rows in fd_dataset_messy.csv (emails
that happened to sign off with a "Name | Phone" signature). Every other
column (account number, amount, dates, rate, status, branch) doesn't exist
in the email text, so it's synthesized with a fixed random seed -- realistic
values, but not real financial data.

Usage:
    python fd_database.py
        -> (re)builds fd_master_database.csv from the dataset

    from fd_database import get_fd_record, insert_fd_record, update_fd_record
        -> CRUD operations once the CSV exists, for later chapters
"""

import re
import random
from datetime import datetime, timedelta

import pandas as pd

SOURCE_DATASET = "fd_dataset_messy.csv"
DB_PATH = "fd_master_database.csv"

TODAY = datetime(2026, 6, 27)  # fixed "current date" so Status is reproducible
BRANCHES = ["Pune", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad", "Kolkata", "Ahmedabad"]
TENURE_OPTIONS_MONTHS = [12, 24, 36, 48, 60]
BASE_RATE_BY_TENURE = {12: 6.50, 24: 6.75, 36: 7.00, 48: 7.00, 60: 7.25}

FD_REF_PATTERN = re.compile(r"BJ(\d{4})FD\d+")
SIGNATURE_PATTERN = re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)\s*\|\s*([6-9]\d{9})")


def extract_seed_records(path: str, n: int) -> list[dict]:
    """Pull real (FD_No, Customer_Name, Mobile_Number) triples from emails
    that contain both an FD reference number and a clean 'Name | Phone'
    signature. Deduplicated by FD reference, full two-word names only."""
    df = pd.read_csv(path)
    seen_fd = set()
    seeds = []

    for _, row in df.iterrows():
        content = str(row["content"])
        fd_match = FD_REF_PATTERN.search(content)
        sig_match = SIGNATURE_PATTERN.search(content)
        if not (fd_match and sig_match):
            continue

        fd_no = fd_match.group(0)
        name = sig_match.group(1).strip()
        if fd_no in seen_fd or len(name.split()) < 2:
            continue

        seen_fd.add(fd_no)
        seeds.append({
            "FD_No": fd_no,
            "Customer_Name": name.title(),
            "Mobile_Number": sig_match.group(2),
            "booking_year": int(fd_match.group(1)),
        })
        if len(seeds) >= n:
            break

    return seeds


def synthesize_record(seed: dict, rng: random.Random, force_closed: bool = False) -> dict:
    """Fill in every column that doesn't exist in the raw email text."""
    booking_date = datetime(seed["booking_year"], rng.randint(1, 12), rng.randint(1, 28))
    tenure_months = rng.choice(TENURE_OPTIONS_MONTHS)
    maturity_date = booking_date + timedelta(days=tenure_months * 30)  # approximate, fine for a mock DB

    base_rate = BASE_RATE_BY_TENURE[tenure_months]
    interest_rate = round(base_rate + rng.uniform(0, 0.5), 2)

    if force_closed:
        status = "Closed (Premature)"
        last_updated = booking_date + timedelta(days=rng.randint(30, tenure_months * 30 - 30))
    elif maturity_date <= TODAY:
        status = "Matured"
        last_updated = maturity_date
    else:
        status = "Active"
        # most Active records were last touched at booking; a few got a
        # recent "update" -- simulating someone calling in to change details
        if rng.random() < 0.25:
            last_updated = TODAY - timedelta(days=rng.randint(1, 90))
        else:
            last_updated = booking_date

    return {
        "FD_No": seed["FD_No"],
        "Customer_Name": seed["Customer_Name"],
        "Mobile_Number": seed["Mobile_Number"],
        "Account_Number": "".join(str(rng.randint(0, 9)) for _ in range(14)),
        "FD_Amount_INR": rng.randint(25, 500) * 1000,
        "Interest_Rate_Percent": interest_rate,
        "Tenure_Months": tenure_months,
        "Booking_Date": booking_date.strftime("%Y-%m-%d"),
        "Maturity_Date": maturity_date.strftime("%Y-%m-%d"),
        "Status": status,
        "Branch": rng.choice(BRANCHES),
        "Last_Updated": last_updated.strftime("%Y-%m-%d"),
    }


def build_database(n_rows: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    seeds = extract_seed_records(SOURCE_DATASET, n_rows)

    # Force exactly 2 rows to "Closed (Premature)" for realistic variety,
    # tying back to the premature-withdrawal language in the prompts.
    closed_indices = set(rng.sample(range(len(seeds)), k=min(2, len(seeds))))

    records = [
        synthesize_record(s, rng, force_closed=(i in closed_indices))
        for i, s in enumerate(seeds)
    ]
    return pd.DataFrame(records)


def save_database(df: pd.DataFrame, path: str = DB_PATH) -> None:
    # Account_Number and Mobile_Number must stay strings -- a CSV reader
    # that infers types will silently turn a leading digit string into a
    # number and can mangle it. dtype is enforced again on read in
    # get_fd_record() below for the same reason.
    df.to_csv(path, index=False)


# ----------------------------------------------------------------------
# CRUD helpers -- the "SQL DB" operations this CSV needs to support:
# look up a record, book a new FD, update an existing one.
# ----------------------------------------------------------------------

def get_fd_record(fd_no: str, path: str = DB_PATH) -> dict | None:
    """The real version of what validate_fd_reference's regex stand-in
    will be replaced with: an actual lookup against the records table."""
    df = pd.read_csv(path, dtype={"Mobile_Number": str, "Account_Number": str})
    match = df[df["FD_No"] == fd_no]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def insert_fd_record(record: dict, path: str = DB_PATH) -> None:
    """Simulates a new FD being booked. `record` must have all the same
    keys as a row produced by synthesize_record()."""
    df = pd.read_csv(path, dtype={"Mobile_Number": str, "Account_Number": str})
    if (df["FD_No"] == record["FD_No"]).any():
        raise ValueError(f"FD_No {record['FD_No']} already exists -- use update_fd_record() instead.")
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(path, index=False)


def update_fd_record(fd_no: str, path: str = DB_PATH, **fields) -> None:
    """Simulates a customer detail update (e.g. mobile number changed).
    Always bumps Last_Updated to today, the same way a real DB trigger would."""
    df = pd.read_csv(path, dtype={"Mobile_Number": str, "Account_Number": str})
    mask = df["FD_No"] == fd_no
    if not mask.any():
        raise ValueError(f"FD_No {fd_no} not found.")
    for key, value in fields.items():
        df.loc[mask, key] = value
    df.loc[mask, "Last_Updated"] = TODAY.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = build_database(n_rows=20, seed=42)
    save_database(df)
    print(f"Wrote {len(df)} rows to {DB_PATH}")
    print(df.to_string(index=False))
