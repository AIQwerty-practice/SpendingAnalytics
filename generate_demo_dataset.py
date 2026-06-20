"""Generate a randomized but financially plausible demo banking dataset.

This file is intentionally separate from generate_synthetic_dataset.py so the
grading/training dataset remains deterministic and unchanged.
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from generate_synthetic_dataset import BANKS, CATEGORY_TAXONOMY, CURRENCY, PROFILES, YEARS
from paths import DATA_DIR, DEMO_DATASET_PATH


MERCHANTS = {
    "Housing": ["Condo Rent", "Mortgage Payment", "Property Tax", "Student Residence"],
    "Groceries": ["Costco", "Loblaws", "Metro", "No Frills", "Sobeys", "Walmart Grocery", "Farm Boy"],
    "Dining": ["Uber Eats", "The Keg", "Freshii", "Subway", "Chipotle", "Boston Pizza", "Local Restaurant"],
    "Coffee": ["Starbucks", "Tim Hortons", "Second Cup", "Balzac's Coffee"],
    "Transportation": ["Presto", "GO Transit", "Uber", "Lyft", "Shell", "Petro Canada", "Canadian Tire Gas"],
    "Utilities": ["Hydro One", "Enbridge Gas", "Bell Internet", "Rogers Mobile"],
    "Subscriptions": ["Netflix", "Spotify", "Amazon Prime", "Disney+", "Adobe", "LinkedIn Premium"],
    "Shopping": ["Apple Store", "Best Buy", "H&M", "Uniqlo", "Canadian Tire", "Amazon"],
    "Healthcare": ["Shoppers Drug Mart", "Dental Clinic", "LifeLabs", "Pharmacy"],
    "Education": ["Campus Bookstore", "Pearson Education", "Coursera", "Tuition Payment"],
    "Childcare": ["Kids & Company", "YMCA Childcare", "School Lunch Program"],
    "Insurance": ["Manulife Insurance", "Sun Life", "Auto Insurance", "Home Insurance"],
    "Travel": ["Air Canada", "Marriott", "Expedia", "VIA Rail"],
    "Savings/Investments": ["Wealthsimple", "RBC InvestEase", "TFSA Transfer", "RESP Transfer"],
}

PROFILE_WEIGHTS = {
    "Student": {
        "Groceries": 0.18,
        "Dining": 0.15,
        "Coffee": 0.14,
        "Transportation": 0.14,
        "Subscriptions": 0.10,
        "Shopping": 0.09,
        "Education": 0.08,
        "Healthcare": 0.04,
        "Travel": 0.03,
        "Savings/Investments": 0.02,
        "Utilities": 0.02,
        "Housing": 0.01,
    },
    "Professional": {
        "Housing": 0.16,
        "Groceries": 0.15,
        "Dining": 0.13,
        "Coffee": 0.10,
        "Transportation": 0.12,
        "Utilities": 0.08,
        "Subscriptions": 0.07,
        "Shopping": 0.07,
        "Savings/Investments": 0.06,
        "Travel": 0.04,
        "Healthcare": 0.02,
    },
    "Family": {
        "Housing": 0.15,
        "Groceries": 0.18,
        "Dining": 0.08,
        "Coffee": 0.06,
        "Transportation": 0.12,
        "Utilities": 0.09,
        "Subscriptions": 0.06,
        "Shopping": 0.06,
        "Healthcare": 0.05,
        "Childcare": 0.06,
        "Insurance": 0.05,
        "Savings/Investments": 0.03,
        "Travel": 0.01,
    },
}

INCOME_RANGES = {
    "Student": (15000, 32000),
    "Professional": (55000, 120000),
    "Family": (85000, 180000),
}

SPENDING_RATIOS = {
    "Student": (0.70, 1.10),
    "Professional": (0.60, 0.95),
    "Family": (0.75, 1.05),
}


def random_date(rng: random.Random, year: int, month: int | None = None) -> date:
    if month is None:
        start = date(year, 1, 1)
        return start + timedelta(days=rng.randrange(366 if year % 4 == 0 else 365))
    start = date(year, month, 1)
    end = date(year + int(month == 12), 1 if month == 12 else month + 1, 1)
    return start + timedelta(days=rng.randrange((end - start).days))


def weighted_category(rng: random.Random, profile: str) -> str:
    weights = PROFILE_WEIGHTS[profile]
    categories = list(weights)
    return rng.choices(categories, weights=[weights[item] for item in categories], k=1)[0]


def description_for(rng: random.Random, merchant: str, category: str, bank: str) -> str:
    if category == "Income":
        return rng.choice([f"Direct deposit from {merchant}", f"{merchant} income deposit"])
    return rng.choice(
        [
            f"{merchant} purchase",
            f"{merchant} debit transaction",
            f"{bank} card transaction at {merchant}",
            f"{category} payment - {merchant}",
        ]
    )


def allocate_counts(total: int, parts: int) -> list[int]:
    base = total // parts
    counts = [base] * parts
    for idx in range(total % parts):
        counts[idx] += 1
    return counts


def income_rows(
    rng: random.Random,
    profile: str,
    banks: list[str],
    year: int,
    annual_income: float,
) -> list[dict[str, object]]:
    income_merchants = {
        "Student": ["Part-Time Payroll", "Scholarship Payment", "Campus Job Payroll"],
        "Professional": ["Payroll Deposit", "Client Transfer", "Bonus Payment"],
        "Family": ["Payroll Deposit", "Canada Child Benefit", "Second Income Deposit"],
    }
    rows = []
    monthly_counts = rng.choices([1, 2], weights=[0.75, 0.25], k=12)
    total_deposits = sum(monthly_counts)
    remaining = annual_income
    for month, count in enumerate(monthly_counts, start=1):
        for deposit_idx in range(count):
            deposits_left = total_deposits - len(rows)
            amount = round(remaining / deposits_left, 2) if deposits_left <= 1 else round(rng.uniform(0.75, 1.25) * annual_income / total_deposits, 2)
            amount = min(amount, remaining)
            remaining = round(remaining - amount, 2)
            merchant = rng.choice(income_merchants[profile])
            bank = rng.choice(banks)
            rows.append(
                {
                    "date": random_date(rng, year, month).isoformat(),
                    "merchant": merchant,
                    "description": description_for(rng, merchant, "Income", bank),
                    "amount": amount,
                    "category": "Income",
                    "profile": profile,
                    "bank": bank,
                    "currency": CURRENCY,
                }
            )
    return rows


def expense_rows(
    rng: random.Random,
    profile: str,
    banks: list[str],
    year: int,
    transaction_count: int,
    target_spending: float,
) -> list[dict[str, object]]:
    rows = []
    recurring_categories = ["Subscriptions", "Utilities"]
    if profile != "Student":
        recurring_categories.append("Housing")
    if profile == "Family":
        recurring_categories.extend(["Childcare", "Insurance"])

    recurring_slots = min(max(0, transaction_count // 5), transaction_count)
    flexible_slots = transaction_count - recurring_slots

    for _ in range(recurring_slots):
        category = rng.choice(recurring_categories)
        merchant = rng.choice(MERCHANTS[category])
        bank = rng.choice(banks)
        rows.append(
            {
                "date": random_date(rng, year, rng.randint(1, 12)).isoformat(),
                "merchant": merchant,
                "description": description_for(rng, merchant, category, bank),
                "amount": -rng.uniform(10, 300),
                "category": category,
                "profile": profile,
                "bank": bank,
                "currency": CURRENCY,
            }
        )

    for _ in range(flexible_slots):
        category = weighted_category(rng, profile)
        merchant = rng.choice(MERCHANTS[category])
        bank = rng.choice(banks)
        rows.append(
            {
                "date": random_date(rng, year).isoformat(),
                "merchant": merchant,
                "description": description_for(rng, merchant, category, bank),
                "amount": -rng.uniform(5, 350),
                "category": category,
                "profile": profile,
                "bank": bank,
                "currency": CURRENCY,
            }
        )

    current_spending = sum(abs(row["amount"]) for row in rows) or 1
    scale = target_spending / current_spending
    for row in rows:
        row["amount"] = -round(abs(row["amount"]) * scale, 2)
    return rows


def generate_demo_transactions(
    profiles: list[str] | None = None,
    banks: list[str] | None = None,
    years: list[int] | None = None,
    transactions_per_profile: int = 500,
    seed: int | None = None,
) -> pd.DataFrame:
    selected_profiles = profiles or PROFILES
    selected_banks = banks or BANKS
    selected_years = years or YEARS
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    for profile in selected_profiles:
        low_target = max(1, int(transactions_per_profile * 0.95))
        high_target = max(low_target, int(transactions_per_profile * 1.05))
        profile_target = rng.randint(low_target, high_target)
        year_counts = allocate_counts(profile_target, len(selected_years))

        for year, year_target in zip(selected_years, year_counts):
            annual_income = rng.uniform(*INCOME_RANGES[profile])
            ratio = rng.uniform(*SPENDING_RATIOS[profile])
            target_spending = annual_income * ratio

            income = income_rows(rng, profile, selected_banks, year, annual_income)
            expense_count = max(1, year_target - len(income))
            expenses = expense_rows(rng, profile, selected_banks, year, expense_count, target_spending)
            rows.extend(income + expenses)

    df = pd.DataFrame(rows)
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_TAXONOMY, ordered=True)
    return df.sort_values(["profile", "bank", "date", "merchant"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a randomized demo banking dataset.")
    parser.add_argument("--output", type=Path, default=DEMO_DATASET_PATH)
    parser.add_argument("--transactions-per-profile", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    df = generate_demo_transactions(transactions_per_profile=args.transactions_per_profile, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df):,} demo transactions")
    print(f"Saved dataset to {args.output}")
    print(df.groupby(["profile", "bank"], observed=False).size().to_string())


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    main()
