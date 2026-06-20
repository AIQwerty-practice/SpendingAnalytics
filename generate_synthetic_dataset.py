"""Generate one consolidated synthetic banking transaction dataset.

The output is intentionally deterministic so the course project can be
retrained and demonstrated consistently.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from paths import DATA_DIR, DATASET_PATH


RANDOM_SEED = 42
PROFILES = ["Student", "Professional", "Family"]
BANKS = ["RBC", "TD", "Scotiabank"]
YEARS = [2024, 2025, 2026]
CURRENCY = "CAD"
CATEGORY_TAXONOMY = [
    "Income",
    "Housing",
    "Groceries",
    "Dining",
    "Coffee",
    "Transportation",
    "Utilities",
    "Subscriptions",
    "Shopping",
    "Healthcare",
    "Education",
    "Childcare",
    "Insurance",
    "Travel",
    "Savings/Investments",
]


@dataclass(frozen=True)
class MerchantRule:
    merchants: list[str]
    category: str
    min_amount: float
    max_amount: float
    monthly_probability: float
    recurring_day: int | None = None


PROFILE_RULES: dict[str, list[MerchantRule]] = {
    "Student": [
        MerchantRule(["Payroll Deposit", "Scholarship Payment"], "Income", 650, 1900, 0.95, 1),
        MerchantRule(["Campus Bookstore", "Pearson Education", "Coursera"], "Education", 25, 240, 0.35),
        MerchantRule(["Metro", "No Frills", "Walmart Grocery"], "Groceries", 18, 120, 0.95),
        MerchantRule(["Starbucks", "Tim Hortons", "Second Cup"], "Coffee", 4, 18, 0.9),
        MerchantRule(["Uber Eats", "Subway", "Chipotle", "Pizza Pizza"], "Dining", 10, 55, 0.8),
        MerchantRule(["Presto", "Uber", "Lyft"], "Transportation", 3, 85, 0.75),
        MerchantRule(["Netflix", "Spotify", "Amazon Prime"], "Subscriptions", 8, 22, 0.9, 15),
        MerchantRule(["Apple Store", "Best Buy", "H&M", "Uniqlo"], "Shopping", 20, 220, 0.45),
    ],
    "Professional": [
        MerchantRule(["Payroll Deposit", "Client Transfer"], "Income", 3200, 7800, 1.0, 1),
        MerchantRule(["Sobeys", "Loblaws", "Costco"], "Groceries", 45, 240, 0.95),
        MerchantRule(["Starbucks", "Tim Hortons", "Balzac's Coffee"], "Coffee", 4, 20, 0.85),
        MerchantRule(["Uber Eats", "The Keg", "Freshii", "Local Restaurant"], "Dining", 18, 160, 0.85),
        MerchantRule(["Condo Rent", "Mortgage Payment"], "Housing", 1500, 2900, 1.0, 2),
        MerchantRule(["Hydro One", "Enbridge Gas", "Bell Internet"], "Utilities", 55, 260, 0.9, 10),
        MerchantRule(["GO Transit", "Uber", "Shell", "Petro Canada"], "Transportation", 12, 170, 0.85),
        MerchantRule(["Netflix", "Spotify", "Adobe", "LinkedIn Premium"], "Subscriptions", 10, 75, 0.9, 15),
        MerchantRule(["Wealthsimple", "RBC InvestEase", "TFSA Transfer"], "Savings/Investments", 150, 1200, 0.7, 20),
        MerchantRule(["Air Canada", "Marriott", "Expedia"], "Travel", 120, 1200, 0.22),
    ],
    "Family": [
        MerchantRule(["Payroll Deposit", "Canada Child Benefit"], "Income", 5200, 9800, 1.0, 1),
        MerchantRule(["Costco", "Loblaws", "Walmart Grocery", "Farm Boy"], "Groceries", 80, 420, 1.0),
        MerchantRule(["Starbucks", "Tim Hortons"], "Coffee", 5, 28, 0.7),
        MerchantRule(["Swiss Chalet", "Boston Pizza", "Uber Eats"], "Dining", 35, 210, 0.75),
        MerchantRule(["Mortgage Payment", "Property Tax"], "Housing", 2100, 4300, 1.0, 2),
        MerchantRule(["Hydro One", "Enbridge Gas", "Bell Internet", "Rogers Mobile"], "Utilities", 80, 420, 1.0, 10),
        MerchantRule(["Shell", "Petro Canada", "Canadian Tire Gas"], "Transportation", 35, 260, 0.95),
        MerchantRule(["Kids & Company", "YMCA Childcare", "School Lunch Program"], "Childcare", 120, 1100, 0.75, 5),
        MerchantRule(["Manulife Insurance", "Sun Life", "Auto Insurance"], "Insurance", 75, 380, 0.95, 18),
        MerchantRule(["Shoppers Drug Mart", "Dental Clinic", "LifeLabs"], "Healthcare", 20, 420, 0.45),
        MerchantRule(["Netflix", "Disney+", "Spotify Family"], "Subscriptions", 10, 35, 0.95, 15),
        MerchantRule(["RESP Transfer", "TFSA Transfer", "Wealthsimple"], "Savings/Investments", 100, 1600, 0.7, 20),
    ],
}


def random_day(year: int, month: int) -> date:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    return start + timedelta(days=random.randrange((end - start).days))


def make_description(merchant: str, category: str, bank: str) -> str:
    templates = [
        "{merchant} purchase",
        "{merchant} debit transaction",
        "{bank} card transaction at {merchant}",
        "{category} payment - {merchant}",
    ]
    if category == "Income":
        templates = ["Direct deposit from {merchant}", "{merchant} income deposit"]
    return random.choice(templates).format(merchant=merchant, category=category, bank=bank)


def signed_amount(category: str, low: float, high: float) -> float:
    amount = round(random.uniform(low, high), 2)
    return amount if category == "Income" else -amount


def generate_transactions(rows_per_profile_bank_year: int = 260) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    configured_categories = {rule.category for rules in PROFILE_RULES.values() for rule in rules}
    unknown_categories = configured_categories - set(CATEGORY_TAXONOMY)
    if unknown_categories:
        raise ValueError(f"Unknown categories in generator rules: {sorted(unknown_categories)}")

    for profile in PROFILES:
        for bank in BANKS:
            for year in YEARS:
                rules = PROFILE_RULES[profile]
                for month in range(1, 13):
                    for rule in rules:
                        if random.random() <= rule.monthly_probability:
                            count = 1 if rule.recurring_day else random.randint(1, 4)
                            for _ in range(count):
                                merchant = random.choice(rule.merchants)
                                transaction_date = (
                                    date(year, month, min(rule.recurring_day, 28))
                                    if rule.recurring_day
                                    else random_day(year, month)
                                )
                                rows.append(
                                    {
                                        "date": transaction_date.isoformat(),
                                        "merchant": merchant,
                                        "description": make_description(merchant, rule.category, bank),
                                        "amount": signed_amount(rule.category, rule.min_amount, rule.max_amount),
                                        "category": rule.category,
                                        "profile": profile,
                                        "bank": bank,
                                        "currency": CURRENCY,
                                    }
                                )

                # Add a small number of irregular realistic transactions.
                while len([r for r in rows if r["profile"] == profile and r["bank"] == bank and str(r["date"]).startswith(str(year))]) < rows_per_profile_bank_year:
                    rule = random.choice(rules)
                    merchant = random.choice(rule.merchants)
                    transaction_date = random_day(year, random.randint(1, 12))
                    rows.append(
                        {
                            "date": transaction_date.isoformat(),
                            "merchant": merchant,
                            "description": make_description(merchant, rule.category, bank),
                            "amount": signed_amount(rule.category, rule.min_amount, rule.max_amount),
                            "category": rule.category,
                            "profile": profile,
                            "bank": bank,
                            "currency": CURRENCY,
                        }
                    )

    df = pd.DataFrame(rows)
    df = df.sort_values(["profile", "bank", "date", "merchant"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consolidated synthetic banking data.")
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument("--rows-per-profile-bank-year", type=int, default=260)
    args = parser.parse_args()

    df = generate_transactions(args.rows_per_profile_bank_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Generated {len(df):,} transactions")
    print(f"Saved dataset to {args.output}")
    print(df.groupby(["profile", "bank"]).size().to_string())


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    main()
