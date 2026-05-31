"""Read-only finance tools backed by the synthetic transaction ledger."""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .schemas import DATA_DIR, TRUE_FACT_VALUES, Transaction, load_transactions


READ_ONLY_TOOL_NAMES = frozenset({
    "lookup_transactions",
    "get_transaction",
    "get_recurring_payments",
    "get_budget_summary",
    "get_account_summary",
    "resolve_fact",
})

_FORBIDDEN_WRITE_PREFIXES = (
    "transfer_",
    "pay_",
    "update_",
    "delete_",
    "create_",
    "write_",
    "send_",
    "modify_",
)


def _assert_read_only_tools() -> None:
    for name in READ_ONLY_TOOL_NAMES:
        lower = name.lower()
        for prefix in _FORBIDDEN_WRITE_PREFIXES:
            if lower.startswith(prefix) or lower == prefix.rstrip("_"):
                raise RuntimeError(f"Write-capable tool name detected: {name}")


class FinanceTools:
    """Deterministic read-only access to synthetic ledger data."""

    def __init__(self, transactions: list[Transaction]):
        self._transactions = list(transactions)

    @classmethod
    def from_data(cls, path: Path | str = DATA_DIR / "transactions.json") -> FinanceTools:
        return cls(load_transactions(path))

    def lookup_transactions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        merchant: Optional[str] = None,
        category: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> list[dict]:
        results: list[Transaction] = []
        for txn in self._transactions:
            if start_date and txn.date < start_date:
                continue
            if end_date and txn.date > end_date:
                continue
            if merchant and merchant.lower() not in txn.merchant.lower():
                continue
            if category and txn.category.lower() != category.lower():
                continue
            if account_id and txn.account_id != account_id:
                continue
            results.append(txn)
        return [asdict(t) for t in results]

    def get_transaction(self, txn_id: str) -> Optional[dict]:
        for txn in self._transactions:
            if txn.txn_id == txn_id:
                return asdict(txn)
        return None

    def get_recurring_payments(self, category: Optional[str] = None) -> list[dict]:
        """Detect recurring payments by merchant frequency (rent, subscriptions)."""
        by_merchant: dict[str, list[Transaction]] = defaultdict(list)
        for txn in self._transactions:
            if txn.amount >= 0:
                continue
            if category and txn.category.lower() != category.lower():
                continue
            by_merchant[txn.merchant].append(txn)

        recurring: list[dict] = []
        for merchant, txns in by_merchant.items():
            if len(txns) < 2:
                continue
            amounts = [abs(t.amount) for t in txns]
            avg_amount = sum(amounts) / len(amounts)
            if max(amounts) - min(amounts) < 0.01:
                recurring.append({
                    "merchant": merchant,
                    "category": txns[0].category,
                    "amount": round(avg_amount, 2),
                    "account_id": txns[0].account_id,
                    "occurrences": len(txns),
                })
        return sorted(recurring, key=lambda r: r["merchant"])

    def get_budget_summary(self, month: str) -> dict:
        """Aggregate spending by category for YYYY-MM."""
        prefix = month[:7] if len(month) >= 7 else month
        by_category: dict[str, float] = defaultdict(float)
        total_income = 0.0
        total_spending = 0.0
        for txn in self._transactions:
            if not txn.date.startswith(prefix):
                continue
            if txn.amount >= 0:
                total_income += txn.amount
                by_category["income"] += txn.amount
            else:
                total_spending += abs(txn.amount)
                by_category[txn.category] += abs(txn.amount)
        return {
            "month": prefix,
            "total_income": round(total_income, 2),
            "total_spending": round(total_spending, 2),
            "by_category": {k: round(v, 2) for k, v in sorted(by_category.items())},
        }

    def get_account_summary(self, account_id: str) -> dict:
        balance = 0.0
        txn_count = 0
        last_date: Optional[str] = None
        for txn in self._transactions:
            if txn.account_id != account_id:
                continue
            balance += txn.amount
            txn_count += 1
            if last_date is None or txn.date > last_date:
                last_date = txn.date
        return {
            "account_id": account_id,
            "balance": round(balance, 2),
            "transaction_count": txn_count,
            "last_transaction_date": last_date,
        }

    def resolve_fact(self, fact_id: str) -> Any:
        """Ground-truth resolver from ledger + canonical fact table."""
        if fact_id == "rent_amount":
            rent = self.get_recurring_payments(category="rent")
            if rent:
                return rent[0]["amount"]
        elif fact_id == "rent_merchant":
            rent = self.get_recurring_payments(category="rent")
            if rent:
                return rent[0]["merchant"]
        elif fact_id == "paycheck_amount":
            payroll = [
                t for t in self._transactions
                if t.category == "income" and "payroll" in t.merchant.lower()
            ]
            if payroll:
                return abs(payroll[0].amount)
        elif fact_id == "subscription_amount_spotify":
            spotify = [
                t for t in self._transactions if "spotify" in t.merchant.lower()
            ]
            if spotify:
                return abs(spotify[0].amount)
        elif fact_id == "budget_goal":
            return TRUE_FACT_VALUES["budget_goal"]
        elif fact_id == "rent_account":
            rent = self.get_recurring_payments(category="rent")
            if rent:
                return rent[0]["account_id"]
        return TRUE_FACT_VALUES.get(fact_id)


_assert_read_only_tools()

# Verify no write methods exist on the class beyond read-only set.
_public_methods = {
    name for name, _ in inspect.getmembers(FinanceTools, predicate=inspect.isfunction)
    if not name.startswith("_")
}
_extra = _public_methods - READ_ONLY_TOOL_NAMES - {"from_data"}
if _extra:
    raise RuntimeError(f"Unexpected public methods on FinanceTools: {_extra}")
