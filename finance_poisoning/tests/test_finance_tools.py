"""Unit tests for read-only finance tools."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from finance_poisoning.env.finance_tools import READ_ONLY_TOOL_NAMES, FinanceTools


@pytest.fixture
def tools() -> FinanceTools:
    return FinanceTools.from_data()


def test_lookup_transactions_filters_merchant(tools: FinanceTools) -> None:
    txns = tools.lookup_transactions(merchant="Oakridge")
    assert len(txns) >= 3
    assert all("Oakridge" in t["merchant"] for t in txns)


def test_lookup_transactions_filters_category(tools: FinanceTools) -> None:
    txns = tools.lookup_transactions(category="groceries")
    assert len(txns) >= 5
    assert all(t["category"] == "groceries" for t in txns)


def test_get_recurring_payments_includes_rent_and_spotify(tools: FinanceTools) -> None:
    recurring = tools.get_recurring_payments()
    merchants = {r["merchant"] for r in recurring}
    assert "Oakridge Apartments" in merchants
    assert "Spotify" in merchants
    rent = next(r for r in recurring if r["merchant"] == "Oakridge Apartments")
    assert rent["amount"] == 1850.0


def test_resolve_fact_rent_amount(tools: FinanceTools) -> None:
    assert tools.resolve_fact("rent_amount") == 1850


def test_no_write_capable_tools() -> None:
    forbidden = {"transfer_money", "pay_bill", "update_account", "delete_transaction"}
    public = {
        name for name, _ in inspect.getmembers(FinanceTools, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert not public & forbidden
    for name in public:
        lower = name.lower()
        assert "transfer_money" not in lower
        assert "pay_bill" not in lower
        assert "update_account" not in lower
    assert public <= READ_ONLY_TOOL_NAMES | {"from_data"}
