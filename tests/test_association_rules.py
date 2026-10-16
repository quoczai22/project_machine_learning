import pandas as pd
import pytest

from src.association_rules import mine_churn_rules


@pytest.fixture
def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Contract=Month-to-month": [1, 1, 1, 1, 0, 0, 0, 0],
            "InternetService=Fiber optic": [1, 1, 1, 0, 1, 0, 0, 0],
            "PaymentMethod=Electronic check": [1, 1, 0, 1, 0, 0, 0, 0],
            "Churn=Yes": [1, 1, 1, 1, 0, 0, 0, 0],
        },
        dtype=bool,
    )


@pytest.mark.parametrize("algorithm", ["apriori", "fpgrowth"])
def test_mined_rules_only_predict_churn_yes(transactions, algorithm):
    rules, summary = mine_churn_rules(
        transactions, algorithm, min_support=0.20, min_confidence=0.50
    )

    assert not rules.empty
    assert rules["consequents"].eq("Churn=Yes").all()
    assert rules["lift"].ge(1.0).all()
    assert summary["churn_rules"] == len(rules)
    assert summary["frequent_itemsets"] > 0


def test_mining_rejects_unknown_algorithm(transactions):
    with pytest.raises(ValueError, match="algorithm"):
        mine_churn_rules(transactions, "unknown")
