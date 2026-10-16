import pandas as pd

from src.association_rules import mine_churn_rules


def test_apriori_rules_only_have_churn_yes_consequent():
    transactions = pd.DataFrame(
        {
            "Contract=Month-to-month": [1, 1, 1, 1, 0, 0, 0, 0],
            "InternetService=Fiber optic": [1, 1, 1, 0, 1, 0, 0, 0],
            "Churn=Yes": [1, 1, 1, 1, 0, 0, 0, 0],
        }, dtype=bool,
    )
    rules, summary = mine_churn_rules(transactions, min_support=0.20, min_confidence=0.50)

    assert not rules.empty
    assert rules["consequents"].eq("Churn=Yes").all()
    assert rules["lift"].ge(1.0).all()
    assert summary["churn_rules"] == len(rules)
