"""Khai phá luật Apriori có consequent ``Churn=Yes``."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from src.association_data import build_transaction_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _format_itemset(items: frozenset[str]) -> str:
    return " + ".join(sorted(items))


def mine_churn_rules(
    transactions: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.40,
    min_lift: float = 1.0,
    max_len: int = 3,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Sinh luật Apriori chỉ có vế phải chính xác là ``Churn=Yes``."""
    if "Churn=Yes" not in transactions.columns:
        raise ValueError("Ma trận giao dịch phải có item 'Churn=Yes'")
    started = perf_counter()
    frequent = apriori(
        transactions, min_support=min_support, use_colnames=True, max_len=max_len
    )
    elapsed = perf_counter() - started
    columns = [
        "antecedents", "consequents", "support", "confidence", "lift",
        "leverage", "conviction", "antecedent_length",
    ]
    if frequent.empty:
        return pd.DataFrame(columns=columns), {
            "algorithm": "Apriori", "frequent_itemsets": 0,
            "churn_rules": 0, "runtime_seconds": elapsed,
        }
    raw_rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    raw_rules = raw_rules[
        (raw_rules["consequents"] == frozenset({"Churn=Yes"}))
        & (raw_rules["lift"] >= min_lift)
    ].copy()
    rules = pd.DataFrame(
        {
            "antecedents": raw_rules["antecedents"].map(_format_itemset),
            "consequents": raw_rules["consequents"].map(_format_itemset),
            "support": raw_rules["support"],
            "confidence": raw_rules["confidence"],
            "lift": raw_rules["lift"],
            "leverage": raw_rules["leverage"],
            "conviction": raw_rules["conviction"],
            "antecedent_length": raw_rules["antecedents"].map(len),
        }
    ).sort_values(["lift", "confidence", "support"], ascending=False, ignore_index=True)
    return rules, {
        "algorithm": "Apriori", "frequent_itemsets": len(frequent),
        "churn_rules": len(rules), "runtime_seconds": elapsed,
    }


def save_top_rules_plot(rules: pd.DataFrame, path: str | Path, top_n: int = 10) -> None:
    """Lưu biểu đồ Lift của các luật Churn tiêu biểu."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    top = rules.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(4, 0.55 * len(top))))
    plt.barh(
        [f"{antecedent} → Churn=Yes" for antecedent in top["antecedents"]],
        top["lift"], color="#2a9d8f",
    )
    plt.axvline(1.0, color="#e76f51", linestyle="--", label="Lift = 1")
    plt.xlabel("Lift")
    plt.title("Các luật Apriori liên hệ với Churn=Yes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apriori cho Telco Churn")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="reports")
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.40)
    parser.add_argument("--min-lift", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=3)
    args = parser.parse_args()

    transactions = build_transaction_table(pd.read_csv(args.data))
    rules, summary = mine_churn_rules(
        transactions, args.min_support, args.min_confidence, args.min_lift, args.max_len
    )
    metrics = Path(args.out) / "metrics" / "association_rules"
    figures = Path(args.out) / "figures" / "association_rules"
    metrics.mkdir(parents=True, exist_ok=True)
    rules.to_csv(metrics / "apriori_churn_yes_rules.csv", index=False)
    pd.DataFrame([summary]).to_csv(metrics / "algorithm_comparison.csv", index=False)
    if not rules.empty:
        save_top_rules_plot(rules, figures / "top_rules_lift_apriori.png")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
