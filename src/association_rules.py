"""Khai phá luật kết hợp liên quan đến ``Churn=Yes``.

Chạy hai thuật toán Apriori và FP-Growth trên cùng ma trận giao dịch để so
sánh kết quả frequent itemsets/số luật và thời gian chạy. Đây là phân tích
mô tả; Support, Confidence và Lift là các độ đo chính, không dùng Accuracy/F1.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from src.association_data import build_transaction_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RULE_COLUMNS = [
    "algorithm",
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "antecedent_length",
]


def _format_itemset(items: frozenset[str]) -> str:
    return " + ".join(sorted(items))


def mine_churn_rules(
    transactions: pd.DataFrame,
    algorithm: str,
    min_support: float = 0.05,
    min_confidence: float = 0.40,
    min_lift: float = 1.0,
    max_len: int = 3,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Tạo luật mà consequent chính xác là ``Churn=Yes``.

    Args:
        transactions: Ma trận Boolean, mỗi dòng là một khách hàng.
        algorithm: ``apriori`` hoặc ``fpgrowth``.
        min_support: Tỷ lệ xuất hiện tối thiểu của itemset.
        min_confidence: Độ tin cậy tối thiểu khi sinh luật.
        min_lift: Chỉ giữ luật có Lift từ mức này trở lên.
        max_len: Số item tối đa trong frequent itemset.
    """
    if algorithm not in {"apriori", "fpgrowth"}:
        raise ValueError("algorithm phải là 'apriori' hoặc 'fpgrowth'")
    if "Churn=Yes" not in transactions.columns:
        raise ValueError("Ma trận giao dịch phải có item 'Churn=Yes'")

    miner = apriori if algorithm == "apriori" else fpgrowth
    started = perf_counter()
    frequent = miner(
        transactions,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    elapsed = perf_counter() - started
    if frequent.empty:
        return pd.DataFrame(columns=RULE_COLUMNS), {
            "algorithm": algorithm,
            "frequent_itemsets": 0,
            "churn_rules": 0,
            "runtime_seconds": elapsed,
        }

    raw_rules = association_rules(
        frequent, metric="confidence", min_threshold=min_confidence
    )
    churn_target = frozenset({"Churn=Yes"})
    raw_rules = raw_rules[
        (raw_rules["consequents"] == churn_target) & (raw_rules["lift"] >= min_lift)
    ].copy()
    if raw_rules.empty:
        rules = pd.DataFrame(columns=RULE_COLUMNS)
    else:
        rules = pd.DataFrame(
            {
                "algorithm": algorithm,
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

    summary = {
        "algorithm": algorithm,
        "frequent_itemsets": len(frequent),
        "churn_rules": len(rules),
        "runtime_seconds": elapsed,
    }
    return rules, summary


def save_top_rules_plot(rules: pd.DataFrame, path: str | Path, top_n: int = 10) -> None:
    """Lưu biểu đồ Lift của tối đa ``top_n`` luật Churn nổi bật."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = rules.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(4, 0.55 * len(selected))))
    labels = [f"{item} → Churn=Yes" for item in selected["antecedents"]]
    plt.barh(labels, selected["lift"], color="#2a9d8f")
    plt.axvline(1.0, color="#e76f51", linestyle="--", label="Lift = 1")
    plt.xlabel("Lift")
    plt.title("Các luật liên hệ với Churn=Yes có Lift cao nhất")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Luật kết hợp Telco Churn")
    parser.add_argument("--data", required=True, help="CSV Telco gốc/đã làm sạch")
    parser.add_argument("--out", default="reports", help="Thư mục reports")
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.40)
    parser.add_argument("--min-lift", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=3)
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    transactions = build_transaction_table(raw)
    metrics_dir = Path(args.out) / "metrics" / "association_rules"
    figures_dir = Path(args.out) / "figures" / "association_rules"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_rules: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for algorithm in ("apriori", "fpgrowth"):
        rules, summary = mine_churn_rules(
            transactions,
            algorithm=algorithm,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift,
            max_len=args.max_len,
        )
        all_rules.append(rules)
        summaries.append(summary)
        rules.to_csv(metrics_dir / f"{algorithm}_churn_yes_rules.csv", index=False)
        if not rules.empty:
            save_top_rules_plot(rules, figures_dir / f"top_rules_lift_{algorithm}.png")

    pd.concat(all_rules, ignore_index=True).to_csv(
        metrics_dir / "association_rules_churn_yes.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        metrics_dir / "algorithm_comparison.csv", index=False
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
