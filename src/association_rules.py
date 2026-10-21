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
import numpy as np
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


def save_algorithm_comparison_plot(summaries_df: pd.DataFrame, path: str | Path) -> None:
    """Lưu biểu đồ so sánh kết quả khai phá và thời gian chạy giữa Apriori và FP-Growth."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle(
        "So sánh Apriori và FP-Growth trên cùng dữ liệu giao dịch",
        fontsize=13,
        fontweight="bold",
        color="#00204a",
        y=0.98,
    )

    algos = ["Apriori", "FP-Growth"]
    freq_items = [
        int(summaries_df.loc[summaries_df["algorithm"] == "apriori", "frequent_itemsets"].values[0]),
        int(summaries_df.loc[summaries_df["algorithm"] == "fpgrowth", "frequent_itemsets"].values[0]),
    ]
    churn_rules = [
        int(summaries_df.loc[summaries_df["algorithm"] == "apriori", "churn_rules"].values[0]),
        int(summaries_df.loc[summaries_df["algorithm"] == "fpgrowth", "churn_rules"].values[0]),
    ]
    x = np.arange(len(algos))
    width = 0.55

    c_freq_apriori = "#1B828C"
    c_freq_fpgrowth = "#E86A24"
    c_rule_apriori = "#85BFC4"
    c_rule_fpgrowth = "#F7AC80"

    ax1.bar(x[0], freq_items[0], width, color=c_freq_apriori, label="Frequent itemsets")
    ax1.bar(x[1], freq_items[1], width, color=c_freq_fpgrowth)
    ax1.bar(x[0], churn_rules[0], width, color=c_rule_apriori, label="Luật → Churn=Yes")
    ax1.bar(x[1], churn_rules[1], width, color=c_rule_fpgrowth)

    ax1.set_title("Kết quả khai phá", fontsize=11, fontweight="bold", color="#00204a", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, fontsize=10)
    ax1.set_ylabel("Số lượng", fontsize=10)
    ax1.set_ylim(0, max(freq_items) * 1.08)

    for i, (f, r) in enumerate(zip(freq_items, churn_rules)):
        ax1.text(x[i], f + 100, f"{f:,}".replace(",", "."), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax1.text(x[i], r + 80, str(r), ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9.5)

    runtime_apriori = float(summaries_df.loc[summaries_df["algorithm"] == "apriori", "runtime_seconds"].values[0])
    runtime_fpgrowth = float(summaries_df.loc[summaries_df["algorithm"] == "fpgrowth", "runtime_seconds"].values[0])

    ax2.bar(x[0], runtime_apriori, width, color=c_freq_apriori)
    ax2.bar(x[1], runtime_fpgrowth, width, color=c_freq_fpgrowth)

    ax2.set_title("Thời gian trong lượt thực nghiệm", fontsize=11, fontweight="bold", color="#00204a", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, fontsize=10)
    ax2.set_ylabel("Thời gian chạy (giây)", fontsize=10)
    ax2.set_ylim(0, max(runtime_fpgrowth * 1.15, 1.0))

    ax2.text(x[0], runtime_apriori + 0.08, f"{runtime_apriori:.2f} s".replace(".", ","), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax2.text(x[1], runtime_fpgrowth + 0.08, f"{runtime_fpgrowth:.2f} s".replace(".", ","), ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.text(0.5, -0.02, "Hai thuật toán cho cùng số frequent itemsets và luật ở cấu hình hiện tại. Thời gian phụ thuộc dữ liệu, ngưỡng và cách cài đặt.", ha="center", fontsize=8.5, color="#444444")

    plt.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rules_quality_scatter(rules: pd.DataFrame, path: str | Path) -> None:
    """Lưu biểu đồ phân tán chất lượng các luật Churn theo Support (%), Confidence (%) và Lift."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    support_pct = rules["support"] * 100
    conf_pct = rules["confidence"] * 100
    lift = rules["lift"]
    ant_len = rules["antecedent_length"]
    sizes = ant_len.map({1: 50, 2: 75, 3: 100}).fillna(70)

    scatter = ax.scatter(
        support_pct, conf_pct, c=lift, cmap="viridis",
        s=sizes, alpha=0.85, edgecolors="none"
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Lift", fontsize=10)

    ax.set_title("Chất lượng các luật có consequent Churn=Yes", fontsize=13, fontweight="bold", color="#00204a", pad=15)
    ax.set_xlabel("Support (%)", fontsize=10)
    ax.set_ylabel("Confidence (%)", fontsize=10)
    ax.set_xlim(4, max(support_pct) * 1.08)
    ax.set_ylim(38, max(conf_pct) * 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(0.45, -0.04, "Mỗi điểm là một luật. Màu sáng hơn biểu thị Lift cao hơn; kích thước phản ánh số item ở vế trái.", ha="center", fontsize=8.5, color="#444444")

    plt.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    figures_dir.mkdir(parents=True, exist_ok=True)

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
    df_summaries = pd.DataFrame(summaries)
    df_summaries.to_csv(
        metrics_dir / "algorithm_comparison.csv", index=False
    )

    # Lưu biểu đồ so sánh Apriori vs FP-Growth và biểu đồ phân tán chất lượng luật
    save_algorithm_comparison_plot(df_summaries, figures_dir / "algorithm_comparison.png")
    if not all_rules[0].empty:
        save_rules_quality_scatter(all_rules[0], figures_dir / "rules_quality_scatter.png")

    print(df_summaries.to_string(index=False))


if __name__ == "__main__":
    main()
