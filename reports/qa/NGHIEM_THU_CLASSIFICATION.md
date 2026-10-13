# Bien ban nghiem thu classification

## Quy trinh tai tao

`src/prepare_classification_data.py` tao cap `Train_Data.csv` va
`Test_Data.csv` tu `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` theo thu tu:

1. Chuyen `TotalCharges` sang so voi `errors="coerce"`.
2. `train_test_split(test_size=0.2, random_state=42, stratify=Churn)`.
3. Fit median imputer, StandardScaler, categorical imputer va OneHotEncoder
   tren Train duy nhat.
4. Transform Test bang cac transformer da fit tren Train.

Metadata tai `data/processed/classification_split_metadata.json` ghi:

- Source: 7.043 dong, SHA-256
  `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
- Train/Test: 5.634 / 1.409 dong.
- `random_state=42`; `stratify=Churn`.
- Imputer, scaler va encoder deu `fit` tren `Train_Data only`.

## Kiem tra tu dong

```text
python -m pytest tests/test_leakage.py -q
```

Ket qua: `2 passed`. Test kiem tra cap train/test dong bo va metadata bat buoc
ghi ro split, random state, stratify va fit-on-train.

## Ket qua GridSearchCV

| Model | Best parameters | CV F1 | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | C=0.1, solver=lbfgs | 0.6335 | 0.7424 | 0.5096 | 0.7834 | 0.6175 |
| Random Forest | n_estimators=100, max_depth=10, min_samples_split=2 | 0.6353 | 0.7530 | 0.5236 | 0.7701 | 0.6234 |

GridSearchCV dung 5 fold, `scoring="f1"`; ca hai model dung
`class_weight="balanced"`, `random_state=42`. CSV va Confusion Matrix nam tai
`data/processed/classification_results.csv` va `reports/figures/`.
