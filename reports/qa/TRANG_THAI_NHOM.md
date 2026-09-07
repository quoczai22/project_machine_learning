# Trang thai nghiem thu va nhanh Git

Cap nhat tren nhanh `feature-result`.

## Nhanh hien con

- `main`: nhanh nop chinh, chua merge ket qua cuoi.
- `feature-result`: nhanh tong hop da nghiem thu code va artifact.

Da xoa sau khi xac nhan da nam trong `feature-result`:

- `feature-eda`
- `branch_bao`
- `nhanh_Tung`
- `test_4`
- `integration-person5`
- `hieu-dev`

## Phan da nghiem thu

### Nguoi 1 - EDA

- Notebook EDA, bao cao insight va 6 hinh EDA da co trong `reports/figures`.

### Nguoi 2 - Preprocessing

- Ban giao cap classification da xu ly: `Train_Data.csv` (5.634 dong) va
  `Test_Data.csv` (1.409 dong).
- Ban giao input clustering 7.043 dong:
  `data/working_raw_7043.csv`.

### Nguoi 3 - Classification

- Logistic Regression va Random Forest duoc GridSearchCV theo F1.
- Ket qua chot nam trong `data/processed/classification_results.csv`.
- Da co hai Confusion Matrix trong `reports/figures/`.

### Nguoi 4 - Clustering

- Quet K-Means va Hierarchical voi k = 2..6.
- Chon K-Means k = 4 de phan tich cuoi: Silhouette 0.4212; Hierarchical k = 4
  dat 0.3457.
- Da co Elbow, Silhouette, PCA, profile 4 cum, ket qua 7.043 dong va bien ban
  nghiem thu tai `reports/qa/NGHIEM_THU_CLUSTERING.md`.
- Unit test clustering: 2 passed.

## Viec con lai truoc khi merge main

Nguoi 5 cap nhat Word/README tong hop bang dung mot bo so lieu tren
`feature-result`:

1. Classification: Logistic Regression F1 = 0.6175; Random Forest F1 = 0.6234.
2. Clustering: ket qua k = 2..6; giai thich k = 2 co Silhouette cao nhat,
   nhung chon K-Means k = 4 vi phan khuc de dien giai hon.
3. Doi chieu Word, CSV va anh artifact de khong con so lieu cu/mau thuan.

Sau khi Word duoc doi chieu va duyet, merge `feature-result` vao `main`.
