# Bien ban nghiem thu clustering

## Pham vi kiem tra

- Nhanh: `test_4`
- Input: `data/working_raw_7043.csv`
- So dong input: 7.043
- Feature dua vao clustering: `tenure`, `MonthlyCharges`, `service_count`
- Khong dua `customerID` va `Churn` vao model.
- StandardScaler duoc fit trong `src/clustering.py`, sau khi doc input chua scale.
- Pham vi danh gia: k = 2..6; phan tich cuoi: k = 4.

## Bang chung input

Input co cac gia tri chua scale: tenure trung binh 32.3711 (do lech chuan
24.5595), MonthlyCharges trung binh 64.7617 (do lech chuan 30.0900),
service_count trung binh 3.3629 (do lech chuan 2.0620). Vi vay day khong phai
la CSV da StandardScaler.

SHA-256 cua `data/working_raw_7043.csv`:
`32C88172E92DC67C3CA86C45DBED559791B8A000A0537ACC8575EEDB4ED4005A`.

## Kiem tra tu dong

Lenh da chay:

```text
python -m pytest tests/test_clustering.py -q
```

Ket qua: `2 passed`. Log nguyen van nam tai `reports/qa/pytest_clustering.txt`.

Hai test kiem tra:

1. K-Means va Hierarchical deu tra ve ket qua cho cac k yeu cau, Silhouette
   nam trong khoang hop le [-1, 1].
2. Ket qua k = 4 co dung 4 nhan cum cho ca hai thuat toan va bang so sanh co
   dung hai dong.

## Ket qua chay tren du lieu 7.043 dong

| Thuật toán | k | Silhouette |
|---|---:|---:|
| K-Means | 2 | 0.4255 |
| K-Means | 4 | 0.4212 |
| Hierarchical | 4 | 0.3457 |

K-Means k = 4 cao hon Hierarchical k = 4. Du k = 2 co Silhouette cao nhat,
chenh lech voi k = 4 la nho; k = 4 duoc chon de tao bon phan khuc khach hang
co y nghia de dien giai.

SHA-256 cua `reports/clustering_metrics.csv`:
`047B8A6F8F39224A44536429D6C1A33DDEF5543010100037F20985EA5C11D356`.

SHA-256 cua `reports/algorithm_comparison_k4.csv`:
`C4D72CA95747649ABE92837B2889AFE6F1724DA1C8D2AFD8C49830F20981405C`.

## Artifact doi chieu

- `reports/clustering_metrics.csv`: bang inertia va Silhouette k = 2..6.
- `reports/elbow_kmeans.png`: do thi Elbow.
- `reports/silhouette_compare.png`: so sanh Silhouette hai thuat toan.
- `reports/pca_kmeans_k4.png`: hinh PCA cua ket qua cuoi.
- `reports/cluster_summary_k4.csv`: profile bon cum.
- `reports/clustering_result_7043.csv`: nhan cum cua toan bo 7.043 khach hang.

## Ket luan nghiem thu

Dat. Code, input, ket qua va artifact deu co the chay lai bang lenh trong
README. Ket qua Word phai su dung dung cac file tren va giai thich ro ly do
chon k = 4 thay vi khang dinh k = 4 la gia tri Silhouette cao nhat.
