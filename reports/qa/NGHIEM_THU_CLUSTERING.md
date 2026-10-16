# Bien ban nghiem thu clustering

## Pham vi kiem tra

- Nhanh hien tai: `main`.
- Input: `data/working_raw_7043.csv`
- So dong input: 7.043
- Feature dua vao clustering: `tenure`, `MonthlyCharges`, `service_count`
- Khong dua `customerID` va `Churn` vao model.
- StandardScaler duoc fit trong `src/clustering.py`, sau khi doc input chua scale.
- Pham vi danh gia: k = 2..10; phan tich cuoi: k = 4.

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

Ket qua: `4 passed`. Log nguyen van nam tai `reports/qa/pytest_clustering.txt`.

Bon test kiem tra:

1. K-Means va Hierarchical deu tra ve ket qua cho cac k yeu cau, Silhouette
   nam trong khoang hop le [-1, 1].
2. Ket qua k = 4 co dung 4 nhan cum cho ca hai thuat toan va bang so sanh co
   dung hai dong.
3. `customerID` va `Churn` khong nam trong feature dua vao clustering.
4. Cau hinh mac dinh quet day du k = 2..10, khong bi lui ve k = 2..6.

## Ket qua chay tren du lieu 7.043 dong

| k | K-Means Silhouette | Hierarchical Silhouette |
|---:|---:|---:|
| 2 | 0.4255 | 0.4037 |
| 3 | 0.4055 | 0.3323 |
| 4 | 0.4212 | 0.3457 |
| 5 | 0.3835 | 0.3478 |
| 6 | 0.3950 | 0.3284 |
| 7 | 0.3798 | 0.3206 |
| 8 | 0.3678 | 0.3078 |
| 9 | 0.3483 | 0.3122 |
| 10 | 0.3539 | 0.2984 |

K-Means k = 4 cao hon Hierarchical k = 4. Tai k = 2, K-Means chi chia thanh
hai nhom tong quat: nhom tenure 46.54 thang, phi 89.44 va 5.28 dich vu; va
nhom tenure 20.89 thang, phi 44.77 va 1.81 dich vu. Du k = 2 co Silhouette cao
nhat, chenh lech voi k = 4 la nho; k = 4 tach them cac nhom lau nam/chi cao,
lau nam/chi thap, moi/chi cao va moi/chi thap de co insight kinh doanh cu the.

SHA-256 cua `reports/clustering_metrics.csv`:
`8103D10EFD8BE32AC40DD1484DB8DDE1E066EFCD8B3DA670E6FB733835195363`.

SHA-256 cua `reports/algorithm_comparison_k4.csv`:
`C4D72CA95747649ABE92837B2889AFE6F1724DA1C8D2AFD8C49830F20981405C`.

## Artifact doi chieu

- `reports/clustering_metrics.csv`: bang inertia va Silhouette k = 2..10.
- `reports/elbow_kmeans.png`: do thi Elbow.
- `reports/silhouette_compare.png`: so sanh Silhouette hai thuat toan.
- `reports/pca_kmeans_k4.png`: hinh PCA cua ket qua cuoi.
- `reports/cluster_summary_k4.csv`: profile bon cum.
- `reports/clustering_result_7043.csv`: nhan cum cua toan bo 7.043 khach hang.

## Ket luan nghiem thu

Dat. Code, input, ket qua va artifact deu co the chay lai bang lenh trong
README. Ket qua Word phai su dung dung cac file tren va giai thich ro ly do
chon k = 4 thay vi khang dinh k = 4 la gia tri Silhouette cao nhat.
