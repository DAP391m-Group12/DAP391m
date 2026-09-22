# Dataset B — Kế hoạch feature selection trước khi huấn luyện

## 1. Mục tiêu và phạm vi

Tài liệu này là nguồn chuẩn cho bằng chứng và quyết định feature governance trước pipeline huấn luyện chính. Contract dữ liệu—cohort/target, 99 cột, candidate universe, metadata và exclusions—nằm tại `docs/Dataset_B_Data_Selection.md`; không lặp lại toàn bộ contract ở đây.

Feature selection không được thực hiện bằng cách preprocessing toàn bộ candidate pool, huấn luyện sáu mô hình rồi quay lại loại feature. Quy trình được chốt theo thứ tự:

1. lập luận nghiệp vụ và thời điểm có dữ liệu;
2. kiểm tra chất lượng dữ liệu tối thiểu;
3. kiểm tra redundancy/association tại các block có khả năng trùng thông tin;
4. thực hiện sensitivity analysis hẹp nếu bằng chứng trước đó chưa đủ;
5. khóa feature set và business-player mapping;
6. sau đó mới fit preprocessing trên train và huấn luyện các mô hình chính.

Mục tiêu của quy trình là tạo được câu trả lời có bằng chứng cho câu hỏi **“tại sao chọn feature này và loại feature kia?”**, đồng thời giữ số business players đủ nhỏ để GSV/Shapley–Lorenz khả thi và dễ diễn giải.

Đây là tài liệu thiết kế và decision protocol, chưa phải kết quả thực nghiệm hay tuyên bố feature importance.

## 2. Contract preconditions

Mọi quyết định ở tài liệu này kế thừa các điều kiện không được thay đổi từ Data Selection: cohort chỉ dùng `action_taken ∈ {1, 2, 3}`, mapping `1/2 → y_approved=1` và `3 → y_approved=0`, raw `action_taken` giữ để truy vết nhưng nằm ngoài `X`, và Gate 1–2 không được mở lại. Test set không dùng để quyết định giữ/bỏ feature.

## 3. Phạm vi quyết định trong tài liệu này

Ba mươi lăm predictor candidates gồm 15 feature ưu tiên, 18 feature cần kiểm tra thêm và 2 representation thay thế. Metadata và target không phải predictor; danh sách vai trò đầy đủ và phép đối soát `35 + 9 + 1 + 54 = 99` chỉ được duy trì tại Data Selection. Mục 4–6 ghi bằng chứng cho 35 candidates; mục 7 quy định cách dùng metadata làm diagnostic ngoài `X`.

## 4. Feature ưu tiên — 15 cột

Nhóm này có prior nghiệp vụ rõ ràng và được kiểm tra trước. “Ưu tiên” không có nghĩa tự động giữ tất cả. Một feature vẫn có thể bị loại nếu không có trước quyết định, chất lượng dữ liệu không đạt, hoặc trùng information content với representation được chọn.

Các kết quả dưới đây được lấy từ saved outputs của `notebooks/eda_datasetB/05_feature_selction.ipynb` trên cohort Dataset B gồm **293.301 hồ sơ**. Cả 15 feature đều có `timing_status = available_before_decision`. Mục 4 mới thực hiện data-quality, distribution và redundancy screening; chưa preprocessing, target association, model training, protected-group sensitivity hoặc mở test set.

| Trạng thái sau mục 4                | Số feature | Feature                                                                                                                                                                                                                                                    |
| -------------------------------------- | ----------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `keep`                               |          12 | `loan_type`, `loan_purpose`, `lien_status`, `reverse_mortgage`, `open-end_line_of_credit`, `business_or_commercial_purpose`, `income`, `debt_to_income_ratio`, `loan_amount`, `loan_to_value_ratio`, `loan_term`, `occupancy_type` |
| `exclude_from_main`                  |           1 | `property_value`                                                                                                                                                                                                                                         |
| `keep_for_representation_comparison` |           2 | `construction_method`, `total_units`                                                                                                                                                                                                                   |

### 4.1. Cấu trúc sản phẩm và mục đích vay

- `loan_type`
- `loan_purpose`
- `lien_status`

Ba cột mô tả loại khoản vay, mục đích sử dụng và vị trí quyền lưu giữ. Cả ba đều có valid-code coverage 100%, không có blank, missing hoặc mã ngoài schema.

| Feature          | Raw categories | Phân phối quan sát được                                                                                                                                                                       | Quyết định |
| ---------------- | -------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `loan_type`    |              4 | Conventional: 265.629 (90,57%); FHA: 21.121 (7,20%); VA: 6.208 (2,12%); RHS/FSA guaranteed: 343 (0,12%)                                                                                             | `keep`      |
| `loan_purpose` |              6 | Home purchase: 139.144 (47,44%); Home improvement: 49.762 (16,97%); Other purpose: 40.360 (13,76%); Cash-out refinancing: 34.595 (11,80%); Refinancing: 29.412 (10,03%); Not applicable: 28 (0,01%) | `keep`      |
| `lien_status`  |              2 | First lien: 204.431 (69,70%); Subordinate lien: 88.870 (30,30%)                                                                                                                                     | `keep`      |

Các category RHS/FSA và Not applicable rất hiếm nhưng vẫn là mã hợp lệ, vì vậy chưa bị gộp hoặc loại ở feature selection. Ba feature được giữ riêng vì biểu diễn ba khái niệm nghiệp vụ khác nhau. `loan_type` và `lien_status` vẫn phải được đối chiếu với `derived_loan_product_type` ở comparison representation sau; hiện tại cả ba được ánh xạ thành ba GSV/SL players riêng.

### 4.2. Đặc điểm giao dịch

- `reverse_mortgage`
- `open-end_line_of_credit`
- `business_or_commercial_purpose`

Ba cột không được dùng để lọc cohort. Cả ba có valid-code coverage 100%, không có blank, missing hoặc mã ngoài schema. Mã `1111 = Exempt` hợp lệ về schema nhưng không được tính là informative code `1` hoặc `2`.

| Feature                            |      Informative |        Exempt | Phân phối informative                                                                               | Quyết định |
| ---------------------------------- | ---------------: | ------------: | ----------------------------------------------------------------------------------------------------- | ------------- |
| `reverse_mortgage`               | 284.665 (97,06%) | 8.636 (2,94%) | Reverse mortgage: 786; Not reverse mortgage: 283.879; minority = 0,28% informative                    | `keep`      |
| `open-end_line_of_credit`        | 284.979 (97,16%) | 8.322 (2,84%) | Open-end: 81.148; Not open-end: 203.831; minority = 28,48% informative                                | `keep`      |
| `business_or_commercial_purpose` | 284.680 (97,06%) | 8.621 (2,94%) | Business/commercial: 25.039; Not primarily business/commercial: 259.641; minority = 8,80% informative | `keep`      |

Mỗi feature có ba raw categories gồm `1`, `2` và `1111`. `reverse_mortgage` gần constant nhưng vẫn có 786 hồ sơ thuộc category reverse mortgage và mô tả một loại giao dịch riêng, nên rarity không đủ để loại feature.

Có 284.656 hồ sơ (97,05% cohort) informative đồng thời ở cả ba cột; 8.645 hồ sơ còn lại không được dùng trong bảng co-occurrence. Sáu pattern quan sát được theo thứ tự `reverse_mortgage` – `open-end_line_of_credit` – `business_or_commercial_purpose` là:

| Pattern     | Số hồ sơ | Tỷ lệ trong population informative |
| ----------- | ----------: | -----------------------------------: |
| `2–2–2` |     179.681 |                               63,12% |
| `2–1–2` |      79.152 |                               27,81% |
| `2–2–1` |      23.863 |                                8,38% |
| `2–1–1` |       1.174 |                                0,41% |
| `1–1–2` |         642 |                                0,23% |
| `1–2–2` |         144 |                                0,05% |

Các pattern cho thấy không cột nào được xác định hoàn toàn bởi hai cột còn lại. Vì vậy không cần mở rộng sang Cramér’s V hoặc sensitivity analysis ở giai đoạn này. Cả ba raw feature được giữ trong model, nhưng được gom thành một GSV/SL business player `transaction_characteristics`. `Exempt` được bảo toàn như một category riêng, không chuyển thành mã `2`.

### 4.3. Năng lực tài chính

- `income`
- `debt_to_income_ratio`

Hai feature đều có valid-domain coverage 100%, không có blank, missing hoặc token ngoài schema.

`income` có 3.007 raw values; 272.723 giá trị numeric (92,98%) và 20.578 `NA` (7,02%). Trong phần numeric có 360 giá trị âm, 3.111 giá trị bằng 0 và 269.252 giá trị dương. Phân phối theo raw unit: minimum −11.229; p01 = 0; Q1 = 75; median = 120; Q3 = 190; p99 = 1.242; maximum = 100.000.000. Đuôi phải rất dài và các giá trị âm/0 phải được xử lý có kiểm soát ở preprocessing fit trên train, không bị tự động xóa, clip hoặc log-transform trong mục 4.

`debt_to_income_ratio` có 21 raw categories: 19 informative bands với 262.193 hồ sơ (89,39%) và 31.108 special values (10,61%), gồm `NA` 22.466 (7,66%) và `Exempt` 8.642 (2,95%). Phân phối của 19 informative bands:

| DTI band     | Số hồ sơ | Tỷ lệ cohort | DTI band     | Số hồ sơ | Tỷ lệ cohort |
| ------------ | ----------: | -------------: | ------------ | ----------: | -------------: |
| `<20%`     |      15.077 |          5,14% | `20%-<30%` |      36.366 |         12,40% |
| `30%-<36%` |      36.082 |         12,30% | `36`       |       6.910 |          2,36% |
| `37`       |       7.359 |          2,51% | `38`       |       7.751 |          2,64% |
| `39`       |       8.295 |          2,83% | `40`       |       8.412 |          2,87% |
| `41`       |       8.943 |          3,05% | `42`       |       9.999 |          3,41% |
| `43`       |       8.761 |          2,99% | `44`       |       9.959 |          3,40% |
| `45`       |       7.729 |          2,64% | `46`       |       7.763 |          2,65% |
| `47`       |       7.992 |          2,72% | `48`       |       8.553 |          2,92% |
| `49`       |      12.372 |          4,22% | `50%-60%`  |      26.226 |          8,94% |
| `>60%`     |      27.644 |          9,43% |              |             |                |

Có 259.767 hồ sơ (88,57% cohort) informative đồng thời ở cả `income` và DTI. Median income nhìn chung giảm từ 196 ở band `<20%` xuống 75 ở band `>60%`, nhưng income vẫn có IQR dương trong cả 19 DTI bands; IQR ở hai đầu lần lượt là 345 và 72. Hai feature có association nhưng không phải representation thay thế nhau: income mô tả quy mô thu nhập, còn DTI mô tả gánh nặng nghĩa vụ nợ tương đối.

Quyết định: `keep` cả `income` và `debt_to_income_ratio` như hai raw features và hai GSV/SL players riêng. Không cần sensitivity analysis bổ sung trong nhóm 4.3.

### 4.4. Quy mô khoản vay và tài sản bảo đảm

- `loan_amount`
- `property_value`
- `loan_to_value_ratio`

Đây là block redundancy quan trọng nhất vì LTV có quan hệ gần toán học với `loan_amount` và `property_value`. Cả ba có valid-domain coverage 100%, không có blank hoặc invalid token.

| Feature                 | Raw unique |  Numeric coverage | Special values                                   | Minimum |      Q1 |  Median |      Q3 |       P99 |       Maximum |
| ----------------------- | ---------: | ----------------: | ------------------------------------------------ | ------: | ------: | ------: | ------: | --------: | ------------: |
| `loan_amount`         |        993 | 293.301 (100,00%) | 0                                                |   5.000 | 105.000 | 205.000 | 435.000 | 2.435.000 |   600.855.000 |
| `property_value`      |      1.223 |  279.550 (95,31%) | `NA`: 5.109 (1,74%); `Exempt`: 8.642 (2,95%) |   5.000 | 255.000 | 495.000 | 775.000 | 4.445.000 | 1.400.005.000 |
| `loan_to_value_ratio` |     80.236 |  274.681 (93,65%) | `NA`: 9.978 (3,40%); `Exempt`: 8.642 (2,95%) |   0,001 |  55,285 |   74,99 |  84,444 |   105,021 |     2.000.000 |

Tất cả giá trị numeric của ba feature đều dương. LTV có 5.261 giá trị lớn hơn 100 (1,92% numeric), 246 lớn hơn 200 (0,0896%), 60 lớn hơn 1.000 (0,0218%); p99,9 = 193,4654. Đây là range diagnostics, không phải quy tắc tự động đánh dấu invalid.

Common-population coverage:

- `loan_amount` + `property_value`: 279.550 hồ sơ (95,31%);
- `loan_amount` + LTV: 274.681 hồ sơ (93,65%);
- `property_value` + LTV: 273.800 hồ sơ (93,35%);
- cả ba numeric: 273.800 hồ sơ (93,35%).

Spearman correlation trên 273.800 hồ sơ common population:

- `loan_amount` – `property_value`: 0,6080;
- `loan_amount` – LTV: 0,2377;
- `property_value` – LTV: −0,2526.

So sánh reported LTV với `loan_amount / property_value × 100` trên cùng 273.800 hồ sơ cho thấy 45,25% nằm trong sai khác 1 điểm và 66,63% nằm trong sai khác 5 điểm. Median absolute difference = 1,285385; p95 absolute difference = 64,81707; maximum absolute difference = 1.999.700. Do đó reported LTV không phải bản sao số học chính xác của hai cột còn lại.

Quyết định:

- `loan_amount`: `keep`, làm đại diện cho quy mô tài trợ tuyệt đối và một GSV/SL player riêng;
- `loan_to_value_ratio`: `keep`, làm đại diện cho mức đòn bẩy tương đối và một GSV/SL player riêng;
- `property_value`: `exclude_from_main`, nhưng vẫn giữ trong source/audit.

`property_value` bị loại để không giữ đồng thời cả ba representation có thông tin chồng lấn. Đây là quyết định parsimony, interpretability và giảm chi phí GSV/SL; correlation 0,6080 không chứng minh feature hoàn toàn vô ích và mục 4 chưa chạy model ablation để kiểm tra incremental predictive value. LTV có numeric coverage thấp hơn `property_value` 1,66 điểm phần trăm, tương đương 4.869 hồ sơ; chênh lệch này không đồng nghĩa các hồ sơ đó sẽ bị xóa vì special values được xử lý ở preprocessing sau.

### 4.5. Điều khoản khoản vay

- `loan_term`

`loan_term` có valid-domain coverage 100%, 338 raw values và không có blank hoặc invalid token. Có 274.648 giá trị numeric (93,64%) và 18.653 special values (6,36%), gồm `NA` 10.360 (3,53%) và `Exempt` 8.293 (2,83%). Toàn bộ numeric values là số nguyên dương.

Phân phối theo raw unit: minimum = 1; p01 = 12; Q1 = 300; median = 360; Q3 = 360; p99 = 360; maximum = 900. Các giá trị phổ biến nhất là 360: 189.844 (69,12% numeric); 240: 27.838 (10,14%); 300: 20.394 (7,43%); 180: 14.051 (5,12%); 120: 9.589 (3,49%).

Range diagnostics ghi nhận 78 giá trị nhỏ hơn 12 (0,0284%), 2.702 lớn hơn 360 (0,9838%), 9 lớn hơn 480 (0,0033%), 1 lớn hơn 600 (0,0004%) và 1.953 không chia hết cho 12 (0,7111%). Notebook không tự suy luận raw unit, không chuyển thành tháng/năm và không coi các trường hợp này là invalid.

Quyết định: `keep` `loan_term` như một raw feature và GSV/SL player riêng vì nó biểu diễn thời hạn hợp đồng, khác với quy mô khoản vay, LTV và năng lực tài chính.

### 4.6. Đặc điểm bất động sản

- `occupancy_type`
- `construction_method`
- `total_units`

Cả ba có valid-code coverage 100%, không có blank, missing hoặc mã ngoài schema.

| Feature                 | Raw categories | Phân phối quan sát được                                                                                                                                                             | Trạng thái                           |
| ----------------------- | -------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `occupancy_type`      |              3 | Principal residence: 257.433 (87,77%); Investment property: 28.768 (9,81%); Second residence: 7.100 (2,42%)                                                                               | `keep`                               |
| `construction_method` |              2 | Site-built: 286.486 (97,68%); Manufactured home: 6.815 (2,32%)                                                                                                                            | `keep_for_representation_comparison` |
| `total_units`         |              9 | 1 unit: 259.784 (88,57%); 2: 23.905 (8,15%); 3: 4.534 (1,55%); 5–24: 2.091 (0,71%); 4: 1.971 (0,67%); 25–49: 405 (0,14%); 50–99: 333 (0,11%); >149: 170 (0,06%); 100–149: 108 (0,04%) | `keep_for_representation_comparison` |

`total_units` được giữ như categorical values theo schema HMDA, không parse các khoảng thành số liên tục. Bảng chéo `construction_method` × `total_units` có đủ 18/18 combinations và tổng số dòng khớp cohort, nhưng support rất không đồng đều:

| Construction method |       1 |      2 |     3 |     4 | 5–24 | 25–49 | 50–99 | 100–149 | >149 |
| ------------------- | ------: | -----: | ----: | ----: | ----: | -----: | -----: | -------: | ---: |
| Site-built          | 253.039 | 23.881 | 4.528 | 1.968 | 2.073 |    398 |    328 |      103 |  168 |
| Manufactured home   |   6.745 |     24 |     6 |     3 |    18 |      7 |      5 |        5 |    2 |

Cell lớn nhất có 253.039 hồ sơ và cell nhỏ nhất chỉ có 2 hồ sơ. Vì vậy sự hiện diện của đủ 18 combinations không được diễn giải là mọi combination đều có support ổn định; chưa có cơ sở để tự động tạo interaction hoặc mở rộng association analysis.

Quyết định: `occupancy_type` được giữ như raw feature và GSV/SL player riêng. `construction_method` và `total_units` đều vượt qua data-quality gate nhưng chưa được khóa thành final raw features hoặc GSV/SL players. Hai cột phải được so sánh với `derived_dwelling_category` ở mục 6/9, và không đưa đồng thời raw representation với derived representation vào main model.

### Tổng hợp chất lượng và phân phối mục 4

- Về schema, cả 15 feature đều đạt valid-domain coverage 100% và không có blank hoặc token/mã ngoài schema. Tuy nhiên, valid coverage không đồng nghĩa informative coverage: các trạng thái hợp lệ như `NA`, `Exempt` và `1111` vẫn chiếm từ khoảng 2,8% đến 10,6% ở một số feature.
- Các categorical features nhìn chung mất cân bằng mạnh: Conventional chiếm 90,57% `loan_type`; Not reverse mortgage chiếm 96,79% cohort; Site-built chiếm 97,68% `construction_method`; Principal residence chiếm 87,77% `occupancy_type`; 1-unit property chiếm 88,57% `total_units`. Category hiếm vẫn được bảo toàn nếu hợp lệ và có ý nghĩa nghiệp vụ; việc gộp chỉ được quyết định trong preprocessing trên train.
- Các numeric features có đuôi phân phối đáng chú ý. `income`, `loan_amount` và `property_value` có maximum rất xa p99; `income` còn có 360 giá trị âm và 3.111 giá trị bằng 0. LTV có 60 giá trị lớn hơn 1.000. `loan_term` tập trung 69,12% tại raw value 360 nhưng vẫn có các giá trị ngoài khoảng phổ biến. Đây là preprocessing diagnostics, không phải bằng chứng tự động xóa dòng hoặc loại feature.
- Về redundancy, `income` và DTI có association nhưng vẫn giữ thông tin riêng; ba transaction flags không xác định lẫn nhau; block `loan_amount` – `property_value` – LTV là nơi duy nhất trong mục 4 đã loại một feature khỏi main set. `construction_method` và `total_units` vẫn là quyết định mở về representation.
- Các quyết định đã ổn định trong mục 4 hiện tạo 10 GSV/SL business players: ba players của nhóm 4.1; `transaction_characteristics`; `income`; `debt_to_income_ratio`; `loan_amount`; `loan_to_value_ratio`; `loan_term`; và `occupancy_type`. Player cho representation nhà ở chỉ được xác định sau khi so sánh với `derived_dwelling_category`.
- Các kết luận trên là feature-governance evidence, không phải feature importance, causal evidence hoặc bằng chứng model performance. Mục 4 chưa dùng target để chọn feature và chưa thực hiện fairness/sensitivity theo protected groups.

## 5. Feature cần kiểm tra thêm — 18 cột

Nhóm này không được đưa vào main feature set chỉ vì có correlation hoặc univariate association với target. Mặc định chúng nằm ngoài main feature set cho đến khi thỏa đồng thời:

- có ý nghĩa nghiệp vụ phù hợp với mục tiêu dự đoán approval;
- có tại thời điểm quyết định;
- có coverage và population áp dụng đủ rõ;
- không chủ yếu là proxy cho `lei`, quy trình báo cáo hoặc missingness hậu quyết định;
- cung cấp information content khác với các feature ưu tiên;
- lợi ích giải thích đủ lớn để chấp nhận chi phí tăng business-player count.

### Trạng thái checkpoint hiện tại

Notebook `notebooks/eda_datasetB/05_feature_selction.ipynb` đã được chạy liên tục từ cell 1 đến 24 trên cohort Dataset B gồm **293.301 hồ sơ** (`action_taken ∈ {1, 2, 3}`), không có stored runtime error. Phần này chỉ thực hiện feature-governance screening: không fit model, không mở test set, không tính GSV/SL chính thức và không đưa `lei` hoặc protected metadata vào `X`.

Mục 5 được **tạm thời đóng ở mức ghi nhận screening** để chuyển sang công việc preprocessing đã được phép cho các representation đã khóa từ mục 4. Đây không phải feature lock cuối cùng: các field có trạng thái `advance_to_grouped_validation` hoặc `hold_for_timing_lineage_confirmation` vẫn không được thêm vào main feature set cho đến khi điều kiện tương ứng được giải quyết.

### 5.1. Kênh nộp hồ sơ và quan hệ tổ chức

- `preapproval`
- `submission_of_application`
- `initially_payable_to_institution`
- `conforming_loan_limit`

Đã hoàn thành schema, timing, applicability, redundancy và LEI diagnostics. `NA` của `conforming_loan_limit` được xác nhận là valid special state, nên bốn field đều đạt valid-code coverage 100% sau khi sửa mapping. `preapproval` đi tiếp như conditional feature trong home-purchase context. `submission_of_application` và `initially_payable_to_institution` đi tiếp như một joint candidate block; association full cohort của chúng không được dùng thay cho informative-only evidence.

`conforming_loan_limit` bị `exclude_from_further_review`: public-derived flag sử dụng reported loan amount có basis khác nhau theo action outcome, vì vậy không được coi là consistently pre-decision safe. Field này vẫn được giữ trong source/audit, nhưng `decision = exclude` và `gsv_player = not_applicable`.

### 5.2. Cấu trúc trả nợ đặc biệt

- `negative_amortization`
- `interest_only_payment`
- `balloon_payment`
- `other_nonamortizing_features`

Đã hoàn thành schema, timing, coverage, overlap, LEI guardrail và protected-metadata coverage diagnostics. Cả bốn field có valid-code coverage 100%; `1111` được giữ là reporting exemption, không diễn giải là “No”. Association full cohort khoảng 0,71 chủ yếu do 8.644 dòng `1111` chung; trên informative-only population, association cao nhất là `interest_only_payment` × `balloon_payment` với Cramér’s V = 0,1701.

`negative_amortization` (401 code-1 rows, tập trung mạnh tại ít LEI) và `other_nonamortizing_features` (411 code-1 rows) bị `exclude_from_further_review` do rarity và organization concentration. `interest_only_payment` và `balloon_payment` đi tiếp như một joint candidate block; final keep/exclude và GSV player của chúng vẫn để trống đến grouped train/validation đã được phê duyệt.

### 5.3. Manufactured home và multifamily

- `manufactured_home_secured_property_type`
- `manufactured_home_land_property_interest`
- `multifamily_affordable_units`

Đã hoàn thành audited screening trên applicable population: schema, numeric validation, applicability cross-check, redundancy, LEI diagnostics và protected coverage/status diagnostics. Ba field đều reconcile đủ 293.301 dòng, không có blank hoặc invalid token. Không có informative value ngoài applicable context; `Not Applicable` và `Exempt` được giữ riêng, không impute thành `0`.

Hai manufactured-home fields có Cramér’s V = 0,9809 trên status trong manufactured context và 0,8393 trên informative-only values, nên chỉ được xem xét như một joint business block. `multifamily_affordable_units` có 2.434 informative rows; tất cả parse được, không âm và là integer, trong đó 2.091 giá trị bằng 0. LEI uplift so với global dominance lần lượt là +37,3441; +41,1445; và +21,6286 điểm phần trăm; đây là proxy-risk diagnostic, không phải quy tắc tự động loại feature. Protected diagnostics sử dụng `applicant_sex` và canonical set của cả năm race slots, chỉ để kiểm tra coverage/status.

Tuy nhiên, HMDA FIG fields 89, 90 và 92 chỉ xác minh reporting domain; notebook chưa có bằng chứng local lineage rằng các giá trị luôn tồn tại tại approval-model prediction point. Vì vậy cả ba field ở trạng thái `hold_for_timing_lineage_confirmation`; `decision` và `gsv_player` giữ `NA`, `timing_gate = False`, và 5.3 không kích hoạt chuyển sang 5.4.

### 5.4. Loại mô hình credit score

- `applicant_credit_score_type`
- `co-applicant_credit_score_type`

Hai cột mô tả loại/tên hệ thống chấm điểm, không phải điểm tín dụng. Cần kiểm tra nguy cơ chúng hoạt động như proxy cho tổ chức hoặc quy trình báo cáo.

Chưa được triển khai trong notebook; chưa có schema audit, timing evidence, LEI/protected diagnostics hoặc screening decision.

### 5.5. Automated Underwriting System

- `aus-1`
- `aus-2`
- `aus-3`
- `aus-4`
- `aus-5`

Năm slot là một nhóm thông tin đa trị, không phải năm biến thứ bậc độc lập. Nếu được giữ, chúng phải ánh xạ thành một business player `aus` và phân biệt rõ valid system, not-applicable, `Exempt`, `NA` và blank.

Chưa được triển khai trong notebook; chưa có slot-order audit, schema/timing evidence, grouped redundancy diagnostics hoặc screening decision.

### Việc còn mở sau checkpoint

1. Xác nhận prediction-time lineage cho ba fields 5.3. Nếu có bằng chứng pre-decision thì rerun gate; nếu không có, giữ `hold` hoặc loại khỏi further review.
2. Hoàn thành screening 5.4 và 5.5 trước khi mở grouped validation cho các candidate còn lại.
3. Chỉ sau khi tất cả screening đã được review mới chốt `feature_decision_table`, `FINAL_MODEL_FEATURES` và `EXPLANATION_GROUPS`; grouped train/validation là một scope tiếp theo, không phải kết quả của mục 5 hiện tại.

## 6. Representation thay thế — 2 cột

- `derived_loan_product_type`
- `derived_dwelling_category`

Mặc định ưu tiên representation gốc:

- `loan_type` + `lien_status` thay cho `derived_loan_product_type`;
- `construction_method` + `total_units` thay cho `derived_dwelling_category`.

Không đưa raw và derived representation tương ứng vào cùng main model. Chỉ thực hiện comparison hẹp nếu quality/association evidence chưa đủ để chọn representation.

## 7. Metadata — chia thành hai loại

Tất cả metadata đều nằm ngoài model feature matrix `X`. Việc “dùng trong feature selection” có nghĩa là dùng làm diagnostic/audit guardrail, không có nghĩa dùng làm predictor.

### 7.1. Metadata được xử lý cùng feature selection — 6 cột

- `applicant_sex`
- `applicant_race-1`
- `applicant_race-2`
- `applicant_race-3`
- `applicant_race-4`
- `applicant_race-5`

#### `applicant_sex`

- Xử lý như protected categorical attribute.
- Kiểm tra valid codes, unknown/not-applicable và coverage.
- Dùng để kiểm tra sự khác biệt về coverage, missingness và distribution của candidate features giữa các nhóm.
- Không đưa vào `X` của main model.

#### `applicant_race-1` đến `applicant_race-5`

- Bảo toàn năm raw slots để truy vết.
- Xử lý chung thành một protected multi-valued group `applicant_race`.
- Không coi năm slot là biến thứ bậc hoặc năm business features độc lập.
- Kiểm tra valid codes, slot consistency, coverage và nhóm có sample size quá nhỏ.
- Dùng cùng `applicant_sex` cho sensitivity/fairness diagnostics trong quá trình chốt feature.
- Không đưa vào `X` của main model.

Các diagnostics tối thiểu với sáu cột này:

1. số lượng và tỷ lệ từng nhóm;
2. tỷ lệ special/missing codes;
3. coverage của candidate features theo sex/race;
4. kiểm tra xem missingness hoặc category hiếm của candidate có tập trung bất thường ở một nhóm hay không;
5. kiểm tra kết luận redundancy có thay đổi mạnh giữa các nhóm hay không.

Các kết quả này chỉ hỗ trợ đánh giá độ nhạy và rủi ro fairness. Chúng không chứng minh quan hệ nhân quả hoặc discrimination.

### 7.2. Metadata chỉ giữ để tham khảo/audit sau — 3 cột

- `lei`
- `county_code`
- `applicant_age`

Ba cột được bảo toàn trong Dataset B để truy vết và phục vụ các phân tích audit/RQ3 về sau. Trong phạm vi feature-selection hiện tại:

- không đưa vào `X`;
- không dùng để quyết định giữ/bỏ predictor;
- không preprocessing cùng model features;
- không tính là business players của GSV/SL.

Nếu sau này cần robustness theo tổ chức, địa lý hoặc tuổi, phải mở một analysis scope riêng thay vì âm thầm đưa chúng vào quy trình hiện tại.

## 8. Minimal data audit trước khi quyết định

Giai đoạn feature selection chỉ thực hiện chuẩn hóa tối thiểu để quan sát dữ liệu; chưa impute, One-Hot Encode, scale hoặc fit model pipeline.

Với mỗi candidate cần ghi nhận:

- dtype và raw tokens;
- valid domain/codes theo schema;
- blank, `NA`, `Exempt`, `1111` và not-applicable như các trạng thái riêng;
- coverage và parse rate;
- số category và category hiếm;
- feature có tồn tại trước quyết định hay không;
- population áp dụng có điều kiện;
- overlap với feature hoặc representation khác.

Nếu xem association với target hoặc protected groups, chỉ dùng phần dữ liệu được chỉ định cho analysis/train; không dùng test set.

## 9. Redundancy và association analysis

Không tạo một ma trận correlation lớn rồi loại feature bằng một threshold duy nhất. Chỉ kiểm tra các block có giả thuyết overlap rõ ràng:

1. `loan_amount` – `property_value` – `loan_to_value_ratio`;
2. `income` – `debt_to_income_ratio`;
3. `loan_type` + `lien_status` – `derived_loan_product_type`;
4. `construction_method` + `total_units` – `derived_dwelling_category`;
5. ba transaction-characteristic flags;
6. năm AUS slots nếu AUS còn được xem xét.

Phép đo được chọn theo loại biến:

- numeric–numeric: Spearman correlation;
- categorical–categorical: Cramér’s V;
- numeric–categorical: correlation ratio hoặc so sánh distribution theo nhóm;
- multi-valued group: overlap/co-occurrence và group-level association.

Association chỉ là bằng chứng về mức độ liên hệ/trùng thông tin, không phải feature importance và không chứng minh quan hệ nhân quả.

## 10. Sensitivity analysis hẹp

Sensitivity analysis chỉ dùng cho quyết định còn mơ hồ sau business rationale, data quality và redundancy analysis. Các comparison ưu tiên:

- `loan_amount + loan_to_value_ratio` so với `loan_amount + property_value`;
- có và không có block `transaction_characteristics`;
- raw representation so với derived representation tương ứng.

Ưu tiên sensitivity ở mức dữ liệu và association. Nếu bắt buộc dùng model để phân xử, chỉ chạy một kiểm tra nhỏ trên train/validation với Logistic Regression và Random Forest; không chạy toàn bộ sáu model và không tính GSV/SL cho các biến thể sensitivity.

Sau sensitivity, feature decision được khóa trước khi pipeline huấn luyện chính bắt đầu.

### 10.1. Grouped validation sau screening

Khi một quyết định vẫn còn mở sau các gate ở mục 8–10, validation chỉ thực hiện trên train/validation với representation và preprocessing giống nhau giữa các phương án. Protocol phải ghi rõ baseline đang dùng các feature đã khóa hay block priority còn trạng thái representation-comparison; không được biến screening hiện tại thành claim đã feature-lock.

1. Baseline bắt đầu từ block 15 feature ưu tiên đã được screening; protocol phải nêu rõ trạng thái quyết định hiện hành của block này. So sánh baseline với grouped addition và grouped ablation theo business block, không thêm rời từng slot của cùng một khái niệm.
2. So sánh raw và derived representation theo cùng split và hyperparameter budget.
3. Kiểm tra ít nhất một mô hình tuyến tính và một mô hình cây; không chạy toàn bộ sáu model hoặc GSV/SL cho sensitivity variants.
4. Nếu protocol đã phê duyệt nhiều seed hoặc cross-validation, báo cáo độ biến thiên; không chỉ báo cáo một con số tốt nhất.
5. Nếu đủ dữ liệu, kiểm tra robustness theo `lei` hoặc held-out-LEI. Fairness sensitivity dùng chín audit metadata ngoài `X`; đây không phải mitigation hoặc causal analysis.

Chỉ số cần báo cáo khi có model-based validation gồm `Δ ROC-AUC`, `Δ PR-AUC` với Denied là lớp cần chú ý, `Δ Brier score` hoặc calibration metric đã khóa, và confidence interval hoặc paired resampling phù hợp cho chênh lệch giữa feature sets. Feature importance đơn lẻ không đủ để quyết định vì có thể phản ánh redundancy, organization proxy, missingness hoặc leakage.

## 11. Quy tắc ra quyết định

### Feature ưu tiên

Mặc định giữ nếu:

- có prior nghiệp vụ rõ;
- có trước quyết định;
- quality/coverage chấp nhận được;
- không bị representation khác thay thế hợp lý hơn.

Chỉ loại khi có bằng chứng cụ thể về timing, leakage, chất lượng hoặc redundancy.

### Feature cần kiểm tra

Mặc định không đưa vào main set. Chỉ giữ khi đồng thời cung cấp một khái niệm mới, có ý nghĩa và timing hợp lệ, coverage/phạm vi áp dụng đủ rõ, giá trị tăng thêm ổn định trên validation thay vì một split, và không chủ yếu phản ánh missingness hoặc proxy `lei`. Lợi ích phải biện minh được chi phí tăng GSV/SL players và không làm suy giảm đáng kể calibration, robustness hoặc fairness sensitivity mà không có lợi ích rõ ràng.

### Representation thay thế

Với mỗi khái niệm raw-versus-derived, chỉ chọn một representation. Ưu tiên raw khi hiệu năng tương đương; chỉ chọn derived khi lợi ích ổn định và vẫn ánh xạ được cho RQ2. Không đưa đồng thời hai representation tương ứng vào main model.

### Metadata dùng trong feature-selection diagnostics

Được xử lý để kiểm tra coverage và sensitivity theo protected groups, nhưng không trở thành predictor.

### Metadata tham khảo

Chỉ bảo toàn cho audit và truy vết sau này; không tác động đến quyết định feature hiện tại.

## 12. Đầu ra bắt buộc của `05_feature_selction.ipynb`

Notebook feature selection phải kết thúc bằng các artifact logic sau:

- `feature_registry`: toàn bộ candidate features và metadata kèm vai trò;
- `priority_feature_audit`: audit của 15 feature ưu tiên;
- `review_feature_audit`: audit của 18 feature cần kiểm tra;
- `protected_metadata_audit`: audit của `applicant_sex` và nhóm `applicant_race`;
- `redundancy_evidence`: kết quả của các block association đã định trước;
- `feature_decision_table`: quyết định `keep`, `exclude` hoặc `alternative` cùng lý do;
- `FINAL_MODEL_FEATURES`: raw columns được khóa cho main model;
- `EXPLANATION_GROUPS`: ánh xạ encoded columns về business players cho GSV/SL;
- `AUDIT_COLUMNS`: sáu protected-metadata columns dùng cho diagnostics;
- `REFERENCE_METADATA`: `lei`, `county_code`, `applicant_age`.

Mỗi dòng trong `feature_decision_table` cần có tối thiểu:

| Trường                        | Nội dung                                            |
| ------------------------------- | ---------------------------------------------------- |
| `feature_or_block`            | Tên feature hoặc block nghiệp vụ                 |
| `business_reason`             | Ý nghĩa đối với approval                        |
| `timing_status`               | Có trước quyết định hay không                 |
| `quality_evidence`            | Coverage, valid codes, special tokens                |
| `redundancy_evidence`         | Feature/representation liên quan                    |
| `protected_group_sensitivity` | Kết quả kiểm tra theo sex/race nếu áp dụng     |
| `validation_evidence`         | Addition/ablation, metric và uncertainty nếu đã được phép chạy |
| `stability_evidence`          | Kết quả qua fold/seed/model family nếu protocol yêu cầu |
| `decision`                    | `keep`, `exclude` hoặc `alternative`          |
| `decision_reason`             | Một câu lý do có thể bảo vệ được           |
| `gsv_player`                  | Business player tương ứng hoặc`not_applicable` |

Notebook này không thực hiện huấn luyện đầy đủ sáu mô hình, không mở test set và không tính GSV/SL chính thức. Sau khi các danh sách trên được review và khóa, chúng mới được đồng bộ sang `config/features.yaml` và pipeline huấn luyện.

## 13. Trạng thái quyết định hiện tại

Đã chốt ở mức thiết kế:

- 15 feature ưu tiên;
- 18 feature cần kiểm tra;
- 2 representation thay thế;
- 6 metadata được xử lý cùng feature-selection diagnostics;
- 3 metadata chỉ giữ để tham khảo/audit;
- target và cohort mapping;
- temporary training feature lock phải được version trước RQ1/CV; final feature lock vẫn bắt buộc trước các deliverable kết luận cuối và GSV/SL chính thức.

Đã chốt từ kết quả audit 15 feature ưu tiên ở mục 4:

- 12 feature có quyết định `keep`; danh sách này là **temporary training feature lock v0.2** cho RQ1/CV. `Dataset_B_Pipeline.md` chỉ tham chiếu section này, không sao chép lại danh sách;
- `property_value` có quyết định `exclude_from_main` nhưng vẫn được giữ trong source/audit;
- block `loan_amount` – `property_value` – LTV đã chọn `loan_amount` + `loan_to_value_ratio` cho main set thay vì giữ cả ba representation;
- `construction_method` và `total_units` đã vượt qua data-quality gate nhưng có trạng thái `keep_for_representation_comparison`;
- các quyết định đã ổn định trong mục 4 hiện ánh xạ thành 10 GSV/SL business players.

Chưa được coi là kết quả cuối:

- final model feature set;
- feature nào trong nhóm 18 đủ điều kiện đi tiếp;
- lựa chọn raw-versus-derived cho `loan_type` + `lien_status` so với `derived_loan_product_type`;
- lựa chọn raw-versus-derived cho `construction_method` + `total_units` so với `derived_dwelling_category`;
- các association checks còn lại, sensitivity analysis hẹp và protected-group/fairness diagnostics;
- kết quả model, validation, test, GSV hoặc Shapley–Lorenz.

## 14. Tài liệu liên quan

- `docs/Dataset_B_Data_Selection.md`
- `docs/Yêu cầu Kỹ thuật Set B cho RQ1 và RQ2 và RQ3 - HMDA NY 2024 - Revised.docx`
- `docs/Yêu cầu Kỹ thuật Set A cho RQ1 và RQ2 - HMDA NY 2024 - Revised.docx`
- `config/features.yaml`
- `notebooks/eda_datasetB/05_feature_selction.ipynb`
