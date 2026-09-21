# Dataset B — Data Selection và kế hoạch kiểm tra feature

**Phạm vi:** HMDA New York 2024, bài toán dự đoán **credit approval**.  
**Trạng thái tài liệu:** chốt vai trò cột sau Gate 1–2; chưa tuyên bố kết quả feature importance vì các kiểm tra thực nghiệm chưa được chạy.

## 1. Mục tiêu và ranh giới diễn giải

Dataset B phục vụ dự đoán xác suất hồ sơ được chấp thuận:

- `y_approved = 1` khi `action_taken` thuộc `{1, 2}`;
- `y_approved = 0` khi `action_taken = 3`;
- chỉ giữ cohort có `action_taken` thuộc `{1, 2, 3}`;
- vẫn bảo toàn `action_taken` gốc để truy vết, nhưng **không đưa vào X**.

Mô hình dự đoán kết quả phê duyệt được ghi nhận trong HMDA. Kết quả không được diễn giải thành xác suất vỡ nợ, mức độ tín nhiệm thực, nguyên nhân bị từ chối, hay bằng chứng nhân quả về phân biệt đối xử.

Ba trường `reverse_mortgage`, `open-end_line_of_credit` và `business_or_commercial_purpose` **không được dùng để lọc cohort**. Đây là các đặc điểm giao dịch có thể ảnh hưởng đến quyết định phê duyệt nên được giữ trong nhóm predictor cốt lõi.

## 2. Đối soát toàn bộ 99 cột

| Vai trò sau Gate 1–2 | Số cột | Trạng thái |
|---|---:|---|
| Predictor candidates | 35 | Đi tiếp sang kiểm tra dữ liệu và thực nghiệm |
| Metadata phục vụ audit/RQ3 | 9 | Giữ ngoài ma trận feature `X` |
| Target source | 1 | Chỉ dùng để tạo `y_approved` |
| Loại khỏi Dataset B | 54 | Không giữ trạng thái chờ |
| **Tổng** | **99** | Đã đối soát |

Ngoài 99 cột HMDA, tài liệu yêu cầu Dataset B có thể cần một `record_id` kỹ thuật để truy vết. Trường này phải được tạo có kiểm soát, không phải predictor và không được tính vào phép đối soát 99 cột.

## 3. Các cột được giữ

### 3.1. Target source — 1 cột

- `action_taken`

Quy tắc bắt buộc: tạo target sau khi xác định cohort; không dùng `action_taken` làm predictor; không thay thế hoặc ghi đè cột gốc.

### 3.2. Metadata phục vụ audit và RQ3 — 9 cột

- `lei`
- `county_code`
- `applicant_race-1`
- `applicant_race-2`
- `applicant_race-3`
- `applicant_race-4`
- `applicant_race-5`
- `applicant_sex`
- `applicant_age`

Các cột này được giữ để phân tích tổ chức, địa lý và nhóm người nộp đơn trong RQ3/fairness audit. Chúng **không đi vào feature matrix của mô hình chính**, trừ khi sau này có một thiết kế thí nghiệm riêng được phê duyệt rõ ràng.

### 3.3. Predictor candidates — 35 cột

#### A. Ưu tiên chọn theo lập luận nghiệp vụ — 15 cột

Đây là baseline feature set nên được kiểm tra trước. Chúng mô tả cấu trúc khoản vay, năng lực tài chính có thể quan sát, quy mô/tài sản bảo đảm và đặc điểm bất động sản — các nhóm thông tin có quan hệ trực tiếp nhất với bài toán phê duyệt tín dụng.

**Cấu trúc và mục đích khoản vay**

- `loan_type`
- `loan_purpose`
- `lien_status`
- `reverse_mortgage`
- `open-end_line_of_credit`
- `business_or_commercial_purpose`

**Năng lực tài chính và quy mô khoản vay**

- `income`
- `debt_to_income_ratio`
- `loan_amount`
- `property_value`
- `loan_to_value_ratio`

**Điều khoản khoản vay**

- `loan_term`

**Đặc điểm bất động sản**

- `occupancy_type`
- `construction_method`
- `total_units`

Việc “ưu tiên” không đồng nghĩa tự động giữ trong feature set cuối. Một cột vẫn phải bị loại nếu sai lineage, không có tại thời điểm quyết định, chất lượng dữ liệu không đạt, hoặc chỉ lặp lại một representation đã chọn.

#### B. Chỉ giữ nếu thực nghiệm chứng minh giá trị tăng thêm — 18 cột

Các cột sau có thể hữu ích, nhưng ý nghĩa hoặc phạm vi áp dụng phụ thuộc mạnh vào quy trình báo cáo, loại hồ sơ và cơ chế missing. Không thể giữ chỉ vì có correlation hoặc feature importance cao.

**Kênh nộp hồ sơ và quan hệ tổ chức**

- `preapproval`
- `submission_of_application`
- `initially_payable_to_institution`
- `conforming_loan_limit`

**Cấu trúc trả nợ đặc biệt**

- `negative_amortization`
- `interest_only_payment`
- `balloon_payment`
- `other_nonamortizing_features`

**Nhà sản xuất và multifamily**

- `manufactured_home_secured_property_type`
- `manufactured_home_land_property_interest`
- `multifamily_affordable_units`

**Loại mô hình credit score được dùng**

- `applicant_credit_score_type`
- `co-applicant_credit_score_type`

Hai trường trên cho biết **loại/tên mô hình chấm điểm**, không phải điểm tín dụng thực tế. Chỉ giữ nếu xác nhận chúng có ý nghĩa trước quyết định và không hoạt động chủ yếu như proxy cho tổ chức hoặc quy trình báo cáo.

**Automated Underwriting System (AUS)**

- `aus-1`
- `aus-2`
- `aus-3`
- `aus-4`
- `aus-5`

Năm slot AUS phải được xem như **một nhóm thông tin đa trị**, không phải năm biến thứ bậc độc lập. Cần phân biệt mã “không áp dụng”, “Exempt” và thiếu dữ liệu theo đúng schema.

#### C. Representation thay thế — 2 cột

- `derived_loan_product_type`
- `derived_dwelling_category`

Không dùng đồng thời representation gốc và derived tương ứng:

- so sánh `derived_loan_product_type` với cặp `loan_type` + `lien_status`;
- so sánh `derived_dwelling_category` với `construction_method` + `total_units`.

Mặc định ưu tiên các cột gốc vì dễ giải thích và kiểm soát encoding hơn. Chỉ chọn representation derived nếu thực nghiệm cho thấy kết quả ổn định hơn mà không làm mất khả năng giải thích cho RQ2.

## 4. Các cột bị loại hoàn toàn — 54 cột

### 4.1. Gate 1: hậu quyết định, outcome-dependent hoặc leakage trực tiếp — 15 cột

- `purchaser_type`
- `interest_rate`
- `rate_spread`
- `hoepa_status`
- `total_loan_costs`
- `total_points_and_fees`
- `origination_charges`
- `discount_points`
- `lender_credits`
- `prepayment_penalty_term`
- `intro_rate_period`
- `denial_reason-1`
- `denial_reason-2`
- `denial_reason-3`
- `denial_reason-4`

Lý do chung: các trường này được hình thành sau hoặc phụ thuộc mạnh vào kết quả/quá trình ra quyết định. Đưa chúng vào mô hình approval tạo data leakage (rò rỉ dữ liệu) hoặc khiến mô hình học dấu vết của outcome thay vì thông tin có sẵn tại thời điểm thẩm định.

### 4.2. Gate 2 và các cột ngoài vai trò đã chốt — 39 cột

**Năm báo cáo và địa lý không được chọn**

- `activity_year`
- `derived_msa-md`
- `state_code`
- `census_tract`

`activity_year` và `state_code` là hằng trong phạm vi NY 2024. `county_code` đã được giữ làm metadata; các mức địa lý/tract còn lại bị loại theo phạm vi đã chốt.

**Derived protected attributes**

- `derived_ethnicity`
- `derived_race`
- `derived_sex`

**Ethnicity và cờ quan sát**

- `applicant_ethnicity-1`
- `applicant_ethnicity-2`
- `applicant_ethnicity-3`
- `applicant_ethnicity-4`
- `applicant_ethnicity-5`
- `co-applicant_ethnicity-1`
- `co-applicant_ethnicity-2`
- `co-applicant_ethnicity-3`
- `co-applicant_ethnicity-4`
- `co-applicant_ethnicity-5`
- `applicant_ethnicity_observed`
- `co-applicant_ethnicity_observed`

**Race của co-applicant và cờ quan sát race**

- `co-applicant_race-1`
- `co-applicant_race-2`
- `co-applicant_race-3`
- `co-applicant_race-4`
- `co-applicant_race-5`
- `applicant_race_observed`
- `co-applicant_race_observed`

**Sex của co-applicant và cờ quan sát sex**

- `co-applicant_sex`
- `applicant_sex_observed`
- `co-applicant_sex_observed`

**Age ngoài metadata đã chọn**

- `co-applicant_age`
- `applicant_age_above_62`
- `co-applicant_age_above_62`

**Thông tin census tract/context**

- `tract_population`
- `tract_minority_population_percent`
- `ffiec_msa_md_median_family_income`
- `tract_to_msa_income_percentage`
- `tract_owner_occupied_units`
- `tract_one_to_four_family_homes`
- `tract_median_age_of_housing_units`

Quyết định phạm vi: ngoài chín metadata đã nêu ở mục 3.2, các cột nhân khẩu học, địa lý và tract còn lại bị loại hoàn toàn; không duy trì trạng thái “pending”.

## 5. Các vấn đề phải kiểm tra trước khi khóa feature set

### 5.1. Schema và lineage

- Đối chiếu chính xác 99 header trong `data/raw/state_NY_2024.csv` với schema HMDA 2024.
- Xác minh mapping của LTV: file raw hiện dùng `loan_to_value_ratio`, trong một số tài liệu/file dẫn xuất có thể gọi `combined_loan_to_value_ratio`. Không được coi hai tên là hai feature khác nhau.
- Ghi lại nguồn, năm dữ liệu, phiên bản extraction và quy tắc tạo `record_id`.
- Bảo đảm tầng Dataset B không sửa dữ liệu raw và mọi biến đổi đều truy vết được.

### 5.2. Cohort và target

- Xác nhận cohort chỉ gồm `action_taken ∈ {1,2,3}` và báo cáo số dòng từng lớp.
- Xác nhận `y_approved` đúng mapping `1/2 → 1`, `3 → 0`.
- Không lọc hồ sơ theo `reverse_mortgage`, `open-end_line_of_credit` hoặc `business_or_commercial_purpose`.
- So sánh phân bố lớp tổng thể và theo `lei`/`county_code` để phát hiện tổ chức hoặc địa bàn chi phối dữ liệu.

### 5.3. Chất lượng dữ liệu của từng candidate

Với mỗi feature, cần kiểm tra:

- kiểu dữ liệu thực tế, miền giá trị và mã hợp lệ theo schema;
- tỷ lệ quan sát được, blank, `NA`, `Exempt`/`1111` và “not applicable” — không gộp mặc định thành một loại missing;
- tỷ lệ missing toàn bộ, theo `action_taken` và theo `lei`;
- category hiếm, category chỉ xuất hiện ở một lớp, hoặc category tập trung tại rất ít tổ chức;
- khả năng parse và đơn vị đo;
- outlier bằng quantile bền vững đối với `income`, `loan_amount`, `property_value`, `loan_to_value_ratio`, `debt_to_income_ratio` và `loan_term`.

Không loại feature chỉ vì missing cao nếu missing phản ánh phạm vi áp dụng hợp lệ. Ngược lại, không giữ feature chỉ vì missingness dự đoán target tốt: đó có thể là dấu hiệu quy trình hoặc leakage.

### 5.4. Timing và leakage

- Với từng candidate, xác nhận thông tin có tồn tại tại thời điểm ra quyết định phê duyệt.
- Đo khả năng dự đoán của **chỉ cờ missingness**. Nếu cờ missing dự đoán quá mạnh, kiểm tra xem missing có được tạo ra sau outcome hay do cách tổ chức báo cáo.
- So sánh missing pattern giữa approved và denied, đồng thời phân tầng theo `lei`; tránh kết luận leakage chỉ từ khác biệt thô toàn bộ dữ liệu.
- Không dùng test set để quyết định giữ/bỏ feature.

### 5.5. Phạm vi áp dụng có điều kiện

- Hai trường manufactured home chỉ được đánh giá trong nhóm hồ sơ phù hợp với loại construction/property.
- `multifamily_affordable_units` chỉ được đánh giá trong phạm vi multifamily phù hợp.
- Các slot AUS phải được chuẩn hóa thành một nhóm có thứ tự slot rõ ràng; mã “không áp dụng” không được coi là một hệ thống AUS thật.
- Khi một feature không áp dụng cho phần lớn hồ sơ, so sánh ba phương án: biểu diễn có điều kiện, gộp nhóm nghiệp vụ, hoặc loại feature; không imputing một giá trị số giả.

### 5.6. Redundancy và representation

- Kiểm tra block `loan_amount` – `property_value` – `loan_to_value_ratio`; LTV liên hệ toán học với hai cột còn lại nhưng có thể mang thông tin báo cáo khác.
- Kiểm tra block `income` – `debt_to_income_ratio`.
- So sánh representation raw và derived nêu tại mục 3.3C; không đưa cả hai vào cùng mô hình chính.
- Dùng association/correlation phù hợp với loại biến để chẩn đoán, nhưng không lấy một ngưỡng correlation đơn lẻ làm quy tắc loại.
- Kiểm tra quan hệ có điều kiện và khả năng đảo chiều giữa nhóm tổ chức, địa lý hoặc loại sản phẩm; không dựa hoàn toàn vào association toàn cục.

## 6. Handoff sang feature governance

Tài liệu này chốt candidate universe và các gate dữ liệu bắt buộc. Evidence, grouped validation, decision rules và decision-record schema được duy trì duy nhất tại `docs/Feature_selection.md`, mục 8–12. Bất kỳ validation nào vẫn chỉ dùng train/validation; test set được niêm phong đến khi feature set và pipeline đã khóa.

## 7. Trạng thái contract hiện tại

Đã chốt:

- target source và cohort logic;
- đúng 9 metadata phục vụ audit/RQ3;
- 54 cột bị loại hoàn toàn;
- 35 predictor candidates còn lại và ba mức xử lý;
- không lọc cohort bằng ba cột loại hình giao dịch;
- test set không tham gia feature selection.

Chưa được coi là đã chứng minh:

- feature nào trong 35 cột có importance cao hoặc giá trị tăng thêm ổn định;
- raw hay derived representation nào sẽ được chọn;
- missingness nào là hợp lệ, proxy quy trình hay leakage;
- feature set cuối cùng cho sáu mô hình.

## 8. Nguồn tham chiếu

### Tài liệu nội bộ

- `docs/Project_planning_recover.docx`
- `docs/Yêu cầu Kỹ thuật Set B cho RQ1 và RQ2 và RQ3 - HMDA NY 2024 - Revised.docx`
- `docs/Yêu cầu Kỹ thuật Set A cho RQ1 và RQ2 - HMDA NY 2024 - Revised.docx`

### HMDA 2024 chính thức

- [2024 HMDA national loan-level dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024)
- [2024 Reportable HMDA Data: Regulatory and Reporting Overview Reference Chart](https://files.consumerfinance.gov/f/documents/cfpb_reportable-hmda-data_regulatory-and-reporting-overview-reference-chart_2024-10.pdf)
- [2024 HMDA Filing Instructions Guide](https://files.ffiec.cfpb.gov/documentation/2024-hmda-fig.pdf)
- [HMDA Data Browser — 2024](https://ffiec.cfpb.gov/data-browser/maps/2024)
