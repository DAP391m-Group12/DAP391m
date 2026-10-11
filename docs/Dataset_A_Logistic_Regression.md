# BÁO CÁO HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH LOGISTIC REGRESSION (DATASET A - MỤC 4 & 5)

## 1. TỔNG QUAN PIPELINE VÀ BỐ CỤC THỰC NGHIỆM

Báo cáo này tổng hợp quy trình xây dựng, tối ưu hóa ngưỡng phân lớp và ước lượng hiệu năng cho mô hình Baseline Hồi quy Logistic (Logistic Regression) trên tập dữ liệu Dataset A thuộc dự án DAP391m.

### Quy trình thực nghiệm 3 giai đoạn chuẩn
1. **Huấn luyện mô hình (Training):** Huấn luyện Logistic Regression trên tập Train với bộ siêu tham số tối ưu thu được từ quá trình tìm kiếm lưới (GridSearchCV).
2. **Tối ưu hóa ngưỡng phân lớp (Threshold Selection):** Quét lưới trên tập Validation độc lập để xác định ngưỡng quyết định $t^*$ tối đa hóa chỉ số Balanced Accuracy.
3. **Đánh giá ước lượng điểm (Point Estimates):** Kiểm thử mô hình với ngưỡng $t^*$ trên tập Test để tính toán các độ đo hiệu năng, kiểm tra hiệu chuẩn xác suất và xuất đồ thị phân tích.

### Quy mô phân chia dữ liệu (Data Splits)
- **Tập Huấn luyện (Train):** $139,671$ mẫu (Tỉ lệ nhãn duyệt vay / từ chối là $105,821 / 33,850$, tương ứng mức mất cân bằng $\approx 3.13 : 1$).
- **Tập Kiểm định (Validation):** $29,930$ mẫu.
- **Tập Kiểm thử (Test):** $29,930$ mẫu.
- **Tổng số quan sát thực nghiệm:** $199,531$ mẫu.

### Danh mục đặc trưng và biến mục tiêu
- **Biến đầu vào (`FEATURE_COLS` - 7 đặc trưng):**
  - *Biến định lượng (Numerical):* `income`, `loan_amount` (áp dụng chuẩn hóa Z-score).
  - *Biến phân loại (Categorical):* `applicant_sex`, `loan_purpose`, `lien_status`, `loan_type`, `derived_race`.
- **Biến mục tiêu (Target):** `action_taken` (Nhãn `1`: Approved - Duyệt vay; Nhãn `0`: Denied - Từ chối).

---

## 2. CHUẨN HÓA ĐẶC TRƯNG (FEATURE SCALING)

Để thuật toán tối ưu Gradient hội tụ ổn định và không bị thiên lệch bởi biên độ thang đo của các biến liên tục, kỹ thuật chuẩn hóa dữ liệu được triển khai nghiêm ngặt.

- **Phương pháp:** Sử dụng `StandardScaler` (Z-score scaling) cho hai biến định lượng `income` và `loan_amount`.
- **Nguyên tắc chống rò rỉ dữ liệu (Data Leakage):**
  - Thực hiện `fit_transform` duy nhất trên tập Train để xác định kỳ vọng ($\mu$) và độ lệch chuẩn ($\sigma$).
  - Chỉ áp dụng phương thức `transform` trên tập Validation và tập Test dựa trên bộ tham số $\mu, \sigma$ của tập Train.
- **Lưu trữ đối tượng:** Xuất ra tệp `scaler.pkl` phục vụ kiểm tra và tái hiện pipeline suy luận.
- **Phân phối của tập Train sau chuẩn hóa:**
  - `income`: $\text{Mean} = 0.000$, $\text{Std} = 1.000$, $\text{Min} = -5.239$, $\text{Max} = 10.345$.
  - `loan_amount`: $\text{Mean} = 0.000$, $\text{Std} = 1.000$, $\text{Min} = -3.476$, $\text{Max} = 4.847$.
  - Các biến phân loại giữ nguyên giá trị mã hóa số nguyên tương ứng.

---

## 3. THIẾT KẾ MÔ HÌNH VÀ TỐI ƯU HÓA SIÊU THAM SỐ

### Chiến lược tìm kiếm siêu tham số
Thực hiện tìm kiếm lưới `GridSearchCV` với kiểm định chéo phân tầng 5 lượt (5-fold Stratified K-Fold) trên tập Train, hàm mục tiêu là tối đa hóa diện tích dưới đường cong ROC (`scoring="roc_auc"`). Không gian tìm kiếm bao gồm: thuật toán giải (`lbfgs`, `liblinear`), hình thức chuẩn hóa trọng số (`l1`, `l2`, `None`), tham số phạt $C \in [0.001, 100]$, và cơ chế gán trọng số lớp (`balanced`, `None`).

### Cấu hình siêu tham số tối ưu (`BEST_PARAMS_LR`)
- `C`: $0.001$ (Mức phạt điều chuẩn mạnh, giúp kiểm soát hiện tượng quá khớp hiệu quả).
- `penalty`: `'l2'` (Điều chuẩn Ridge).
- `solver`: `'lbfgs'` (Giải thuật tối ưu quasi-Newton bậc hai).
- `class_weight`: `'balanced'` (Tự động bù trừ trọng số theo tần suất nghịch đảo của từng lớp).
- `fit_intercept`: `True`.
- `max_iter`: $1000$ (Đảm bảo điều kiện hội tụ tuyệt đối).
- `random_state`: $42$.
- **Thời gian huấn luyện:** $0.13$ giây.

---

## 4. LỰA CHỌN NGƯỠNG PHÂN LỚP TRÊN TẬP VALIDATION (MỤC 4)

Trong điều kiện mất cân bằng nhãn ($75\%$ hồ sơ được duyệt), việc sử dụng ngưỡng cắt phân loại mặc định $t = 0.5$ sẽ dẫn đến tình trạng thiên vị nhãn đa số. Do đó, cần tìm ngưỡng tối ưu $t^*$ để cân bằng khả năng dự đoán trên cả hai nhóm nhãn.

- **Phương pháp:** Quét lưới 99 giá trị ngưỡng xác suất cách đều trong đoạn $[0.01, 0.99]$ dựa trên xác suất dự báo của mô hình trên tập Validation (`lr_prob_val`).
- **Hàm tối ưu:** $\text{argmax}_{t} \left( \text{Balanced Accuracy}(y_{val}, \hat{y}(t)) \right)$.
- **Kết quả:**
  - Ngưỡng quyết định tối ưu: $t^* = 0.5200$.
  - Balanced Accuracy đạt được trên tập Validation: $0.6532$ ($65.32\%$).

---

## 5. ĐÁNH GIÁ ƯỚC LƯỢNG ĐIỂM TRÊN TẬP TEST (MỤC 5)

Áp dụng mô hình Hồi quy Logistic với ngưỡng tối ưu $t^* = 0.5200$ trên tập Test độc lập gồm $29,930$ quan sát.

### 5.1. Bảng tổng hợp độ đo đánh giá (Point Estimates)

| Chỉ số đo lường | Giá trị đạt được | Nhận xét chi tiết |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.7037** | Khả năng phân loại và xếp hạng hồ sơ tín dụng ở mức khá tốt. |
| **PR-AUC (Denied)** | **0.4342** | Diện tích đường cong Precision-Recall của lớp thiểu số (Bị từ chối). |
| **Brier Score** | **0.2187** | Đo lường mức độ tin cậy của xác suất (càng thấp càng tốt). |
| **Cal Intercept** | **1.1333** | Hệ số chặn hiệu chuẩn xác suất (đo độ lệch kỳ vọng). |
| **Cal Slope** | **1.1246** | Độ dốc đường cong hiệu chuẩn bám sát $1.0$, xác suất dự báo tin cậy. |
| **Balanced Accuracy** | **0.6573** | Đạt $65.73\%$, thể hiện tính ổn định cao so với tập Validation ($65.32\%$). |

### 5.2. Hiệu năng phân loại chi tiết theo từng lớp nhãn

- **Lớp Chấp thuận (Approved - Nhãn 1):**
  - **Precision:** $0.8606$ ($86.06\%$ hồ sơ dự báo được duyệt là chính xác thực tế).
  - **Recall:** $0.6372$ (Thu hồi được $63.72\%$ tổng số hồ sơ đủ tiêu chuẩn giải ngân).
  - **F1-Score:** $0.7323$.
- **Lớp Từ chối (Denied - Nhãn 0):**
  - **Precision:** $0.3740$ ($37.40\%$ hồ sơ dự báo từ chối là rủi ro chính xác).
  - **Recall:** $0.6774$ (Nhận diện thành công $67.74\%$ các hồ sơ có nguy cơ nợ xấu thực tế).
  - **F1-Score:** $0.4819$.

### 5.3. Tốc độ thực thi
- **Thời gian huấn luyện (Train time):** $0.13$ giây.
- **Thời gian suy luận (Inference time):** $0.0041$ giây cho toàn bộ $29,930$ hồ sơ tập Test ($\approx 0.14\ \mu\text{s}$ / hồ sơ).

---

## 6. PHÂN TÍCH HỆ THỐNG BIỂU ĐỒ HIỆU NĂNG

### Ma trận nhầm lẫn (`lr_confusion_matrix.png`)
Biểu diễn số lượng dự đoán và tỉ lệ thu hồi chuẩn hóa (Normalized Recall). Với ngưỡng tối ưu $t^* = 0.5200$, mô hình bảo vệ danh mục tín dụng tốt nhờ tỷ lệ phát hiện các khoản nợ xấu thực tế lên tới $67.74\%$.

### Đường cong ROC và Precision-Recall (`lr_roc_pr_curves.png`)
- Đường cong ROC tách rời đáng kể so với đường chéo ngẫu nhiên ($AUC = 0.7037$).
- Đường cong PR của lớp Approved vượt trội so với baseline ngẫu nhiên ($\text{Baseline} \approx 0.75$).

### Phân tích độ nhạy theo ngưỡng quyết định (`lr_threshold_analysis.png`)
- Trực quan hóa tương quan đánh đổi giữa Precision, Recall và F1-Score khi ngưỡng quyết định dịch chuyển từ $0.01$ đến $0.99$.
- Điểm F1 cực đại nằm ở ngưỡng thấp ($\approx 0.22$), trong khi ngưỡng $t^* = 0.5200$ (tối ưu Balanced Accuracy) đạt được điểm thỏa hiệp cân bằng nhất giữa mở rộng tệp khách hàng vay và kiểm soát rủi ro từ chối.

---

## 7. ĐÓNG GÓI ARTIFACTS VÀ KẾT LUẬN

### Danh mục tệp tin artifacts đã lưu trữ
- `lr_model.pkl`: Mô hình Logistic Regression đã huấn luyện sẵn sàng tích hợp.
- `lr_meta.pkl`: Metadata bao gồm ngưỡng quyết định $t^* = 0.5200$, thời gian thực thi, siêu tham số tối ưu và điểm số CV.
- `scaler.pkl`: Đối tượng `StandardScaler` dùng cho pipeline tiền xử lý dữ liệu mới.

### Kết luận
Mô hình Logistic Regression đáp ứng đầy đủ tiêu chí của một mô hình Baseline chuẩn: tốc độ tính toán siêu nhanh ($0.13$ giây), xác suất có độ hiệu chuẩn tốt (Calibration Slope $\approx 1.12$), và Balanced Accuracy đạt $65.73\%$ trên tập Test. Đây là hệ quy chiếu vững chắc để so sánh với các thuật toán phức tạp hơn ở các giai đoạn thực nghiệm tiếp theo.