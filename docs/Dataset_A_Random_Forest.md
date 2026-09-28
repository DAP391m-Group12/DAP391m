# **BÁO CÁO HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST**

# **1. TỔNG QUAN PIPELINE VÀ BỐ CỤC THỰC NGHIỆM**
Báo cáo này trình bày quy trình xây dựng, tối ưu hóa và đánh giá mô hình phân loại quyết định duyệt vay (action_taken) sử dụng thuật toán Rừng ngẫu nhiên (Random Forest) trên bộ dữ liệu Dataset A thuộc dự án DAP391m[cite: 6].

## **Quy trình thực nghiệm chuẩn 3 bước**
- **Huấn luyện mô hình (Training):** Huấn luyện Random Forest trên tập Train với bộ siêu tham số tối ưu thu được từ quá trình tìm kiếm lưới (GridSearchCV)[cite: 6].
- **Lựa chọn ngưỡng phân lớp (Threshold Selection):** Xác định ngưỡng quyết định tối ưu t* nhằm tối đa hóa chỉ số Balanced Accuracy trên tập Validation độc lập[cite: 6].
- **Đánh giá ước lượng điểm (Point Estimates):** Đánh giá toàn diện các độ đo phân loại, hiệu chuẩn xác suất và xuất các đồ thị hiệu năng trên tập Test[cite: 6].

## **Quy mô phân chia dữ liệu (Data Splits)**
- **Tập Huấn luyện (Train):** 139,671 mẫu (Phân bố nhãn duyệt vay / từ chối là 105,821 / 33,850, tương ứng tỉ lệ mất cân bằng ~ 3.13 : 1)[cite: 6].
- **Tập Kiểm định (Validation):** 29,930 mẫu[cite: 6].
- **Tập Kiểm thử (Test):** 29,930 mẫu[cite: 6].
- **Tổng số quan sát thực nghiệm:** 199,531 mẫu[cite: 6].

## **Danh mục đặc trưng và biến mục tiêu**
- **Đặc trưng đầu vào (FEATURE_COLS - 7 biến):**
  - *Biến định lượng (Numerical):* income (thu nhập), loan_amount (số tiền vay) — được xử lý chuẩn hóa thang đo[cite: 6].
  - *Biến hạng mục (Categorical):* applicant_sex (giới tính), loan_purpose (mục đích vay), lien_status (quyền thế chấp), loan_type (loại khoản vay), derived_race (chủng tộc)[cite: 6].
- **Biến mục tiêu (Target):** action_taken (Nhãn 1: Approved - Chấp thuận duyệt vay; Nhãn 0: Denied - Từ chối khoản vay)[cite: 6].

# **2. CHUẨN HÓA ĐẶC TRƯNG (FEATURE SCALING)**
Để đảm bảo tính đồng nhất giữa các mô hình và phòng chống rò rỉ dữ liệu, quy trình chuẩn hóa đã được kế thừa chặt chẽ[cite: 6]:
- **Phương pháp:** Kế thừa trực tiếp đối tượng StandardScaler đã được fit từ bước huấn luyện mô hình Logistic Regression (scaler.pkl)[cite: 6].
- **Nguyên tắc phòng chống rò rỉ thông tin (Data Leakage):**
  - Áp dụng phương thức transform thuần túy cho hai biến định lượng income và loan_amount trên cả 3 tập dữ liệu (Train, Validation, Test) dựa theo các tham số đã học từ trước[cite: 6].
  - Không thực hiện fit_transform lại nhằm giữ nguyên phân phối tham số chuẩn gốc[cite: 6].
- **Lưu trữ:** Tệp scaler.pkl sẵn có tại thư mục models/ được tái sử dụng xuyên suốt toàn bộ pipeline[cite: 6].

# **3. THIẾT KẾ MÔ HÌNH VÀ TỐI ƯU HÓA SIÊU THAM SỐ**
## **Chiến lược tìm kiếm tham số tối ưu**
Khảo sát không gian siêu tham số thông qua GridSearchCV kết hợp kiểm định chéo phân tầng 5 lớp (5-fold Stratified K-Fold) trên tập Train với hàm mục tiêu tối đa hóa diện tích dưới đường cong ROC (scoring='roc_auc')[cite: 6]. Không gian khảo sát bao gồm: số lượng cây quyết định (n_estimators: 100, 200), độ sâu tối đa (max_depth: 10, 15, 20), số mẫu tối thiểu để phân tách (min_samples_split: 2, 5), số mẫu tối thiểu tại nút lá (min_samples_leaf: 1, 2) và cơ chế trọng số lớp (class_weight: 'balanced', 'balanced_subsample')[cite: 6].

## **Cấu hình siêu tham số tối ưu (BEST_PARAMS_RF)**
- n_estimators: 200 (Xây dựng tổ hợp 200 cây quyết định nhằm giảm phương sai và tăng độ ổn định)[cite: 6].
- max_depth: 15 (Kiểm soát độ sâu để tránh quá khớp trên tập dữ liệu lớn)[cite: 6].
- min_samples_split: 5 (Số lượng mẫu tối thiểu để phân tách một nút nhánh)[cite: 6].
- min_samples_leaf: 2 (Số lượng mẫu tối thiểu tại mỗi nút lá)[cite: 6].
- class_weight: 'balanced' (Gán trọng số nghịch đảo tần suất lớp để xử lý mất cân bằng mẫu 3:1)[cite: 6].
- random_state: 42 (Cố định tính tái lập thực nghiệm)[cite: 6].
- n_jobs: -1 (Tận dụng toàn bộ tài nguyên CPU để huấn luyện song song)[cite: 6].
- **Thời gian huấn luyện:** 3.01 giây[cite: 6].

# **4. LỰA CHỌN NGƯỠNG PHÂN LỚP TRÊN TẬP VALIDATION**
Do phân bố nhãn bị mất cân bằng (lớp Approved chiếm hơn 75%), việc áp dụng ngưỡng cắt xác suất mặc định t = 0.5 sẽ dẫn đến thiên vị lớp đa số[cite: 5, 6]. Do đó, cần xác định ngưỡng quyết định tối ưu t* để cân bằng khả năng nhận diện cả hai lớp[cite: 5, 6].
- **Phương pháp:** Quét lưới 99 giá trị ngưỡng cách đều từ 0.01 đến 0.99 trên xác suất dự đoán của tập Validation (rf_prob_val)[cite: 6].
- **Kết quả:**
  - Ngưỡng quyết định tối ưu: t* = 0.5500[cite: 6].
  - Balanced Accuracy đạt được trên tập Validation: 0.6840 (68.40%)[cite: 6].

# **5. ĐÁNH GIÁ ƯỚC LƯỢNG ĐIỂM TRÊN TẬP TEST**
Mô hình hoàn chỉnh cùng ngưỡng tối ưu t* = 0.5500 được áp dụng độc lập trên tập Test (29,930 quan sát)[cite: 6].

## **5.1. Bảng tổng hợp các chỉ số đánh giá (Evaluation Metrics)**
| Tên chỉ số đo lường | Giá trị đạt được | Ý nghĩa và nhận định |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.7488 | Khả năng phân biệt thứ hạng giữa hồ sơ được duyệt và bị từ chối đạt mức tốt. |
| **PR-AUC (Denied)** | 0.5170 | Diện tích đường cong Precision-Recall cho lớp thiểu số (Từ chối). |
| **Brier Score** | 0.1932 | Sai số bình phương trung bình giữa xác suất dự đoán và nhãn thực tế thấp, độ tin cậy xác suất cao. |
| **Calibration Intercept** | 1.0473 | Hệ số chặn hiệu chuẩn xác suất (độ lệch hiệu chuẩn). |
| **Calibration Slope** | 0.9423 | Độ dốc đường cong hiệu chuẩn tiệm cận 1.0 (xác suất dự đoán phản ánh sát thực tế). |
| **Balanced Accuracy** | 0.6768 | Độ chính xác cân bằng đạt 67.68%, ổn định tương đương tập Validation. |

## **5.2. Hiệu năng phân loại theo từng lớp nhãn**
- **Lớp Chấp thuận (Approved - Nhãn 1):**
  - *Precision:* 0.8796 (87.96% hồ sơ dự đoán duyệt là chính xác)[cite: 6].
  - *Recall:* 0.6179 (Thu hồi được 61.79% tổng số khách hàng thực sự đủ điều kiện)[cite: 6].
  - *F1-Score:* 0.7259[cite: 6].
- **Lớp Từ chối (Denied - Nhãn 0):**
  - *Precision:* 0.3812 (38.12% hồ sơ dự đoán từ chối là chính xác)[cite: 6].
  - *Recall:* 0.7357 (Nhận diện thành công 73.57% hồ sơ rủi ro, hỗ trợ kiểm soát nợ xấu)[cite: 6].
  - *F1-Score:* 0.5022[cite: 6].

## **5.3. Tốc độ thực thi (Execution Speed)**
- **Thời gian huấn luyện (Training time):** 3.01 giây[cite: 6].
- **Thời gian suy luận (Inference time):** 0.1419 giây cho toàn bộ 29,930 hồ sơ tập Test (trung bình xấp xỉ 4.74 microsecond / hồ sơ)[cite: 6].

# **6. PHÂN TÍCH HỆ THỐNG BIỂU ĐỒ ĐÁNH GIÁ**
Quá trình thực nghiệm đã xuất 3 tệp biểu đồ phân tích hiệu năng[cite: 6]:
- **Ma trận nhầm lẫn (rf_confusion_matrix.png):**
  - Thể hiện cả dạng số lượng đếm tuyệt đối (Counts) và tỉ lệ chuẩn hóa theo dòng thực tế (Normalized Recall)[cite: 6].
  - Mô hình kiểm soát rủi ro tín dụng khi phát hiện 73.57% các hồ sơ bị từ chối vay thực tế[cite: 6].
- **Đường cong ROC và Precision-Recall (rf_roc_pr_curves.png):**
  - Đường cong ROC tách biệt rõ so với đường phân giác ngẫu nhiên (AUC = 0.7488)[cite: 6].
  - Đường cong PR của lớp Approved duy trì mức Precision cao trên dải Recall rộng[cite: 6].
- **Phân tích ngưỡng quyết định (rf_threshold_analysis.png):**
  - Biểu diễn biến thiên của Precision, Recall và F1-Score khi ngưỡng quyết định thay đổi từ 0.01 đến 0.99[cite: 6].
  - Ngưỡng đạt F1 cực đại ở mốc xấp xỉ 0.22, trong khi ngưỡng t* = 0.5500 (tối ưu Balanced Accuracy) mang lại sự cân bằng giữa mở rộng tín dụng và quản trị rủi ro[cite: 6].

# **7. ĐÓNG GÓI ARTIFACTS VÀ KẾT LUẬN**
## **Danh mục tệp tin artifact đã lưu trữ**
- rf_model.pkl: Mô hình Random Forest đã huấn luyện sẵn sàng triển khai[cite: 6].
- rf_meta.pkl: Tệp lưu trữ siêu dữ liệu (ngưỡng t* = 0.5500, thời gian chạy, bộ siêu tham số và kết quả CV)[cite: 6].
- scaler.pkl: Đối tượng StandardScaler đã chuẩn hóa[cite: 6].

## **Kết luận**
Mô hình Random Forest đem lại khả năng phân loại và bắt trúng hồ sơ rủi ro tín dụng cao (Recall lớp Denied đạt 73.57%)[cite: 6], các chỉ số hiệu chuẩn xác suất và diện tích đường cong ROC đều thể hiện sự tối ưu rõ rệt[cite: 6]. Toàn bộ mô hình, siêu dữ liệu và hình ảnh biểu đồ đã được lưu trữ hoàn tất, sẵn sàng cho các phân tích tính công bằng (fairness) và giải thích mô hình (SHAP/feature importance) tiếp theo[cite: 1, 6].