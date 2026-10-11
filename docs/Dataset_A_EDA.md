# BÁO CÁO TIỀN XỬ LÝ & KHÁM PHÁ DỮ LIỆU KHOẢN VAY MUA NHÀ (HMDA NEW YORK 2024)

## 1. TỔNG QUAN DỰ ÁN VÀ NGUỒN DỮ LIỆU

- **Nguồn dữ liệu:** Bộ dữ liệu Đạo luật Tiết lộ Thông tin Vay Mua nhà (Home Mortgage Disclosure Act - HMDA) bang New York năm 2024 (`state_NY_2024.csv`).
- **Mục tiêu:** Thực hiện tiền xử lý dữ liệu (data preprocessing), phát hiện và làm sạch các giá trị dị biệt/bất thường (anomalies/outliers), mã hóa các biến phân loại và biến mục tiêu, đồng thời chuẩn hóa phân phối các thuộc tính định lượng phục vụ bài toán phân loại quyết định duyệt vay (`action_taken`).

### Danh sách 8 đặc trưng cốt lõi được lựa chọn

| Tên đặc trưng | Loại biến | Mô tả chi tiết |
| :--- | :--- | :--- |
| `applicant_sex` | Categorical | Giới tính của người nộp đơn vay (Male / Female) |
| `income` | Numerical | Thu nhập hàng năm của người nộp đơn |
| `loan_amount` | Numerical | Tổng số tiền khoản vay đề nghị |
| `loan_purpose` | Categorical | Mục đích sử dụng khoản vay |
| `lien_status` | Categorical | Tình trạng quyền giữ tài sản thế chấp (Lien status) |
| `loan_type` | Categorical | Loại hình khoản vay (Thông thường, FHA, VA,...) |
| `derived_race` | Categorical | Chủng tộc / dân tộc của người nộp đơn |
| `action_taken` | Target Label | Quyết định phê duyệt từ phía tổ chức tài chính |

---

## 2. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

### 2.1. Xử lý giá trị thiếu (Missing Values)
Thực hiện loại bỏ toàn bộ các bản ghi chứa giá trị rỗng (`NaN`) trên 8 cột thuộc tính quan sát thông qua phương thức `dropna(subset=cols)`. Bước này đảm bảo tính toàn vẹn của dữ liệu mẫu, triệt tiêu nhiễu và hạn chế mất mát thông tin khi huấn luyện mô hình.

### 2.2. Lọc đối tượng và làm sạch phạm vi nghiên cứu
- **Loại bỏ hồ sơ không áp dụng:** Loại bỏ các bản ghi có `loan_purpose == 5` (các khoản vay đại diện cho "Pháp nhân", tổ chức kinh tế thay vì cá nhân người vay).
- **Lọc thông tin giới tính cá nhân:** Chỉ giữ lại các hồ sơ có giới tính người nộp đơn xác định gồm `1` (Nam / Male) và `2` (Nữ / Female), loại bỏ các trường hợp không khai báo hoặc không áp dụng để đồng nhất đối tượng nghiên cứu là thể nhân.

### 2.3. Mã hóa thuộc tính hạng mục (Categorical Encoding)

#### Chuẩn hóa mục đích vay (`loan_purpose`)
Ánh xạ lại hệ số mã hóa theo thứ tự trực quan và chuẩn hóa ngữ nghĩa:
- **0**: Refinancing (Tái tài trợ khoản vay - mã gốc `31`)
- **1**: Home purchase (Mua nhà - mã gốc `1`)
- **2**: Home improvement (Cải tạo/sửa chữa nhà ở - mã gốc `2`)
- **3**: Cash-out refinancing (Tái tài trợ rút tiền mặt - mã gốc `32`)
- **4**: Other purpose (Mục đích khác - mã gốc `4`)

#### Mã hóa nhị phân chủng tộc (`derived_race`)
Lọc và mã hóa nhị phân cho hai nhóm nhân khẩu học chính nhằm phục vụ việc phân tích công bằng tín dụng:
- `Black or African American` $\rightarrow 1$
- `White` $\rightarrow 0$

### 2.4. Định nghĩa và mã hóa biến mục tiêu (`action_taken`)
- Gom các nhóm quyết định tích cực (mã `1`: Khoản vay được giải ngân; mã `2`: Khoản vay được chấp thuận nhưng người vay không nhận) thành nhãn `1` (**Accepted** - Chấp thuận).
- Nhóm hồ sơ bị từ chối (mã `3`) được ánh xạ thành nhãn `0` (**Denied** - Từ chối).

#### Phân tích mất cân bằng nhãn (Class Imbalance)
Tỉ lệ giữa lớp Accepted và Denied đạt mức xấp xỉ $3 : 1$ (khoảng $75\%$ được duyệt : $25\%$ bị từ chối). Đây là mức mất cân bằng nhẹ, không đòi hỏi các giải pháp can thiệp tái lấy mẫu nhân tạo cực đoan (như SMOTE nặng) gây biến dạng phân phối ban đầu.

---

## 3. PHÂN TÍCH PHÂN PHỐI VÀ BIẾN ĐỔI BIẾN ĐỊNH LƯỢNG

### 3.1. Kỹ thuật biến đổi Sign-Log Transform
Các thuộc tính tài chính (`income`, `loan_amount`) thường có độ lệch phải (right-skewed) rất lớn và độ phân tán thang đo rộng.
Áp dụng công thức biến đổi:  
$$y = \text{sign}(x) \times \ln(1 + |x|)$$

Triển khai bằng mã nguồn `np.sign(x) * np.log1p(np.abs(x))`, bảo toàn được chiều âm/dương của dữ liệu đồng thời nén biên độ giá trị lớn về thang đo logarithm tự nhiên.

### 3.2. Phân tích phân phối khoản vay (`loan_amount`)
Sau khi áp dụng biến đổi logarithm tự nhiên qua hàm `log1p(loan_amount)`:

| Chỉ số thống kê | Giá trị |
| :--- | :--- |
| **Số lượng quan sát ($N$)** | $201,777$ mẫu |
| **Giá trị trung bình ($\mu$)** | $12.06$ |
| **Độ lệch chuẩn ($\sigma$)** | $1.02$ |
| **Phân vị 25% ($Q_1$)** | $11.35$ |
| **Trung vị ($Median$)** | $12.07$ |
| **Phân vị 75% ($Q_3$)** | $12.81$ |
| **Khoảng giá trị ($Min - Max$)** | $8.52 - 16.99$ |

Phân phối của `loan_amount` sau khi biến đổi tiệm cận gần như hoàn hảo dạng phân phối chuẩn hình chuông (Gaussian distribution), tạo điều kiện thuận lợi cho các thuật toán học máy tuyến tính hoạt động hiệu quả.

### 3.3. Phát hiện bất thường nghiêm trọng ở biến Thu nhập (`income`)

#### Vấn đề phát hiện
Khi trực quan hóa phân phối biến đổi Sign-Log của `income`, biểu đồ xuất hiện một đỉnh nhọn dị biệt tại điểm $0$ và một dải đuôi âm kéo dài bất thường đạt mức $Z \approx -15.28\sigma$ so với trung bình.

#### Đánh giá thực tế và tác động thuật toán
- **Về mặt nghiệp vụ ngân hàng:** Người vay thế chấp mua nhà sở hữu thu nhập âm cực đoan là điều phi lý, thường do hạch toán lỗ kinh doanh hoặc lỗi sai lệch trong khâu nhập liệu hành chính.
- **Về mặt thống kê & mô hình:** Mức dị biệt tới $-15.28\sigma$ vi phạm nghiêm trọng giả định phân phối chuẩn đối xứng, có khả năng làm sai lệch vector trọng số hồi quy (Logistic Regression).

#### Giải pháp xử lý
1. Lọc điều kiện bắt buộc: `income > 0` (loại bỏ triệt để giá trị $\le 0$).
2. Áp dụng trực tiếp hàm `np.log1p(income)`.

**Kết quả:** Biểu đồ tần suất thu nhập sau khi lọc dịch chuyển hoàn toàn về phân phối chuẩn đối xứng với đỉnh phân phối tập trung quanh giá trị $4.5 - 5.0$, loại bỏ toàn bộ các điểm kỳ dị âm.

---

## 4. KẾT QUẢ VÀ BƯỚC TIẾP THEO

### Lưu trữ dữ liệu sạch
Dữ liệu sau khi hoàn thành quy trình làm sạch và chuyển đổi phân phối được kết xuất vào tệp tin:
`state_NY_processed_2024.csv`.

### Khuyến nghị cho Pipeline Machine Learning
- **Chuẩn hóa thang đo (Scaling):** Tiến hành chuẩn hóa `StandardScaler` cho hai thuộc tính liên tục `income` và `loan_amount` trước khi đưa vào mô hình hóa.
- **Xây dựng mô hình:** Sử dụng Hồi quy Logistic làm Baseline đối sánh hiệu năng với các mô hình cây quyết định và tổ hợp cây (Random Forest).
- **Đánh giá tính công bằng (Model Fairness):** Phân tích SHAP Beeswarm Plot trên các biến nhân khẩu học nhạy cảm (`applicant_sex`, `derived_race`) để giám sát hiện tượng thiên lệch (algorithmic bias) trong phê duyệt tín dụng.