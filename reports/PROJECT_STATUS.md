# Tiến trình dự án — 18/09/2026

## Kết luận hiện tại

Dự án đang ở giai đoạn **hoàn thiện nền tảng Set A**. Phần cấu hình phạm vi và
tiền xử lý cốt lõi đã được triển khai; pipeline nghiên cứu hoàn chỉnh để trả lời
RQ1/RQ2 chưa có. Vì vậy, mã hiện tại **chưa đáp ứng đầy đủ yêu cầu ban đầu**.

## Đối chiếu với yêu cầu đã khóa

| Hạng mục yêu cầu | Trạng thái | Bằng chứng / ghi chú |
| --- | --- | --- |
| Mục tiêu `y_approved`: action 1/2 là 1, action 3 là 0; loại các action khác | Đạt | `prepare_set_a_data` lọc `{1,2,3}` và không giữ `action_taken` trong đầu vào. |
| Set A dùng đúng 11 cột, gồm đủ 5 race slots | Đạt | Cấu hình và tiền xử lý kiểm tra bắt buộc toàn bộ cột. |
| Race slots là multi-hot không có thứ tự | Đạt | `ApplicantRaceMultiHotEncoder` gộp category từ cả 5 slots. |
| Bảy nhóm giải thích RQ2 | Đạt ở mức chuẩn bị | Preprocessor lưu ánh xạ feature đã mã hóa sang đủ 7 nhóm; chưa có thuật toán giải thích. |
| Hai mô hình Logistic Regression và Random Forest, cùng split | Chưa triển khai | `src/pipeline/train.py` rỗng; split trong config vẫn là `to_be_locked_before_execution`. |
| Tuning CV chỉ trong train và chọn/freeze threshold theo balanced accuracy ở validation | Chưa triển khai | Chỉ có yêu cầu trong YAML, chưa có code thực thi/test. |
| RQ1: xác suất, ROC-AUC, Denied PR-AUC, Brier, calibration, metric theo threshold và thời gian | Chưa triển khai | `src/pipeline/evaluate.py` rỗng. |
| Bootstrap ghép cặp 1.000 lần, CI percentile 95%, không retrain | Chưa triển khai | Chỉ được khai báo trong YAML. |
| RQ2: LR/RF × GSV/Shapley-Lorenz, chuẩn hóa theo nhóm và bootstrap | Chưa triển khai | Chưa có module giải thích hoặc kiểm thử. |
| Lưu/tải model, dự đoán `P(approved)` và FastAPI | Chưa triển khai | `predict.py` và toàn bộ `src/api/` rỗng. |
| Kiểm thử tự động | Một phần | Có 5 unit test cho tiền xử lý, gồm assert hợp đồng và runtime logging; chưa có test train/evaluate/explanation/API. File API test hiện tên `test_api..py` và rỗng. |

## Dữ liệu và kiểm thử đã xác nhận

- `data/raw/state_NY.csv` có 383.614 dòng, 99 cột và đủ 12 cột cần cho việc tạo target/Set A.
- Phân bố `action_taken`: 1=208.818, 2=9.942, 3=74.541, 4=42.186, 5=18.377,
  6=28.198, 7=281, 8=1.271. Tập Set A hợp lệ trước xử lý có 293.301 dòng.
- `python -m unittest discover -v` chạy **5/5 test đạt** trong môi trường tạm có
  `PyYAML`, NumPy, pandas và scikit-learn.
- Preprocessing ghi log ra terminal và `reports/logs/preprocess.log`, gồm số dòng
  đầu vào/hợp lệ, số Approved/Denied, shape sau biến đổi và lỗi thiếu cột. Các
  assert kiểm tra contract Set A, target, output encoding và mapping nhóm RQ2.
- Python mặc định trên máy hiện thiếu `PyYAML`; vì vậy chạy test trực tiếp hiện
  thất bại trước khi import mã. `requirements.txt` đã liệt kê `pyyaml`, nhưng
  dependencies chưa được cài vào interpreter đang được gọi.

## Việc cần làm tiếp theo (ưu tiên)

1. Tạo môi trường dự án, cài dependency từ `requirements.txt`, và ghi rõ lệnh
   chạy test bằng `unittest` (hoặc bổ sung `pytest` nếu muốn dùng pytest).
2. Khóa seed và tỉ lệ train/validation/test trong `config/config.yaml`; triển
   khai data loader và split dùng chung, có stratification và artifact tái lập.
3. Triển khai `train.py`: CV trong train, Logistic Regression và Random Forest,
   chọn threshold theo balanced accuracy ở validation, rồi freeze model/threshold.
4. Triển khai `evaluate.py`: toàn bộ metric RQ1, calibration không hiệu chỉnh
   xác suất, thời gian chạy và paired bootstrap 1.000 lần trên test set.
5. Triển khai RQ2 theo bảy nhóm đã ánh xạ, kèm bốn phân tích yêu cầu và bootstrap
   riêng theo explanation observations.
6. Hoàn thiện lưu/tải artifact, prediction/API và các test end-to-end; đổi tên
   `tests/test_api..py` thành tên discovery hợp lệ khi bổ sung test API.

## Lưu ý phạm vi

Các kết quả sau này chỉ được diễn giải là mô hình hóa quyết định phê duyệt lịch
sử. Không dùng chúng để dự báo vỡ nợ, suy luận creditworthiness, hoặc kết luận
nhân quả/pháp lý về phân biệt đối xử.
