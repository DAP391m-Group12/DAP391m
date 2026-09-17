# DAP391m-Group12/
```text
DAP391m/
│
├── .env.example                  # Mẫu cấu hình môi trường (DB URL, Port, Secret Key)
├── .gitignore                    # Bỏ qua data nặng, trọng số model, .venv, mlruns, cache
├── README.md                     # Hướng dẫn cài đặt, chạy pipeline và API
├── requirements.txt              # Danh sách thư viện (fastapi, scikit-learn, mlflow, pyyaml,...)
│
├── config/                       # Toàn bộ cấu hình hệ thống & tham số dạng YAML
│   ├── config.yaml               # Cấu hình chung (seed, đường dẫn data/model, tỷ lệ split)
│   ├── model_config.yaml         # Siêu tham số cho từng thuật toán (RF, XGBoost, LightGBM,...)
│   └── features.yaml             # Danh sách cột: target, categorical, numerical, drop_cols
│
├── data/                         # Dữ liệu dự án
│   ├── raw/                      # Dữ liệu gốc thu thập được (chỉ đọc, không ghi đè)
│   │   └── state_NY.csv
│   └── processed/                # Dữ liệu sau khi làm sạch, tiền xử lý và chia train/test
│
├── model/                        # Lưu các artifact trọng số model tốt nhất (.pkl, .joblib, .onnx)
│
├── notebooks/                    # Jupyter Notebooks phục vụ EDA và thử nghiệm nhanh
│   ├── 01_eda.ipynb              # Khám phá và trực quan hóa dữ liệu ban đầu
│   └── 02_experiments.ipynb      # Thử nghiệm các thuật toán, feature selection
│
├── reports/                      # Báo cáo kết quả và biểu đồ trực quan
│   └── figures/                  # Lưu ảnh biểu đồ (confusion matrix, ROC-AUC, feature importance)
│
├── sql/                          # Các câu lệnh truy vấn trích xuất dữ liệu từ Database
│
├── src/                          # Source code chính của dự án (được đóng gói thành module)
│   ├── __init__.py
│   │
│   ├── api/                      # Phần triển khai API Serving (FastAPI)
│   │   ├── __init__.py
│   │   ├── main.py               # Khởi tạo FastAPI app, định nghĩa các endpoints (/predict, /health)
│   │   ├── schemas.py            # Pydantic schemas (validate dữ liệu input/output của request)
│   │   └── dependencies.py       # Các hàm load model hoặc xử lý logic chung cho API
│   │
│   ├── pipeline/                 # Pipeline xử lý dữ liệu và huấn luyện model
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Đọc dữ liệu từ file/DB theo config
│   │   ├── preprocess.py         # Xử lý missing value, encode, scale features
│   │   ├── train.py              # Huấn luyện model + log tham số/metric vào MLflow
│   │   ├── evaluate.py           # Tính toán các chỉ số đánh giá (Accuracy, F1, Precision, Recall)
│   │   └── predict.py            # Load model đã lưu để thực hiện dự đoán (dùng cho API)
│   │
│   └── utils/                    # Các hàm tiện ích dùng chung
│       ├── __init__.py
│       ├── config_loader.py      # Hàm đọc và parse các file cấu hình YAML
│       └── logger.py             # Cấu hình format log hệ thống
│
├── tests/                        # Kiểm thử tự động
│   ├── __init__.py
│   ├── test_api.py               # Test các endpoint API
│   └── test_pipeline.py          # Test hàm tiền xử lý và inference
│
└── mlruns/                       # (Tự động sinh ra khi chạy MLflow) Quản lý lịch sử train model
```