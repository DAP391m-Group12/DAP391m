# Dataset B — Bàn giao preprocessing Phase 1

## 1. Trạng thái bàn giao

Tài liệu này là hướng dẫn vận hành preprocessing cho ba baseline Phase 1: Logistic Regression (LR), Random Forest (RF) và XGBoost. Nó ghi lại contract dữ liệu, quy tắc biến đổi, artifact đã xuất và cách sử dụng trong notebook.

| Hạng mục | Giá trị hiện hành |
| --- | --- |
| Dataset | `dataset_b` |
| Feature contract | `temporary_training_lock_v0_2` — 12 raw features |
| Split contract | `split_v1` — Train/Validation/Test cố định `70%/15%/15%` |
| CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` chỉ trên Train |
| Preprocessing version | `dataset_b_preprocessing_v2` |
| Encoded width | 66 cột theo thứ tự cố định |
| Artifact CV hiện hành | `data/preprocessing/dataset_b_v1/cv_preprocessing_v2/` |
| Profile LR | Two-channel + OHE + `RobustScaler` cho bốn kênh số |
| Profile RF/XGBoost | Two-channel + OHE, không scale |
| Class handling | Không sample weight, không resampling, không class weighting |

Preprocessing CV đã được triển khai và artifact năm fold đã được xuất từ Train. Chưa chạy estimator LR/RF/XGBoost, chưa refit preprocessing trên toàn Train, chưa transform Validation/Test bằng fitted full-Train pipeline và chưa có kết quả RQ1.

Nguồn điều khiển:

- [Feature Selection](Feature_selection.md#13-trạng-thái-quyết-định-hiện-tại): danh sách feature và bằng chứng screening;
- [Dataset B Pipeline](Dataset_B_Pipeline.md): ranh giới Train/CV/Validation/Test;
- `config/dataset_b/feature_set_v1.yaml`: feature và target contract;
- `config/dataset_b/split_v1.yaml`: split và CV contract;
- `config/dataset_b/model_protocol_v1.yaml`: ba baseline cố định;
- `data/preprocessing/dataset_b_v1/cv_preprocessing_v2/manifest.json`: contract của ma trận đã xuất.

## 2. Ranh giới dữ liệu bắt buộc

1. Không chia dữ liệu lại. Mọi bước downstream dùng `split_v1` và `record_id` hiện có.
2. `X` chỉ gồm 12 raw features trong feature contract. Không đưa `record_id`, `action_taken`, `y_approved`, chín audit columns hoặc candidate khác vào `X`.
3. CV chỉ dùng Train. Mỗi preprocessing pipeline được fit riêng trên fold-train và chỉ transform fold-validation.
4. Sau CV, tạo pipeline mới và fit trên toàn Train. Pipeline này chỉ transform Validation và Test.
5. Validation chỉ chọn threshold. Test chỉ đánh giá sau khi model, preprocessing và threshold đã khóa.
6. Không lấy thống kê median, category vocabulary hoặc scaling từ Validation/Test.

Ba partition cố định hiện có:

| Partition | Số dòng | Vai trò |
| --- | ---: | --- |
| Train | 205.310 | CV và full-Train refit |
| Validation | 43.995 | Chọn threshold sau refit |
| Test | 43.996 | Final evaluation |

Target là `y_approved = 1` khi `action_taken ∈ {1, 2}` và `0` khi `action_taken = 3`. `action_taken` chỉ phục vụ tạo target và truy vết, không thuộc model input.

## 3. Contract của 12 raw features

| Raw feature | Representation đã chốt |
| --- | --- |
| `loan_type` | Categorical theo schema; thiếu thật → `MISSING` |
| `loan_purpose` | Categorical; giữ mã *Not applicable*; thiếu thật → `MISSING` |
| `lien_status` | Categorical; thiếu thật → `MISSING` |
| `reverse_mortgage` | Giữ riêng `1`, `2`, `1111`; thiếu thật → `MISSING` |
| `open-end_line_of_credit` | Giữ riêng `1`, `2`, `1111`; thiếu thật → `MISSING` |
| `business_or_commercial_purpose` | Giữ riêng `1`, `2`, `1111`; thiếu thật → `MISSING` |
| `income` | Two-channel; `income < 0` được map thành `0`; giữ số `0` thật; `NA` và `MISSING` là hai status khác nhau |
| `debt_to_income_ratio` | Categorical; giữ các band/mức cùng `NA`, `Exempt`, `MISSING` riêng biệt |
| `loan_amount` | Two-channel; giá trị thiếu thật dùng median fold-train cho value và `MISSING` cho status |
| `loan_to_value_ratio` | Two-channel; giữ `NA`, `Exempt`, `MISSING`; giữ nguyên giá trị số `> 1000%` |
| `loan_term` | Two-channel; giữ `NA`, `Exempt`, `MISSING` |
| `occupancy_type` | Categorical theo schema; thiếu thật → `MISSING` |

Quy tắc token:

- `NaN`/null thực sự, chuỗi rỗng và whitespace-only → `MISSING`.
- Literal `NA` và `Exempt` là nhãn hợp lệ ở các cột cho phép, không phải lỗi dữ liệu.
- `1111` ở ba feature exemption là category hợp lệ; `income = 1111` vẫn là số.
- Token ngoài miền của từng cột bị từ chối; không tự đổi thành `MISSING`.
- Không xóa dòng, clip, log-transform hoặc tự sửa outlier khác.

## 4. Kiến trúc pipeline đã triển khai

Factory công khai:

```python
from src.dataset_b.preprocessing import make_preprocessing_pipeline

lr_preprocessor = make_preprocessing_pipeline("lr")
rf_preprocessor = make_preprocessing_pipeline("rf")
xgb_preprocessor = make_preprocessing_pipeline("xgb")
```

Mỗi lời gọi trả về một pipeline mới, chưa fit, tương thích scikit-learn:

```text
12 raw columns
    → DatasetBRawParserStep
    → DatasetBNumericTwoChannel
    → DatasetBOneHotEncoder
    → DatasetBNumericValueRobustScaler     profile="lr"
      hoặc passthrough                    profile="rf"/"xgb"
    → sparse CSR matrix (66 columns)
```

### 4.1. Parser

- Kiểm tra đúng 12 cột và đúng thứ tự contract.
- Chuẩn hóa missing token theo miền của từng feature.
- Map `income < 0` thành `0`.
- Giữ income bằng `0` và LTV cực đoan.
- Không sửa DataFrame đầu vào.

### 4.2. Numeric two-channel

Bốn feature `income`, `loan_amount`, `loan_to_value_ratio`, `loan_term` được tách thành:

- `<feature>_value`: số thực; `NA`/`Exempt`/`MISSING` nhận median fit trên fold-train hoặc full Train;
- `<feature>_status`: trạng thái `OBSERVED`, `NA`, `Exempt` hoặc `MISSING` theo schema của cột.

Median chỉ là placeholder kỹ thuật cho value channel. Status channel giữ nguyên lý do không có giá trị số.

### 4.3. OHE 66 cột

Đầu ra gồm:

- 4 numeric value columns ở bốn vị trí đầu;
- 62 OHE columns theo schema cố định;
- tổng cộng 66 cột, cùng thứ tự cho LR, RF và XGBoost.

Bốn dummy `OBSERVED` của numeric status bị bỏ làm reference category. Các dummy `MISSING` vẫn được giữ dù không xuất hiện trong split hiện tại. Tám raw categorical features tiếp tục dùng full OHE.

Tên và nguồn của toàn bộ 66 cột nằm trong `manifest.json` tại:

- `output_columns`;
- `output_raw_sources`;
- `output_categories`.

### 4.4. Scaling theo model

| Model | Profile | Scaling |
| --- | --- | --- |
| LR | `lr` | `RobustScaler` chỉ trên bốn numeric value columns |
| RF | `rf` | Không scale |
| XGBoost | `xgb` | Không scale |

RF và XGBoost có cùng representation nên artifact CV dùng chung prefix `tree`. LR có artifact riêng vì bốn numeric value columns đã được scale.

## 5. Artifact CV đã bàn giao

```text
data/preprocessing/dataset_b_v1/cv_preprocessing_v2/
├── manifest.json
├── fold_01/
│   ├── lr_train.npz
│   ├── lr_fold_validation.npz
│   ├── tree_train.npz
│   ├── tree_fold_validation.npz
│   └── rows.npz
├── fold_02/
├── fold_03/
├── fold_04/
└── fold_05/
```

Mỗi `rows.npz` chứa:

- `train_positions`;
- `fold_validation_positions`;
- `train_record_ids`;
- `fold_validation_record_ids`;
- `train_y`;
- `fold_validation_y`.

Artifact này có `fit_scope = fold_train_only` và `use = cross_validation_only`. Nó không chứa fitted pipeline object và không dùng để transform Validation/Test. Không cố tái dựng fitted preprocessor từ median/scaler summary trong manifest.

Bản 70 cột tại `data/preprocessing/dataset_b_v1/cv/` chỉ được giữ để truy vết và không dùng cho baseline v2.

## 6. Notebook — thiết lập dùng chung

Notebook nên chạy bằng environment của repository. Cell đầu xác định project root, mở import path và bật logger:

```python
from pathlib import Path
import logging
import sys

PROJECT_ROOT = next(
    path
    for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "config" / "dataset_b" / "model_protocol_v1.yaml").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
```

Đọc model protocol và manifest trước khi train:

```python
import json
import yaml

with (PROJECT_ROOT / "config/dataset_b/model_protocol_v1.yaml").open(
    encoding="utf-8"
) as handle:
    protocol = yaml.safe_load(handle)

cv_root = PROJECT_ROOT / protocol["contracts"]["cv_artifact"]
with (cv_root / "manifest.json").open(encoding="utf-8") as handle:
    preprocessing_manifest = json.load(handle)

assert protocol["contracts"]["preprocessing_version"] == "dataset_b_preprocessing_v2"
assert preprocessing_manifest["preprocessing_version"] == "dataset_b_preprocessing_v2"
assert preprocessing_manifest["encoded_feature_count"] == 66
assert preprocessing_manifest["source_partition"] == "train"
assert preprocessing_manifest["fit_scope"] == "fold_train_only"
assert preprocessing_manifest["use"] == "cross_validation_only"
```

## 7. Notebook — chạy CV từ artifact đã xuất

Đây là đường đi ngắn nhất cho baseline hiện tại. Ma trận đã được preprocess đúng fold boundary, nên estimator được fit trực tiếp trên `.npz`. Không ghép thêm `make_preprocessing_pipeline` vào route này.

### 7.1. Tạo estimator từ protocol

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ESTIMATOR_CLASS = {
    "lr": LogisticRegression,
    "rf": RandomForestClassifier,
    "xgb": XGBClassifier,
}

MATRIX_PREFIX = {
    "lr": "lr",
    "rf": "tree",
    "xgb": "tree",
}

def make_estimator(model_name: str):
    model_contract = protocol["models"][model_name]
    return ESTIMATOR_CLASS[model_name](**model_contract["parameters"])
```

Không thêm `class_weight`, `sample_weight`, resampling hoặc hyperparameter ngoài protocol trong baseline này.

### 7.2. Fit và đánh giá năm fold

```python
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score

def run_preprocessed_cv(model_name: str):
    prefix = MATRIX_PREFIX[model_name]
    fold_results = []

    for fold_contract in preprocessing_manifest["folds"]:
        fold_dir = cv_root / fold_contract["directory"]
        X_fold_train = load_npz(fold_dir / f"{prefix}_train.npz")
        X_fold_validation = load_npz(
            fold_dir / f"{prefix}_fold_validation.npz"
        )

        with np.load(fold_dir / "rows.npz", allow_pickle=False) as rows:
            y_fold_train = rows["train_y"]
            y_fold_validation = rows["fold_validation_y"]
            train_record_ids = rows["train_record_ids"]
            fold_validation_record_ids = rows["fold_validation_record_ids"]

        assert X_fold_train.shape == (len(y_fold_train), 66)
        assert X_fold_validation.shape == (len(y_fold_validation), 66)
        assert len(train_record_ids) == len(y_fold_train)
        assert len(fold_validation_record_ids) == len(y_fold_validation)

        estimator = make_estimator(model_name)
        estimator.fit(X_fold_train, y_fold_train)
        probability = estimator.predict_proba(X_fold_validation)[:, 1]
        fold_auc = roc_auc_score(y_fold_validation, probability)

        fold_results.append(
            {
                "fold": fold_contract["fold"],
                "roc_auc": float(fold_auc),
                "validation_rows": len(y_fold_validation),
            }
        )

    auc_values = np.asarray([item["roc_auc"] for item in fold_results])
    summary = {
        "model": model_name,
        "mean_roc_auc": float(auc_values.mean()),
        "std_roc_auc": float(auc_values.std()),
        "folds": fold_results,
    }
    return summary

lr_cv = run_preprocessed_cv("lr")
rf_cv = run_preprocessed_cv("rf")
xgb_cv = run_preprocessed_cv("xgb")
```

`record_id` phải được giữ cùng prediction nếu notebook lưu out-of-fold result. Không suy ra thứ tự hàng từ DataFrame khác; dùng arrays trong `rows.npz`.

## 8. Notebook — lấy raw pipeline và train trực tiếp

Route này dùng khi cần CV từ raw input, full-Train refit hoặc dự báo Validation/Test.

### 8.1. Load fixed partitions

```python
from src.dataset_b.preprocessing import DatasetBSplitLoader

loader = DatasetBSplitLoader()
train = loader.load("train")
validation = loader.load("validation")
test = loader.load("test")

assert train.X.shape[1] == 12
assert validation.X.shape[1] == 12
assert test.X.shape[1] == 12
```

Mỗi partition cung cấp:

- `.X`: đúng 12 raw model features;
- `.y`: `y_approved`;
- `.record_ids`: ID theo đúng thứ tự manifest;
- `.audit`: chín audit columns, nằm ngoài `X`;
- `.action_taken`: target source để truy vết, nằm ngoài `X`.

### 8.2. Ghép preprocessor với estimator

Protocol dùng `tree` làm tên profile artifact chung cho RF/XGBoost, còn factory công khai nhận `lr`, `rf` hoặc `xgb`. Notebook dùng mapping sau:

```python
from sklearn.pipeline import Pipeline
from src.dataset_b.preprocessing import make_preprocessing_pipeline

PIPELINE_PROFILE = {
    "lr": "lr",
    "rf": "rf",
    "xgb": "xgb",
}

def make_model_pipeline(model_name: str):
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessing_pipeline(PIPELINE_PROFILE[model_name]),
            ),
            ("estimator", make_estimator(model_name)),
        ]
    )
```

### 8.3. CV trực tiếp từ raw Train

```python
from sklearn.base import clone
from src.dataset_b.splits import DatasetBSplitter

splitter = DatasetBSplitter.from_yaml().cv_splitter()
template = make_model_pipeline("lr")

for fold_number, (fit_idx, hold_idx) in enumerate(
    splitter.split(train.X, train.y), start=1
):
    fold_model = clone(template)
    fold_model.fit(train.X.iloc[fit_idx], train.y.iloc[fit_idx])
    fold_probability = fold_model.predict_proba(
        train.X.iloc[hold_idx]
    )[:, 1]
```

Mỗi `clone(template)` tạo fitted state độc lập. Không tái sử dụng model hoặc fitted preprocessor giữa các fold.

### 8.4. Refit toàn Train và transform holdouts

```python
model_name = "lr"  # hoặc "rf", "xgb"
final_model = make_model_pipeline(model_name)

final_model.fit(train.X, train.y)
validation_probability = final_model.predict_proba(validation.X)[:, 1]
```

Sau khi threshold được chọn và khóa trên Validation:

```python
test_probability = final_model.predict_proba(test.X)[:, 1]
```

Không gọi lại `.fit`, `.fit_transform` hoặc thay pipeline giữa Validation và Test. Không dùng `.y` của Test để điều chỉnh preprocessing, estimator, configuration hoặc threshold.

## 9. Threshold và ranh giới với model phase

Protocol hiện chốt tối đa hóa Balanced Accuracy trên Validation nhưng chưa chốt tie-break nếu nhiều threshold đạt cùng giá trị tối đa. Notebook phải lưu toàn bộ candidate score và không tự giấu một tie-break chưa được phê duyệt.

Phần preprocessing bàn giao kết thúc tại fitted model pipeline và xác suất `P(y_approved = 1)`. Việc tính threshold, metric RQ1, confidence interval và lưu model/prediction artifact thuộc model/evaluation phase tiếp theo.

## 10. Logging và artifact tối thiểu cho notebook

Code preprocessing đã log các bước load, fit median, two-channel, encode, scale và export. Notebook không nên tắt logger. Với mỗi model/fold, tối thiểu cần lưu hoặc hiển thị:

- model name và exact parameters;
- preprocessing version;
- feature version và split version;
- fold number;
- số dòng train/fold-validation;
- matrix shape;
- `record_id` đi cùng prediction;
- fold ROC-AUC;
- warning hội tụ hoặc lỗi estimator;
- phiên bản thư viện từ manifest.

Không ghi đè artifact v2. Model outputs phải đi sang thư mục artifact của model run, không ghi vào `data/preprocessing/dataset_b_v1/cv_preprocessing_v2/`.

## 11. Checklist trước khi train

- [ ] Notebook tìm đúng project root và import được `src.dataset_b`.
- [ ] Đọc `model_protocol_v1.yaml` và manifest v2.
- [ ] Xác nhận preprocessing version là `dataset_b_preprocessing_v2`.
- [ ] Xác nhận mỗi matrix có 66 cột.
- [ ] LR dùng prefix `lr`; RF/XGBoost dùng prefix `tree` khi đọc artifact CV.
- [ ] Không chạy preprocessing lần nữa trên `.npz` đã preprocess.
- [ ] Khi dùng raw input, luôn ghép một pipeline mới cho mỗi fold/model.
- [ ] Không split lại và không dùng Validation/Test trong CV.
- [ ] Không class weighting, sample weighting hoặc resampling trong baseline.
- [ ] Giữ `record_id` cùng mọi prediction.
- [ ] Không fit lại sau khi đã dùng Validation để chọn threshold.
- [ ] Chỉ đánh giá Test sau khi threshold đã khóa.

## 12. Giới hạn đã biết

- `notebooks/02_model_experiments.ipynb` hiện chưa có model implementation.
- Artifact CV lưu ma trận, nhãn và hàng; không lưu fitted preprocessing pipeline object.
- Chưa có full-Train preprocessing artifact hoặc fitted estimator.
- Chưa có model runner tự đọc `model_protocol_v1.yaml`.
- Tie-break của threshold vẫn cần được chốt trong model/evaluation phase.
- Feature lock vẫn là temporary training lock, chưa phải `FINAL_MODEL_FEATURES`.
- RQ2/RQ3 và các neural models nằm ngoài bàn giao Phase 1 này.
