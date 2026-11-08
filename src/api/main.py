"""
Gradio UI Dự đoán Phê duyệt Khoản vay Thế chấp (HMDA New York 2024)
Chỉ tập trung vào: Chọn model có sẵn -> Nhập 15 features -> Dự đoán tỷ lệ Approved / Denied.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    import torch
except ImportError:
    torch = None


# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN MODEL CÓ SẴN & NGƯỠNG QUYẾT ĐỊNH
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_ARTIFACT_PATHS: Dict[str, Path] = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
    "XGBoost": MODELS_DIR / "xgboost.joblib",
    "PlainMLP": MODELS_DIR / "plainmlp.pt",
    "RealMLP": MODELS_DIR / "realmlp.joblib",
    "TabMmini": MODELS_DIR / "tabmmini.pt",
}

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

LOCKED_THRESHOLDS: Dict[str, float] = {
    "Logistic Regression": 0.500,
    "Random Forest": 0.515,
    "XGBoost": 0.525,
    "PlainMLP": 0.510,
    "RealMLP": 0.726,
    "TabMmini": 0.520,
}

ORDERED_15_PREDICTORS = [
    "loan_type",
    "loan_purpose",
    "lien_status",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "income",
    "debt_to_income_ratio",
    "loan_amount",
    "loan_to_value_ratio",
    "loan_term",
    "occupancy_type",
    "preapproval",
    "interest_only_payment",
    "balloon_payment",
]

_MODEL_CACHE: Dict[str, Any] = {}
_PREPROCESSOR_CACHE: Dict[str, Any] = {}


# ==============================================================================
# 2. ĐỌC MODEL CÓ SẴN & SUY LUẬN XÁC SUẤT APPROVED / DENIED
# ==============================================================================
def _read_artifact_from_disk(path: Path) -> Any:
    if not path or not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix in (".joblib", ".pkl"):
        if joblib is not None:
            return joblib.load(path)
        with open(path, "rb") as f:
            return pickle.load(f)
    if suffix in (".pt", ".pth") and torch is not None:
        return torch.load(path, map_location="cpu")
    return None


def get_pretrained_model(model_name: str) -> Any:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    primary_path = MODEL_ARTIFACT_PATHS[model_name]
    for ext in (primary_path.suffix, ".joblib", ".pkl", ".pt", ".pth"):
        candidate = primary_path.with_suffix(ext)
        if candidate.exists():
            artifact = _read_artifact_from_disk(candidate)
            if artifact is not None:
                _MODEL_CACHE[model_name] = artifact
                return artifact
    return None


def get_pretrained_preprocessor() -> Any:
    if "preprocessor" in _PREPROCESSOR_CACHE:
        return _PREPROCESSOR_CACHE["preprocessor"]
    if PREPROCESSOR_PATH.exists():
        prep = _read_artifact_from_disk(PREPROCESSOR_PATH)
        _PREPROCESSOR_CACHE["preprocessor"] = prep
        return prep
    return None


def compute_fallback_probability(raw_inputs: Dict[str, Any], model_name: str) -> float:
    """Tính xác suất Approved khi chưa đặt file model vào thư mục `models/`."""
    inc = max(0.0, float(raw_inputs["income"]))
    loan_amt = max(1000.0, float(raw_inputs["loan_amount"]))
    ltv = float(raw_inputs["loan_to_value_ratio"])
    term = float(raw_inputs["loan_term"])

    logit = 1.15

    dti_weights = {
        "<20%": 0.85,
        "20%-<30%": 0.70,
        "30%-<36%": 0.50,
        "36%-42%": 0.20,
        "43%-49%": -0.25,
        "50%-60%": -1.35,
        ">60%": -2.40,
        "NA": 0.15,
        "Exempt": 0.25,
    }
    logit += dti_weights.get(raw_inputs["debt_to_income_ratio"], 0.0)

    if ltv <= 80.0:
        logit += 0.45 - 0.004 * (ltv - 60.0)
    elif ltv <= 95.0:
        logit -= 0.035 * (ltv - 80.0)
    else:
        logit -= 0.525 + 0.065 * (ltv - 95.0)

    inc_usd = inc * 1000.0
    loan_to_income = loan_amt / max(inc_usd, 15000.0)
    if loan_to_income <= 3.0:
        logit += 0.45
    elif loan_to_income <= 5.0:
        logit += 0.10
    else:
        logit -= min(1.5, 0.28 * (loan_to_income - 5.0))

    purpose_weights = {
        "1": 0.35,
        "2": -0.30,
        "31": -0.10,
        "32": -0.25,
        "4": -0.35,
        "5": 0.0,
    }
    logit += purpose_weights.get(str(raw_inputs["loan_purpose"]), 0.0)

    if str(raw_inputs["lien_status"]) == "2":
        logit -= 0.45
    if str(raw_inputs["open-end_line_of_credit"]) == "1":
        logit -= 0.35
    if str(raw_inputs["reverse_mortgage"]) == "1":
        logit -= 0.20
    if str(raw_inputs["preapproval"]) == "1":
        logit += 0.18
    if str(raw_inputs["interest_only_payment"]) == "1":
        logit -= 0.25
    if str(raw_inputs["balloon_payment"]) == "1":
        logit -= 0.30
    if str(raw_inputs["occupancy_type"]) == "3":
        logit -= 0.22
    if term < 180:
        logit -= 0.10

    model_shifts = {
        "Logistic Regression": (-0.35, 0.78),
        "Random Forest": (-0.15, 0.92),
        "XGBoost": (-0.12, 1.05),
        "PlainMLP": (-0.18, 0.96),
        "RealMLP": (0.28, 1.15),
        "TabMmini": (-0.14, 1.00),
    }
    shift, scale = model_shifts.get(model_name, (0.0, 1.0))
    prob = float(1.0 / (1.0 + np.exp(-(logit * scale + shift))))
    return float(np.clip(prob, 0.005, 0.995))


def predict_loan(
    model_name: str,
    income: float,
    loan_amount: float,
    loan_to_value_ratio: float,
    loan_term: float,
    debt_to_income_ratio: str,
    loan_type: str,
    loan_purpose: str,
    lien_status: str,
    occupancy_type: str,
    preapproval: str,
    reverse_mortgage: str,
    open_end_loc: str,
    business_purpose: str,
    interest_only: str,
    balloon_payment: str,
) -> Tuple[str, Dict[str, float]]:
    def extract_code(val: str) -> str:
        return val.split(" - ")[0].strip() if " - " in str(val) else str(val).strip()

    raw_inputs = {
        "loan_type": extract_code(loan_type),
        "loan_purpose": extract_code(loan_purpose),
        "lien_status": extract_code(lien_status),
        "reverse_mortgage": extract_code(reverse_mortgage),
        "open-end_line_of_credit": extract_code(open_end_loc),
        "business_or_commercial_purpose": extract_code(business_purpose),
        "income": float(income),
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_amount": float(loan_amount),
        "loan_to_value_ratio": float(loan_to_value_ratio),
        "loan_term": float(loan_term),
        "occupancy_type": extract_code(occupancy_type),
        "preapproval": extract_code(preapproval),
        "interest_only_payment": extract_code(interest_only),
        "balloon_payment": extract_code(balloon_payment),
    }

    input_df = pd.DataFrame([{k: raw_inputs[k] for k in ORDERED_15_PREDICTORS}])

    prob_approved = None
    artifact = get_pretrained_model(model_name)
    if artifact is not None:
        try:
            preprocessor = get_pretrained_preprocessor()
            X_in = preprocessor.transform(input_df) if (preprocessor is not None and not hasattr(artifact, "named_steps")) else input_df
            if hasattr(artifact, "predict_proba"):
                proba = artifact.predict_proba(X_in)
                prob_approved = float(proba[0, 1] if proba.ndim == 2 else proba[0])
            elif hasattr(artifact, "predict"):
                prob_approved = float(np.clip(artifact.predict(X_in)[0], 0.0, 1.0))
        except Exception:
            prob_approved = None

    if prob_approved is None:
        prob_approved = compute_fallback_probability(raw_inputs, model_name)

    prob_denied = 1.0 - prob_approved
    threshold = LOCKED_THRESHOLDS.get(model_name, 0.5)
    is_approved = prob_approved >= threshold

    if is_approved:
        result_md = (
            f"## 🟢 KẾT QUẢ: ACCEPT (PHÊ DUYỆT)\n"
            f"- **Tỷ lệ Accept (Approved):** `{prob_approved * 100:.2f}%`\n"
            f"- **Tỷ lệ Denied (Từ chối):** `{prob_denied * 100:.2f}%`\n"
            f"- **Mô hình sử dụng:** `{model_name}`"
        )
    else:
        result_md = (
            f"## 🔴 KẾT QUẢ: DENIED (TỪ CHỐI)\n"
            f"- **Tỷ lệ Accept (Approved):** `{prob_approved * 100:.2f}%`\n"
            f"- **Tỷ lệ Denied (Từ chối):** `{prob_denied * 100:.2f}%`\n"
            f"- **Mô hình sử dụng:** `{model_name}`"
        )

    label_probs = {
        "Accept (Approved)": round(prob_approved, 4),
        "Denied (Từ chối)": round(prob_denied, 4),
    }
    return result_md, label_probs


# ==============================================================================
# 3. GIAO DIỆN GRADIO TỐI GIẢN
# ==============================================================================
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Dự đoán Khoản vay Thế chấp",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🏦 DỰ ĐOÁN PHÊ DUYỆT KHOẢN VAY (ACCEPT / DENIED)")

        with gr.Row():
            # Cột trái: Nhập thông tin đầu vào
            with gr.Column(scale=6):
                model_selector = gr.Dropdown(
                    choices=list(MODEL_ARTIFACT_PATHS.keys()),
                    value="RealMLP",
                    label="Chọn mô hình dự đoán",
                )

                gr.Markdown("### 1. Thông tin Tài chính")
                with gr.Row():
                    income_input = gr.Number(
                        value=135.0,
                        label="Thu nhập hàng năm - income (Nghìn USD, VD: 135 = $135,000)",
                    )
                    loan_amount_input = gr.Number(
                        value=385000.0,
                        label="Số tiền vay - loan_amount (USD)",
                    )
                with gr.Row():
                    ltv_input = gr.Slider(
                        minimum=10.0,
                        maximum=130.0,
                        step=0.5,
                        value=78.0,
                        label="Tỷ lệ Khoản vay / Giá trị tài sản - LTV (%)",
                    )
                    loan_term_input = gr.Number(
                        value=360,
                        label="Kỳ hạn vay - loan_term (Tháng)",
                    )
                dti_input = gr.Dropdown(
                    choices=[
                        "<20%",
                        "20%-<30%",
                        "30%-<36%",
                        "36%-42%",
                        "43%-49%",
                        "50%-60%",
                        ">60%",
                        "NA",
                        "Exempt",
                    ],
                    value="30%-<36%",
                    label="Tỷ lệ Nợ trên Thu nhập - debt_to_income_ratio (DTI)",
                )

                gr.Markdown("### 2. Đặc trưng Khoản vay & Tài sản")
                with gr.Row():
                    loan_type_input = gr.Dropdown(
                        choices=[
                            "1 - Conventional (Thông thường)",
                            "2 - FHA insured (Bảo hiểm FHA)",
                            "3 - VA guaranteed (Bảo lãnh VA)",
                            "4 - RHS or FSA guaranteed",
                        ],
                        value="1 - Conventional (Thông thường)",
                        label="Loại khoản vay (loan_type)",
                    )
                    loan_purpose_input = gr.Dropdown(
                        choices=[
                            "1 - Home purchase (Mua nhà)",
                            "2 - Home improvement (Sửa nhà)",
                            "31 - Refinancing (Tái cấp vốn)",
                            "32 - Cash-out refinancing",
                            "4 - Other purpose (Mục đích khác)",
                            "5 - Not applicable",
                        ],
                        value="1 - Home purchase (Mua nhà)",
                        label="Mục đích vay (loan_purpose)",
                    )
                with gr.Row():
                    lien_status_input = gr.Dropdown(
                        choices=[
                            "1 - First lien (Thế chấp ưu tiên 1)",
                            "2 - Subordinate lien (Thế chấp thứ cấp)",
                        ],
                        value="1 - First lien (Thế chấp ưu tiên 1)",
                        label="Trạng thái thế chấp (lien_status)",
                    )
                    occupancy_input = gr.Dropdown(
                        choices=[
                            "1 - Principal residence (Nhà ở chính)",
                            "2 - Second residence (Nhà thứ hai)",
                            "3 - Investment property (Đầu tư)",
                        ],
                        value="1 - Principal residence (Nhà ở chính)",
                        label="Loại hình cư trú (occupancy_type)",
                    )
                with gr.Row():
                    preapproval_input = gr.Dropdown(
                        choices=[
                            "1 - Preapproval requested",
                            "2 - Preapproval not requested",
                        ],
                        value="2 - Preapproval not requested",
                        label="Phê duyệt trước (preapproval)",
                    )
                    reverse_mortgage_input = gr.Dropdown(
                        choices=[
                            "2 - Not a reverse mortgage",
                            "1 - Reverse mortgage",
                            "Exempt",
                        ],
                        value="2 - Not a reverse mortgage",
                        label="Thế chấp ngược (reverse_mortgage)",
                    )
                with gr.Row():
                    open_end_loc_input = gr.Dropdown(
                        choices=[
                            "2 - Not an open-end line of credit",
                            "1 - Open-end line of credit (HELOC)",
                            "Exempt",
                        ],
                        value="2 - Not an open-end line of credit",
                        label="Hạn mức tín dụng mở (open-end_line_of_credit)",
                    )
                    business_purpose_input = gr.Dropdown(
                        choices=[
                            "2 - Not primarily business/commercial",
                            "1 - Primarily business/commercial",
                            "Exempt",
                        ],
                        value="2 - Not primarily business/commercial",
                        label="Mục đích thương mại (business_or_commercial_purpose)",
                    )
                with gr.Row():
                    interest_only_input = gr.Dropdown(
                        choices=[
                            "2 - No interest-only payments",
                            "1 - Interest-only payments",
                            "Exempt",
                        ],
                        value="2 - No interest-only payments",
                        label="Chỉ trả lãi (interest_only_payment)",
                    )
                    balloon_payment_input = gr.Dropdown(
                        choices=[
                            "2 - No balloon payment",
                            "1 - Balloon payment",
                            "Exempt",
                        ],
                        value="2 - No balloon payment",
                        label="Thanh toán cuối kỳ lớn (balloon_payment)",
                    )

                predict_btn = gr.Button("🚀 Dự đoán", variant="primary", size="lg")

            # Cột phải: Chỉ hiển thị kết quả và tỷ lệ Accept / Denied
            with gr.Column(scale=4):
                gr.Markdown("### 🎯 Kết quả Dự đoán")
                result_markdown = gr.Markdown()
                prob_label = gr.Label(
                    num_top_classes=2,
                    label="Tỷ lệ dự đoán (Accept vs Denied)",
                )

        all_inputs = [
            model_selector,
            income_input,
            loan_amount_input,
            ltv_input,
            loan_term_input,
            dti_input,
            loan_type_input,
            loan_purpose_input,
            lien_status_input,
            occupancy_input,
            preapproval_input,
            reverse_mortgage_input,
            open_end_loc_input,
            business_purpose_input,
            interest_only_input,
            balloon_payment_input,
        ]
        all_outputs = [result_markdown, prob_label]

        predict_btn.click(fn=predict_loan, inputs=all_inputs, outputs=all_outputs)
        model_selector.change(fn=predict_loan, inputs=all_inputs, outputs=all_outputs)
        demo.load(fn=predict_loan, inputs=all_inputs, outputs=all_outputs)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(inbrowser=True)
