import gradio as gr
import os
import joblib
import pandas as pd
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "stacking_model.pkl")

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
REPLY_PATTERN = re.compile(r'\b(re:|reply|respond)\b', flags=re.IGNORECASE)
FORWARD_PATTERN = re.compile(r'\b(fwd:|fw:|forward)\b', flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r'<[^>]+>')


def _safe_mean_word_length(text: str) -> float:
    words = text.split()
    return float(np.mean([len(word) for word in words])) if words else 0.0


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()


def build_features(subject: str, body: str) -> dict:
    """Build all engineered features for model prediction."""
    subject_text = subject.strip() if subject else "no_subject"
    body_text = body.strip() if body else ""
    text = f"{subject_text} {body_text}".strip()

    uppercase_count = sum(ch.isupper() for ch in f"{subject_text}{body_text}")
    total_length = max(len(text), 1)

    return {
        "Text": text,
        "subject_len": len(subject_text),
        "body_len": len(body_text),
        "text_len": len(text),
        "subject_word_count": len(subject_text.split()),
        "body_word_count": len(body_text.split()),
        "text_word_count": len(text.split()),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "url_count": len(URL_PATTERN.findall(text)),
        "digit_count": sum(ch.isdigit() for ch in text),
        "uppercase_ratio": uppercase_count / total_length,
        "contains_reply": int(bool(REPLY_PATTERN.search(subject_text))),
        "contains_forward": int(bool(FORWARD_PATTERN.search(subject_text))),
        "has_html": int(bool(HTML_PATTERN.search(text))),
        "mean_word_len": _safe_mean_word_length(body_text),
    }


def predict_class(subject: str, body: str) -> str:
    if not body.strip():
        return "❌ EMAIL BODY CANNOT BE EMPTY"

    subject_text = subject.strip() if subject else ""
    body_text = body.strip()

    # Use stacking classifier for inference
    feature_row = build_features(subject_text, body_text)
    pred = model.predict(pd.DataFrame([feature_row]))[0]

    result = "🚨 Spam" if int(pred) == 1 else "✅ Ham"
    return result

interface = gr.Interface(
    fn=predict_class,
    inputs=[
        gr.Textbox(
            label="📧 Email Subject",
            lines=1,
            placeholder="Enter email subject...",
        ),
        gr.Textbox(
            label="📝 Email Body",
            lines=6,
            placeholder="Enter email body...",
        ),
    ],
    outputs=gr.Textbox(
        label="🔍 Classification Result",
        interactive=False,
    ),
    title="🛡️ Email Spam Classifier",
    description="Powered by Stacking Classifier - trained on Enron Email Dataset\n\nThis tool analyzes emails and classifies them as either legitimate (Ham) or spam based on advanced machine learning techniques.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    interface.launch()
