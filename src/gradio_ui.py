import gradio as gr
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    "Logistic Regression": os.path.join(BASE_DIR, "models", "log_model.pkl"),
    "SVM": os.path.join(BASE_DIR, "models", "svm_model.pkl"),
}

def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        models[name] = joblib.load(path)
    return models

models = load_models()

def predict_class(subject: str, body: str, model_name: str) -> str:
    if not body.strip():
        return "EMAIL BODY CANNOT BE EMPTY"

    subject = subject.lower() if subject else "no_subject"
    body = body.lower()

    full_text = f"{subject} {body}"

    model = models[model_name]
    pred = model.predict([full_text])[0]

    return "Spam" if int(pred) == 1 else "Ham"

interface = gr.Interface(
    fn=predict_class,
    inputs=[
        gr.Textbox(label="Subject", lines=1),
        gr.Textbox(label="Body", lines=5),
        gr.Radio(
            choices=list(MODEL_PATHS.keys()),
            value="Logistic Regression",
            label="Choose Model"
        ),
    ],
    outputs="text",
    title="Email Spam Classifier",
    description="This model is trained on the Enron Email Spam Classification dataset."
)

if __name__ == "__main__":
    interface.launch()
