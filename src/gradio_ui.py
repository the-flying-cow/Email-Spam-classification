import gradio as gr
from main import main
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH= os.path.join(BASE_DIR, 'models', 'log_model.pkl')
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return main()

model= load_model()

def predict_class(Subject: str, Body: str) -> str:
    if Subject=="":
        Subject= 'no_subject'
    else:
        Subject= Subject.lower()

    if Body=="":
        return "EMAIL BODY CANNOT BE EMPTY"
    else:
        Body= Body.lower()
    full_text= f"{Subject} {Body}"
    pred= model.predict([full_text])[0]
    return "Spam" if int(pred) == 1 else "Ham"

input_to_model= [
    gr.Textbox(label= 'Subject', lines= 1),
    gr.Textbox(label= 'Body', lines=5)
    ]

with gr.Blocks() as demo:
    interface= gr.Interface(fn= predict_class, inputs= input_to_model, outputs= 'text',
             title= 'Email Spam Classifier', description= 'This model is trained on the Enron Email Spam Classification dataset. <3')

if __name__=='__main__':
    demo.launch()
