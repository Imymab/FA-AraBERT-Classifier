import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Hugging Face model name
MODEL_NAME = "imaneumabderahmane/Arabertv2-classifier-FA"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def predict_first_aid(query):
    """
    Classify an Arabic query as FIRST_AID or NOT_FIRST_AID.
    """
    if not query.strip():
        return "Please enter a valid Arabic query."

    inputs = tokenizer(query, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]

    return predicted_label


# Create Gradio interface
interface = gr.Interface(
    fn=predict_first_aid,
    inputs=gr.Textbox(label="Enter your Arabic query"),
    outputs=gr.Textbox(label="Prediction"),
    title="Arabic First-Aid Query Classifier (FA-AraBERT)",
    description="This demo classifies whether an Arabic query is related to first aid or not."
)

# Launch app
if __name__ == "__main__":
    interface.launch()
