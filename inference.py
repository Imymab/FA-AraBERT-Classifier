import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

def load_model_and_tokenizer(model_name: str):
    """
    Load a pretrained FA-AraBERT classifier and its tokenizer from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  
    return tokenizer, model

def predict(text: str, tokenizer, model):
    """
    Run inference on a single text input and return the predicted label.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_class

def main():
    parser = argparse.ArgumentParser(description="FA-AraBERT Arabic First-Aid Query Classifier")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Hugging Face model identifier for FA-AraBERT (e.g., imaneuabderahmane/Arabertv2-classifier-FA)")
    parser.add_argument("--text", type=str, required=True, help="Arabic query text to classify")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_name)
    prediction = predict(args.text, tokenizer, model)

    print(f"Input text: {args.text}")
    print(f"Predicted label: {prediction} (0 = Non-FA, 1 = FA)")

if __name__ == "__main__":
    main()
