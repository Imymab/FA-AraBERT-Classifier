# FA-AraBERT: Arabic First-Aid Query Classifier

This repository provides the code, documentation, and usage examples associated with **FA-AraBERT**, an Arabic binary text classification model designed to detect whether a user query is related to first aid. The classifier constitutes the intent detection and safety filtering component of an MSA First-Aid Chatbot pipeline.

Two models are provided and evaluated in this project: **FA-AraBERTv2** and **FA-AraBERTv0.2**, both fine-tuned from AraBERT base models. These classifiers were systematically compared under multiple training configurations in order to select the most suitable model for deployment.

Importantly, this GitHub repository is **intentionally lightweight**. The trained model weights (≈500 MB) and tokenizer artifacts are hosted on the Hugging Face Model Hub, which is the recommended platform for distributing large pretrained models.

## Overview
- [System architecture](#system-architecture)
- [Model Description](#model-description)
- [Dataset](#dataset)
- [Training](#training)
- [Hugging Face models and chatbot](#hugging-face-models-and-chatbot)
- [Usage](#usage)
- [Files](#files)
- [Reproducibility and extension](#reproducibility-and-extension)
- [Ethical considerations](#ethical-considerations)


## System architecture

<p align=center>
<img width="528" height="275" alt="Chatbot structure" src="https://github.com/user-attachments/assets/cb58f77d-27ab-47e8-b6e7-bb5749499a4c" align="middle" /></p>
<p align=center><i>Overview of the Chatbot System Architecture</i></p>

The FA-AraBERT classifiers operate as the first component of an end-to-end MSA First-Aid Chatbot. A user query is first processed by the FA-AraBERT classifier to determine whether it is related to first aid. If the query is classified as first-aid related, it is forwarded to a large language model responsible for response generation. Otherwise, the system returns a warning indicating that the query falls outside the chatbot’s scope.

Conceptually, the pipeline follows the sequence: User Query → FA-AraBERT Classifier → First-Aid LLM (if applicable) → Generated Response.


## Model Description

* Base model: **AraBERT**
* Task: Binary text classification
* Language: Arabic
* Labels:

<table align="center">
  <tr>
    <th>Label</th>
    <th>FA Case</th>
  </tr>
  <tr>
    <td>LABEL_0</td>
    <td>Non first aid case (NFA)</td>
  </tr>
  <tr>
    <td>LABEL_1</td>
    <td>First aid case (FA)</td>
  </tr>
</table>
<p align=center>Table 1 : Classification labels</p>

The project evaluates two AraBERT-based classifiers :

* [FA-AraBERTv0.2](https://huggingface.co/imaneumabderahmane/Arabertv02-classifier-FA) is built on AraBERTv0.2-base, which processes raw Arabic text directly without morphological pre-segmentation, relying solely on WordPiece tokenization.

* [FA-AraBERTv2](https://huggingface.co/imaneumabderahmane/Arabertv2-classifier-FA) is built on AraBERTv2-base and requires an explicit preprocessing step based on the Farasa segmenter to ensure consistency with its pre-training configuration.

Both models were fine-tuned for binary sequence classification to distinguish first-aid related queries from non–first-aid queries.

## Dataset

The FA-AraBERT classifiers were trained and evaluated on the FALAH-Mix dataset, which contains 1,028 Arabic question–answer pairs. The dataset exhibits a strong class imbalance, with approximately 90% non–first-aid queries and 10% first-aid queries. The data were split into training, development, and test sets while preserving the original class distribution.

To mitigate class imbalance, the training set was augmented with additional first-aid samples from external datasets, including the Mayo Clinic First Aid dataset and the AHD dataset. This resulted in a balanced training set of 1,184 samples, while the development and test sets remained unchanged.

<table border="1" align="center">
  <tr>
    <th>Dataset</th>
    <th>QA pairs</th>
    <th>FA-QA pairs</th>
    <th>%</th>
    <th>NFA-QA pairs</th>
    <th>%</th>
  </tr>
  <tr>
    <td>Training</td>
    <td>1184</td>
    <td>517</td>
    <td>44%</td>
    <td>667</td>
    <td>56%</td>
  </tr>
  <tr>
    <td>Development</td>
    <td>131</td>
    <td>13</td>
    <td>10%</td>
    <td>118</td>
    <td>90%</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>155</td>
    <td>16</td>
    <td>10%</td>
    <td>139</td>
    <td>90%</td>
  </tr>
  <tr>
    <td>All</td>
    <td>1470</td>
    <td>546</td>
    <td>37%</td>
    <td>924</td>
    <td>63%</td>
  </tr>
</table>
<p align=center><i>Table 2 : Distribution of QA pairs after balancing the FALAH-Mix training set</i></p>


## Training

The models were fine-tuned using supervised learning with the following configuration:

* Optimizer: AdamW
* Learning rate: 3 × 10⁻⁵
* Batch size: 16
* Epochs: 3
* Loss function: Cross-entropy
* Mixed-precision training enabled

**Evaluation metrics:**

* **Macro F1-score**, chosen to account for class imbalance
* **BERTScore** for downstream response generation evaluation by the LLM, measures semantic similarity between generated responses and reference answers

Experiments were conducted on [Google Colab](https://colab.research.google.com/drive/1em7S-Gk9AGW4HyLYAHQGBTG7BkN29dYc?usp=sharing) using A100 and T4 GPUs. All models were implemented using the Hugging Face Transformers library.

### Results

Table 3 summarizes the Macro F1 scores obtained by FA-AraBERTv2 and FA-AraBERTv0.2 under different training configurations.

<table border="1" align="center">
  <tr>
    <th>Training configuration</th>
    <th>FA-AraBERTv2</th>
    <th>FA-AraBERTv0.2</th>
  </tr>
  <tr>
    <td>Baseline few-shot</td>
    <td>0.4728</td>
    <td>0.4728</td>
  </tr>
  <tr>
    <td>Balanced training</td>
    <td>0.5588</td>
    <td>0.5578</td>
  </tr>
  <tr>
    <td>Fine-tuning (imbalanced)</td>
    <td>0.5453</td>
    <td>0.5634</td>
  </tr>
  <tr>
    <td>Balanced fine-tuning + class weights</td>
    <td>0.6379</td>
    <td>0.6361</td>
  </tr>
</table>
<p  align="center">Table 3: Macro F1 scores on the FALAH-Mix test set</p>

The best performance was achieved when fine-tuning on the balanced FALAH-Mix training set with class weighting. **FA-AraBERTv2** achieved a Macro F1 score of **0.6379**, slightly outperforming FA-AraBERTv0.2. Due to this consistent advantage, FA-AraBERTv2 was selected for deployment in the final chatbot system.

## Hugging Face Models and Chatbot

The trained models and tokenizer files are available on the Hugging Face Model Hub:

* [FA-AraBERTv2](https://huggingface.co/imaneumabderahmane/Arabertv2-classifier-FA)
* [FA-AraBERTv0.2](https://huggingface.co/imaneumabderahmane/Arabertv02-classifier-FA)

First-Aid Chatbot (Deployed Space)

An interactive Arabic first-aid chatbot integrating FA-AraBERT as the intent detection module:

- [Hugging Face Space](https://huggingface.co/spaces/khaoula1972/Strm-firstaid)
- [GitHub repository](https://github.com/khaoula1972/first-aid-chatbot)
- [Vercel deployment](https://first-aid-chatbot.vercel.app/)

The chatbot routes first-aid queries through FA-AraBERT for classification. If the classifier misclassifies a query, the Mistral LLM acts as a “parent” classifier to reclassify or correct the query before generating a response. This ensures higher accuracy and reduces false positives/negatives in the pipeline.

## Usage

Both models can be loaded using the Transformers library by specifying the corresponding Hugging Face identifier.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "imaneumabderahmane/Arabertv2-classifier-FA"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "ما هي الإسعافات الأولية لحروق الدرجة الأولى؟"
inputs = tokenizer(text, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

print(prediction)
```
Label mappings are defined in the model configuration file on Hugging Face.

## Files

This repository contains only lightweight files required for documentation and reproducibility:

- README.md: Project documentation and usage instructions.

- requirements.txt: Python dependencies required for inference and evaluation.

- inference.py: Example script demonstrating how to load and run the classifier.

Optional files may include evaluation scripts or notebooks illustrating experimental results.

### Files not included

The following files are intentionally not included in this GitHub repository due to their size and are instead hosted on Hugging Face:

- model.safetensors
- config.json
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json
- vocab.txt

Hosting these artifacts on Hugging Face ensures efficient distribution, versioning, and compatibility with the Transformers ecosystem.

## Reproducibility and extension

This repository is intended to support reproducibility and further research. Users may fine-tune the model on additional data, evaluate it on new benchmarks, or integrate it into larger dialogue or triage systems. For full experimental details, including dataset composition, training configuration, and evaluation metrics, please refer to the associated thesis or technical report.

## Ethical considerations

Although these models do not generate medical advice, incorrect classification may lead to inappropriate downstream handling of user queries. The FA-AraBERT classifiers perform intent classification only and do not provide medical advice, assess urgency, or guarantee correctness. According to our testing misclassifications may occur, so for real-world deployment, these models should be integrated into a broader safety-aware framework with additional validation mechanisms and, where appropriate, human oversight.

**Disclaimer: This model is intended for research and educational purposes only.**

**It is not a medical device and must not be used as a substitute for professional medical advice or emergency services.**
