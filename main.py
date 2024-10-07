#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 07:26:44 2024

@author: macbookpro13zoll
"""

import pandas as pd
import random

# Define categories of risks
risk_categories = {
    0: "Financial Risk",
    1: "Operational Risk",
    2: "Regulatory Risk",
    3: "Reputational and legal Risk",
    4: "Cybersecurity Risk",
    5: "There is no Risk"
}

# Sample text descriptions for each risk category
risk_descriptions = {
    0: [
        "The company's liquidity has deteriorated due to decreasing cash flows.",
        "High debt levels have raised concerns about a potential default.",
        "exchange rates and inflations",
        "Falling stock prices suggest potential insolvency issues."
    ],
    1: [
        "The manufacturing unit experienced a major disruption due to equipment failure.",
        "Supply chain issues have delayed product deliveries by weeks.",
        "instability of business caused by wars or conflicts in a region",
        "A shortage of skilled labor is hampering daily operations."
    ],
    2: [
        "The company is facing penalties for non-compliance with environmental regulations.",
        "New tax regulations could lead to increased financial burden.",
        "staff non-complaince to company's polies and processes",
        "An investigation by the antitrust authority is looming over the company."
    ],
    3: [
        "Recent negative media coverage has tarnished the company's public image.",
        "Consumer complaints about product quality are growing.",
        "Consumers take legal actions against company over effect of products.",
        "The CEO's involvement in a scandal has damaged the company's reputation."
    ],
    4: [
        "A recent phishing attack compromised sensitive customer data.",
        "holes that poses vulnerability in software or hardware programs",
        "Outdated security protocols exposed the company to a potential ransomware attack.",
        "Hackers gained unauthorized access to the company's financial systems."
    ],
    5: [
        "The Company is very Liquid and Profitable and there is general company growth.",
        "there is smooth running of the companies business without shortage of any form of resources.",
        "the company is compliant and there is no form of investigation or Scandal.",
        "The Company's Cybersecurity is Strong and there is perfect protection of database."
    ]
}

# Generate synthetic data
data = []
num_samples = 350  # Number of data points to generate

for _ in range(num_samples):
    label = random.choice(list(risk_categories.keys()))  # Randomly choose a risk category
    description = random.choice(risk_descriptions[label])  # Randomly choose a description
    data.append([description, label])
    
# Create a DataFrame
df = pd.DataFrame(data, columns=['text', 'label'])

# Save to CSV file
df.to_csv('company_risk_data.csv', index=False)

print("Synthetic company risk data generated and saved to 'company_risk_data.csv'.")

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate 
from sklearn.model_selection import train_test_split

dataset = load_dataset('csv', data_files='company_risk_data.csv')
dataset = dataset['train'].train_test_split(test_size=0.2)  # Split into training and test sets
print(dataset["train"][:100])
# 2. Preprocess the Data (Tokenization)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Define Labels and Set Format
label_names = dataset['train'].features['label']  # Assuming you have pre-defined labels
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")  # Rename to 'labels' as required by Trainer
tokenized_datasets.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

# 4. Load Pre-Trained Model (BERT) for Classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels= 6)

# 5. Define Metrics for Evaluation
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.tensor(predictions)
   
   # Apply argmax to get predicted classes
    predictions = torch.argmax(predictions, dim=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    return {"accuracy": accuracy['accuracy'], "f1": f1['f1']}

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load the model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("./roberta-risk-classifier")
tokenizer = RobertaTokenizer.from_pretrained("./roberta-risk-classifier")

app = FastAPI()

class RiskInput(BaseModel):
    text: str

@app.post("/predict")
def predict_risk(input_data: RiskInput):
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Ensure logits is a torch tensor before applying argmax
    predicted_class = torch.argmax(logits, dim=1).item()  # Get the predicted class
    return {"predicted_class": predicted_class}


# 6. Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,     
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics
)

# 8. Fine-Tune the Model
trainer.train()
                         
# Save the fine-tuned model
model.save_pretrained("./roberta-risk-classifier")
tokenizer.save_pretrained("./roberta-risk-classifier")

# 9. Evaluate the Model
trainer.evaluate()
# Save the model
model.save_pretrained("/Users/macbookpro13zoll")
tokenizer.save_pretrained("/Users/macbookpro13zoll")
 
model.eval()
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('./Users/macbookpro13zoll')
model = RobertaForSequenceClassification.from_pretrained('./roberta-risk-classifier')

joblib.dump(model, 'model.pkl')
# Set the model to evaluation mode
model.eval()

# Sample text input
text = "This is a test sentence for classification."

# Preprocess input: Tokenization + conversion to input IDs
inputs = tokenizer(text, return_tensors="pt")

# Disable gradient computation during inference
with torch.no_grad():
    outputs = model(**inputs)

# Get logits (raw output) and convert to probabilities using softmax
logits = outputs.logits
probabilities = torch.softmax(logits, dim=1)

# Get predicted class
predicted_class = torch.argmax(probabilities, dim=1)
print(f"Predicted class: {predicted_class.item()}")


model.eval()
import os


#

# Define category labels (adjust according to your classification problem)
category_labels = {
   0: "Financial Risk",
   1: "Operational Risk",
   2: "Regulatory Risk",
   3: "Reputational and legal Risk",
   4: "Cybersecurity Risk",
   5: "No Risk"
}

def prepare_input(text):
    """Preprocess and tokenize the input text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

def predict(text):
    """Predict the risk category for the given text."""
    # Prepare the input
    inputs = prepare_input(text)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the logits (model outputs)
    logits = outputs.logits

    # Debugging: print logits
    print(f"Logits: {logits}")
    
    # Get the predicted class index
    predicted_class_value = torch.argmax(logits, dim=1).item()
    
    # Debugging: print predicted class index
    print(f"Predicted class index: {predicted_class_value}")
    
    # Ensure the predicted class index is valid
    if predicted_class_value in category_labels:
        predicted_category = category_labels[predicted_class_value]
    else:
        predicted_category = "Unknown Category"  # In case of an unexpected index
    
    return predicted_category

# Example usage:
 text = "company is owing the bank"
predicted_category = predict(text)
print(f"Predicted Risk Category: {predicted_category}")
# Get the current working directory
print(os.getcwd())
model.save_pretrained("./Users/macbookpro13zoll")
tokenizer.save_pretrained("./Users/macbookpro13zoll")
import joblib
joblib.dump(model, 'model.pkl')
