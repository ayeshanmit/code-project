# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 11:42:19 2025

@author: user
"""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import numpy as np

class ABSADataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model(train_texts, train_labels):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    train_dataset = ABSADataset(train_texts, train_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir="./models",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="no",
        logging_dir="./logs",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    return model, tokenizer

def evaluate_model(model, tokenizer, test_texts, test_labels):
    test_dataset = ABSADataset(test_texts, test_labels, tokenizer)

    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=["Negative", "Neutral", "Positive"]))
