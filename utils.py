# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 11:41:14 2025

@author: user
"""

import pandas as pd
from datasets import load_dataset

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

import pandas as pd


def load_local_absa_csv(domain):
    path = f"data/{domain}_reviews.csv"  # e.g., laptop_reviews.csv
    df = pd.read_csv(path)

    df = df.dropna(subset=["sentence", "aspect", "polarity"])
    df["input"] = df["aspect"] + " - " + df["sentence"]
    df["label_id"] = df["polarity"].map(label_map)

    return df


def flatten_absa_dataset(dataset_split):
    records = []
    for entry in dataset_split:
        for aspect in entry['aspects']:
            records.append({
                'text': entry['text'],
                'aspect': aspect['term'],
                'label': aspect['polarity']
            })
    return pd.DataFrame(records)

def load_hf_dataset(domain):
    """Loads 'laptop' or 'restaurant' from huggingface."""
    ds = load_dataset("jakartaresearch/semeval-absa", domain, trust_remote_code=True, download_mode="force_redownload")
    train_df = flatten_absa_dataset(ds["train"])
    test_df = flatten_absa_dataset(ds["test"])

    train_df["input"] = train_df.apply(lambda row: f"{row['aspect']} - {row['text']}", axis=1)
    test_df["input"] = test_df.apply(lambda row: f"{row['aspect']} - {row['text']}", axis=1)

    train_df["label_id"] = train_df["label"].map(label_map)
    test_df["label_id"] = test_df["label"].map(label_map)

    return train_df, test_df
