# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 11:42:41 2025

@author: user
"""

from utils import load_local_absa_csv
from sentiment_classifier import train_model, evaluate_model

if __name__ == "__main__":
    for domain in ["laptop", "restaurant"]:
        print(f"\n========== DOMAIN: {domain.upper()} ==========")
        df = load_local_absa_csv(domain)

        # 80/20 split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        model, tokenizer = train_model(train_df["input"].tolist(), train_df["label_id"].tolist())
        evaluate_model(model, tokenizer, test_df["input"].tolist(), test_df["label_id"].tolist())
