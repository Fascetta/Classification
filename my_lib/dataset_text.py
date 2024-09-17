from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from functools import partial


def dataset_from_pandas(df: pd.DataFrame, model_name: str, batched=True, test=False):

    def preprocess(df: pd.DataFrame):
        df["description"] = df["object"] + ": " + df["description"]
        return df

    def get_tokenizer(model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(
        example, tokenizer, content_column="description", label_column="target"
    ):
        out_dict = tokenizer(
            example[content_column], truncation=True, padding=True
        )  # truncation and padding ensure equal lenght sequences
        if not test:
            out_dict["label"] = example[label_column]
        return out_dict

    df = preprocess(df)
    tokenizer = get_tokenizer(model_name)

    return Dataset.from_pandas(df).map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=batched,
        batch_size=len(df),
    )
