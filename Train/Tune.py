from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from multiprocessing import cpu_count
from tqdm.auto import tqdm


def analyze_token_lengths(
    ds,
    text_columns: Union[str, List[str]],
    model_name: str,
    percentile: float = 99.0,
    add_special_tokens: bool = True,
    visualize: bool = True,
    bins: int = 50,
    num_proc: int = None,
    batch_size: int = 1000,
):
    """
    Multi-core CPU token length analyzer (HF Dataset version).
    """

    if isinstance(text_columns, str):
        text_columns = [text_columns]

    if num_proc is None:
        num_proc = max(cpu_count() - 1, 1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print(f"\nUsing {num_proc} CPU cores")
    print("Tokenizing dataset in parallel...\n")

    def tokenize_batch(batch):
        texts = [
            " ".join(str(batch[col][i]) for col in text_columns if col in batch)
            for i in range(len(batch[text_columns[0]]))
        ]

        tokens = tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        return {"token_length": [len(ids) for ids in tokens["input_ids"]]}

    ds = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    token_lengths = np.array(ds["token_length"])

    # Stats
    pctl_value = int(np.percentile(token_lengths, percentile))
    max_len = int(token_lengths.max())
    mean_len = float(token_lengths.mean())
    median_len = int(np.median(token_lengths))

    # ---------------- PRINT SUMMARY ----------------
    print("\n========== Token Length Statistics ==========")
    print(f"Model: {model_name}")
    print(f"Samples: {len(token_lengths)}")
    print(f"Mean length: {mean_len:.2f}")
    print(f"Median length: {median_len}")
    print(f"{percentile}th percentile: {pctl_value}")
    print(f"Max length: {max_len}")
    print(f"Recommended max_seq_length: {pctl_value}")
    print("==============================================\n")

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.hist(token_lengths, bins=bins)
        plt.axvline(pctl_value, linestyle='--', linewidth=2,
                    label=f"p{int(percentile)}")
        plt.axvline(max_len, linestyle='--', linewidth=2,
                    label="max")
        plt.title("Token Length Distribution")
        plt.xlabel("Token Length")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "mean": mean_len,
        "median": median_len,
        "pctl": pctl_value,
        "max": max_len,
        "recommended_max_length": pctl_value,
    }
