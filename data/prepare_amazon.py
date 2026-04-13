"""
Prepare Amazon Reviews 2023 data for campaign auto-targeting experiments.

Downloads the specified Amazon Reviews 2023 category, computes item and user
embeddings, and saves them as CSVs in the format expected by the algorithms.

Item embeddings: pre-computed text embeddings from product metadata
                 (768-dim, from the Amazon Reviews 2023 release by Hou et al. 2024).
User embeddings: mean of item embeddings for all items a user has reviewed.

Usage
-----
    python data/prepare_amazon.py --category Musical_Instruments \
        --output_dir data/musical_instruments

    python data/prepare_amazon.py --category Baby_Products \
        --output_dir data/baby_products --core_filter 5

Supported categories (from Amazon Reviews 2023):
    Musical_Instruments, Baby_Products, Sports_and_Outdoors,
    Electronics, Clothing_Shoes_and_Jewelry, ...
    Full list: https://amazon-reviews-2023.github.io/

Output files
------------
    <output_dir>/amazon_embeddings.csv       — items (id, title, embedding)
    <output_dir>/amazon_embeddings_users.csv — users (id, user_embedding)

Requirements
------------
    pip install datasets huggingface_hub
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


HF_DATASET_REPO = "McAuley-Lab/Amazon-Reviews-2023"


def load_item_embeddings(category: str) -> pd.DataFrame:
    """
    Load item metadata with pre-computed text embeddings for a given category.
    Returns a DataFrame with columns [parent_asin, title, embedding].
    """
    from datasets import load_dataset

    print(f"Loading item metadata for '{category}' ...")
    ds = load_dataset(
        HF_DATASET_REPO,
        f"raw_meta_{category}",
        split="full",
        trust_remote_code=True,
    )
    df = ds.to_pandas()

    # The dataset provides 'text_embeddings' (list of floats, 768-dim)
    if "text_embeddings" not in df.columns:
        raise ValueError(
            f"Column 'text_embeddings' not found. Available: {list(df.columns)}"
        )

    df = df[["parent_asin", "title", "text_embeddings"]].dropna(subset=["text_embeddings"])
    df = df.rename(columns={"text_embeddings": "embedding"})

    # Serialize embedding lists to JSON strings (format expected by load_data)
    df["embedding"] = df["embedding"].apply(
        lambda v: json.dumps(list(v)) if not isinstance(v, str) else v
    )
    df = df.rename(columns={"parent_asin": "id"})
    print(f"  Loaded {len(df):,} items with embeddings.")
    return df


def load_reviews(category: str) -> pd.DataFrame:
    """Load user-item review interactions (user_id, parent_asin)."""
    from datasets import load_dataset

    print(f"Loading reviews for '{category}' ...")
    ds = load_dataset(
        HF_DATASET_REPO,
        f"raw_review_{category}",
        split="full",
        trust_remote_code=True,
    )
    df = ds.to_pandas()[["user_id", "parent_asin"]].dropna()
    print(f"  Loaded {len(df):,} reviews.")
    return df


def apply_core_filter(reviews: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively remove users and items with fewer than k interactions."""
    print(f"Applying {k}-core filter ...")
    prev_len = -1
    while prev_len != len(reviews):
        prev_len = len(reviews)
        item_counts = reviews["parent_asin"].value_counts()
        reviews = reviews[reviews["parent_asin"].isin(item_counts[item_counts >= k].index)]
        user_counts = reviews["user_id"].value_counts()
        reviews = reviews[reviews["user_id"].isin(user_counts[user_counts >= k].index)]
    print(f"  Retained {reviews['user_id'].nunique():,} users and "
          f"{reviews['parent_asin'].nunique():,} items after {k}-core filter.")
    return reviews


def compute_user_embeddings(reviews: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user embeddings as the mean of item embeddings for reviewed items.

    Only users who reviewed at least one item with a known embedding are kept.
    """
    print("Computing user embeddings (mean of reviewed item embeddings) ...")

    # Build item_id -> embedding array lookup
    item_emb_map = {}
    for _, row in item_df.iterrows():
        try:
            vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
            item_emb_map[row["id"]] = vec
        except Exception:
            continue

    # Filter reviews to items with known embeddings
    reviews = reviews[reviews["parent_asin"].isin(item_emb_map)]

    user_records = []
    for user_id, group in reviews.groupby("user_id"):
        vecs = [item_emb_map[asin] for asin in group["parent_asin"] if asin in item_emb_map]
        if vecs:
            user_emb = np.mean(vecs, axis=0)
            user_records.append({
                "id": user_id,
                "user_embedding": json.dumps(user_emb.tolist()),
            })

    users_df = pd.DataFrame(user_records)
    print(f"  Computed embeddings for {len(users_df):,} users.")
    return users_df


def subsample(df: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """Randomly subsample a fraction of rows."""
    return df.sample(frac=fraction, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare Amazon Reviews 2023 embeddings.")
    parser.add_argument(
        "--category",
        type=str,
        default="Musical_Instruments",
        help="Amazon Reviews 2023 category name (e.g. Musical_Instruments, Baby_Products).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/musical_instruments",
        help="Directory to write output CSV files.",
    )
    parser.add_argument(
        "--core_filter",
        type=int,
        default=0,
        help="Apply k-core filtering (0 = disabled). Use 5 for Baby Products.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of interactions to keep (e.g. 0.7 for Musical Instruments).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Item embeddings
    item_df = load_item_embeddings(args.category)

    # 2. Reviews
    reviews = load_reviews(args.category)

    # 3. Subsample (Musical Instruments uses 70%)
    if args.subsample < 1.0:
        n_before = len(reviews)
        reviews = subsample(reviews, args.subsample, seed=args.seed)
        print(f"Subsampled to {len(reviews):,} reviews (from {n_before:,}, "
              f"fraction={args.subsample}).")

    # 4. Core filter
    if args.core_filter > 0:
        reviews = apply_core_filter(reviews, args.core_filter)

    # 5. Keep only items that appear in reviews
    active_item_ids = set(reviews["parent_asin"].unique())
    item_df = item_df[item_df["id"].isin(active_item_ids)].reset_index(drop=True)
    print(f"Retaining {len(item_df):,} items present in reviews.")

    # 6. User embeddings
    users_df = compute_user_embeddings(reviews, item_df)

    # 7. Save
    items_path = os.path.join(args.output_dir, "amazon_embeddings.csv")
    users_path = os.path.join(args.output_dir, "amazon_embeddings_users.csv")
    item_df.to_csv(items_path, index=False)
    users_df.to_csv(users_path, index=False)

    print(f"\nSaved:")
    print(f"  Items : {items_path}  ({len(item_df):,} rows)")
    print(f"  Users : {users_path}  ({len(users_df):,} rows)")
    print(f"\nEmbedding dimension: "
          f"{len(json.loads(item_df['embedding'].iloc[0]))}")


if __name__ == "__main__":
    main()
