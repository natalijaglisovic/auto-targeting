"""
End-to-end campaign auto-targeting experiment.

Loads user and item embeddings from CSV, runs biclustering and/or bandit
optimization, and prints per-campaign quality, lift, and Gini metrics.

Usage
-----
    python examples/run_experiment.py \
        --users  data/musical_instruments/amazon_embeddings_users.csv \
        --items  data/musical_instruments/amazon_embeddings.csv \
        --n_campaigns 5 \
        --max_users   10000 \
        --min_users   3000 \
        --items_per_ad 5 \
        --method both

    # Bandit only, Thompson Sampling, 30 rounds
    python examples/run_experiment.py \
        --users  data/baby_products/amazon_embeddings_users.csv \
        --items  data/baby_products/amazon_embeddings.csv \
        --n_campaigns 5 --min_users 5000 --max_users 20000 \
        --method bandit --bandit_strategy thompson --n_rounds 30

    # Synthetic data (no CSV required)
    python examples/run_experiment.py --synthetic
"""

import argparse
import sys
import os
import time

import numpy as np
import pandas as pd

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.data import load_data
from algorithm_biclustering import ConstrainedSpectralBicluster, gini_coefficient, get_bicluster_score
from algorithm_bandit import BanditCampaignOptimizer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(affinity, user_assignments, item_assignments, n_campaigns):
    """
    Compute quality, utility, lift, and Gini from flat assignment arrays.

    Args:
        affinity:          (n_items, n_users) affinity matrix
        user_assignments:  (n_users,) array of campaign indices (TOMBSTONE = n_campaigns)
        item_assignments:  (n_items,) array of campaign indices
        n_campaigns:       number of campaigns

    Returns:
        dict with keys: quality, utility, lift, gini
    """
    global_mean = np.mean(affinity)
    total_quality = 0.0
    total_utility = 0.0
    valid = 0
    per_user_utils = np.zeros(affinity.shape[1])

    for k in range(n_campaigns):
        u_idx = np.where(user_assignments == k)[0]
        i_idx = np.where(item_assignments == k)[0]
        if len(u_idx) == 0 or len(i_idx) == 0:
            continue
        sub = affinity[np.ix_(i_idx, u_idx)]
        quality = np.mean(sub)
        total_quality += quality
        total_utility += np.sum(sub)
        per_user_utils[u_idx] = np.mean(sub, axis=0)
        valid += 1

    avg_quality = total_quality / valid if valid > 0 else 0.0
    avg_lift = avg_quality / global_mean if global_mean > 0 else 0.0
    assigned = user_assignments != n_campaigns
    gini = gini_coefficient(per_user_utils[assigned])

    return {
        "quality": total_quality,
        "utility": total_utility,
        "lift": avg_lift,
        "gini": gini,
    }


def print_metrics(name, metrics, elapsed):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Quality (sum of means): {metrics['quality']:.4f}")
    print(f"  Utility (total sum):    {metrics['utility']:.2f}")
    print(f"  Avg Lift:               {metrics['lift']:.2f}x")
    print(f"  Gini coefficient:       {metrics['gini']:.4f}")
    print(f"  Runtime:                {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Biclustering
# ---------------------------------------------------------------------------

def run_biclustering(affinity, n_campaigns, min_users, max_users, items_per_ad,
                     search_alpha=3):
    print("\n--- Running Constrained Spectral Biclustering ---")
    t0 = time.time()

    model = ConstrainedSpectralBicluster()
    model.fit(
        affinity,
        n_clusters=n_campaigns,
        min_items=1,
        min_users=min_users,
        max_items=items_per_ad,
        max_users=max_users,
        search_alpha=search_alpha,
    )

    elapsed = time.time() - t0

    n_items, n_users = affinity.shape
    user_assignments = np.full(n_users, n_campaigns, dtype=int)
    item_assignments = np.full(n_items, n_campaigns, dtype=int)

    for k, (row_idx, col_idx) in enumerate(model.biclusters):
        item_assignments[row_idx] = k
        user_assignments[col_idx] = k

    metrics = compute_metrics(affinity, user_assignments, item_assignments, n_campaigns)
    print_metrics("Biclustering", metrics, elapsed)
    return metrics


# ---------------------------------------------------------------------------
# Bandit
# ---------------------------------------------------------------------------

def run_bandit(user_matrix, item_matrix, item_biases, n_campaigns, min_users,
               max_users, items_per_ad, strategy="thompson", n_rounds=50,
               ucb_c=2.0, stats_window=10):
    print(f"\n--- Running Bandit Optimizer (strategy={strategy}) ---")
    t0 = time.time()

    ad_configs = [
        {"id": k, "min_users": min_users, "max_users": max_users}
        for k in range(n_campaigns)
    ]

    optimizer = BanditCampaignOptimizer(
        user_embeds=user_matrix,
        item_embeds=item_matrix,
        item_biases=item_biases,
        ad_configs=ad_configs,
        items_per_ad=items_per_ad,
    )

    best_users, best_items, best_score = optimizer.solve(
        n_rounds=n_rounds,
        strategy=strategy,
        ucb_c=ucb_c,
        stats_window=stats_window,
    )

    elapsed = time.time() - t0

    affinity = optimizer.affinity_matrix  # (n_users, n_items)
    # Bandit stores affinity as (n_users, n_items); transpose to (n_items, n_users)
    affinity_T = affinity.T
    metrics = compute_metrics(affinity_T, best_users, best_items, n_campaigns)
    print_metrics(f"Bandit ({strategy})", metrics, elapsed)
    return metrics


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

def run_synthetic_demo():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from data.gauss_l2 import generate_gauss_data, print_report

    print("\n=== Synthetic data demo ===")
    item_emb, user_emb, affinity, item_ids, user_ids, baseline, remaining = generate_gauss_data(
        items=[5, 5, 5],
        users=[200, 200, 200],
        num_items=500,
        num_users=5000,
        emb_dim=2,
        seed=42,
    )

    print("\nBaseline (ground-truth campaign structure):")
    print_report(baseline, affinity, item_ids, user_ids)

    print("\nRunning biclustering on synthetic data ...")
    t0 = time.time()
    model = ConstrainedSpectralBicluster()
    model.fit(affinity, n_clusters=3, min_items=1, min_users=10,
              max_items=5, max_users=300, search_alpha=3)
    print(f"Biclustering done in {time.time()-t0:.1f}s, "
          f"found {len(model.biclusters)} campaigns.")
    for k, (row_idx, col_idx) in enumerate(model.biclusters):
        q = np.mean(affinity[np.ix_(row_idx, col_idx)])
        print(f"  Campaign {k}: {len(row_idx)} items x {len(col_idx)} users, quality={q:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run campaign auto-targeting experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--users",       type=str, help="Path to users CSV.")
    parser.add_argument("--items",       type=str, help="Path to items CSV.")
    parser.add_argument("--n_campaigns", type=int, default=5,   help="Number of campaigns.")
    parser.add_argument("--min_users",   type=int, default=100,  help="Min users per campaign.")
    parser.add_argument("--max_users",   type=int, default=10000, help="Max users per campaign.")
    parser.add_argument("--items_per_ad",type=int, default=5,   help="Items per campaign.")
    parser.add_argument("--method",      type=str, default="both",
                        choices=["biclustering", "bandit", "both"],
                        help="Which method(s) to run.")
    parser.add_argument("--bandit_strategy", type=str, default="thompson",
                        choices=["thompson", "ucb"],
                        help="Bandit exploration strategy.")
    parser.add_argument("--n_rounds",    type=int, default=50, help="Bandit optimization rounds.")
    parser.add_argument("--ucb_c",       type=float, default=2.0, help="UCB exploration constant.")
    parser.add_argument("--stats_window",type=int, default=10, help="Bandit rolling window size.")
    parser.add_argument("--search_alpha",type=int, default=3,
                        help="Biclustering over-partitioning factor.")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Run a synthetic data demo instead of loading CSVs.")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_demo()
        return

    if not args.users or not args.items:
        parser.error("--users and --items are required unless --synthetic is set.")

    # Load data
    print(f"Loading data from:\n  users: {args.users}\n  items: {args.items}")
    users_df = pd.read_csv(args.users)
    items_df = pd.read_csv(args.items)
    user_matrix, item_matrix, item_biases = load_data(users_df, items_df)

    print(f"\nLoaded {len(user_matrix):,} users and {len(item_matrix):,} items "
          f"(dim={user_matrix.shape[1]})")

    # Affinity matrix: A = exp(I @ U^T), shape (n_items, n_users)
    print("Computing affinity matrix ...")
    affinity = np.exp(item_matrix @ user_matrix.T)

    results = {}

    if args.method in ("biclustering", "both"):
        results["biclustering"] = run_biclustering(
            affinity,
            n_campaigns=args.n_campaigns,
            min_users=args.min_users,
            max_users=args.max_users,
            items_per_ad=args.items_per_ad,
            search_alpha=args.search_alpha,
        )

    if args.method in ("bandit", "both"):
        results[f"bandit_{args.bandit_strategy}"] = run_bandit(
            user_matrix, item_matrix, item_biases,
            n_campaigns=args.n_campaigns,
            min_users=args.min_users,
            max_users=args.max_users,
            items_per_ad=args.items_per_ad,
            strategy=args.bandit_strategy,
            n_rounds=args.n_rounds,
            ucb_c=args.ucb_c,
            stats_window=args.stats_window,
        )

    if len(results) > 1:
        print("\n=== Summary ===")
        print(f"{'Method':<25} {'Quality':>10} {'Lift':>8} {'Gini':>8}")
        print("-" * 55)
        for method, m in results.items():
            print(f"{method:<25} {m['quality']:>10.4f} {m['lift']:>8.2f}x {m['gini']:>8.4f}")


if __name__ == "__main__":
    main()
