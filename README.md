# Auto-Targeting: Constrained User–Item Allocation for E-Commerce Marketing Campaigns

This repository contains the code for the paper:

> **Constrained user–item allocation for e-commerce marketing campaigns**

## Overview

We formalize the task of **jointly selecting users and items to form marketing campaigns** from learned embeddings, without any predefined campaign structure. Given user and item embeddings in a shared space, the goal is to partition them into campaigns where each campaign groups users and items with high mutual affinity.

This is a combinatorial optimization problem (see Eq. 1 in the paper). We propose two complementary approaches:

| Method | Description | Strengths |
|---|---|---|
| **Constrained Spectral Biclustering** | Identifies dense regions in the user–item affinity matrix via spectral co-clustering with over-partitioning and pruning | Highest quality; best lift and Gini on all tested datasets |
| **Multi-Armed Bandit** (UCB1 / Thompson Sampling) | Treats campaigns as arms and alternates user/item assignment with exploration bonuses | Fastest; scales to very large datasets |

The affinity between user $u$ and item $i$ is defined as:

$$A_{iu} = \exp(I_i \cdot U_u^\top)$$

where $I_i, U_u \in \mathbb{R}^D$ are item and user embedding vectors in a shared $D$-dimensional space.

---

## Repository structure

```
campaign_match_public/
├── src/
│   ├── algorithm_biclustering.py   # Constrained spectral biclustering
│   ├── algorithm_bandit.py         # Multi-armed bandit optimizer (UCB1, Thompson)
│   ├── eval.py                     # Evaluation metrics (quality, lift, Gini)
│   ├── data/
│   │   └── gauss_l2.py             # Synthetic Gaussian data generator
│   └── utils/
│       └── data.py                 # CSV data loading and embedding parsing
├── brute_force/
│   ├── brute_force.py              # Exact exhaustive solver (tiny datasets only)
│   └── brute_force.ipynb           # Demo notebook with Amazon data
├── data/
│   └── prepare_amazon.py           # Script to download and prepare Amazon Reviews 2023
├── examples/
│   └── run_experiment.py           # End-to-end example for both algorithms
├── environment.yml
└── README.md
```

---

## Setup

**1. Create and activate the conda environment:**

```bash
conda env create --prefix ./env --file environment.yml
conda activate ./env
```

**2. Update environment (if needed):**

```bash
conda env update --prefix ./env --file environment.yml --prune
```

---

## Data

We evaluate on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) benchmark dataset.

We use two categories:
- **Musical Instruments** — 70% of the full dataset, no filtering
- **Baby Products** — 5-core filtered (users and items with fewer than 5 interactions removed)

### Preparing the data

Run the data preparation script to download and process the Amazon Reviews 2023 dataset:

```bash
python data/prepare_amazon.py --category Musical_Instruments --output_dir data/musical_instruments
python data/prepare_amazon.py --category Baby_Products --output_dir data/baby_products --core_filter 5
```

This produces two CSV files per category in the output directory:
- `amazon_embeddings.csv` — item embeddings (columns: `id`, `title`, `embedding`)
- `amazon_embeddings_users.csv` — user embeddings (columns: `id`, `user_embedding`)

For this code, the user embeddings are computed as the **mean of item embeddings** for all items a user has reviewed, which is a standard collaborative-filtering approach for embedding-based user representations. Other embedding approaches can also be used. 

The script requires `requests` and `datasets` (HuggingFace), which are installed automatically or can be added via:
```bash
pip install requests datasets
```

### Data format

The algorithms expect:
- **Items CSV**: must have an `embedding` column (JSON list string, e.g. `"[0.12, -0.34, ...]"`)
- **Users CSV**: must have a `user_embedding` column (same format)

---

## Usage

### Quick start

```python
import numpy as np
import pandas as pd
import sys; sys.path.insert(0, 'src')

from utils.data import load_data
from algorithm_biclustering import ConstrainedSpectralBicluster
from algorithm_bandit import BanditCampaignOptimizer

# Load embeddings
users_df = pd.read_csv("data/musical_instruments/amazon_embeddings_users.csv")
items_df = pd.read_csv("data/musical_instruments/amazon_embeddings.csv")
user_matrix, item_matrix, item_biases = load_data(users_df, items_df)

# Compute affinity matrix: A = exp(I @ U^T)
affinity = np.exp(item_matrix @ user_matrix.T)  # shape: (n_items, n_users)
```

### Biclustering

```python
model = ConstrainedSpectralBicluster()
model.fit(
    affinity,
    n_clusters=5,
    min_items=3,
    min_users=100,
    max_items=5,
    max_users=10000,
    search_alpha=3,
)

for row_idx, col_idx in model.biclusters:
    quality = np.mean(affinity[np.ix_(row_idx, col_idx)])
    print(f"Campaign: {len(row_idx)} items x {len(col_idx)} users, quality={quality:.4f}")
```

### Bandit 

```python
# Define campaign capacity constraints
n_campaigns = 5
ad_configs = [
    {'id': k, 'min_users': 100, 'max_users': 10000}
    for k in range(n_campaigns)
]

optimizer = BanditCampaignOptimizer(
    user_embeds=user_matrix,
    item_embeds=item_matrix,
    item_biases=item_biases,
    ad_configs=ad_configs,
    items_per_ad=5,
)

best_users, best_items, best_score = optimizer.solve(
    n_rounds=50,
    strategy='thompson',   # 'thompson' | 'ucb'
    stats_window=10,
)

optimizer.print_report()
optimizer.plot_convergence()
```

### Synthetic data

```python
from src.data.gauss_l2 import generate_gauss_data, print_report

item_emb, user_emb, affinity, item_ids, user_ids, baseline, remaining = generate_gauss_data(
    items=[5, 5, 5],        # 5 seed items per campaign
    users=[200, 200, 200],  # 200 seed users per campaign
    num_items=1000,
    num_users=10000,
    emb_dim=2,
    seed=42,
)
print_report(baseline, affinity, item_ids, user_ids)
```

### End-to-end example

```bash
python examples/run_experiment.py \
    --users data/musical_instruments/amazon_embeddings_users.csv \
    --items data/musical_instruments/amazon_embeddings.csv \
    --n_campaigns 5 \
    --max_users 10000 \
    --items_per_ad 5 \
    --method both
```

---

## Evaluation Metrics

All metrics are computed from the affinity matrix $A$. See Section 3.4 of the paper for full definitions.

| Metric | Description |
|---|---|
| **Quality** | Mean affinity within each campaign (Eq. 4) |
| **Utility** | Total affinity summed across all campaigns (Eq. 3) |
| **Lift** | Quality relative to global mean affinity — values > 1 indicate above-random assignment (Eq. 5) |
| **Gini** | Fairness of utility distribution across users — 0 is perfectly equal (Eq. 6) |

