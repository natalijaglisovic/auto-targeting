import numpy as np
import itertools
import pandas as pd
import json

def brute_force_optimize(u_mat, i_mat, i_bias, ad_configs, items_per_ad=5):
    """
    Brute-force assignment of users and items to ads.
    - Can leave users/items unassigned.
    - Respects items_per_ad target.
    """

    n_users = u_mat.shape[0]
    n_items = i_mat.shape[0]
    n_ads = len(ad_configs)
    TOMBSTONE_IDX = n_ads  # Unassigned index

    # Compute affinity matrix like the engine
    user_norms = np.linalg.norm(u_mat, axis=1, keepdims=True)
    user_norms[user_norms == 0] = 1.0
    users_normed = u_mat / user_norms
    dot_products = np.dot(users_normed, i_mat.T)
    affinity_matrix = np.exp(dot_products + i_bias)  # exactly like your engine

    # Generate all possible item assignments per ad
    ad_ids = list(range(n_ads))
    # Include tombstone as an option
    possible_item_assignments = ad_ids + [TOMBSTONE_IDX]

    # Limit items per ad: for brute force, we generate combinations per ad
    # For small datasets only!
    best_score = -np.inf
    best_user_assignments = None
    best_item_assignments = None

    # Iterate over all possible item assignments
    for item_assignment in itertools.product(possible_item_assignments, repeat=n_items):
        item_assignment = np.array(item_assignment)

        # Check items_per_ad constraint
        valid = True
        for ad_idx in ad_ids:
            count = np.sum(item_assignment == ad_idx)
            if count > items_per_ad:
                valid = False
                break
        if not valid:
            continue

        # Iterate over all possible user assignments (include tombstone)
        possible_user_assignments = ad_ids + [TOMBSTONE_IDX]
        for user_assignment in itertools.product(possible_user_assignments, repeat=n_users):
            user_assignment = np.array(user_assignment)

            # Check min/max users per ad
            valid = True
            for config in ad_configs:
                ad_idx = config['id']
                assigned_count = np.sum(user_assignment == ad_idx)
                if assigned_count < config.get('min_users', 0) or assigned_count > config.get('max_users', n_users):
                    valid = False
                    break
            if not valid:
                continue

            # Compute score like engine
            total_score = 0.0
            for u, ad in enumerate(user_assignment):
                if ad == TOMBSTONE_IDX:
                    continue
                items_in_ad = np.where(item_assignment == ad)[0]
                if len(items_in_ad) > 0:
                    total_score += np.sum(affinity_matrix[u, items_in_ad])

            # Update best
            if total_score > best_score:
                best_score = total_score
                best_user_assignments = user_assignment.copy()
                best_item_assignments = item_assignment.copy()

    return best_user_assignments, best_item_assignments, best_score

def parse_vector_string(s):
    """
    Parses strings like "[-0.2603, 0.156...]" into a numpy array.
    """
    try:
        if pd.isna(s) or s == '': return np.zeros(32) # Fallback
        # Determine if it's a list or numpy print format
        if ',' not in s and '\n' in s: # Looks like numpy print output
             s = s.replace('\n', ' ').replace('[', '').replace(']', '')
             return np.fromstring(s, sep=' ')
        else: # Looks like standard JSON list
            return np.array(json.loads(s), dtype=np.float32)
    except Exception as e:
        print(f"Warning: formatting error in vector: {str(e)[:50]}...")
        return np.zeros(32)

def load_data(users_df, items_df):
    """
    Parses DataFrames based on your specific CSV columns.
    """
    # Parse User Embeddings (Column: 'user_embedding')
    print("Parsing User Vectors...")
    if 'user_embedding' not in users_df.columns:
        raise ValueError("User CSV missing 'user_embedding' column")
    
    user_matrix = np.stack(users_df['user_embedding'].apply(parse_vector_string).values)

    # Parse Item Embeddings (Column: 'embedding')
    print("Parsing Item Vectors...")
    if 'embedding' not in items_df.columns:
        raise ValueError("Item CSV missing 'embedding' column")

    item_matrix = np.stack(items_df['embedding'].apply(parse_vector_string).values)

    # Parse Item Bias (Column: 'bias' or 'itemScore')
    # print("Parsing Item Biases...")
    # if 'bias' in items_df.columns:
    #     item_biases = items_df['bias'].fillna(0).values.astype(np.float32)
    # elif 'itemScore' in items_df.columns:
    #     item_biases = items_df['itemScore'].fillna(0).values.astype(np.float32)
    # else:
    #     print("WARNING: No 'bias' column found in items. Defaulting to 0.")
    #     item_biases = np.zeros(len(items_df), dtype=np.float32)
    item_biases = np.zeros(len(items_df), dtype=np.float32)

    return user_matrix, item_matrix, item_biases

def print_brute_force_results(users_df, items_df, user_assignments, item_assignments, ad_configs):
    TOMBSTONE_IDX = len(ad_configs)  # same as in brute force

    # Assign the results to the DataFrames
    users_df['Assigned_Ad_ATEM'] = user_assignments
    items_df['Assigned_Ad_ATEM'] = item_assignments

    print("\n" + "="*40)
    print("      FINAL AD ASSIGNMENTS      ")
    print("="*40)

    # Iterate through each ad
    for config in ad_configs:
        ad_id = config['id']
        assigned_users = users_df[users_df['Assigned_Ad_ATEM'] == ad_id]['id'].tolist()
        assigned_items = items_df[items_df['Assigned_Ad_ATEM'] == ad_id]

        print(f"\n--- [ Ad {ad_id} ] ------------------------")
        print(f"Capacity: {len(assigned_users)} Users / {len(assigned_items)} Items")

        # Users
        if assigned_users:
            print(f"Users: {assigned_users}")
        else:
            print("Users: (None)")

        # Items
        if not assigned_items.empty:
            print("Items:")
            for _, row in assigned_items.iterrows():
                label = row['title'] if 'title' in row and pd.notna(row['title']) else row['id']
                print(f"  - {label}")
        else:
            print("Items: (None)")

    # Show unassigned (TOMBSTONE)
    unassigned_users = users_df[users_df['Assigned_Ad_ATEM'] == TOMBSTONE_IDX]['id'].tolist()
    unassigned_items = items_df[items_df['Assigned_Ad_ATEM'] == TOMBSTONE_IDX]

    print(f"\n--- [ UNASSIGNED / BACKLOG ] -------------------")
    print(f"Users left behind: {unassigned_users if unassigned_users else '(None)'}")
    print("Items left behind:")
    if not unassigned_items.empty:
        for _, row in unassigned_items.iterrows():
            label = row['title'] if 'title' in row and pd.notna(row['title']) else row['id']
            print(f"  - {label}")
    else:
        print("  (None)")