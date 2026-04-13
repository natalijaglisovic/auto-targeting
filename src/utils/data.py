import pandas as pd
import numpy as np
import json

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
    print("Parsing Item Biases...")
    if 'bias' in items_df.columns:
        item_biases = items_df['bias'].fillna(0).values.astype(np.float32)
    elif 'itemScore' in items_df.columns:
        item_biases = items_df['itemScore'].fillna(0).values.astype(np.float32)
    else:
        print("WARNING: No 'bias' column found in items. Defaulting to 0.")
        item_biases = np.zeros(len(items_df), dtype=np.float32)

    return user_matrix, item_matrix, item_biases
    

def compute_log_linear_scores(users, items, biases = 0):
        #print("Computing Affinity Matrix...")
        user_norms = np.linalg.norm(users, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1.0 
        users_normed = users / user_norms
        dot_products = np.dot(users_normed, items.T)
        scores = dot_products + biases
        scores = np.clip(scores, -30, 30) 
        return np.exp(scores)