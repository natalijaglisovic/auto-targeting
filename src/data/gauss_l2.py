import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import paired_distances

def sample_uniform(num, dim, clip = 1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(size=[num, dim], low=-clip, high=clip)

def sample_normal(nums, dim, means, clip = 1.0, var = 0.01, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    covs = np.eye(dim)
    covs = np.expand_dims(covs, 0)
    covs = np.repeat(covs, len(nums), axis=0)
    vars = var if isinstance(var, list) else [var,] * len(nums)
    covs = covs
    embs = []
    for mean, cov, var, user in zip(means, covs, vars, nums):
        emb = rng.multivariate_normal(mean=mean, cov=cov*var, size=user)
        emb = np.clip(emb, a_min = -clip, a_max = clip)
        embs.append(emb)
    return embs

def generate_gauss_data(items, users, num_items = -1, num_users = -1, emb_dim = 2, clip = 1.0, item_var = 0.01, user_var = 0.01, seed = None):
    rng = np.random.default_rng(seed=seed)
    items_independent = max(num_items-sum(items), 0)
    users_idependent = max(num_users-sum(users), 0)

    means = sample_uniform(len(users), emb_dim, clip=clip, rng=rng)
    item_emb_list = sample_normal(items, emb_dim, means, clip=clip, var=item_var, rng=rng)
    user_emb_list = sample_normal(users, emb_dim, means, clip=clip, var=user_var, rng=rng)

    item_emb_other = sample_uniform(items_independent, emb_dim, clip=clip, rng=rng)
    user_emb_other = sample_uniform(users_idependent, emb_dim, clip=clip, rng=rng)
    item_emb_list.append(item_emb_other)
    user_emb_list.append(user_emb_other)

    item_emb = np.concatenate(item_emb_list, axis=0)
    user_emb = np.concatenate(user_emb_list, axis=0)

    num_users = user_emb.shape[0]
    num_items = item_emb.shape[0]

    item_emb2d = np.expand_dims(item_emb, 1)
    item_emb2d = np.repeat(item_emb2d, num_users, axis=1)
    user_emb2d = np.expand_dims(user_emb, 0)
    user_emb2d = np.repeat(user_emb2d, num_items, axis=0)
    item_emb2d = np.reshape(item_emb2d, [num_users * num_items, emb_dim])
    user_emb2d = np.reshape(user_emb2d, [num_users * num_items, emb_dim])
    #max_dist = 2*clip*np.sqrt(2)
    # expected dist?
    affinity = paired_distances(item_emb2d, user_emb2d, metric='manhattan') #1.0- (np.log(paired_distances(item_emb2d, user_emb2d) + 1e-10) / np.log()
    affinity = (clip*2*2 - affinity)/(clip*2*2)
    affinity = np.reshape(affinity, [num_items, num_users])
    affinity = np.exp(affinity)

    # the predefined campaigns are always a valid and reasonable solution
    # however, there might be a better one due to overlap and independent items
    baseline_solution = []
    item_idx = 0
    user_idx = 0
    item_id = np.zeros((len(item_emb,)), dtype=int)
    user_id = np.zeros((len(user_emb,)), dtype=int)
    for id, (item, user) in enumerate(zip(items, users)):
        baseline_solution.append({'items' : list(range(item_idx, item_idx+item)), 'users' : list(range(user_idx, user_idx+user))})
        item_id[baseline_solution[-1]['items']] = id + 1
        user_id[baseline_solution[-1]['users']] = id + 1
        item_idx, user_idx = item_idx + item, user_idx + user

    remaining = {'items' : list(range(item_idx, len(item_emb))), 'users' : list(range(user_idx, len(user_emb)))}

    return item_emb, user_emb, affinity, item_id, user_id, baseline_solution, remaining



################################ Evaluation ###############################################

def get_bicluster_score(matrix, row_indices, col_indices, global_mean):
    """Calculate score, utility, and lift for a bicluster."""
    submatrix = matrix[np.ix_(row_indices, col_indices)]
    score = np.mean(submatrix)      # Quality (density)
    utility = np.sum(submatrix)     # Total utility
    lift = score / global_mean if global_mean > 0 else 0.0
    return score, utility, lift

def print_report(campaigns, affinity, item_ids, user_ids):
    # Calculate metrics for ML approach: Total Score, Utility, Avg Lift
    total_score_ml = 0.0
    total_utility_ml = 0.0
    global_mean_ml = np.mean(affinity)

    print(f"{'Ad':<10} | {'Shape':<20} | {'Score':<12} | {'Utility':<14} | {'Lift':<8}")
    print("-" * 75)

    for ad_idx, campaign in enumerate(campaigns):
        user_indices = campaign['users']
        item_indices = campaign['items']

        if len(user_indices) > 0 and len(item_indices) > 0:
            # affinity is (items x users)
            score, utility, lift = get_bicluster_score(
                affinity,
                item_indices,   # rows = items
                user_indices,   # cols = users
                global_mean_ml
            )

            total_score_ml += score
            total_utility_ml += utility

            shape = f"{len(item_indices)} Items x {len(user_indices)} Users"
            print(f"{ad_idx+1:<10} | {shape:<20} | {score:<12.4f} | {utility:<14.2f} | {lift:<8.2f}x")

            found_items = item_ids[item_indices]
            found_users = user_ids[user_indices]

            print(f"  Items ({len(found_items)}): {found_items}")
            print(f"  Users ({len(found_users)}): {found_users[:10]}...")

        else:
            print(f"{ad_idx+1:<10} | Empty Ad            | -            | -              | -")

    print("-" * 75)

    avg_lift_ml = (total_score_ml / len(campaigns)) / global_mean_ml if global_mean_ml > 0 else 0.0

    print(f"Total Score: {total_score_ml:.4f}")
    print(f"Total Utility: {total_utility_ml:.4f}")
    print(f"AVG Lift: {avg_lift_ml:.2f}x")


def plot_embeddings(item_emb, user_emb, solution, clip, name, remaining=True):
    item_emb_list = [item_emb[s['items']] for s in solution]
    user_emb_list = [user_emb[s['users']] for s in solution]

    remaining_items = np.ones(len(item_emb), dtype=bool)
    remaining_users = np.ones(len(user_emb), dtype=bool)
    for s in solution:
        remaining_items[s['items']] = False
        remaining_users[s['users']] = False
    item_emb_list.append(item_emb[remaining_items])
    user_emb_list.append(user_emb[remaining_users])

    # only plot first two emb dims
    plt.figure()
    plt.title(f'{name} Campaign Users')
    for emb in user_emb_list[:-1]:
        plt.scatter(emb[:, 0], emb[:, 1], s=16)
    if remaining:
        emb = user_emb_list[-1]
        plt.scatter(emb[:, 0], emb[:, 1], s=1, c='gray')
    plt.xlim(-clip, clip)
    plt.ylim(-clip, clip)

    plt.figure()
    plt.title(f'{name} Campaign Items')
    for emb in item_emb_list[:-1]:
        plt.scatter(emb[:, 0], emb[:, 1], s=16)
    if remaining:
        emb = item_emb_list[-1]
        plt.scatter(emb[:, 0], emb[:, 1], s=1, c='gray')
    plt.xlim(-clip, clip)
    plt.ylim(-clip, clip)




################################ Example ###############################################

def main():
    clip = 1.0

    items = [5, 5, 5]
    users = [200, 200, 200]
    num_items = 1000
    num_users = 10000

    item_emb, user_emb, affinity, item_ids, user_ids, baseline_solution, remaining = generate_gauss_data(
        items = items,
        users = users,
        num_items = num_items,
        num_users = num_users,
        emb_dim = 2,
        clip = clip,
        item_var = 0.01,
        user_var = 0.05,
        seed=1)

    # affinity is normalized manhatten between 0 and 1 right now
    # we might change it to make it more similar to embedding based affinity
    print_report(baseline_solution, affinity, item_ids, user_ids)

    plot_embeddings(item_emb, user_emb, baseline_solution, clip, 'Baseline', remaining=True)

    plt.figure()
    plt.title('Affinity Scores')
    sns.heatmap(affinity, annot=False)

    plt.figure()
    plt.title('Affinity Scores in Campaigns')
    sns.heatmap(affinity[:sum(items), :sum(users)], annot=False)

    plt.show()

if __name__ == '__main__':
    main()
