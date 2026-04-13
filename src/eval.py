import numpy as np
from sklearn.metrics import auc
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from tqdm import tqdm


def eval_l2(users, items):
    """
    Measure the distance in the embedding space with respect to the L2 norm.
    We usually optimize for this norm.
    
    :param users:
    :param items:
    """
    return pairwise_distances(users, items)


def eval_dprod(users, items):
    """
    Measures the dot product similarity.
    It values the angle between the embedding vectors in comparison to both vector lengths.
    
    :param users: 
    :param items: 
    """
    return np.dot(users, items.T)


def eval_cos(users, items):
    """
    Measures the cosine similarity which roughly approximates the inverse of the L2 norm in high-dimensional embedding spaces.
    It is the normalized do product similarity.
    
    :param users: 
    :param items: 
    """
    return cosine_similarity(users, items)


def eval_loglin(users, items):
    """
    The default measure is defined as length of the item vector projected on the normalized user vector.
    This is roughly the cosine similarity scaled by the length of the item vector.
    The scaling values more important items higher.
    Lastly, the value is clipped and transformed to R+ with the exponential function.

    y = exp(clip(u/|u| @ i^T, 30))

    :param users:
    :param items:
    """
    user_norms = np.linalg.norm(users, axis=1, keepdims=True)
    user_norms[user_norms == 0] = 1.0 
    users_normed = users / user_norms
    dot_products = np.dot(users_normed, items.T)
    scores = dot_products
    # clipping only for bias relevant?
    scores = np.clip(scores, -30, 30) 
    return np.exp(scores)


def eval(users, items, kind='loglin'):
    match kind:
        case 'loglin' : return eval_loglin(users, items)
        case 'cos' : return eval_cos(users, items)
        case 'dprod' : return eval_dprod(users, items)
        case 'l2' : return eval_l2(users, items)
        case _ : raise ValueError(f'Unsupported kind: {kind}')




def eval_curve(users, items, selected_users, selected_items, kind='loglin'):
    ads = set([a for a,i in selected_items])
    ad_to_item = {a : [] for a in ads}
    ad_to_user = {a : [] for a in ads}
    for a, i in selected_items:
        ad_to_item[a].append(i)

    scores = []
    last = 0
    final_score = 0
    for ad, user in tqdm(selected_users):
        ad_to_user[ad].append(user)
        score = 0
        for ad in ads:
            i = np.array([items[i] for i in ad_to_item[ad]])
            u = np.array([users[u] for u in ad_to_user[ad]])
            if len(u) == 0:
                continue
            s = eval(u, i, kind=kind)
            s = s.sum()
            score += s
            
        final_score = score
        score -= last
        last += score
        scores.append(score)

    return auc(np.arange(1, len(scores)+1), scores)/len(scores), final_score, scores



def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n