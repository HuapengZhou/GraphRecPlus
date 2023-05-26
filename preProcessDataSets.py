import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
# Load the .mat file.
mat2 = scipy.io.loadmat('data/Ciao/rating.mat')
# Print the keys of the mat dictionary to see the variables stored in the .mat file.
mat1 = scipy.io.loadmat('data/Ciao/trustnetwork.mat')

edges = mat1['trustnetwork']
ratings = mat2['rating']
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pickle

def preprocess_data(ratings, edges):
    """
    Preprocess the ratings and edges data.

    Parameters
    ----------
    ratings : np.ndarray
        The ratings data. It should have five columns: userid, productid, categoryid, rating, helpfulness.

    edges : np.ndarray
        The trust network edges. It should have two columns: userid1, userid2.

    Returns
    -------
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists : dict
        User purchase history and rating history.

    train_u, train_v, train_r : np.ndarray
        Training set data.

    test_u, test_v, test_r : np.ndarray
        Testing set data.

    social_adj_lists : dict
        User social adjacency lists.
    """
    # Preprocess ratings data
    unique_users = np.unique(ratings[:, 0])
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    mapped_user_ids = np.array([user_id_map[old_id] for old_id in ratings[:, 0]])

    unique_items = np.unique(ratings[:, 1])
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    mapped_item_ids = np.array([item_id_map[old_id] for old_id in ratings[:, 1]])

    ratings[:, 0] = mapped_user_ids
    ratings[:, 1] = mapped_item_ids

    # Preprocess edges data
    mapped_edges = np.array([user_id_map[old_id] for old_id in edges.flatten()]).reshape(edges.shape)

    # Generate purchase history and rating history
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists = generate_history(ratings)

    # Split into training set and testing set
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    train_u = train_ratings[:, 0]
    train_v = train_ratings[:, 1]
    train_r = train_ratings[:, 3]

    test_u = test_ratings[:, 0]
    test_v = test_ratings[:, 1]
    test_r = test_ratings[:, 3]

    # Generate user social adjacency lists
    social_adj_lists = generate_social(mapped_edges)

    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
           train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists


def generate_history(ratings):
    """
    Generate user purchase history and rating history.

    Parameters
    ----------
    ratings : np.ndarray
        The ratings data. It should have five columns: userid, productid, categoryid, rating, helpfulness.

    Returns
    -------
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists : dict
        User purchase history and rating history.
    """
    history_u_lists = defaultdict(list)
    history_ur_lists = defaultdict(list)
    history_v_lists = defaultdict(list)
    history_vr_lists = defaultdict(list)
    # add the catagory info
    history_vc_lists = defaultdict(list)
    for u, v, c, r, _ in ratings:
        history_u_lists[u].append(v)
        history_ur_lists[u].append(r)
        history_v_lists[v].append(u)
        history_vr_lists[v].append(r)
        history_vc_lists[v].append(c)

    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists


def generate_social(edges):
    """
    Generate user social adjacency lists.

    Parameters
    ----------
    edges : np.ndarray
        The trust network edges. It should have two columns: userid1, userid2.

    Returns
    -------
    social_adj_lists : dict
        User social adjacency lists.
    """
    social_adj_lists = defaultdict(list)

    for u, v in edges:
        social_adj_lists[u].append(v)

    return social_adj_lists


def save_to_pickle(filepath, data):
    """
    Save data to a pickle file.

    Parameters
    ----------
    filepath : str
        Path to the pickle file.

    data : tuple
        Data to be saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)



# 计算 ratings_list
ratings_list = np.unique(ratings[:, 3])
# Preprocess data and generate required information
history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists = preprocess_data(ratings, edges)

# Save the data to a pickle file
data = (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
        train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list)

save_to_pickle('./data/Ciao80_no_cat.pickle', data)

print("finished")
