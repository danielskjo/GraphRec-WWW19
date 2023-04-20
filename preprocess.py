from collections import defaultdict
import itertools
import os
import pickle
import random

import networkx as nx
from networkx.algorithms import bipartite as bi
import numpy as np
from scipy.io import loadmat


def preprocess(path):
    uSet_u2u = set()
    uSet_u2b = set()
    bSet_u2b = set()

    social_adj_lists = defaultdict(set)
    history_u_lists = defaultdict(list)
    history_v_lists = defaultdict(list)

    history_ur_lists = defaultdict(list)
    history_vr_lists = defaultdict(list)

    G = nx.Graph()
    G.name = 'ciao'

    ratings_f = loadmat(path + 'epinions/rating.mat')['rating']
    trust_f = loadmat(path + 'epinions/trustnetwork.mat')['trustnetwork']

    users = []

    for s in ratings_f:
        uid = s[0]
        users.append(uid)

    offset = len(set(users))

    for s in ratings_f:
        uid = s[0]
        iid = s[1] + offset
        rating = s[3]
        uSet_u2b.add(uid)
        bSet_u2b.add(iid)
        G.add_edge(uid, iid, type='u2b', rating=rating)

    for s in trust_f:
        uid = s[0]
        fid = s[1]
        uSet_u2u.add(uid)
        uSet_u2u.add(fid)
        G.add_edge(uid, fid, type='u2u')

    print(G)
    print("uSet of u2u, size: " + str(len(uSet_u2u)))
    print("uSet of u2b, size: " + str(len(uSet_u2b)))
    print("bSet of u2b, size: " + str(len(bSet_u2b)))

    G = nx.convert_node_labels_to_integers(
        G, first_label=0, ordering='default', label_attribute="name")

    node_names = nx.get_node_attributes(G, 'name')
    inv_map = {v: k for k, v in node_names.items()}

    uSet_u2u = set([inv_map.get(name) for name in uSet_u2u])
    uSet_u2b = set([inv_map.get(name) for name in uSet_u2b])
    bSet_u2b = set([inv_map.get(name) for name in bSet_u2b])

    edge_list_uv = []
    edge_list_vu = []

    for node in G:
        for nbr in G[node]:
            if G[node][nbr]['type'] == 'u2u':
                social_adj_lists[node].add(nbr)

            if G[node][nbr]['type'] == 'u2b':
                r = G[node][nbr]['rating']

                if node in uSet_u2b and nbr in bSet_u2b:
                    history_u_lists[node].append(nbr)
                    history_v_lists[nbr].append(node)
                    history_ur_lists[node].append(r)
                    history_vr_lists[nbr].append(r)
                    edge_list_uv.append((node, nbr, r))
                    edge_list_vu.append((nbr, node, r))

                if nbr in uSet_u2b and node in bSet_u2b:
                    history_u_lists[nbr].append(node)
                    history_v_lists[node].append(nbr)
                    history_ur_lists[nbr].append(r)
                    history_vr_lists[node].append(r)
                    edge_list_uv.append((nbr, node, r))
                    edge_list_vu.append((node, nbr, r))

    data = []
    for (u, v) in G.edges():
        if G[u][v]['type'] == 'u2b':
            r = G[u][v]['rating']

            if u in uSet_u2b:
                data.append((u, v, r))
            else:
                data.append((v, u, r))

    random.shuffle(data)

    size = len(data)
    train_data = data[:int(0.8 * size)]
    test_data = data[int(0.8 * size):]

    with open(path + 'dataset.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

    train_u, train_v, train_r, test_u, test_v, test_r = [], [], [], [], [], []

    for u, v, r in train_data:
        train_u.append(u)
        train_v.append(v)
        train_r.append(r)

    for u, v, r in test_data:
        test_u.append(u)
        test_v.append(v)
        test_r.append(r)

    ratings_list = [1, 2, 3, 4, 5]

    _social_adj_lists = defaultdict(set)
    _history_u_lists = defaultdict(list)
    _history_v_lists = defaultdict(list)

    _history_ur_lists = defaultdict(list)
    _history_vr_lists = defaultdict(list)
    _train_u, _train_v, _train_r, _test_u, _test_v, _test_r = [], [], [], [], [], []

    social_id_dic = {v: k for k, v in dict(
        enumerate(social_adj_lists.keys())).items()}
    user_id_dic = {v: k for k, v in dict(
        enumerate(history_u_lists.keys())).items()}
    item_id_dic = {v: k for k, v in dict(
        enumerate(history_v_lists.keys())).items()}

    for u in history_u_lists:
        _history_u_lists[user_id_dic[u]] = [item_id_dic[v]
                                            for v in history_u_lists[u]]

    for v in history_v_lists:
        _history_v_lists[item_id_dic[v]] = [user_id_dic[u]
                                            for u in history_v_lists[v]]

    for u in history_ur_lists:
        _history_ur_lists[user_id_dic[u]] = history_ur_lists[u]

    for v in history_vr_lists:
        _history_vr_lists[item_id_dic[v]] = history_vr_lists[v]

    for u in social_adj_lists:
        _social_adj_lists[social_id_dic[u]] = [social_id_dic[us]
                                               for us in social_adj_lists[u]]

    for u, v, r in train_data:
        if u in user_id_dic.keys() and v in item_id_dic.keys():
            _train_u.append(user_id_dic[u])
            _train_v.append(item_id_dic[v])
            _train_r.append(r)

    for u, v, r in test_data:
        if u in user_id_dic.keys() and v in item_id_dic.keys():
            _test_u.append(user_id_dic[u])
            _test_v.append(item_id_dic[v])
            _test_r.append(r)

    with open(path + 'list.pkl', 'wb') as f:
        pickle.dump(_history_u_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_history_ur_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_history_v_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_history_vr_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_train_u, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_train_v, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_train_r, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_test_u, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_test_v, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_test_r, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_social_adj_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ratings_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    path = 'data/'
    preprocess(path)
