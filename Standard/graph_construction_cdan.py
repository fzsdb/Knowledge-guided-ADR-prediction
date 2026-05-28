import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from torch_geometric.data import Data
import config_cdan as config


def build_adr_graph(data_train, data_test, device, jaccard_percentile=None):
    jaccard_percentile = jaccard_percentile or config.JACCARD_PERCENTILE
    adr_data = pd.read_csv('adrid_with_multiinfo.csv')
    adr_indices = np.unique(np.concatenate([data_train[:, 1], data_test[:, 1]])).astype(int)
    index_to_adrecs = {}
    for idx in adr_indices:
        if idx >= len(adr_data):
            print(f"Warning: adr_index {idx} out of range in adrid_with_multiinfo.csv")
            continue
        adrecs_ids = adr_data.iloc[idx, 1]
        if pd.isna(adrecs_ids):
            print(f"Warning: Empty ADReCS IDs for index {idx}")
            continue
        index_to_adrecs[idx] = adrecs_ids.split(';')

    def parse_adr_hierarchy(adr_id):
        levels = adr_id.split('.')
        if len(levels) != 4:
            print(f"Warning: Invalid ADReCS ID format: {adr_id}")
            return None
        soc = levels[0]
        hlgt = f"{levels[0]}.{levels[1]}"
        hlt = f"{levels[0]}.{levels[1]}.{levels[2]}"
        pt = adr_id
        return {"SOC": soc, "HLGT": hlgt, "HLT": hlt, "PT": pt}

    nodes = set()
    edges = []
    hlt_to_pts = defaultdict(list)

    for idx, adrecs_ids in index_to_adrecs.items():
        for adrecs_id in adrecs_ids:
            hierarchy = parse_adr_hierarchy(adrecs_id)
            if hierarchy is None:
                continue
            nodes.add(hierarchy["SOC"])
            nodes.add(hierarchy["HLGT"])
            nodes.add(hierarchy["HLT"])
            nodes.add(hierarchy["PT"])
            edges.append((hierarchy["PT"], hierarchy["HLT"], "child_of", 1.0))
            edges.append((hierarchy["HLT"], hierarchy["HLGT"], "child_of", 1.0))
            edges.append((hierarchy["HLGT"], hierarchy["SOC"], "child_of", 1.0))
            edges.append((hierarchy["HLT"], hierarchy["PT"], "parent_of", 1.0))
            edges.append((hierarchy["HLGT"], hierarchy["HLT"], "parent_of", 1.0))
            edges.append((hierarchy["SOC"], hierarchy["HLGT"], "parent_of", 1.0))
            hlt_to_pts[hierarchy["HLT"]].append(hierarchy["PT"])

    for hlt, pts in hlt_to_pts.items():
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                edges.append((pts[i], pts[j], "sibling_of", 1.0))
                edges.append((pts[j], pts[i], "sibling_of", 1.0))

    adr_to_drugs = defaultdict(set)
    train_sample = pd.DataFrame(data_train, columns=['drug_index', 'adr_index', 'label'])
    for _, row in train_sample[train_sample['label'] == 1].iterrows():
        drug_idx = int(row['drug_index'])
        adr_idx = int(row['adr_index'])
        adrecs_ids = index_to_adrecs.get(adr_idx, [])
        for adrecs_id in adrecs_ids:
            hierarchy = parse_adr_hierarchy(adrecs_id)
            if hierarchy:
                adr_to_drugs[hierarchy["PT"]].add(drug_idx)

    pt_cooccur_weights = {}
    jaccard_values = []

    for idx1, idx2 in combinations(index_to_adrecs.keys(), 2):
        adrecs_ids1 = index_to_adrecs[idx1]
        adrecs_ids2 = index_to_adrecs[idx2]
        drugs1 = set().union(*(adr_to_drugs.get(adrecs_id, set()) for adrecs_id in adrecs_ids1 if parse_adr_hierarchy(adrecs_id)))
        drugs2 = set().union(*(adr_to_drugs.get(adrecs_id, set()) for adrecs_id in adrecs_ids2 if parse_adr_hierarchy(adrecs_id)))
        intersection = len(drugs1 & drugs2)
        union = len(drugs1 | drugs2)
        if union > 0:
            jaccard = intersection / union
            if jaccard > 0:
                jaccard_values.append((jaccard, idx1, idx2, adrecs_ids1, adrecs_ids2))

    if jaccard_values:
        jaccard_scores = [v[0] for v in jaccard_values]
        jaccard_arr = np.array(jaccard_scores)
        dynamic_threshold = np.percentile(jaccard_arr, jaccard_percentile)
        
        print("\n=== ADR Graph (PT Co-occurrence) Jaccard Statistics ===")
        print(f"Total co-occurrence pairs calculated: {len(jaccard_values)}")
        print(f"Dynamic threshold (percentile {jaccard_percentile}): {dynamic_threshold:.6f}")
        
        for jaccard, idx1, idx2, adrecs_ids1, adrecs_ids2 in jaccard_values:
            if jaccard >= dynamic_threshold:
                for pt1 in adrecs_ids1:
                    if not parse_adr_hierarchy(pt1): continue
                    for pt2 in adrecs_ids2:
                        if not parse_adr_hierarchy(pt2): continue
                        pt_cooccur_weights[(pt1, pt2)] = jaccard
                        pt_cooccur_weights[(pt2, pt1)] = jaccard
    else:
        print("\n=== ADR Graph: No co-occurrence pairs found ===")

    for (pt1, pt2), weight in pt_cooccur_weights.items():
        edges.append((pt1, pt2, "co_occurrence", weight))

    nodes = list(nodes)
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    edge_index = [[node2idx[child], node2idx[parent]] for child, parent, _, _ in edges]
    edge_type = [
        0 if rel == "child_of" else
        1 if rel == "parent_of" else
        2 if rel == "sibling_of" else
        3 for _, _, rel, _ in edges
    ]
    edge_weight = [weight for _, _, _, weight in edges]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type, dtype=torch.long).to(device)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(device)

    x = torch.randn(len(nodes), 128).to(device)
    graph = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight).to(device)
    
    return graph, node2idx, index_to_adrecs


def build_atc_graph_global(data_train, index_to_atc_global, device, jaccard_percentile=None):
    jaccard_percentile = jaccard_percentile or config.JACCARD_PERCENTILE
    
    def parse_atc_hierarchy(atc_code):
        if len(atc_code) != 7:
            return None
        return {"L1": atc_code[0], "L2": atc_code[:3], "L3": atc_code[:4], "L4": atc_code[:5], "L5": atc_code}

    nodes = set()
    edges = []
    level4_to_level5 = defaultdict(list)

    for drug_idx, atc_codes in index_to_atc_global.items():
        for atc_code in atc_codes:
            hier = parse_atc_hierarchy(atc_code)
            if hier is None:
                continue
            for level in ["L1", "L2", "L3", "L4", "L5"]:
                nodes.add(hier[level])
            edges.append((hier["L5"], hier["L4"], "child_of", 1.0))
            edges.append((hier["L4"], hier["L3"], "child_of", 1.0))
            edges.append((hier["L3"], hier["L2"], "child_of", 1.0))
            edges.append((hier["L2"], hier["L1"], "child_of", 1.0))
            edges.append((hier["L4"], hier["L5"], "parent_of", 1.0))
            edges.append((hier["L3"], hier["L4"], "parent_of", 1.0))
            edges.append((hier["L2"], hier["L3"], "parent_of", 1.0))
            edges.append((hier["L1"], hier["L2"], "parent_of", 1.0))
            level4_to_level5[hier["L4"]].append(hier["L5"])

    for level4, level5s in level4_to_level5.items():
        for i in range(len(level5s)):
            for j in range(i + 1, len(level5s)):
                edges.append((level5s[i], level5s[j], "sibling_of", 1.0))
                edges.append((level5s[j], level5s[i], "sibling_of", 1.0))

    level5_to_adrs = defaultdict(set)
    train_df = pd.DataFrame(data_train, columns=['drug_index', 'adr_index', 'label'])
    for _, row in train_df[train_df['label'] == 1].iterrows():
        drug_idx = int(row['drug_index'])
        adr_idx = int(row['adr_index'])
        atc_codes = index_to_atc_global.get(drug_idx, [])
        for atc_code in atc_codes:
            hier = parse_atc_hierarchy(atc_code)
            if hier:
                level5_to_adrs[hier["L5"]].add(adr_idx)

    jaccard_values = []
    for l5_1, l5_2 in combinations(level5_to_adrs.keys(), 2):
        adrs1 = level5_to_adrs[l5_1]
        adrs2 = level5_to_adrs[l5_2]
        inter = len(adrs1 & adrs2)
        union = len(adrs1 | adrs2)
        if union > 0:
            jacc = inter / union
            if jacc > 0:
                jaccard_values.append((jacc, l5_1, l5_2))

    cooccur_edges = {}
    if jaccard_values:
        jaccard_scores = [v[0] for v in jaccard_values]
        jaccard_arr = np.array(jaccard_scores)
        dynamic_threshold = np.percentile(jaccard_arr, jaccard_percentile)
        
        print(f"\n=== ATC Graph Co-occurrence Stats (Jaccard > percentile {jaccard_percentile}) ===")
        print(f"Total pairs: {len(jaccard_values)}, Threshold: {dynamic_threshold:.6f}")
        
        for jacc, l5_1, l5_2 in jaccard_values:
            if jacc >= dynamic_threshold:
                cooccur_edges[(l5_1, l5_2)] = jacc
                cooccur_edges[(l5_2, l5_1)] = jacc

    for (l5_1, l5_2), weight in cooccur_edges.items():
        edges.append((l5_1, l5_2, "co_occurrence", weight))

    nodes = list(nodes)
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    edge_index_list = []
    edge_type_list = []
    edge_weight_list = []
    rel_to_id = {"child_of": 0, "parent_of": 1, "sibling_of": 2, "co_occurrence": 3}

    for child, parent, rel, weight in edges:
        if child not in node2idx or parent not in node2idx:
            continue
        edge_index_list.append([node2idx[child], node2idx[parent]])
        edge_type_list.append(rel_to_id[rel])
        edge_weight_list.append(weight)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long).to(device)
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float).to(device)
    x = torch.randn(len(nodes), 128).to(device)

    graph = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight).to(device)
    return graph, node2idx


def get_all_atc_l2_codes(index_to_atc):
    l2_codes = set()
    for atc_codes in index_to_atc.values():
        for atc_code in atc_codes:
            if len(atc_code) >= 3:
                l2_codes.add(atc_code[:3])
    return sorted(list(l2_codes))
