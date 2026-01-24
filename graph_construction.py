import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from torch_geometric.data import Data


def build_adr_graph(data_train, data_test, device, jaccard_percentile=75):
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
        print(f"Jaccard Statistics (all pairs):")
        print(f"  Mean: {jaccard_arr.mean():.6f}")
        print(f"  Std:  {jaccard_arr.std():.6f}")
        print(f"  Min:  {jaccard_arr.min():.6f}")
        print(f"  Max:  {jaccard_arr.max():.6f}")
        print(f"  Median: {np.median(jaccard_arr):.6f}")
        print(f"\nDynamic threshold (percentile {jaccard_percentile}): {dynamic_threshold:.6f}")
        
        filtered_count = 0
        filtered_jaccard_values = []
        for jaccard, idx1, idx2, adrecs_ids1, adrecs_ids2 in jaccard_values:
            if jaccard >= dynamic_threshold:
                for pt1 in adrecs_ids1:
                    if not parse_adr_hierarchy(pt1): continue
                    for pt2 in adrecs_ids2:
                        if not parse_adr_hierarchy(pt2): continue
                        pt_cooccur_weights[(pt1, pt2)] = jaccard
                        pt_cooccur_weights[(pt2, pt1)] = jaccard
                filtered_jaccard_values.append(jaccard)
                filtered_count += 1
        
        if filtered_jaccard_values:
            filtered_arr = np.array(filtered_jaccard_values)
            print(f"\nAfter filtering (>= {dynamic_threshold:.6f}):")
            print(f"  Number of edges: {len(filtered_jaccard_values)}")
            print(f"  Percentage retained: {len(filtered_jaccard_values)/len(jaccard_values)*100:.2f}%")
            print(f"  Mean Jaccard: {filtered_arr.mean():.6f}")
            print(f"  Std Jaccard:  {filtered_arr.std():.6f}")
            print(f"  Min/Max Jaccard: {filtered_arr.min():.6f} / {filtered_arr.max():.6f}")
    else:
        print("\n=== ADR Graph: No co-occurrence pairs found ===")
        dynamic_threshold = 0.0

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


def build_atc_graph(data_train, data_test, device, jaccard_percentile=75):
    atc_data = pd.read_csv('result_with_atc_traintest.csv')
    drug_smiles = pd.read_csv('smiles_traintest.csv')
    if 'drugid' in drug_smiles.columns:
        drug_index_to_id = {i: drug_smiles['drugid'].iloc[i] for i in range(len(drug_smiles))}
    else:
        drug_index_to_id = {i: f"BADD_D{str(i).zfill(5)}" for i in range(len(drug_smiles))}

    drug_indices = np.unique(np.concatenate([data_train[:, 0], data_test[:, 0]])).astype(int)
    index_to_atc = {}
    for idx in drug_indices:
        drug_id = drug_index_to_id.get(idx)
        if drug_id is None:
            print(f"Warning: drug_index {idx} not found in drug_smiles mapping")
            continue
        matching_rows = atc_data[atc_data['DRUG_ID'] == drug_id]
        if not matching_rows.empty:
            atc_codes = matching_rows['ATC_CODE'].iloc[0].split(';')
            valid_atc_codes = [atc_code.strip() for atc_code in atc_codes if len(atc_code.strip()) == 7]
            if valid_atc_codes:
                index_to_atc[idx] = valid_atc_codes
            else:
                print(f"Warning: No valid ATC codes for DRUG_ID {drug_id}")
        else:
            print(f"Warning: DRUG_ID {drug_id} not found in result_with_atc_traintest.csv")

    def parse_atc_hierarchy(atc_code):
        if len(atc_code) != 7:
            print(f"Warning: Invalid ATC code format: {atc_code}")
            return None
        return {"L1": atc_code[0], "L2": atc_code[:3], "L3": atc_code[:4], "L4": atc_code[:5], "L5": atc_code}

    nodes = set()
    edges = []
    level4_to_level5 = defaultdict(list)

    for idx, atc_codes in index_to_atc.items():
        for atc_code in atc_codes:
            hierarchy = parse_atc_hierarchy(atc_code)
            if hierarchy is None:
                continue
            for level in ["L1", "L2", "L3", "L4", "L5"]:
                nodes.add(hierarchy[level])
            edges.append((hierarchy["L5"], hierarchy["L4"], "child_of", 1.0))
            edges.append((hierarchy["L4"], hierarchy["L3"], "child_of", 1.0))
            edges.append((hierarchy["L3"], hierarchy["L2"], "child_of", 1.0))
            edges.append((hierarchy["L2"], hierarchy["L1"], "child_of", 1.0))
            edges.append((hierarchy["L4"], hierarchy["L5"], "parent_of", 1.0))
            edges.append((hierarchy["L3"], hierarchy["L4"], "parent_of", 1.0))
            edges.append((hierarchy["L2"], hierarchy["L3"], "parent_of", 1.0))
            edges.append((hierarchy["L1"], hierarchy["L2"], "parent_of", 1.0))
            level4_to_level5[hierarchy["L4"]].append(hierarchy["L5"])

    for level4, level5s in level4_to_level5.items():
        for i in range(len(level5s)):
            for j in range(i + 1, len(level5s)):
                edges.append((level5s[i], level5s[j], "sibling_of", 1.0))
                edges.append((level5s[j], level5s[i], "sibling_of", 1.0))

    level5_to_adrs = defaultdict(set)
    train_sample = pd.DataFrame(data_train, columns=['drug_index', 'adr_index', 'label'])
    for _, row in train_sample[train_sample['label'] == 1].iterrows():
        drug_idx = int(row['drug_index'])
        adr_idx = int(row['adr_index'])
        atc_codes = index_to_atc.get(drug_idx, [])
        for atc_code in atc_codes:
            hierarchy = parse_atc_hierarchy(atc_code)
            if hierarchy:
                level5_to_adrs[hierarchy["L5"]].add(adr_idx)

    level5_cooccur_weights = {}
    jaccard_values = []

    for l5_1, l5_2 in combinations(level5_to_adrs.keys(), 2):
        adrs1 = level5_to_adrs[l5_1]
        adrs2 = level5_to_adrs[l5_2]
        intersection = len(adrs1 & adrs2)
        union = len(adrs1 | adrs2)
        if union > 0:
            jaccard = intersection / union
            if jaccard > 0:
                jaccard_values.append((jaccard, l5_1, l5_2))

    if jaccard_values:
        jaccard_scores = [v[0] for v in jaccard_values]
        jaccard_arr = np.array(jaccard_scores)
        
        dynamic_threshold = np.percentile(jaccard_arr, jaccard_percentile)
        
        print("\n=== ATC Graph (L5 Co-occurrence) Jaccard Statistics ===")
        print(f"Total co-occurrence pairs calculated: {len(jaccard_values)}")
        print(f"Jaccard Statistics (all pairs):")
        print(f"  Mean: {jaccard_arr.mean():.6f}")
        print(f"  Std:  {jaccard_arr.std():.6f}")
        print(f"  Min:  {jaccard_arr.min():.6f}")
        print(f"  Max:  {jaccard_arr.max():.6f}")
        print(f"  Median: {np.median(jaccard_arr):.6f}")
        print(f"\nDynamic threshold (percentile {jaccard_percentile}): {dynamic_threshold:.6f}")
        
        filtered_jaccard_values = []
        for jaccard, l5_1, l5_2 in jaccard_values:
            if jaccard >= dynamic_threshold:
                level5_cooccur_weights[(l5_1, l5_2)] = jaccard
                level5_cooccur_weights[(l5_2, l5_1)] = jaccard
                filtered_jaccard_values.append(jaccard)
        
        if filtered_jaccard_values:
            filtered_arr = np.array(filtered_jaccard_values)
            print(f"\nAfter filtering (>= {dynamic_threshold:.6f}):")
            print(f"  Number of edges: {len(filtered_jaccard_values)}")
            print(f"  Percentage retained: {len(filtered_jaccard_values)/len(jaccard_values)*100:.2f}%")
            print(f"  Mean Jaccard: {filtered_arr.mean():.6f}")
            print(f"  Std Jaccard:  {filtered_arr.std():.6f}")
            print(f"  Min/Max Jaccard: {filtered_arr.min():.6f} / {filtered_arr.max():.6f}")
    else:
        print("\n=== ATC Graph: No co-occurrence pairs found ===")
        dynamic_threshold = 0.0

    for (l5_1, l5_2), weight in level5_cooccur_weights.items():
        edges.append((l5_1, l5_2, "co_occurrence", weight))

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
    
    return graph, node2idx, index_to_atc


def get_all_atc_l2_codes(index_to_atc):
    l2_codes = set()
    for atc_codes in index_to_atc.values():
        for atc_code in atc_codes:
            if len(atc_code) >= 3:
                l2_codes.add(atc_code[:3])
    return sorted(list(l2_codes))
