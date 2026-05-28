import pandas as pd
import numpy as np
import torch
from molecular_features import build_molecule_graphs, extract_morgan_fingerprints
from graph_construction_cdan import build_adr_graph, build_atc_graph_global, get_all_atc_l2_codes
import config_cdan as config


def read_raw_data(data_train, data_test, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    smiles_train = pd.read_csv('smiles_traintest.csv')
    n_train = len(smiles_train)
    
    smiles_val = pd.read_csv('smiles_val.csv')
    n_val = len(smiles_val)
    
    total_drugs = n_train + n_val
    offset = n_train
    
    print(f"\n=== Drug Dataset Statistics ===")
    print(f"Training drugs: {n_train}")
    print(f"Validation drugs: {n_val}")
    print(f"Total drugs: {total_drugs}")
    
    smiles_all = (
        smiles_train['Canonical_SMILES'].str.replace(" ", "", regex=False).tolist() +
        smiles_val['Canonical_SMILES'].str.replace(" ", "", regex=False).tolist()
    )
    
    print(f"Building fingerprints and graphs for {total_drugs} drugs...")
    fingerprints = extract_morgan_fingerprints(smiles_all, device)
    mol_graphs = build_molecule_graphs(smiles_all)


    smiles_all_cdan = (
    smiles_train['Canonical_SMILES'].str.replace(" ", "", regex=False).tolist() +
    smiles_val_cdan['Canonical_SMILES'].str.replace(" ", "", regex=False).tolist()
    )
    fingerprints_cdan = extract_morgan_fingerprints(smiles_all_cdan, device)
    mol_graphs_cdan = build_molecule_graphs(smiles_all_cdan)

    print(f"\n=== Loading MolFormer Representations ===")
    molformer_train = pd.read_csv('molformer_train.csv')
    molformer_val = pd.read_csv('molformer_val.csv')
    
    molformer_train_features = molformer_train.iloc[:, 1:].values
    molformer_val_features = molformer_val.iloc[:, 1:].values
    
    molformer_all = np.vstack([molformer_train_features, molformer_val_features])
    molformer_representations = torch.FloatTensor(molformer_all).to(device)
    
    print(f"Loaded MolFormer representations: {molformer_representations.shape}")

    molformer_all_cdan = np.vstack([molformer_train.iloc[:, 1:].values,
                                 molformer_val_cdan.iloc[:, 1:].values])
    molformer_representations_cdan = torch.FloatTensor(molformer_all_cdan).to(device)

    
    index_to_atc_global = {}
    atc_train = pd.read_csv('result_with_atc_traintest.csv')
    train_drug_ids = {
        i: smiles_train.iloc[i]['drugid'] if 'drugid' in smiles_train.columns else f"D{str(i).zfill(5)}"
        for i in range(n_train)
    }
    for idx in range(n_train):
        drug_id = train_drug_ids[idx]
        row = atc_train[atc_train['DRUG_ID'] == drug_id]
        if not row.empty:
            codes = [c.strip() for c in row.iloc[0]['ATC_CODE'].split(';') if len(c.strip()) == 7]
            if codes:
                index_to_atc_global[idx] = codes
    
    atc_val = pd.read_csv('result_with_atc_val.csv')
    val_drug_ids = {
        i: smiles_val.iloc[i]['drugid'] if 'drugid' in smiles_val.columns else f"VAL_{i}"
        for i in range(n_val)
    }
    for local_idx in range(n_val):
        global_idx = offset + local_idx
        drug_id = val_drug_ids[local_idx]
        row = atc_val[atc_val['DRUG_ID'] == drug_id]
        if not row.empty:
            codes = [c.strip() for c in row.iloc[0]['ATC_CODE'].split(';') if len(c.strip()) == 7]
            if codes:
                index_to_atc_global[global_idx] = codes


    index_to_atc_cdan = {}
    for idx in range(n_train):
        drug_id = train_drug_ids[idx]
        row = atc_train[atc_train['DRUG_ID'] == drug_id]
        if not row.empty:
            codes = [c.strip() for c in row.iloc[0]['ATC_CODE'].split(';') if len(c.strip()) == 7]
            if codes:
                index_to_atc_cdan[idx] = codes
    val_drug_ids_cdan = {
        i: smiles_val_cdan.iloc[i]['drugid'] if 'drugid' in smiles_val_cdan.columns else f"VAL_{i}"
        for i in range(len(smiles_val_cdan))
    }
    for local_idx in range(len(smiles_val_cdan)):
        global_idx = offset + local_idx
        drug_id = val_drug_ids_cdan[local_idx]
        row = atc_val_cdan[atc_val_cdan['DRUG_ID'] == drug_id]
        if not row.empty:
            codes = [c.strip() for c in row.iloc[0]['ATC_CODE'].split(';') if len(c.strip()) == 7]
            if codes:
                index_to_atc_cdan[global_idx] = codes
    
    data_test_offset = data_test.copy()
    data_test_offset[:, 0] += offset
    
    adr_graph, adr_node2idx, index_to_adrecs = build_adr_graph(data_train, data_test_offset, device)
    atc_graph, atc_node2idx = build_atc_graph_global(data_train, index_to_atc_global, device)
    
    adr_graph = adr_graph.to(device)
    atc_graph = atc_graph.to(device)
    
    return (fingerprints, mol_graphs, molformer_representations,
            adr_graph, adr_node2idx, index_to_adrecs,
            atc_graph, atc_node2idx, index_to_atc_global, offset,
            fingerprints_cdan, mol_graphs_cdan, molformer_representations_cdan,
            index_to_atc_cdan)
