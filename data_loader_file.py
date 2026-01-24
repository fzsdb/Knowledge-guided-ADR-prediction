import pandas as pd
import torch
from molecular_features import build_molecule_graphs, extract_morgan_fingerprints
from graph_construction import build_adr_graph, build_atc_graph, get_all_atc_l2_codes


def read_raw_data(data_train, data_test, args):
    try:
        drug_smiles = pd.read_csv('smiles_traintest.csv')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    Smiles = drug_smiles['Canonical_SMILES'].str.replace(" ", "", regex=False).tolist()
    fingerprints = extract_morgan_fingerprints(Smiles, device)
    
    print("Building molecular graphs...")
    mol_graphs = build_molecule_graphs(Smiles)
    print(f"Built {len(mol_graphs)} molecular graphs")
    
    adr_graph, adr_node2idx, index_to_adrecs = build_adr_graph(data_train, data_test, device)
    atc_graph, atc_node2idx, index_to_atc = build_atc_graph(data_train, data_test, device)
    adr_graph = adr_graph.to(device)
    atc_graph = atc_graph.to(device)
    
    return (fingerprints, mol_graphs, adr_graph, adr_node2idx, index_to_adrecs, 
            atc_graph, atc_node2idx, index_to_atc)
