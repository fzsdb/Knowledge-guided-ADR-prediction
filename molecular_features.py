import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data
from config import MORGAN_FP_DIM


def get_atom_features(atom):
    atom_types = [
        'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I',
        'B', 'Si', 'Se', 'As', 'Al', 'Zn'
    ]
    atom_type_enc = [1 if atom.GetSymbol() == t else 0 for t in atom_types]
    atom_type_enc.append(1 if atom.GetSymbol() not in atom_types else 0)
    
    degree = min(atom.GetDegree(), 5)
    degree_enc = [1 if i == degree else 0 for i in range(6)]
    
    total_h = min(atom.GetTotalNumHs(), 4)
    total_h_enc = [1 if i == total_h else 0 for i in range(5)]
    
    formal_charge = max(-2, min(atom.GetFormalCharge(), 2))
    
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hybrid_enc = [1 if atom.GetHybridization() == h else 0 for h in hybridization_types]
    hybrid_enc.append(1 if atom.GetHybridization() not in hybridization_types else 0)
    
    is_aromatic = 1 if atom.GetIsAromatic() else 0
    is_in_ring = 1 if atom.IsInRing() else 0
    
    valence = min(atom.GetTotalValence(), 6)
    valence_enc = [1 if i == valence else 0 for i in range(7)]
    
    chiral_tag = atom.GetChiralTag()
    chiral_enc = [
        1 if chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0,
        1 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0,
        1 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0
    ]
    
    features = (atom_type_enc + degree_enc + total_h_enc + 
                [formal_charge] + hybrid_enc + [is_aromatic, is_in_ring] + 
                valence_enc + chiral_enc)
    
    return features


def mol_to_graph_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Data(x=torch.zeros(1, 46), edge_index=torch.empty(2, 0, dtype=torch.long))
    
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append([i, j])
        edge_list.append([j, i])
    
    if len(edge_list) == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


def build_molecule_graphs(smiles_list):
    graphs = []
    for smiles in smiles_list:
        graphs.append(mol_to_graph_data(smiles))
    return graphs


def extract_morgan_fingerprints(smiles_list, device):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_FP_DIM)
    fingerprints = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"Warning: Invalid SMILES string: {smile}")
            fingerprints.append(np.zeros(MORGAN_FP_DIM))
        else:
            fp = morgan_gen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
    fingerprints = torch.FloatTensor(np.array(fingerprints)).to(device)
    return fingerprints
