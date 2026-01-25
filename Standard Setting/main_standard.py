import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import set_seed, count_parameters
from data_loader_cdan import read_raw_data
from graph_construction_cdan import get_all_atc_l2_codes
from models_cdan import SimpleMLPModel_CDAN
from train_eval_cdan import train, model_test
import config_cdan as config


def train_test(data_train, data_test, args):
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    
    (fingerprints, mol_graphs, molformer_representations,
     adr_graph, adr_node2idx, index_to_adrecs, 
     atc_graph, atc_node2idx, index_to_atc, offset) = read_raw_data(data_train, data_test, args)
    
    data_test_offset = data_test.copy()
    data_test_offset[:, 0] += offset
    
    atc_l2_list = get_all_atc_l2_codes(index_to_atc)
    print(f"\n=== ATC L2 Statistics ===")
    print(f"Total unique L2 codes: {len(atc_l2_list)}")
    
    all_classes = atc_l2_list + [None]
    
    trainset = torch.utils.data.TensorDataset(
        torch.LongTensor(data_train[:, 0]),
        torch.LongTensor(data_train[:, 1]),
        torch.LongTensor(data_train[:, 2]),
    )
    testset = torch.utils.data.TensorDataset(
        torch.LongTensor(data_test_offset[:, 0]),
        torch.LongTensor(data_test_offset[:, 1]),
        torch.LongTensor(data_test_offset[:, 2]),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    pin_memory = True if device.type == "cuda" else False
    num_workers = 8 if device.type == "cuda" else 0
    
    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                                        pin_memory=pin_memory, num_workers=num_workers)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, 
                                       pin_memory=pin_memory, num_workers=num_workers)
    _target = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,
                                         pin_memory=pin_memory, num_workers=num_workers)

    model = SimpleMLPModel_CDAN(
        fp_dim=config.MORGAN_FP_DIM, 
        adr_feature_dim=64, 
        atc_feature_dim=64, 
        embed_dim=args.embed_dim, 
        atc_l2_list=atc_l2_list, 
        mol_graphs=mol_graphs,
        dropout=args.droprate,
        use_cdan=args.use_cdan
    ).to(device)
    
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {total_params}")
    
    model.set_graph(adr_graph, adr_node2idx, index_to_adrecs, 
                    atc_graph, atc_node2idx, index_to_atc, 
                    molformer_representations)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
    )
    
    best_test_f1 = 0
    best_model_path = 'best_model_cdan.pth'
    endure_count = 0
    start = time.time()
    
    print(f"\n{'='*100}")
    print(f"Training with CDAN: {args.use_cdan}")
    if args.use_cdan:
        print(f"Domain loss weight: {args.domain_loss_weight}")
    print(f"{'='*100}\n")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, train_precision, train_recall, train_roc_auc, train_pr_auc, train_accuracy, train_lb_loss, train_domain_loss, lambda_p = train(
            fingerprints, adr_graph, adr_node2idx, index_to_adrecs, atc_graph, atc_node2idx, index_to_atc, 
            model, _train, _target, optimizer, criterion, device, atc_l2_list, all_classes, 
            epoch, args.epochs, args.use_cdan, args.domain_loss_weight)
        
        test_loss, test_f1, test_precision, test_recall, test_roc_auc, test_pr_auc, test_accuracy = model_test(
            fingerprints, adr_graph, adr_node2idx, index_to_adrecs, atc_graph, atc_node2idx, index_to_atc, 
            model, _test, device, atc_l2_list, all_classes)
        
        scheduler.step(test_loss)
        time_cost = time.time() - start
        
        print(f"Time: {time_cost:.2f}s | Epoch: {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f}", end="")
        if args.use_cdan:
            print(f" | λ: {lambda_p:.4f}")
        else:
            print()
        
        print(f"[Train] Loss: {train_loss:.4f} | LB: {train_lb_loss:.4f}", end="")
        if args.use_cdan:
            print(f" | Domain: {train_domain_loss:.4f}", end="")
        print(f" | F1: {train_f1:.4f} | ROC-AUC: {train_roc_auc:.4f} | "
              f"PR-AUC: {train_pr_auc:.4f} | Prec: {train_precision:.4f} | Rec: {train_recall:.4f} | "
              f"Acc: {train_accuracy:.4f}")
        print(f"[Test]  Loss: {test_loss:.4f} | F1: {test_f1:.4f} | ROC-AUC: {test_roc_auc:.4f} | "
              f"PR-AUC: {test_pr_auc:.4f} | Prec: {test_precision:.4f} | Rec: {test_recall:.4f} | "
              f"Acc: {test_accuracy:.4f}")
        print("-" * 100)
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_roc_auc = test_roc_auc
            endure_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
            }, best_model_path)
            print(f"✓ Model saved at epoch {epoch} with Test F1: {test_f1:.4f}, ROC-AUC: {test_roc_auc:.4f}\n")
        else:
            endure_count += 1
        
        if endure_count > config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        adr_embeddings = model.adr_rgcn(model.adr_graph.to(device))
        atc_embeddings = model.atc_rgcn(model.atc_graph.to(device))
    
    pt_embeddings = {}
    for idx, adrecs_ids in index_to_adrecs.items():
        for adrecs_id in adrecs_ids:
            pt_idx = adr_node2idx.get(adrecs_id, None)
            if pt_idx is not None:
                pt_embeddings[f"{idx}_{adrecs_id}"] = adr_embeddings[pt_idx].cpu().numpy()
    
    atc_level5_embeddings = {}
    for idx, atc_codes in index_to_atc.items():
        for atc_code in atc_codes:
            atc_idx = atc_node2idx.get(atc_code, None)
            if atc_idx is not None:
                atc_level5_embeddings[(idx, atc_code)] = atc_embeddings[atc_idx].cpu().numpy()
    
    embedding_df = pd.DataFrame.from_dict(pt_embeddings, orient="index")
    embedding_df.to_csv("adr_embeddings_learned_cdan.csv")
    atc_embedding_df = pd.DataFrame.from_dict(atc_level5_embeddings, orient="index")
    atc_embedding_df.to_csv("atc_embeddings_learned_cdan.csv")
    
    return best_test_roc_auc


def main():
    parser = argparse.ArgumentParser(description='Drug-ADR Prediction with CDAN')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=config.EMBED_DIM, help='embedding dimension')
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY, help='weight decay')
    parser.add_argument('--droprate', type=float, default=config.DROPOUT_RATE, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=config.TEST_BATCH_SIZE, help='input batch size for testing')
    parser.add_argument('--rawpath', type=str, default='/data/', help='rawpath')
    parser.add_argument('--seed', type=int, default=config.SEED, help='random seed for reproducibility')
    parser.add_argument('--use_cdan', action='store_true', default=config.USE_CDAN, 
                       help='Use Conditional Domain Adversarial Network')
    parser.add_argument('--domain_loss_weight', type=float, default=config.DOMAIN_LOSS_WEIGHT, 
                       help='Weight for domain adversarial loss')
    
    args, _ = parser.parse_known_args()
    
    set_seed(args.seed)
    
    print('='*100)
    print('HYPERPARAMETERS')
    print('='*100)
    print(f'Random seed: {args.seed}')
    print(f'Weight decay: {args.weight_decay}')
    print(f'Dropout rate: {args.droprate}')
    print(f'Learning rate: {args.lr}')
    print(f'Embedding dimension: {args.embed_dim}')
    print(f'Use CDAN: {args.use_cdan}')
    if args.use_cdan:
        print(f'Domain loss weight: {args.domain_loss_weight}')
    print('='*100)
    print('FEATURE DIMENSIONS')
    print('='*100)
    print(f'Morgan fingerprint: {config.MORGAN_FP_DIM}')
    print(f'Morgan reduced: {config.MORGAN_REDUCED_DIM}')
    print(f'GNN output: {config.GNN_OUTPUT_DIM}')
    print(f'MolFormer original: {config.MOLFORMER_DIM}')
    print(f'MolFormer reduced: {config.MOLFORMER_REDUCED_DIM}')
    print(f'Fused drug feature: {config.FUSED_DRUG_DIM}')
    print(f'Atom feature: 46')
    print('='*100)
    
    try:
        data_train = pd.read_csv('train.csv')
        data_test = pd.read_csv('val_bal.csv')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e}")
    
    data_train = np.array(data_train).astype(int)
    data_test = np.array(data_test).astype(int)
    
    association_auc = train_test(data_train, data_test, args)
    
    print(f"\n{'='*100}")
    print(f"FINAL RESULTS")
    print(f"{'='*100}")
    print(f"Best Test ROC-AUC: {association_auc:.5f}")
    print(f"{'='*100}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
