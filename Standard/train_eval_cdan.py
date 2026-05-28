import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
import config_cdan as config


def train(fingerprints, adr_graph, adr_node2idx, index_to_adrecs, atc_graph, atc_node2idx, 
          index_to_atc, model, train_loader, target_loader, optimizer, loss_function, 
          device, atc_l2_list, all_classes, epoch, total_epochs, use_cdan=True, 
          domain_loss_weight=None):
    domain_loss_weight = domain_loss_weight or config.DOMAIN_LOSS_WEIGHT
    
    model.train()
    
    if use_cdan:
        p = float(epoch) / float(total_epochs)
        lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
        model.set_lambda(lambda_p)
    
    avg_loss = 0.0
    avg_domain_loss = 0.0
    avg_lb_loss = 0.0
    pred = []
    a_truth = []
    
    target_iter = iter(target_loader) if use_cdan else None
    
    for i, source_data in enumerate(train_loader, 0):
        index_drug_s, index_side_s, batch_a_s = [d.to(device) for d in source_data]
        
        optimizer.zero_grad()
        batch_fp_s = fingerprints[index_drug_s].to(device)
        a_score_s, batch_gate_weights_s, attention_analysis_s, domain_pred_s = model(
            batch_fp_s, index_side_s, index_drug_s, device
        )
        
        main_loss = loss_function(a_score_s, batch_a_s)
        lb_loss = attention_analysis_s['load_balance_loss']
        
        domain_loss = torch.tensor(0.0, device=device)
        if use_cdan and domain_pred_s is not None:
            try:
                target_data = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data = next(target_iter)
            
            index_drug_t, index_side_t, _ = [d.to(device) for d in target_data]
            batch_fp_t = fingerprints[index_drug_t].to(device)
            
            _, _, _, domain_pred_t = model(batch_fp_t, index_side_t, index_drug_t, device)
            
            if domain_pred_t is not None:
                domain_label_s = torch.zeros(domain_pred_s.size(0), 1).to(device)
                domain_label_t = torch.ones(domain_pred_t.size(0), 1).to(device)
                
                domain_loss_s = F.binary_cross_entropy(domain_pred_s, domain_label_s)
                domain_loss_t = F.binary_cross_entropy(domain_pred_t, domain_label_t)
                domain_loss = domain_loss_s + domain_loss_t
        
        total_loss = main_loss + config.LOAD_BALANCE_WEIGHT * lb_loss + domain_loss_weight * domain_loss
        
        total_loss.backward()
        optimizer.step()
        
        avg_loss += main_loss.item()
        avg_domain_loss += domain_loss.item()
        avg_lb_loss += lb_loss.item()
        pred.append(a_score_s.data.cpu().numpy())
        a_truth.append(batch_a_s.data.cpu().numpy())
    
    avg_loss = avg_loss / len(train_loader)
    avg_domain_loss = avg_domain_loss / len(train_loader)
    avg_lb_loss = avg_lb_loss / len(train_loader)
    pred = np.concatenate(pred, axis=0)
    a_truth = np.concatenate(a_truth, axis=0)
    pred_binary = np.argmax(pred, axis=1)
    
    f1 = f1_score(a_truth, pred_binary)
    precision = precision_score(a_truth, pred_binary)
    recall = recall_score(a_truth, pred_binary)
    accuracy = accuracy_score(a_truth, pred_binary)
    try:
        roc_auc = roc_auc_score(a_truth, pred[:, 1])
        pr_auc = average_precision_score(a_truth, pred[:, 1])
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0
    
    return avg_loss, f1, precision, recall, roc_auc, pr_auc, accuracy, avg_lb_loss, avg_domain_loss, lambda_p if use_cdan else 0.0


def model_test(fingerprints, adr_graph, adr_node2idx, index_to_adrecs, atc_graph, atc_node2idx, 
               index_to_atc, model, test_loader, device, atc_l2_list, all_classes):
    model.eval()
    pred = []
    a_truth = []
    total_loss = 0.0
    loss_function = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for index_drug, index_side, test_a in test_loader:
            index_drug, index_side, test_a = index_drug.to(device), index_side.to(device), test_a.to(device)
            test_fp = fingerprints[index_drug].to(device)
            scores, _, _, _ = model(test_fp, index_side, index_drug, device)
            loss = loss_function(scores, test_a)
            total_loss += loss.item()
            pred.append(scores.data.cpu().numpy())
            a_truth.append(test_a.data.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    pred = np.concatenate(pred, axis=0)
    a_truth = np.concatenate(a_truth, axis=0)
    pred_binary = np.argmax(pred, axis=1)
    
    f1 = f1_score(a_truth, pred_binary)
    precision = precision_score(a_truth, pred_binary)
    recall = recall_score(a_truth, pred_binary)
    accuracy = accuracy_score(a_truth, pred_binary)
    roc_auc = roc_auc_score(a_truth, pred[:, 1])
    pr_auc = average_precision_score(a_truth, pred[:, 1])
    
    return avg_loss, f1, precision, recall, roc_auc, pr_auc, accuracy
