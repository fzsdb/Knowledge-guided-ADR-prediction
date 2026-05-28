import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
import config


def train(fingerprints, adr_graph, adr_node2idx, index_to_adrecs, atc_graph, atc_node2idx, index_to_atc, 
          model, train_loader, optimizer, loss_function, device, atc_l2_list, all_classes):
    model.train()
    avg_loss = 0.0
    avg_lb_loss = 0.0
    pred = []
    a_truth = []
    
    for i, data in enumerate(train_loader, 0):
        index_drug, index_side, batch_a = [d.to(device) for d in data]
        optimizer.zero_grad()
        batch_fp = fingerprints[index_drug].to(device)
        a_score, batch_gate_weights, attention_analysis = model(batch_fp, index_side, index_drug, device)
        
        main_loss = loss_function(a_score, batch_a)
        lb_loss = attention_analysis['load_balance_loss']
        total_loss = main_loss + config.LOAD_BALANCE_WEIGHT * lb_loss
        
        total_loss.backward()
        optimizer.step()
        
        avg_loss += main_loss.item()
        avg_lb_loss += lb_loss.item()
        pred.append(a_score.data.cpu().numpy())
        a_truth.append(batch_a.data.cpu().numpy())
    
    avg_loss = avg_loss / len(train_loader)
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
    
    return avg_loss, f1, precision, recall, roc_auc, pr_auc, accuracy, avg_lb_loss


def model_test(fingerprints, adr_graph, adr_node2idx, index_to_adrecs, atc_graph, atc_node2idx, index_to_atc, 
               model, test_loader, device, atc_l2_list, all_classes):
    model.eval()
    pred = []
    a_truth = []
    total_loss = 0.0
    loss_function = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for index_drug, index_side, test_a in test_loader:
            index_drug, index_side, test_a = index_drug.to(device), index_side.to(device), test_a.to(device)
            test_fp = fingerprints[index_drug].to(device)
            scores, _, _ = model(test_fp, index_side, index_drug, device)
            loss = loss_function(scores, test_a)
            total_loss += loss.item()
            pred.append(scores.data.cpu().numpy())
            a_truth.append(test_a.data.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    pred = np.concatenate(pred, axis=0)
    pred1 = torch.softmax(torch.tensor(pred), dim=1).numpy()
    a_truth = np.concatenate(a_truth, axis=0)
    pred_binary = np.argmax(pred, axis=1)
    
    f1 = f1_score(a_truth, pred_binary)
    precision = precision_score(a_truth, pred_binary)
    recall = recall_score(a_truth, pred_binary)
    accuracy = accuracy_score(a_truth, pred_binary)
    roc_auc = roc_auc_score(a_truth, pred[:, 1])
    pr_auc = average_precision_score(a_truth, pred[:, 1])
    return avg_loss, f1, precision, recall, roc_auc, pr_auc, accuracy, pred1[:, 1]
