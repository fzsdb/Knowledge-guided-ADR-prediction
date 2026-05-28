import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from models import MoleculeGIN, WeightedADRRGCN, MoE_Expert, DynamicPromptCrossAttention
from cdan_modules import GradientReversal, ConditionalDomainDiscriminator
import config_cdan as config


class DualMoE_CDAN(nn.Module):
    def __init__(self, fused_drug_dim, adr_feature_dim, atc_feature_dim,
                 hidden_dim, output_dim, atc_l2_list, dropout=None):
        super(DualMoE_CDAN, self).__init__()
        dropout = dropout or config.DROPOUT_RATE
        self.atc_l2_list = atc_l2_list
        
        self.num_major_experts = len(atc_l2_list)
        self.num_general_experts = config.NUM_GENERAL_EXPERTS
        self.top_k_general = config.TOP_K_GENERAL
        self.expert_mapping = {atc_l2: idx for idx, atc_l2 in enumerate(atc_l2_list)}
        
        print(f"\n=== DualMoE Configuration ===")
        print(f"Number of L2-based Major Experts: {self.num_major_experts}")
        print(f"Number of General Experts: {self.num_general_experts}")
        print(f"Top-K for General Experts: {self.top_k_general}")
        
        self.shared_attention = DynamicPromptCrossAttention(
            drug_dim=fused_drug_dim,
            adr_dim=adr_feature_dim * 4,
            hidden_dim=config.ATTENTION_HIDDEN_DIM,
            num_heads=config.ATTENTION_NUM_HEADS,
            num_prompts=config.ATTENTION_NUM_PROMPTS,
            dropout=dropout
        )
        
        expert_input_dim = config.ATTENTION_HIDDEN_DIM + fused_drug_dim + adr_feature_dim * 4
        gate_major_input_dim = atc_feature_dim * 6 + adr_feature_dim * 4
        gate_general_input_dim = atc_feature_dim * 6 + adr_feature_dim * 4
        
        self.major_experts = nn.ModuleList([
            MoE_Expert(
                input_dim=expert_input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim // 2,
                dropout=dropout
            )
            for _ in range(self.num_major_experts)
        ])
        
        self.gate_major = nn.Sequential(
            nn.Linear(gate_major_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_major_experts),
            nn.Softmax(dim=-1)
        )
        
        self.general_experts = nn.ModuleList([
            MoE_Expert(
                input_dim=expert_input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim // 2,
                dropout=dropout
            )
            for _ in range(self.num_general_experts)
        ])
        
        self.gate_general = nn.Sequential(
            nn.Linear(gate_general_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_general_experts),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, fused_drug_fp, batch_adr_features, batch_atc_features, atc_codes_list):
        batch_size = batch_atc_features.size(0)
        
        attention_output, drug_adr_weights, prompt_weights = self.shared_attention(
            fused_drug_fp, batch_adr_features
        )
        
        expert_input = torch.cat([attention_output, fused_drug_fp, batch_adr_features], dim=1)
        
        gate_input_major = torch.cat([batch_atc_features, batch_adr_features], dim=1)
        gate_input_general = torch.cat([batch_atc_features, batch_adr_features], dim=1)
        
        gate_weights_major = self.gate_major(gate_input_major)
        major_experts_mask = torch.zeros(batch_size, self.num_major_experts, 
                                        device=batch_atc_features.device)
        
        for i in range(batch_size):
            atc_codes = atc_codes_list[i] if i < len(atc_codes_list) else []
            has_major_atc = False
            if not atc_codes:
                atc_codes = ['Unknown']
            
            for atc_code in atc_codes:
                if isinstance(atc_code, str) and len(atc_code) >= 3:
                    atc_l2 = atc_code[:3]
                    if atc_l2 in self.expert_mapping:
                        expert_idx = self.expert_mapping[atc_l2]
                        major_experts_mask[i, expert_idx] = 1.0
                        has_major_atc = True
            
            if not has_major_atc:
                top_expert_idx = torch.argmax(gate_weights_major[i])
                major_experts_mask[i, top_expert_idx] = 1.0
        
        gate_weights_major = gate_weights_major * major_experts_mask
        gate_weights_major = gate_weights_major / (gate_weights_major.sum(dim=1, keepdim=True) + 1e-10)
        
        major_expert_outputs = []
        for expert in self.major_experts:
            expert_out = expert(expert_input)
            major_expert_outputs.append(expert_out)
        
        major_expert_outputs = torch.stack(major_expert_outputs, dim=1)
        gate_weights_major_expanded = gate_weights_major.unsqueeze(-1)
        major_output = torch.sum(major_expert_outputs * gate_weights_major_expanded, dim=1)
        
        gate_logits_general = self.gate_general[:-1](gate_input_general)
        
        top_k_values, top_k_indices = torch.topk(gate_logits_general, self.top_k_general, dim=1)
        
        general_experts_mask = torch.zeros_like(gate_logits_general)
        general_experts_mask.scatter_(1, top_k_indices, 1.0)
        
        masked_logits = gate_logits_general.masked_fill(general_experts_mask == 0, float('-inf'))
        gate_weights_general = torch.softmax(masked_logits, dim=-1)
        
        if self.training:
            expert_usage = general_experts_mask.mean(dim=0)
            load_balance_loss = torch.var(expert_usage) * self.num_general_experts
        else:
            load_balance_loss = torch.tensor(0.0, device=gate_weights_general.device)
        
        general_expert_outputs = []
        for expert in self.general_experts:
            expert_out = expert(expert_input)
            general_expert_outputs.append(expert_out)
        
        general_expert_outputs = torch.stack(general_expert_outputs, dim=1)
        gate_weights_general_expanded = gate_weights_general.unsqueeze(-1)
        general_output = torch.sum(general_expert_outputs * gate_weights_general_expanded, dim=1)
        
        output = torch.cat([major_output, general_output], dim=-1)
        combined_gate_weights = torch.cat([gate_weights_major, gate_weights_general], dim=-1)
        
        attention_analysis = {
            'shared_drug_adr_weights': drug_adr_weights,
            'shared_prompt_weights': prompt_weights,
            'load_balance_loss': load_balance_loss,
            'general_expert_usage': general_experts_mask.sum(dim=0) / batch_size
        }
        
        return output, combined_gate_weights, attention_analysis


class SimpleMLPModel_CDAN(nn.Module):
    def __init__(self, fp_dim, adr_feature_dim, atc_feature_dim, 
                 embed_dim, atc_l2_list, mol_graphs, dropout=None, use_cdan=True): 
        super(SimpleMLPModel_CDAN, self).__init__()
        dropout = dropout or config.DROPOUT_RATE
        
        self.adr_feature_dim = adr_feature_dim
        self.atc_feature_dim = atc_feature_dim
        self.atc_l2_list = atc_l2_list
        self.max_adr_ids = config.MAX_ADR_IDS
        self.max_atc_codes = config.MAX_ATC_CODES
        self.mol_graphs = mol_graphs
        self.use_cdan = use_cdan
        
        self.morgan_projection = nn.Sequential(
            nn.Linear(config.MORGAN_FP_DIM, config.MORGAN_REDUCED_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(config.MORGAN_REDUCED_DIM),
            nn.Dropout(dropout)
        )
        
        self.molformer_projection = nn.Sequential(
            nn.Linear(config.MOLFORMER_DIM, config.MOLFORMER_REDUCED_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(config.MOLFORMER_REDUCED_DIM),
            nn.Dropout(dropout)
        )
        
        self.mol_gnn = MoleculeGIN(
            input_dim=46,
            hidden_dim=config.GIN_HIDDEN_DIM,
            output_dim=config.GNN_OUTPUT_DIM,
            num_layers=config.GIN_NUM_LAYERS,
            dropout=dropout
        )
        
        self.moe = DualMoE_CDAN(
            fused_drug_dim=config.FUSED_DRUG_DIM,
            adr_feature_dim=adr_feature_dim,
            atc_feature_dim=atc_feature_dim,
            hidden_dim=embed_dim, 
            output_dim=embed_dim, 
            atc_l2_list=atc_l2_list, 
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim, momentum=0.5),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim, momentum=0.5),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2)
        )
        
        if self.use_cdan:
            self.gradient_reversal = GradientReversal(lambda_=1.0)
            self.domain_discriminator = ConditionalDomainDiscriminator(
                feature_dim=embed_dim,
                num_classes=2,
                hidden_dim=config.CDAN_DISCRIMINATOR_HIDDEN_DIM,
                dropout=dropout
            )
            print("\n=== CDAN Enabled ===")
            print(f"Domain discriminator input: {embed_dim} * 2 = {embed_dim * 2}")
        
        self.adr_rgcn = WeightedADRRGCN(
            input_dim=config.RGCN_INPUT_DIM, 
            hidden_dim=config.RGCN_HIDDEN_DIM, 
            output_dim=adr_feature_dim, 
            num_relations=config.NUM_RELATIONS, 
            dropout=dropout
        )
        self.atc_rgcn = WeightedADRRGCN(
            input_dim=config.RGCN_INPUT_DIM, 
            hidden_dim=config.RGCN_HIDDEN_DIM, 
            output_dim=atc_feature_dim, 
            num_relations=config.NUM_RELATIONS, 
            dropout=dropout
        )
        
        self.adr_node2idx = None
        self.index_to_adrecs = None
        self.atc_node2idx = None
        self.index_to_atc = None
        self.molformer_representations = None

    def set_graph(self, adr_graph, adr_node2idx, index_to_adrecs, atc_graph, atc_node2idx, 
                  index_to_atc, molformer_representations):
        self.adr_graph = adr_graph
        self.adr_node2idx = adr_node2idx
        self.index_to_adrecs = index_to_adrecs
        self.atc_graph = atc_graph
        self.atc_node2idx = atc_node2idx
        self.index_to_atc = index_to_atc
        self.adr_id_to_idx = {idx: [adr_node2idx.get(adrecs_id, None) for adrecs_id in adrecs_ids] for idx, adrecs_ids in index_to_adrecs.items()}
        self.atc_id_to_idx = {idx: [atc_node2idx.get(atc_code, None) for atc_code in atc_codes] for idx, atc_codes in index_to_atc.items()}
        self.molformer_representations = molformer_representations
    
    def set_lambda(self, lambda_):
        if self.use_cdan:
            self.gradient_reversal.lambda_ = lambda_
        
    def forward(self, drug_fp, adr_indices, drug_indices, device):
        batch_size = drug_fp.size(0)
        
        morgan_reduced = self.morgan_projection(drug_fp)
        
        batch_mol_graphs = [self.mol_graphs[idx.item()].to(device) for idx in drug_indices]
        batch_data = Batch.from_data_list(batch_mol_graphs).to(device)
        gnn_features = self.mol_gnn(batch_data)
        
        batch_molformer = self.molformer_representations[drug_indices].to(device)
        molformer_reduced = self.molformer_projection(batch_molformer)
        
        fused_drug_fp = torch.cat([morgan_reduced, gnn_features, molformer_reduced], dim=1)
        
        adr_embeddings = self.adr_rgcn(self.adr_graph.to(device))
        batch_adr_features = torch.zeros(batch_size, self.adr_feature_dim * self.max_adr_ids, device=device)
        for batch_idx, adr_idx in enumerate(adr_indices.cpu().numpy()):
            adrecs_ids = self.index_to_adrecs.get(adr_idx, [])
            node_indices = self.adr_id_to_idx.get(adr_idx, [])
            valid_indices = [idx for idx in node_indices if idx is not None]
            if not valid_indices:
                continue
            adr_embeds = adr_embeddings[torch.tensor(valid_indices, device=device)]
            if len(valid_indices) < self.max_adr_ids:
                padding = torch.zeros(self.max_adr_ids - len(valid_indices), self.adr_feature_dim, device=device)
                adr_embeds = torch.cat([adr_embeds, padding], dim=0)
            batch_adr_features[batch_idx] = adr_embeds.view(-1)[:self.adr_feature_dim * self.max_adr_ids]
       
        atc_embeddings = self.atc_rgcn(self.atc_graph.to(device))
        batch_atc_features = torch.zeros(batch_size, self.atc_feature_dim * self.max_atc_codes, device=device)
        batch_atc_codes_list = []
        for batch_idx, drug_idx in enumerate(drug_indices.cpu().numpy()):
            atc_codes = self.index_to_atc.get(drug_idx, [])
            batch_atc_codes_list.append(atc_codes)
            atc_indices = self.atc_id_to_idx.get(drug_idx, [])
            valid_indices = [idx for idx in atc_indices if idx is not None]
            if not valid_indices:
                continue
            atc_embeds = atc_embeddings[torch.tensor(valid_indices, device=device)]
            if len(valid_indices) < self.max_atc_codes:
                padding = torch.zeros(self.max_atc_codes - len(valid_indices), self.atc_feature_dim, device=device)
                atc_embeds = torch.cat([atc_embeds, padding], dim=0)
            batch_atc_features[batch_idx] = atc_embeds.view(-1)[:self.atc_feature_dim * self.max_atc_codes]
       
        moe_output, gate_weights, attention_analysis = self.moe(
            fused_drug_fp, batch_adr_features, batch_atc_features, batch_atc_codes_list
        )
        
        adr_pred = self.classifier(moe_output)
        
        domain_pred = None
        if self.use_cdan and self.training:
            reversed_features = self.gradient_reversal(moe_output)
            domain_pred = self.domain_discriminator(reversed_features, adr_pred)
        
        return adr_pred, gate_weights, attention_analysis, domain_pred
