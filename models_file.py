import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from config import MORGAN_FP_DIM, MORGAN_REDUCED_DIM, GNN_OUTPUT_DIM, FUSED_DRUG_DIM


class MoleculeGIN(nn.Module):
    def __init__(self, input_dim=46, hidden_dim=64, output_dim=32, num_layers=3, dropout=0.3):
        super(MoleculeGIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(self._gin_conv(nn1))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(self._gin_conv(nn_layer))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        nn_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.convs.append(self._gin_conv(nn_final))
        self.bns.append(nn.BatchNorm1d(output_dim))
        
    def _gin_conv(self, nn_module):
        class GINConv(nn.Module):
            def __init__(self, nn, eps=0.0):
                super().__init__()
                self.nn = nn
                self.eps = eps
                
            def forward(self, x, edge_index):
                row, col = edge_index
                if edge_index.size(1) > 0:
                    aggr = scatter(x[col], row, dim=0, dim_size=x.size(0), reduce='sum')
                else:
                    aggr = torch.zeros_like(x)
                out = (1 + self.eps) * x + aggr
                return self.nn(out)
        
        return GINConv(nn_module)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_add_pool(x, batch)
        return x


class WeightedRGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, aggr='mean'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.aggr = aggr

        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(num_relations)
        ])
        self.lin_self = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_self.reset_parameters()

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        out = self.lin_self(x)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            edge_index_r = edge_index[:, mask]
            edge_weight_r = edge_weight[mask]

            row, col = edge_index_r
            deg = scatter(edge_weight_r, col, dim=0, dim_size=x.size(0), reduce='sum')
            deg_inv = 1.0 / deg.clamp(min=1e-12)
            deg_inv[deg == 0] = 0

            h = self.lins[r](x)
            msg = h[row] * edge_weight_r.unsqueeze(-1)
            msg = msg * deg_inv[col].unsqueeze(-1)
            out += scatter(msg, col, dim=0, dim_size=x.size(0), reduce='sum')

        return out


class WeightedADRRGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, dropout=0.5):
        super(WeightedADRRGCN, self).__init__()
        self.conv1 = WeightedRGCNConv(input_dim, hidden_dim, num_relations, aggr="mean")
        self.conv2 = WeightedRGCNConv(hidden_dim, output_dim, num_relations, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        x = self.conv1(x, edge_index, edge_type, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type, edge_weight=edge_weight)
        x = self.dropout(x)
        return x


class MoE_Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=32, dropout=0.3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, momentum=0.5),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, fused_feat):
        return self.mlp(fused_feat)


class DynamicPromptCrossAttention(nn.Module):
    def __init__(self, drug_dim=288, adr_dim=256, hidden_dim=128,
                 num_heads=4, num_prompts=8, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.num_prompts = num_prompts
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.prompt_pool = nn.Parameter(torch.randn(num_prompts, hidden_dim))
        nn.init.xavier_uniform_(self.prompt_pool)
        
        self.prompt_selector = nn.Sequential(
            nn.Linear(drug_dim + adr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_prompts),
            nn.Softmax(dim=-1)
        )
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_drug = nn.Linear(drug_dim, hidden_dim)
        self.W_k_adr = nn.Linear(adr_dim, hidden_dim)
        self.W_v_drug = nn.Linear(drug_dim, hidden_dim)
        self.W_v_adr = nn.Linear(adr_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, drug_feat, adr_feat):
        batch_size = drug_feat.size(0)
        
        context = torch.cat([drug_feat, adr_feat], dim=1)
        prompt_weights = self.prompt_selector(context)
        
        selected_prompt = torch.matmul(
            prompt_weights.unsqueeze(1),
            self.prompt_pool
        ).squeeze(1)
        
        Q = self.W_q(selected_prompt)
        
        K_drug = self.W_k_drug(drug_feat)
        K_adr = self.W_k_adr(adr_feat)
        K = torch.stack([K_drug, K_adr], dim=1)
        
        V_drug = self.W_v_drug(drug_feat)
        V_adr = self.W_v_adr(adr_feat)
        V = torch.stack([V_drug, V_adr], dim=1)
        
        Q = Q.view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)
        K = K.view(batch_size, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V).squeeze(2)
        attended = attended.contiguous().view(batch_size, -1)
        
        output = self.out_proj(attended)
        output = self.layer_norm(output)
        
        drug_adr_weights = attention_weights.squeeze(2).mean(dim=1)
        
        return output, drug_adr_weights, prompt_weights


class DualMoE(nn.Module):
    def __init__(self, fused_drug_dim, adr_feature_dim, atc_feature_dim,
                 hidden_dim, output_dim, atc_l2_list, dropout=0.6):
        super(DualMoE, self).__init__()
        self.atc_l2_list = atc_l2_list
        
        self.num_major_experts = len(atc_l2_list)
        self.num_general_experts = 10
        self.top_k_general = 5
        self.expert_mapping = {atc_l2: idx for idx, atc_l2 in enumerate(atc_l2_list)}
        
        print(f"\n=== DualMoE Configuration ===")
        print(f"Number of L2-based Major Experts: {self.num_major_experts}")
        print(f"Number of General Experts: {self.num_general_experts}")
        print(f"Top-K for General Experts: {self.top_k_general}")
        print(f"Sample L2 mappings: {dict(list(self.expert_mapping.items())[:5])}")
        
        self.shared_attention = DynamicPromptCrossAttention(
            drug_dim=fused_drug_dim,
            adr_dim=adr_feature_dim * 4,
            hidden_dim=128,
            num_heads=4,
            num_prompts=8,
            dropout=dropout
        )
        
        expert_input_dim = 128 + fused_drug_dim + adr_feature_dim * 4
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
            target_usage = self.top_k_general / self.num_general_experts
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


class SimpleMLPModel(nn.Module):
    def __init__(self, fp_dim, adr_feature_dim, atc_feature_dim, 
                 embed_dim, atc_l2_list, mol_graphs, dropout=0.3): 
        super(SimpleMLPModel, self).__init__()
        self.adr_feature_dim = adr_feature_dim
        self.atc_feature_dim = atc_feature_dim
        self.atc_l2_list = atc_l2_list
        self.max_adr_ids = 4
        self.max_atc_codes = 6
        self.mol_graphs = mol_graphs
        
        self.morgan_projection = nn.Sequential(
            nn.Linear(MORGAN_FP_DIM, MORGAN_REDUCED_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(MORGAN_REDUCED_DIM),
            nn.Dropout(dropout)
        )
        
        self.mol_gnn = MoleculeGIN(
            input_dim=46,
            hidden_dim=64,
            output_dim=GNN_OUTPUT_DIM,
            num_layers=3,
            dropout=dropout
        )
        
        self.moe = DualMoE(
            fused_drug_dim=FUSED_DRUG_DIM,
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
        
        self.adr_rgcn = WeightedADRRGCN(input_dim=128, hidden_dim=64, output_dim=adr_feature_dim, num_relations=4, dropout=dropout)
        self.atc_rgcn = WeightedADRRGCN(input_dim=128, hidden_dim=64, output_dim=atc_feature_dim, num_relations=4, dropout=dropout)
        self.adr_node2idx = None
        self.index_to_adrecs = None
        self.atc_node2idx = None
        self.index_to_atc = None

    def set_graph(self, adr_graph, adr_node2idx, index_to_adrecs, atc_graph, atc_node2idx, index_to_atc):
        self.adr_graph = adr_graph
        self.adr_node2idx = adr_node2idx
        self.index_to_adrecs = index_to_adrecs
        self.atc_graph = atc_graph
        self.atc_node2idx = atc_node2idx
        self.index_to_atc = index_to_atc
        self.adr_id_to_idx = {idx: [adr_node2idx.get(adrecs_id, None) for adrecs_id in adrecs_ids] for idx, adrecs_ids in index_to_adrecs.items()}
        self.atc_id_to_idx = {idx: [atc_node2idx.get(atc_code, None) for atc_code in atc_codes] for idx, atc_codes in index_to_atc.items()}
        
    def forward(self, drug_fp, adr_indices, drug_indices, device):
        batch_size = drug_fp.size(0)
        
        morgan_reduced = self.morgan_projection(drug_fp)
        
        batch_mol_graphs = [self.mol_graphs[idx.item()].to(device) for idx in drug_indices]
        batch_data = Batch.from_data_list(batch_mol_graphs).to(device)
        gnn_features = self.mol_gnn(batch_data)
        
        fused_drug_fp = torch.cat([morgan_reduced, gnn_features], dim=1)
        
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
       
        moe_output, gate_weights, attention_analysis = self.moe(fused_drug_fp, batch_adr_features, batch_atc_features, batch_atc_codes_list)
        output = self.classifier(moe_output)
        return output, gate_weights, attention_analysis
