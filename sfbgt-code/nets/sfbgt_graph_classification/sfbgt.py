import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mlp_readout_layer import MLPReadout


def _masked_diagonal_like(matrix):
    num_nodes = matrix.size(-1)
    diagonal_mask = torch.eye(num_nodes, device=matrix.device, dtype=matrix.dtype)
    return diagonal_mask.unsqueeze(0)


class ConnectivityChannelProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.functional_scale = nn.Parameter(torch.tensor(1.0))
        self.structural_scale = nn.Parameter(torch.tensor(1.0))

    def _prepare_channel(self, adjacency, scale):
        diagonal_mask = _masked_diagonal_like(adjacency)
        adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
        adjacency = adjacency * (1.0 - diagonal_mask)
        return adjacency * scale

    def forward(self, pair_features, functional_channel, structural_channel):
        functional_adjacency = pair_features[..., functional_channel]
        structural_adjacency = pair_features[..., structural_channel]

        functional_adjacency = self._prepare_channel(functional_adjacency, self.functional_scale)
        structural_adjacency = self._prepare_channel(structural_adjacency, self.structural_scale)
        return functional_adjacency, structural_adjacency


class ModalityAwareNodeProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.shared_norm = nn.LayerNorm(input_dim)
        self.shared_projection = nn.Linear(input_dim, hidden_dim)
        self.functional_projection = nn.Linear(input_dim, hidden_dim)
        self.structural_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features):
        normalized_features = self.shared_norm(node_features)
        shared_embedding = self.shared_projection(normalized_features)
        functional_embedding = shared_embedding + self.functional_projection(normalized_features)
        structural_embedding = shared_embedding + self.structural_projection(normalized_features)
        return self.dropout(functional_embedding), self.dropout(structural_embedding)


class DenseResidualMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, adjacency):
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)

        signed_degree = adjacency.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        signed_messages = torch.matmul(adjacency, hidden_states) / signed_degree

        unsigned_adjacency = adjacency.abs()
        unsigned_degree = unsigned_adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        unsigned_messages = torch.matmul(unsigned_adjacency, hidden_states) / unsigned_degree

        message_context = torch.cat(
            (
                hidden_states,
                signed_messages,
                unsigned_messages,
                signed_messages - unsigned_messages,
            ),
            dim=-1,
        )
        candidate_states = self.message_mlp(message_context)
        gating_values = self.gate(message_context)
        updated_states = gating_values * candidate_states + (1.0 - gating_values) * hidden_states
        updated_states = residual + self.output_dropout(updated_states)
        return self.post_norm(updated_states)


class GraphBranchEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DenseResidualMessagePassingLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, initial_states, adjacency):
        hidden_states = initial_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, adjacency)
        return hidden_states


class NodeRelationExtractor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.feature_temperature = math.sqrt(float(hidden_dim))
        self.relation_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, functional_hidden, structural_hidden):
        functional_centered = functional_hidden - functional_hidden.mean(dim=-1, keepdim=True)
        structural_centered = structural_hidden - structural_hidden.mean(dim=-1, keepdim=True)

        covariance = torch.matmul(functional_centered, structural_centered.transpose(-1, -2))
        covariance = covariance / self.feature_temperature

        functional_std = functional_centered.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        structural_std = structural_centered.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        denominator = torch.matmul(functional_std, structural_std.transpose(-1, -2)).clamp_min(1e-6)

        correlation = (covariance / denominator).clamp(-1.0, 1.0)
        assortativity = torch.exp(correlation).unsqueeze(-1)
        return self.relation_mlp(assortativity).squeeze(-1)


class CommonEdgeCoupling(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        hidden_dim = max(output_dim, 8)
        self.projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, functional_adjacency, structural_adjacency, coupling_threshold):
        diagonal_mask = _masked_diagonal_like(functional_adjacency).bool()

        functional_edges = (functional_adjacency.abs() >= coupling_threshold) & ~diagonal_mask
        structural_edges = (structural_adjacency.abs() >= coupling_threshold) & ~diagonal_mask

        common_edges = (functional_edges & structural_edges).float()
        union_edges = (functional_edges | structural_edges).float()

        common_degree = common_edges.sum(dim=-1, keepdim=True)
        union_degree = union_edges.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agreement_ratio = common_degree / union_degree

        edge_statistics = torch.cat((common_degree, agreement_ratio), dim=-1)
        return self.projection(edge_statistics)


class StructuralFunctionalCoupling(nn.Module):
    def __init__(self, hidden_dim, branch_layers, edge_relation_dim, dropout):
        super().__init__()
        self.input_projector = ModalityAwareNodeProjector(hidden_dim, hidden_dim, dropout)
        self.functional_encoder = GraphBranchEncoder(hidden_dim, branch_layers, dropout)
        self.structural_encoder = GraphBranchEncoder(hidden_dim, branch_layers, dropout)
        self.node_relation_extractor = NodeRelationExtractor(hidden_dim)
        self.common_edge_coupling = CommonEdgeCoupling(edge_relation_dim)

    def forward(self, shared_node_embedding, functional_adjacency, structural_adjacency, coupling_threshold):
        functional_seed, structural_seed = self.input_projector(shared_node_embedding)
        functional_hidden = self.functional_encoder(functional_seed, functional_adjacency)
        structural_hidden = self.structural_encoder(structural_seed, structural_adjacency)

        joint_representation = torch.cat((functional_hidden, structural_hidden), dim=-1)
        node_relation_embedding = self.node_relation_extractor(functional_hidden, structural_hidden)
        edge_relation_embedding = self.common_edge_coupling(
            functional_adjacency,
            structural_adjacency,
            coupling_threshold,
        )
        return joint_representation, node_relation_embedding, edge_relation_embedding


class JointRepresentationFusion(nn.Module):
    def __init__(self, joint_dim, edge_relation_dim, output_dim, dropout):
        super().__init__()
        fused_input_dim = joint_dim + edge_relation_dim
        self.shortcut = nn.Linear(fused_input_dim, output_dim)
        self.content_projection = nn.Sequential(
            nn.Linear(fused_input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.gating_projection = nn.Sequential(
            nn.Linear(fused_input_dim, output_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, joint_representation, edge_relation_embedding):
        fusion_input = torch.cat((joint_representation, edge_relation_embedding), dim=-1)
        shortcut = self.shortcut(fusion_input)
        transformed = self.content_projection(fusion_input)
        gate = self.gating_projection(fusion_input)
        fused_representation = gate * transformed + (1.0 - gate) * shortcut
        return self.output_norm(self.output_dropout(fused_representation))


class RelationBiasedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"context_hidden_dim ({hidden_dim}) must be divisible by context_heads ({num_heads})."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        self.relation_head_scale = nn.Parameter(torch.ones(num_heads))
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, relation_bias):
        batch_size, num_nodes, _ = hidden_states.shape

        queries = self.query_projection(hidden_states).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = self.key_projection(hidden_states).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = self.value_projection(hidden_states).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_logits = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        relation_bias = relation_bias.unsqueeze(1) * self.relation_head_scale.view(1, self.num_heads, 1, 1)
        attention_logits = attention_logits + relation_bias

        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        attended_values = self.output_projection(attended_values)
        return self.output_dropout(attended_values)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout):
        super().__init__()
        self.input_projection = nn.Linear(hidden_dim, ffn_dim * 2)
        self.output_projection = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        projected = self.input_projection(hidden_states)
        value_stream, gate_stream = projected.chunk(2, dim=-1)
        gated_values = value_stream * torch.sigmoid(gate_stream)
        gated_values = self.dropout(gated_values)
        return self.output_projection(gated_values)


class RelationAwareContextLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.relation_attention = RelationBiasedMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_dim, ffn_dim, dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, relation_bias):
        attention_input = self.attention_norm(hidden_states)
        hidden_states = hidden_states + self.relation_attention(attention_input, relation_bias)

        feed_forward_input = self.feed_forward_norm(hidden_states)
        hidden_states = hidden_states + self.output_dropout(self.feed_forward(feed_forward_input))
        return hidden_states


class GraphReadout(nn.Module):
    def __init__(self, hidden_dim, readout):
        super().__init__()
        self.readout = readout
        self.summary_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, node_states):
        if self.readout == "sum":
            pooled = node_states.sum(dim=1)
        elif self.readout == "max":
            pooled = node_states.max(dim=1).values
        else:
            pooled = node_states.mean(dim=1)

        max_pooled = node_states.max(dim=1).values
        return self.summary_projection(torch.cat((pooled, max_pooled), dim=-1))


class SfBGT(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        node_feat_dim = net_params["node_feat_dim"]
        num_classes = net_params["num_classes"]

        branch_hidden_dim = net_params.get("branch_hidden_dim", 64)
        branch_layers = net_params.get("branch_layers", 2)
        edge_relation_dim = net_params.get("edge_relation_dim", 16)
        context_hidden_dim = net_params.get("context_hidden_dim", 64)
        context_layers = net_params.get("context_layers", 4)
        context_heads = net_params.get("context_heads", 4)
        interaction_hidden_dim = net_params.get("interaction_hidden_dim", context_hidden_dim * 2)
        coupling_threshold = net_params.get("coupling_threshold", 0.2)
        dropout = net_params.get("dropout", 0.1)
        readout = net_params.get("readout", "sum")
        functional_channel = net_params.get("functional_channel", 0)
        structural_channel = net_params.get("structural_channel", 1)

        self.coupling_threshold = coupling_threshold
        self.functional_channel = functional_channel
        self.structural_channel = structural_channel

        self.node_feature_norm = nn.LayerNorm(node_feat_dim)
        self.node_feature_projection = nn.Linear(node_feat_dim, branch_hidden_dim)
        self.connectivity_projector = ConnectivityChannelProjector()
        self.structural_functional_coupling = StructuralFunctionalCoupling(
            hidden_dim=branch_hidden_dim,
            branch_layers=branch_layers,
            edge_relation_dim=edge_relation_dim,
            dropout=dropout,
        )
        self.joint_representation_fusion = JointRepresentationFusion(
            joint_dim=branch_hidden_dim * 2,
            edge_relation_dim=edge_relation_dim,
            output_dim=context_hidden_dim,
            dropout=dropout,
        )
        self.context_layers = nn.ModuleList(
            [
                RelationAwareContextLayer(
                    hidden_dim=context_hidden_dim,
                    num_heads=context_heads,
                    ffn_dim=interaction_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(context_layers)
            ]
        )
        self.graph_readout = GraphReadout(context_hidden_dim, readout)
        self.classifier = MLPReadout(context_hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, node_features, pair_features):
        normalized_node_features = self.node_feature_norm(node_features)
        shared_node_embedding = self.node_feature_projection(normalized_node_features)

        functional_adjacency, structural_adjacency = self.connectivity_projector(
            pair_features,
            self.functional_channel,
            self.structural_channel,
        )
        (
            joint_representation,
            node_relation_embedding,
            edge_relation_embedding,
        ) = self.structural_functional_coupling(
            shared_node_embedding,
            functional_adjacency,
            structural_adjacency,
            self.coupling_threshold,
        )

        fused_context = self.joint_representation_fusion(
            joint_representation,
            edge_relation_embedding,
        )
        hidden_states = fused_context

        for context_layer in self.context_layers:
            hidden_states = context_layer(hidden_states, node_relation_embedding)

        graph_representation = self.graph_readout(hidden_states)
        return self.classifier(graph_representation)

    def loss(self, logits, targets):
        return self.criterion(logits, targets.long())
