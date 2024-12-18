import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(InvertedEmbedding, self).__init__()
        self.embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)

        if x_mark is None:
            x = self.embedding(x)
        else:
            x = self.embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        
        return self.dropout(x)
    


class VariableSelectionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(VariableSelectionNetwork, self).__init__()
        
        self.gating_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embeddings):
        gates = self.gating_network(embeddings)
        gates = self.softmax(gates)

        selected_embeddings = embeddings * gates

        return selected_embeddings
    



class EmbeddingWithSelection(nn.Module):
    def __init__(self, c_in, d_model, hidden_dim, dropout=0.1):
        super(EmbeddingWithSelection, self).__init__()
        self.inverted_embedding = InvertedEmbedding(c_in, d_model, dropout)
        self.vsn = VariableSelectionNetwork(d_model, hidden_dim)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, x_mark=None):
        original_embedding = self.inverted_embedding(x, x_mark)

        selected_embeddings = self.vsn(original_embedding)

        return self.layer_norm(original_embedding + selected_embeddings)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask




class Attention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(Attention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / np.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L,H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)

        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, gate_bias_init=-2.0):
        super(GatedLinearUnit, self).__init__()
        self.gate_layer = nn.Linear(input_dim, input_dim)
        self.activation_layer = nn.Linear(input_dim, input_dim)

        with torch.no_grad():
            nn.init.zeros_(self.gate_layer.weight)
            if self.gate_layer.bias is not None:
                self.gate_layer.bias.data.fill_(gate_bias_init)
    
    def forward(self, x):
        gate = self.gate_layer(x)
        activation = self.activation_layer(x)

        return activation * torch.sigmoid(gate)



class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        gated_output = self.gate(output)

        return self.layer_norm(x + output * gated_output)



class TransformerLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(TransformerLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.glu1 = GatedLinearUnit(d_model)
        self.glu2 = GatedLinearUnit(d_model)
        self.grn = GatedResidualNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        new_x = self.glu1(new_x)
        x = x + new_x

        y = x = self.norm1(x)
        y = self.grn(y)

        y = self.glu2(y)

        return self.norm2(x + y), attn
    


class Transformer(nn.Module):
    def __init__(self, attn_layers, grn_layers=None, norm_layer=None):
        super(Transformer, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.grn_layers = nn.ModuleList(grn_layers) if grn_layers is not None else None
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        attns = []
        if self.grn_layers is not None:
            for (attn_layer, grn_layer) in zip(self.attn_layers, self.grn_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = grn_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns