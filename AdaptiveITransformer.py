import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Transformer, TransformerLayer, Attention, AttentionLayer, EmbeddingWithSelection


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.embedding = EmbeddingWithSelection(configs.seq_len, configs.d_model, 1024, configs.dropout)
        
        self.class_strategy = configs.class_strategy

        self.transformer = Transformer(
            [
                TransformerLayer(AttentionLayer(Attention(False, attention_dropout=configs.dropout,
                                                          output_attention=configs.output_attention),
                                                configs.d_model,
                                                configs.n_heads),
                                configs.d_model,
                                configs.d_ff,
                                dropout=configs.dropout
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)


    def forecast(self, x, x_mark):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        
        _, _, N = x.shape

        enc_out = self.embedding(x, x_mark)
        enc_out, attns = self.transformer(enc_out, attn_mask=None)

        output = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return output
    

    def forward(self, x, x_mark):
        output = self.forecast(x, x_mark)
        return output[:, -self.pred_len:, :]