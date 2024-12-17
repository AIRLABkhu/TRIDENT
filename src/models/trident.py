import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, num_heads=4):
        super(ChannelAttentionModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self.query_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, q, k, v):
        # x: (batch, dim) -> (batch, num_heads, head_dim)
        query = self.query_proj(q).view(-1, self.num_heads, self.head_dim)
        key = self.key_proj(k).view(-1, self.num_heads, self.head_dim)
        value = self.value_proj(v).view(-1, self.num_heads, self.head_dim)

        attn_weights = torch.einsum("bhd,bhd->bh", query, key) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = attn_weights.unsqueeze(-1) * value

        attended = attn_output.view(-1, self.num_heads * self.head_dim)
        output = self.norm(self.output_proj(attended))
        return output

class TRIDENTModule(nn.Module):
    def __init__(self):
        super(TRIDENTModule, self).__init__()

        self.domain_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=1024)
        self.class_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=1024)
        self.att_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=1024)

        self.domain_cross_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=1024)
        self.class_cross_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=1024)

    def forward(self, x):
        domain_vector = self.domain_attention(x, x, x)
        class_vector = self.class_attention(x, x, x)
        att_vector = self.att_attention(x, x, x)

        dom_att_vector = self.domain_cross_attention(domain_vector, att_vector, att_vector)
        cls_att_vector = self.class_cross_attention(class_vector, att_vector, att_vector)

        return domain_vector, class_vector, dom_att_vector, cls_att_vector

    def refined_attribute(self, dom, x):
        att_vector = self.att_attention(x, x, x)
        dom_att_vector = self.domain_cross_attention(dom, att_vector, att_vector)
        return dom_att_vector