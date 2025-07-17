import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        output_dim = output_dim or input_dim           
        hidden_dim = hidden_dim or input_dim     
        self.fc1 = nn.Linear(input_dim, hidden_dim)       # Downconv
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, output_dim)      # UPconv
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)            
        x = self.act(x)            
        x = self.drop(x)
        x = self.fc2(x)           
        x = self.drop(x)   
        return x

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

class TRIDENT(nn.Module):
    def __init__(self):
        super(TRIDENT, self).__init__()
        self.domain_module = SimpleModule(input_dim=1024)
        self.class_module = SimpleModule(input_dim=1024)
        self.att_module = SimpleModule(input_dim=1024)
        self.domain_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=512)
        self.class_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=512)

    def forward(self, x):
        domain_vector = self.domain_module(x)
        class_vector = self.class_module(x)
        att_vector = self.att_module(x)

        dom_att_vector = self.domain_attention(domain_vector, att_vector, att_vector)
        cls_att_vector = self.class_attention(class_vector, att_vector, att_vector)

        return domain_vector, class_vector, att_vector, dom_att_vector, cls_att_vector

    def refined_attribute(self, domain_vector, class_vector, att_vector):
        dom_att_vector = self.domain_attention(domain_vector, att_vector, att_vector)
        cls_att_vector = self.class_attention(class_vector, att_vector, att_vector)
        return dom_att_vector, cls_att_vector
        
class NON_TRIDENT(nn.Module):
    def __init__(self):
        super(NON_TRIDENT, self).__init__()
        self.domain_module = SimpleModule(input_dim=1024)
        self.class_module = SimpleModule(input_dim=1024)
        self.att_module = SimpleModule(input_dim=1024)
        self.domain_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=512)
        self.class_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=512)

    def forward(self, x):
        domain_vector = self.domain_module(x)
        class_vector = self.class_module(x)
        att_vector = self.att_module(x)

        dom_att_vector = self.domain_attention(att_vector, att_vector, att_vector)
        cls_att_vector = self.class_attention(att_vector, att_vector, att_vector)

        return domain_vector, class_vector, dom_att_vector + cls_att_vector

class DuTRIDENT(nn.Module):
    def __init__(self):
        super(DuTRIDENT, self).__init__()
        self.class_module = SimpleModule(input_dim=1024)
        self.att_module = SimpleModule(input_dim=1024)
        self.class_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=512)

    def forward(self, x):
        class_vector = self.class_module(x)
        att_vector = self.att_module(x)
        cls_att_vector = self.class_attention(att_vector, att_vector, att_vector)

        return class_vector, cls_att_vector


class DU_TRIDENT(nn.Module):
    def __init__(self):
        super(DU_TRIDENT, self).__init__()
        self.domain_module = SimpleModule(input_dim=1024)
        self.class_module = SimpleModule(input_dim=1024)
        self.att_module = SimpleModule(input_dim=1024)
        self.domain_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=512)
        self.class_attention = ChannelAttentionModule(input_dim=1024, hidden_dim=512)

    def forward(self, x):
        domain_vector = self.domain_module(x)
        class_vector = self.class_module(x)
        att_vector = self.att_module(x)

        dom_att_vector = self.domain_attention(att_vector, att_vector, att_vector)
        cls_att_vector = self.class_attention(att_vector, att_vector, att_vector)

        return domain_vector + dom_att_vector, class_vector + cls_att_vector