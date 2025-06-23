import torch
import torch.nn as nn
import torch.nn.functional as F
from global_configs import *

class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()
        print("Initializing MAG with beta_shift:{} hidden_prob:{}".format(beta_shift, dropout_prob))

        self.W_hv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

# 做normalization，在scale上一致
    def forward(self, text_embedding, visual, acoustic):
        # visual [48,50,47]  acoustic [48,50,74] text_embedding [48,50,768]
        eps = 1e-6
        tmp = torch.cat((visual, text_embedding), dim=-1)
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        # 计算文本特征和融合特征的2范数
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        # 避免除以0
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        # 平衡文本模态和融合模态信息的强度
        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)
        
        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        # 确保融合特征的贡献始终受控，不会超过文本特征的能量级（alpha在[0,1]之间）
        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(self.LayerNorm(acoustic_vis_embedding + text_embedding))
        # embedding_output [48,50,768]
        return embedding_output

class ConcatFusion(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.intermediate_dim = 512
        # self.text_projection = nn.Linear(TEXT_DIM, self.intermediate_dim)
        self.visual_projection = nn.Linear(VISUAL_DIM, self.intermediate_dim)
        self.acoustic_projection = nn.Linear(ACOUSTIC_DIM, self.intermediate_dim)

        # 计算拼接后的总维度
        # self.concat_dim = TEXT_DIM + VISUAL_DIM + ACOUSTIC_DIM
        # 投影回BERT原始维度
        self.final_projection = nn.Linear(self.intermediate_dim * 2, TEXT_DIM)

        self.LayerNorm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, text, visual, acoustic):
        # text_embedding = self.text_projection(text)
        visual_embedding = self.visual_projection(visual)
        acoustic_embedding = self.acoustic_projection(acoustic)

        # [batch_size, seq_len, TEXT_DIM + VISUAL_DIM + ACOUSTIC_DIM]
        concat_embedding = torch.cat((visual_embedding, acoustic_embedding), dim=-1)
        
        # 投影回原始BERT维度
        projected_embedding = self.final_projection(concat_embedding)
    
        embedding_output = self.dropout(self.LayerNorm(projected_embedding + text))
        
        return embedding_output
