
import torch
from torch import nn
import torch.nn.functional as F

# -------------------------------------------------------------------
# 模块一：CDA (CrossDomain Alignment) 跨域对齐
# 作用：解决语义模型特征与几何描述子之间的分布差异
# -------------------------------------------------------------------
class CDA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Conv1d(dim, dim, 1)
        self.k_proj = nn.Conv1d(dim, dim, 1)
        self.v_proj = nn.Conv1d(dim, dim, 1)
        self.out_proj = nn.Conv1d(dim, dim, 1)

    def forward(self, desc, feat):
        # desc 和 feat 的维度均为 [B, C, N]
        q = self.q_proj(desc)
        k = self.k_proj(feat)
        v = self.v_proj(feat)
        
        # 计算注意力权重: [B, N, C] x [B, C, N] -> [B, N, N]
        attn = torch.matmul(q.transpose(1, 2), k) * (q.shape[1] ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # 应用注意力: [B, C, N] x [B, N, N] -> [B, C, N]
        aligned_feat = self.out_proj(torch.matmul(v, attn.transpose(1, 2)))
        return feat + aligned_feat  # 残差连接


# -------------------------------------------------------------------
# 模块二：SAF (Semantic-Aware Fusion) 语义感知融合
# 作用：将对齐后的语义消息注入到几何特征中
# -------------------------------------------------------------------
class SAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wq = nn.Conv1d(dim, dim, 1)
        self.Wkv = nn.Conv1d(dim, 2 * dim, 1)
        self.ffn = nn.Sequential(
            nn.Conv1d(2 * dim, 2 * dim, 1),
            nn.InstanceNorm1d(2 * dim),
            nn.GELU(),
            nn.Conv1d(2 * dim, dim, 1)
        )

    def forward(self, desc, feature):
        q = self.Wq(desc)
        # 将语义特征拆分为 Key 和 Value
        kv = self.Wkv(feature).chunk(2, dim=1)
        k, v = kv[0], kv[1]
        
        # 计算交叉注意力
        attn = torch.matmul(q.transpose(1, 2), k) * (q.shape[1] ** -0.5)
        message = torch.matmul(v, attn.softmax(dim=-1).transpose(1, 2))
        
        # 将几何描述子与语义消息拼接后融合 (对应 SemaGlue 的 Concat 逻辑)
        return desc + self.ffn(torch.cat([desc, message], dim=1))


# -------------------------------------------------------------------
# 模块三：SemanticCNNet (集成语义的 PARSAC 核心网络)
# 作用：完全替代原版的 CNNet
# -------------------------------------------------------------------
class SemanticCNNet(nn.Module):
    def __init__(self, input_dim, output_dim, sem_dim=480, blocks=5, batch_norm=True, separate_weights=True):
        super(SemanticCNNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.separate_probs = separate_weights

        # 1. 几何特征提取 (保持 PARSAC 原逻辑)
        self.p_in = nn.Conv1d(self.input_dim, 128, 1, 1, 0)

        # 2. 语义感知流 (SemaGlue 核心注入)
        self.sem_proj = nn.Conv1d(sem_dim, 128, 1)  # 将 SegNext 的 480 维降维对齐到 128 维
        self.cda = CDA(128)
        self.saf = SAF(128)

        # 3. 核心残差块 (保持 PARSAC 原逻辑)
        self.res_blocks = nn.ModuleList()
        for i in range(blocks):
            if batch_norm:
                self.res_blocks.append(nn.ModuleList([
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.BatchNorm1d(128),
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.BatchNorm1d(128)
                ]))
            else:
                self.res_blocks.append(nn.ModuleList([
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.Conv1d(128, 128, 1, 1, 0)
                ]))

        # 4. 并行权重预测输出层
        self.p_out = nn.Conv1d(128, output_dim, 1, 1, 0)
        if self.separate_probs:
            self.p_out2 = nn.Conv1d(128, output_dim, 1, 1, 0)

    def forward(self, inputs, sem_features=None):
        '''
        Forward pass.
        inputs: 3D data tensor (BxNxC) - 原始观测坐标
        sem_features: 3D data tensor (BxNxSemDim) - 从 SegNext 提取的点级语义
        '''
        # 转换维度以适应 Conv1d: BxNxC -> BxCxN
        inputs_ = torch.transpose(inputs, 1, 2)
        x = inputs_[:, 0:self.input_dim]
        
        # 初始特征提取
        x = F.relu(self.p_in(x))

        # ------------------------------------------------------
        # 执行语义融合 (当传入了语义特征时)
        # ------------------------------------------------------
        if sem_features is not None:
            # 同样将语义特征转换为 BxCxN
            sem_f = torch.transpose(sem_features, 1, 2)
            
            # 投影降维 -> 跨域对齐 -> 语义感知注入
            sem_f = F.relu(self.sem_proj(sem_f))
            sem_f = self.cda(x, sem_f)
            x = self.saf(x, sem_f)
        # ------------------------------------------------------

        # 经过残差块迭代增强
        for r in self.res_blocks:
            res = x
            if self.batch_norm:
                x = F.relu(r[1](F.instance_norm(r[0](x))))
                x = F.relu(r[3](F.instance_norm(r[2](x))))
            else:
                x = F.relu(F.instance_norm(r[0](x)))
                x = F.relu(F.instance_norm(r[1](x)))
            x = x + res

        # 输出采样权重 log_p
        log_ng = F.logsigmoid(self.p_out(x))
        log_ng = torch.transpose(log_ng, 1, 2)
        normalizer = torch.logsumexp(log_ng, dim=-1, keepdim=True)
        log_p = log_ng - normalizer

        # 输出内点权重 log_q
        if self.separate_probs:
            log_ng2 = F.logsigmoid(self.p_out2(x))
            log_ng2 = torch.transpose(log_ng2, 1, 2)
            normalizer2 = torch.logsumexp(log_ng2, dim=-2, keepdim=True)
            log_q = log_ng2 - normalizer2
        else:
            log_q = log_p

        return log_p, log_q

