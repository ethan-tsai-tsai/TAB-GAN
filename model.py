import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 改進的位置編碼，使用 sin/cos 而不是學習
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        return self.activation(x + self.block(x))

class generator(nn.Module):
    def __init__(self, condition_size, noise_size, output_size, device, args):
        super(generator, self).__init__()
        self.device = device
        self.hidden_layer_size = [args.hidden_dim_g] * args.num_layers_g
        
        # 加入梯度裁剪和縮放
        self.gradient_clip = args.gradient_clip if hasattr(args, 'gradient_clip') else 1.0
        
        # 使用 GRU 替代 LSTM，減少參數量並避免梯度消失
        self.gru_list = nn.ModuleList(
            [nn.GRU(condition_size if i == 0 else self.hidden_layer_size[i-1] * 2, 
                    self.hidden_layer_size[i], 
                    batch_first=True, 
                    bidirectional=True,
                    dropout=0.1 if i < len(self.hidden_layer_size)-1 else 0) 
             for i in range(len(self.hidden_layer_size))]
        )
        
        self.position_encoding = PositionalEncoding(
            self.hidden_layer_size[-1] * 2,
            dropout=0.1
        )
        
        assert self.hidden_layer_size[-1] * 2 % args.num_head_g == 0, "Hidden size must be divisible by num_heads"
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_layer_size[-1] * 2,
            nhead=args.num_head_g,
            dim_feedforward=self.hidden_layer_size[-1] * 4,
            dropout=0.1,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=2
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_layer_size[-1] * 2)
            for _ in range(2)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_layer_size[-1] * 2 + noise_size, self.hidden_layer_size[-1] * 2),
            nn.LayerNorm(self.hidden_layer_size[-1] * 2),
            nn.Dropout(0.1),
            nn.GELU(),
            ResidualBlock(self.hidden_layer_size[-1] * 2),
            nn.Linear(self.hidden_layer_size[-1] * 2, output_size),
        )
        
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用 Kaiming 初始化
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)  # 使用正交初始化
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, cond, noise):
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
        
        for i in range(len(self.gru_list)):
            h0 = torch.zeros(2, cond.size(0), self.hidden_layer_size[i]).to(self.device)
            cond, _ = self.gru_list[i](cond, h0)
        
        cond = self.position_encoding(cond)
        
        # 加入殘差連接
        for block in self.residual_blocks:
            cond = block(cond)
            
        cond = cond.permute(1, 0, 2)
        cond_encoded = self.transformer_encoder(cond)
        cond_encoded = cond_encoded.permute(1, 0, 2)
        
        # 使用注意力池化而不是簡單取最後一個時間步
        attn_weights = torch.softmax(torch.matmul(cond_encoded, cond_encoded.transpose(-2, -1)) / math.sqrt(cond_encoded.size(-1)), dim=-1)
        cond_latent = torch.matmul(attn_weights, cond_encoded).mean(dim=1)
        
        fc_input = torch.cat((cond_latent, noise), dim=1)
        out = self.fc(fc_input)
        return out

class discriminator(nn.Module):
    def __init__(self, cond_dim, x_dim, device, args):
        super(discriminator, self).__init__()
        self.device = device
        
        # 擴展嵌入維度
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_dim, args.hidden_dim_d),
            nn.LayerNorm(args.hidden_dim_d),
            nn.GELU(),
            ResidualBlock(args.hidden_dim_d)
        )
        
        self.x_embedding = nn.Sequential(
            nn.Linear(1, args.hidden_dim_d),
            nn.LayerNorm(args.hidden_dim_d),
            nn.GELU(),
            ResidualBlock(args.hidden_dim_d)
        )
        
        self.position_encoding = PositionalEncoding(args.hidden_dim_d, dropout=0.1)
        
        # 使用更深的 Transformer 架構
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim_d,
            nhead=args.num_head_d,
            dim_feedforward=args.hidden_dim_d * 4,
            dropout=0.2,
            activation='gelu'
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=args.num_layers_d,
            norm=nn.LayerNorm(args.hidden_dim_d)
        )
        
        # 加入自注意力機制的輸出層
        self.attention = nn.MultiheadAttention(
            embed_dim=args.hidden_dim_d,
            num_heads=args.num_head_d,
            dropout=0.1
        )
        
        self.fc = nn.Sequential(
            ResidualBlock(args.hidden_dim_d),
            nn.Linear(args.hidden_dim_d, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.TransformerEncoderLayer):
            for name, param in m.named_parameters():
                if 'in_proj_weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'fc1.weight' in name:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                elif 'fc2.weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, cond, x):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        cond_embedded = self.cond_embedding(cond)
        x_embedded = self.x_embedding(x.unsqueeze(-1))
        
        # 連接條件和輸入
        combined = torch.cat((cond_embedded, x_embedded), dim=1)
        combined = self.position_encoding(combined)
        
        # Transformer 編碼
        encoded = combined.permute(1, 0, 2)
        encoded = self.encoder(encoded)
        
        # 自注意力機制
        attn_output, _ = self.attention(encoded[-1:], encoded, encoded)
        
        # 最終分類
        out = self.fc(attn_output.squeeze(0))
        return out