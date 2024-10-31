import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x
    
class generator(nn.Module):
    def __init__(self, condition_size, noise_size, output_size, device, args):
        super(generator, self).__init__()
        self.device = device
        self.hidden_layer_size = [args.hidden_dim_g] * args.num_layers_g
        # LSTM model list
        self.lstm_list = nn.ModuleList(
            [nn.LSTM(condition_size if i == 0 else self.hidden_layer_size[i-1] * 2, self.hidden_layer_size[i], batch_first=True, bidirectional=True) 
             for i in range(len(self.hidden_layer_size))]
        )
        # transformer
        self.position_encoding = PositionalEncoding(self.hidden_layer_size[-1] * 2)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_layer_size[-1] * 2,
            nhead=args.num_head_g,
            dim_feedforward=self.hidden_layer_size[-1] * 4,
            dropout=0.1,
            activation='relu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_layer_size[-1] * 2 + noise_size, self.hidden_layer_size[-1] * 2),
            nn.LayerNorm(self.hidden_layer_size[-1] * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer_size[-1] * 2, output_size),
        )
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, cond, noise):
        for i in range(len(self.lstm_list)):
            h0 = torch.zeros(2, cond.size(0), self.hidden_layer_size[i]).to(self.device)
            c0 = torch.zeros(2, cond.size(0), self.hidden_layer_size[i]).to(self.device)
            cond, _ = self.lstm_list[i](cond, (h0, c0))
        
        cond = self.position_encoding(cond)
        cond = cond.permute(1, 0, 2)  # (batch_size, seq_len, feature_size) -> (seq_len, batch_size, feature_size)
        cond_encoded = self.transformer_encoder(cond)
        cond_encoded = cond_encoded.permute(1, 0, 2)
        
        cond_latent = cond_encoded[:, -1, :]
        fc_input = torch.cat((cond_latent, noise), dim=1)
        out = self.fc(fc_input)
        return out
    
class discriminator(nn.Module):
    def __init__(self, cond_dim, x_dim, device, args):
        super(discriminator, self).__init__()
        self._args = args
        self.device = device
        
        # condition embedding and target embedding
        self.cond_embedding = nn.Linear(cond_dim, args.hidden_dim_d)
        self.x_embedding = nn.Linear(1, args.hidden_dim_d)
        
        # position encoding
        self.position_encoding = PositionalEncoding(args.hidden_dim_d)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = args.hidden_dim_d,
            nhead = args.num_head_d,
            dim_feedforward = args.hidden_dim_d * 4,
            dropout = 0.2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.num_layers_d)
        
        self.fc = nn.Linear(args.hidden_dim_d, 1)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier初始化線性層
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.TransformerEncoderLayer):
            # Transformer的自注意力和前饋層
            for name, param in m.named_parameters():
                if 'in_proj_weight' in name:  # 自注意力權重
                    nn.init.xavier_uniform_(param)
                elif 'fc1.weight' in name:  # 前饋層的第一層
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'fc2.weight' in name:  # 前饋層的第二層
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, cond, x):
        cond_embedded = self.cond_embedding(cond)
        x_embedded = self.x_embedding(x.unsqueeze(-1))
        input = torch.cat((cond_embedded, x_embedded), dim=1)  # (batch_size, seq_len_condition + seq_len_target, d_model)
        input = self.position_encoding(input).permute(1, 0, 2) # (seq_len, batch_size, d_model)
        out = self.encoder(input)
        out = out[-1, :, :]
        out = self.fc(out)
        
        return out