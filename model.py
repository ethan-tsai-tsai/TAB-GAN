import math
import torch
from torch import nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride, padding, dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride, padding, dilation))
        self.net = nn.Sequential(
            self.conv1,
            Chomp1d(padding),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            self.conv2,
            Chomp1d(padding),
            nn.LeakyReLU(),
            nn.Dropout(dropout)            
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i==0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size, padding, dropout)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
    
    def forward(self, x, channel_last=True):
        y = self.tcn(x.transpose(1, 2) if channel_last else x)
        return self.linear(y.transpose(1, 2))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加一個batch_size的維度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x的size是(batch_size, seq_len, d_model)，直接加上位置編碼
        x = x + self.pe[:, :x.size(1), :]
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
            nhead=8,
            dim_feedforward=self.hidden_layer_size[-1] * 4,
            dropout=0.1,
            activation='relu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_layer_size[-1] * 2+noise_size, self.hidden_layer_size[-1] * 2),
            # nn.BatchNorm1d(self.hidden_layer_size[-1] * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer_size[-1] * 2, output_size),
        )
    
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
        out = self.fc(fc_input).unsqueeze(-1)
        return out

# class discriminator(nn.Module):
#     def __init__(self, cond_dim, x_dim, device, args):
#         super(discriminator, self).__init__()
#         num_channels = [args.hidden_dim_d] * args.num_layers_d
#         input_size = args.hidden_dim_d
#         self.cond_embedding = nn.Linear(cond_dim, args.hidden_dim_d)
#         self.x_embedding = nn.Linear(1, args.hidden_dim_d)
#         self.device = device
#         self.tcn = TemporalConvNet(input_size, num_channels, 2, 0.1)
#         self.linear = nn.Linear(num_channels[-1], 1)
#         self.init_weights()
    
#     def init_weights(self):
#         self.linear.weight.data.normal_(0, 0.01)
    
#     def forward(self, cond, x):
#         cond_embedded = self.cond_embedding(cond)
#         x_embedded = self.x_embedding(x.unsqueeze(-1))
#         input = torch.cat((cond_embedded, x_embedded), dim=1).permute(0, 2, 1)  # (batch_size, seq_len_condition + seq_len_target, d_model)
#         # cond, x = cond.view(cond.size(0), -1), x.view(x.size(0), -1)
#         # input = torch.cat([cond, x], axis=1).unsqueeze(1)
#         out = self.tcn(input)
#         # weight = torch.exp(torch.linspace(1, -3, out.shape[2])).to(self.device)
#         # out = out * weight
#         out = self.linear(out.transpose(1, 2))
        
#         return out
    
class discriminator(nn.Module):
    def __init__(self, cond_dim, x_dim, device, args):
        super(discriminator, self).__init__()
        self._args = args
        self.device = device
        self.cond_embedding = nn.Linear(cond_dim, args.hidden_dim_d)
        self.x_embedding = nn.Linear(1, args.hidden_dim_d)
        # position encoding
        seq_len = args.window_size * (270 // args.time_step) + 1
        target_len = args.target_length // args.time_step
        self.position_encode = nn.Parameter(torch.zeros(1, seq_len + target_len, args.hidden_dim_d))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = args.hidden_dim_d,
            nhead = args.num_layers_d,
            dim_feedforward = args.hidden_dim_d * 4,
            dropout = 0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.num_layers_d)
        
        self.fc = nn.Linear(args.hidden_dim_d, 1)
    
    def forward(self, cond, x):
        cond_embedded = self.cond_embedding(cond)
        x_embedded = self.x_embedding(x.unsqueeze(-1))
        input = torch.cat((cond_embedded, x_embedded), dim=1)  # (batch_size, seq_len_condition + seq_len_target, d_model)
        input = (input + self.position_encode).permute(1, 0, 2) # (seq_len, batch_size, d_model)
        out = self.encoder(input)
        out = out[-1, :, :]
        out = self.fc(out)
        
        return out