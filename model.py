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

class generator(nn.Module):
    def __init__(self, condition_size, noise_size, output_size, device):
        super(generator, self).__init__()
        self.device = device
        self.hidden_layer_size = [128]
        
        # 定義LSTM層的ModuleList
        self.lstm_list = nn.ModuleList(
            [nn.LSTM(condition_size if i == 0 else self.hidden_layer_size[i-1], self.hidden_layer_size[i], batch_first=True) 
             for i in range(len(self.hidden_layer_size))]
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_layer_size[-1]+noise_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )
    
    def forward(self, cond, noise):
        for i in range(len(self.lstm_list)):
            h0 = torch.zeros(1, cond.size(0), self.hidden_layer_size[i]).to(self.device)
            c0 = torch.zeros(1, cond.size(0), self.hidden_layer_size[i]).to(self.device)
            cond, _ = self.lstm_list[i](cond, (h0, c0))
            # cond = self.dropout(cond)
        
        cond_latent = cond[:, -1, :]  # 取出最後一個時間步的輸出
        fc_input = torch.cat((cond_latent, noise), dim=1)
        out = self.fc(fc_input)
        out = out.unsqueeze(2)
        return out

class discriminator(nn.Module):
    def __init__(self, cond_dim, x_dim, device, args):
        super(discriminator, self).__init__()
        num_channels = [8]*8
        input_size = 1
        self.device = device
        self.tcn = TemporalConvNet(input_size, num_channels, 2, 0.1)
        self.linear = nn.Linear(num_channels[-1], 1)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
    
    def forward(self, cond, x, channel_last=True):
        cond, x = cond.view(cond.size(0), -1), x.view(x.size(0), -1)
        input = torch.cat([cond, x], axis=1).unsqueeze(1)
        y = self.tcn(input)
        return self.linear(y.transpose(1, 2))
    
    