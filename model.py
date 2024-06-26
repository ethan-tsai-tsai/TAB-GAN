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
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            Chomp1d(padding),
            nn.ReLU(),
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
    def __init__(self, input_size, output_size, device):
        super(generator, self).__init__()
        self.device = device
        self.num_layers = 2
        self.hidden_layer_size = 128
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, self.hidden_layer_size, self.num_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(self.hidden_layer_size, output_size)
    
    def forward(self, x):
        batch_size, seq_len, num_features = x.shape[0], x.shape[1], x.shape[2]
        x = self.bn(x.view(-1, num_features)).view(batch_size, seq_len, num_features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(self.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(self.device).requires_grad_()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        out = out[:, -1, :]
        out = out.unsqueeze(2)
        return out

class discriminator(nn.Module):
    def __init__(self, input_size):
        super(discriminator, self).__init__()
        self.num_channels = [16, 16, 16]
        self.tcn = TCN(input_size, 1, self.num_channels, 2, 0.5)
    def forward(self, x):
        output = self.tcn(x)
        return output