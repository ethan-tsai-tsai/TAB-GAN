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
        self.relu = nn.LeakyReLU()
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
        self.input_size = input_size
        self.device = device
        self.hidden_layer_size = [512, 256, 128
                                  
                                  ]
        
        # 定義LSTM層的ModuleList
        self.lstm_list = nn.ModuleList(
            [nn.LSTM(input_size if i == 0 else self.hidden_layer_size[i-1], self.hidden_layer_size[i], batch_first=True) 
             for i in range(len(self.hidden_layer_size))]
        )
        # self.tcn = TCN(input_size, 256, [64]*10, 2, 0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_layer_size[-1], output_size)
    
    def forward(self, x):
        for i in range(len(self.lstm_list)):
            h0 = torch.zeros(1, x.size(0), self.hidden_layer_size[i]).to(self.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_layer_size[i]).to(self.device)
            out, _ = self.lstm_list[i](x, (h0, c0))
            x = self.dropout(out)
        # out = self.tcn(x)
        
        out = out[:, -1, :]  # 取出最後一個時間步的輸出
        out = self.fc(out)
        out = out.unsqueeze(2)
        return out


# class discriminator(nn.Module):
#     def __init__(self, cond_dim, x_dim):
#         super(discriminator, self).__init__()
#         self.num_channels = [32, 32, 32, 32, 32, 32, 32, 32]
#         self.tcn = TCN(1, 1, self.num_channels, 2, 0.2)
#     def forward(self, cond, x):
#         cond, x = cond.view(cond.size(0), -1), x.view(x.size(0), -1)
#         input = torch.cat([cond, x], axis=1).unsqueeze(2)
#         output = self.tcn(input)
#         return output
    
class discriminator(nn.Module):
    def __init__(self, cond_dim, x_dim, args):
        super().__init__()
        target_length = 270 // args.time_step
        self.conv1 = nn.Conv1d(1, 32, kernel_size = 5, stride = 1, padding = 'same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 5, stride = 1, padding = 'same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 5, stride = 1, padding = 'same')
        self.linear1 = nn.Linear(128*(target_length*(cond_dim)*args.window_size+x_dim*target_length), 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()

    def forward(self, cond, x):
        cond, x = cond.view(cond.size(0), -1), x.view(x.size(0), -1)
        input = torch.cat([cond, x], axis=1).unsqueeze(1)
        conv1 = self.conv1(input)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x =  conv3.reshape(conv3.shape[0], conv3.shape[1]*conv3.shape[2])
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out = self.linear3(out_2)
        return out