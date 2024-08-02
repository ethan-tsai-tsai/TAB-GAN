import torch
from torch import nn
from torch.nn.utils import weight_norm

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
        self.fc = nn.Linear(self.hidden_layer_size[-1]+noise_size, output_size)
    
    def forward(self, cond, noise):
        for i in range(len(self.lstm_list)):
            h0 = torch.zeros(1, cond.size(0), self.hidden_layer_size[i]).to(self.device)
            c0 = torch.zeros(1, cond.size(0), self.hidden_layer_size[i]).to(self.device)
            cond_latent, _ = self.lstm_list[i](cond, (h0, c0))
            # cond = self.dropout(cond_latent)
        
        cond_latent = cond_latent[:, -1, :]  # 取出最後一個時間步的輸出
        fc_input = torch.cat((cond_latent, noise), dim=1)
        out = self.fc(fc_input)
        out = out.unsqueeze(2)
        print(out.shape)
        return out

class discriminator(nn.Module):
    def __init__(self, cond_dim, x_dim, device, args):
        super(discriminator, self).__init__()
        self.device = device
        self.hidden_layer_size = [128]
        
        # 定義LSTM層的ModuleList
        self.lstm_list = nn.ModuleList(
            [nn.LSTM(cond_dim+x_dim if i == 0 else self.hidden_layer_size[i-1], self.hidden_layer_size[i], batch_first=True) 
             for i in range(len(self.hidden_layer_size))]
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_layer_size[-1], 1)
    
    def forward(self, cond, x):
        cond, x = cond.view(cond.size(0), -1), x.view(x.size(0), -1)
        input = torch.cat([cond, x], axis=1).unsqueeze(1)
        
        for i in range(len(self.lstm_list)):
            h0 = torch.zeros(1, input.size(0), self.hidden_layer_size[i]).to(self.device)
            c0 = torch.zeros(1, input.size(0), self.hidden_layer_size[i]).to(self.device)
            out, _ = self.lstm_list[i](input, (h0, c0))
            # input = self.dropout(out)
        out = out[:, -1, :]  # 取出最後一個時間步的輸出
        out = self.fc(out)
        return out
    
    