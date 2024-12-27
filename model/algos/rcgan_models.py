import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, condition_size, noise_dim, hidden_dim, output_size, num_layers):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        self.lstm = nn.LSTM(
            input_size=condition_size + noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, condition, noise):
        batch_size, seq_len, _ = condition.size()
        noise = torch.randn(batch_size, seq_len, self.noise_dim, device=noise.device)
        combined_input = torch.cat((condition, noise), dim=-1)
        lstm_out, _ = self.lstm(combined_input)
        output = self.output_layer(lstm_out)
        output = output[:, -1, :]
        return output

class Discriminator(nn.Module):
    def __init__(self, condition_size, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.seq_embedding = nn.Linear(1, condition_size)
        self.lstm = nn.LSTM(
            input_size=condition_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, sequence, condition):
        sequence = self.seq_embedding(sequence.unsqueeze(-1))
        combined_input = torch.cat((condition, sequence), dim=1)
        lstm_out, _ = self.lstm(combined_input)
        lstm_out = lstm_out[:, -1, :]
        outputs = self.output_layer(lstm_out)
        return outputs