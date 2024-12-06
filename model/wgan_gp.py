# import packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

# import files
from model.network.wgan_gp_models import VAE, Generator, Discriminator

class WGAN_GP:
    def __init__(self, input_size=27, latent_dim=10, batch_size=128, learning_rate=0.000115, device=None):
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Initialize models
        hidden_dims = [27, 400, 400, 400, 10]
        vae_dims = [self.input_size] + hidden_dims + [self.latent_dim]
        self.vae = VAE(vae_dims, self.latent_dim).to(self.device)
        self.generator = Generator(self.input_size + 10).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
    def _preprocess_data(self, data, train_split=0.8):
        x = data.iloc[:, :self.input_size].values
        y = data.iloc[:, -1].values
        
        split = int(data.shape[0] * train_split)
        train_x, test_x = x[:split, :], x[split - 20:, :]
        train_y, test_y = y[:split], y[split - 20:]
        
        # Scale data
        train_x = self.x_scaler.fit_transform(train_x)
        test_x = self.x_scaler.transform(test_x)
        train_y = self.y_scaler.fit_transform(train_y.reshape(-1, 1))
        test_y = self.y_scaler.transform(test_y.reshape(-1, 1))
        
        return train_x, train_y, test_x, test_y
    
    def _sliding_window(self, x, y, window_size):
        x_, y_, y_gan = [], [], []
        for i in range(window_size, x.shape[0]):
            x_.append(x[i - window_size:i, :])
            y_.append(y[i])
            y_gan.append(y[i - window_size:i + 1])
        
        return (torch.FloatTensor(np.array(x_)), 
                torch.FloatTensor(np.array(y_)), 
                torch.FloatTensor(np.array(y_gan)))
    
    def train_vae(self, train_x, epochs=300):
        """Train VAE model"""
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.00003)
        
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_x)), 
                                batch_size=self.batch_size, 
                                shuffle=False)
        
        for epoch in range(epochs):
            total_loss = 0
            for (x,) in train_loader:
                x = x.to(self.device)
                output, z, mu, logVar = self.vae(x)
                kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                loss = F.binary_cross_entropy(output, x) + kl_divergence
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 50 == 0:
                print(f'[{epoch+1}/{epochs}] VAE Loss: {total_loss}')
    
    def train(self, data, num_epochs=100, window_size=3):
        """Main training function"""
        # Preprocess data
        train_x, train_y, test_x, test_y = self._preprocess_data(data)
        
        # Train VAE
        self.train_vae(train_x)
        
        # Prepare sliding window data
        train_x_slide, train_y_slide, train_y_gan = self._sliding_window(train_x, train_y, window_size)
        
        # Setup optimizers
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, 
                                betas=(0.0, 0.9), weight_decay=1e-3)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, 
                                betas=(0.0, 0.9), weight_decay=1e-3)
        
        train_loader = DataLoader(TensorDataset(train_x_slide, train_y_gan), 
                                batch_size=self.batch_size, shuffle=False)
        
        for epoch in range(num_epochs):
            g_losses, d_losses = [], []
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Train discriminator
                fake_data = self.generator(x)
                fake_data = torch.cat([y[:, :3, :], fake_data.reshape(-1, 1, 1)], axis=1)
                
                d_real = self.discriminator(y)
                d_fake = self.discriminator(fake_data)
                d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
                
                opt_d.zero_grad()
                d_loss.backward(retain_graph=True)
                opt_d.step()
                
                # Train generator
                g_fake = self.discriminator(fake_data)
                g_loss = -torch.mean(g_fake)
                
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f'[{epoch+1}/{num_epochs}] Loss_D: {sum(d_losses):.4f} Loss_G: {sum(g_losses):.4f}')
    
    def test(self, test_data):
        """Test function - implement your testing logic here"""
        # Add your testing implementation
        pass
    
