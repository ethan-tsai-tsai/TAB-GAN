import os
import torch
import pickle
import numpy as np
from torch import nn
from lib.calc import calc_kld
from lib.utils import save_model
from lib.visulization import save_loss_curve
from model.algos.rcgan_models import Generator, Discriminator

class RCGAN:
    def __init__(self, datasets, args):
        device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
        self.model_g = Generator(
            condition_size=datasets.num_features - 1,
            noise_dim=args.noise_dim,
            hidden_dim=args.hidden_dim_g,
            output_size=9,
            num_layers=args.num_layers_g
        ).to(device)
        
        self.model_d = Discriminator(
            condition_size=datasets.num_features - 1,
            hidden_dim=args.hidden_dim_d,
            num_layers=args.num_layers_d
        ).to(device)
        
        self.g_optimizer = torch.optim.Adam(self.model_g.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.model_d.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
        self.noise_dim = args.noise_dim
        self.args = args
        
        self.model_path = f'./model_saved/{args.model}/{args.stock}_{args.name}'
        
    def train(self, train_loader, val_loader):
        results = {'loss_d': [], 'loss_g': [], 'test_loss_d': [], 'test_loss_g': [], 'test_kld': []}
        for epoch in range(self.args.epoch):
            self.model_g.train()
            self.model_d.train()
            total_loss_d, total_loss_g = 0, 0
            for _, (condition, real_sequence) in enumerate(train_loader):
                batch_size = real_sequence.size(0)
                condition, real_sequence = condition.to(self.device), real_sequence.to(self.device)
                # train disctiminator
                for _ in range(self.args.d_iter):
                    self.d_optimizer.zero_grad()
                    
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)
                    
                    real_output = self.model_d(real_sequence, condition)
                    d_loss_real = self.criterion(real_output, real_labels)
                    
                    noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                    fake_sequence = self.model_g(condition, noise)
                    fake_output = self.model_d(fake_sequence.detach(), condition)
                    d_loss_fake = self.criterion(fake_output, fake_labels)

                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                # train generator
                self.g_optimizer.zero_grad()
                
                fake_output = self.model_d(fake_sequence, condition)
                g_loss = self.criterion(fake_output, real_labels)
                
                g_loss.backward()
                self.g_optimizer.step()
            total_loss_d += d_loss.cpu().detach().numpy()
            total_loss_g += g_loss.cpu().detach().numpy()
            test_loss_d, test_loss_g, kld = self.validation(val_loader)
            results['loss_d'].append(total_loss_d)
            results['loss_g'].append(total_loss_g)
            results['test_loss_d'].append(test_loss_d)
            results['test_loss_g'].append(test_loss_g)
            results['test_kld'].append(kld)
            
            if (epoch+1)%(self.args.epoch//10)==0:
                print(f'Epoch: {epoch+1}/{self.args.epoch}, loss_d: {total_loss_d:.2f}, loss_g: {total_loss_g:.2f}, test loss_d: {test_loss_d:.2f}, test loss_g: {test_loss_g:.2f}')
        
        # save model
        save_model(self.model_d, self.model_g, self.args, f'{self.model_path}/final.pth')
        if self.args.mode == 'train': save_loss_curve(results, self.args)
        return results

    def validation(self, val_loader):
        self.model_g.eval()
        with torch.inference_mode():
            total_loss_d, total_loss_g = 0, 0
            for _, (condition, real_sequence) in enumerate(val_loader):
                batch_size = real_sequence.size(0)
                condition, real_sequence = condition.to(self.device), real_sequence.to(self.device)
                # add noise
                self.model_g.eval()
                self.model_d.eval()
                
                # evaluate discriminator
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                real_output = self.model_d(real_sequence, condition)
                d_loss_real = self.criterion(real_output, real_labels)
                
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_sequence = self.model_g(condition, noise)
                fake_output = self.model_d(fake_sequence.detach(), condition)
                d_loss_fake = self.criterion(fake_output, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                total_loss_d += d_loss.cpu().detach().numpy()
                # evaluate generator
                fake_output = self.model_d(fake_sequence, condition)
                g_loss = self.criterion(fake_output, real_labels)
                
                total_loss_g += g_loss.cpu().detach().numpy()
                
                # update best model (use kld)
                kld = calc_kld(fake_output.cpu().detach().numpy(), real_sequence.cpu().detach().numpy())
                
        return total_loss_d, total_loss_g, kld
    
    def predict(self, condition, real_sequence):
        with open(f'./data/{self.args.stock}/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        check_point = torch.load(f'./model_saved/{self.args.model}/{self.args.stock}_{self.args.name}/final.pth')
        self.model_g.load_state_dict(check_point['model_g'])
        
        # prediction
        y_preds = np.array([])
        self.model_g.eval()
        with torch.no_grad():
            condition = condition.to(self.device)
            for _ in range(self.args.pred_times):
                noise = torch.randn(condition.shape[0], self.args.noise_dim).to(self.device)
                y_pred = self.model_g(condition, noise).cpu().detach().numpy()
                y_pred = scaler_y.inverse_transform(y_pred) 
                y_pred = np.expand_dims(y_pred, axis=2)
                y_preds = y_pred if y_preds.size==0 else np.concatenate((y_preds, y_pred), axis=2)
                
            y_trues = scaler_y.inverse_transform(real_sequence)
            
        return y_preds, y_trues