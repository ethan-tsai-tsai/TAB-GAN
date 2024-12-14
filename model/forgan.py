import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn

from lib.calc import calc_kld
from lib.utils import save_model
from model.algos.forgan_models import Generator, Discriminator
# Fixing random seeds
torch.manual_seed(1368)
rs = np.random.RandomState(1368)

class ForGAN:
    def __init__(self, stock_data, args):
        self.args = args
        self.FILE_NAME = f'{args.stock}_{args.name}'
        self.device = torch.device(f"cuda:{self.args.cuda}") if torch.cuda.is_available() else torch.device("cpu")

        data_mean = np.array(stock_data.X).mean()
        data_std = np.array(stock_data.X).std()
        # Defining GAN components
        self.model_g = Generator(noise_size=args.noise_dim,
                                   condition_size=stock_data.num_features - 1,
                                   generator_latent_size=args.hidden_dim_g,
                                   cell_type=self.args.cell_type,
                                   mean=data_mean,
                                   std=data_std)


        self.model_d = Discriminator(condition_size=stock_data.num_features - 1,
                                           discriminator_latent_size=args.hidden_dim_d,
                                           cell_type=self.args.cell_type,
                                           mean=data_mean,
                                           std=data_std)

        self.model_g = self.model_g.to(self.device)
        self.model_d = self.model_d.to(self.device)
        
        self.model_path = f'./model_saved/{args.model}/{args.stock}_{args.name}'
        img_path = f'./img/{args.model}/{args.stock}_{args.name}'
        if not os.path.exists(img_path): os.makedirs(img_path)
    def train(self, train_dataset, test_dataset):
        x_train = np.array(train_dataset.X)
        y_train = np.array(train_dataset.y)
        x_val = np.array(test_dataset.X)
        y_val = np.array(test_dataset.y)
        
        x_train = torch.tensor(x_train, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(y_train, device=self.device, dtype=torch.float32)
        x_val = torch.tensor(x_val, device=self.device, dtype=torch.float32)
        best_kld = np.inf
        optimizer_g = torch.optim.RMSprop(self.model_g.parameters(), lr=self.args.lr_g)
        optimizer_d = torch.optim.RMSprop(self.model_d.parameters(), lr=self.args.lr_d)
        adversarial_loss = nn.BCELoss()
        adversarial_loss = adversarial_loss.to(self.device)

        for step in range(self.args.epoch):
            d_loss = 0
            for _ in range(self.args.d_iter):
                # train discriminator on real data
                idx = rs.choice(x_train.shape[0], self.args.batch_size)
                condition = x_train[idx]
                real_data = y_train[idx]
                self.model_d.zero_grad()
                d_real_decision = self.model_d(real_data, condition)
                d_real_loss = adversarial_loss(d_real_decision,
                                               torch.full_like(d_real_decision, 1, device=self.device))
                d_real_loss.backward()
                d_loss += d_real_loss.detach().cpu().numpy()
                # train discriminator on fake data
                noise_batch = torch.tensor(rs.normal(0, 1, (condition.size(0), self.args.noise_dim)),
                                           device=self.device, dtype=torch.float32)
                x_fake = self.model_g(noise_batch, condition).detach()
                d_fake_decision = self.model_d(x_fake, condition)
                d_fake_loss = adversarial_loss(d_fake_decision,
                                               torch.full_like(d_fake_decision, 0, device=self.device))
                d_fake_loss.backward()

                optimizer_d.step()
                d_loss += d_fake_loss.detach().cpu().numpy()

            d_loss = d_loss / (2 * self.args.d_iter)

            self.model_g.zero_grad()
            noise_batch = torch.tensor(rs.normal(0, 1, (self.args.batch_size, self.args.noise_dim)), device=self.device,
                                       dtype=torch.float32)
            x_fake = self.model_g(noise_batch, condition)
            d_g_decision = self.model_d(x_fake, condition)
            # Mackey-Glass works best with Minmax loss in our expriements while other dataset
            # produce their best result with non-saturated loss
            g_loss = -1 * adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 0, device=self.device))
            g_loss.backward()
            optimizer_g.step()

            g_loss = g_loss.detach().cpu().numpy()

            # Validation
            noise_batch = torch.tensor(rs.normal(0, 1, (x_val.size(0), self.args.noise_dim)), device=self.device,
                                       dtype=torch.float32)
            preds = self.model_g(noise_batch, x_val).detach().cpu().numpy().flatten()

            kld = calc_kld(preds, y_val)

            if kld <= best_kld and kld != np.inf:
                best_kld = kld
                print("step : {} , KLD : {}, RMSE : {}".format(step, best_kld,
                                                               np.sqrt(np.square(preds - y_val.flatten()).mean())))
                save_model(self.model_d, self.model_g, self.args, f'./{self.model_path}/final.pth')
            
            if step % 100 == 0:
                print("step : {} , d_loss : {} , g_loss : {}".format(step, d_loss, g_loss))

    def validation(self, val_loader):
        adversarial_loss = nn.BCELoss()
        adversarial_loss = adversarial_loss.to(self.device)
        self.model_g.eval()
        with torch.inference_mode():
            total_loss_d, total_loss_g = 0, 0
            for _, (X, y) in enumerate(val_loader):
                X, y = X.to(self.device), y.to(self.device)
                # add noise
                self.model_g.eval()
                self.model_d.eval()
                d_loss = 0
                # evaluate discriminator
                d_real_decision = self.model_d(y, X)
                d_real_loss = adversarial_loss(d_real_decision,
                                               torch.full_like(d_real_decision, 1, device=self.device))
                
                d_loss += d_real_loss.detach().cpu().numpy()
                # train discriminator on fake data
                noise_batch = torch.tensor(rs.normal(0, 1, (X.size(0), self.args.noise_dim)),
                                           device=self.device, dtype=torch.float32)
                x_fake = self.model_g(noise_batch, X).detach()
                d_fake_decision = self.model_d(x_fake, X)
                d_fake_loss = adversarial_loss(d_fake_decision,
                                               torch.full_like(d_fake_decision, 0, device=self.device))
                d_loss += d_fake_loss
                total_loss_d += d_loss.detach().cpu().numpy()
                
                # evaluate generator
                noise_batch = torch.tensor(rs.normal(0, 1, (X.size(0), self.args.noise_dim)), device=self.device,
                                       dtype=torch.float32)
                x_fake = self.model_g(noise_batch, X)
                d_g_decision = self.model_d(x_fake, X)
                g_loss = -1 * adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 0, device=self.device))
                total_loss_g += g_loss
                # update best model (use kld)
                kld = calc_kld(x_fake.cpu().detach().numpy(), y.cpu().detach().numpy())
                
        return total_loss_d, total_loss_g, kld
    
    def predict(self, X, y=None):
        # load parameters
        with open(f'./data/{self.args.stock}/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        check_point = torch.load(f'{self.model_path}/final.pth')
        self.model_g.load_state_dict(check_point['model_g'])
        
        # prediction
        y_preds = np.array([])
        self.model_g.eval()
        with torch.inference_mode():
            X = X.to(self.device)
            for _ in range(self.args.pred_times):
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                y_pred = self.model_g(noise, X).cpu().detach().numpy()
                y_pred = scaler_y.inverse_transform(y_pred) 
                y_pred = np.expand_dims(y_pred, axis=2)
                y_preds = y_pred if y_preds.size==0 else np.concatenate((y_preds, y_pred), axis=2)
                
            y_trues = scaler_y.inverse_transform(y)
        return y_preds, y_trues
