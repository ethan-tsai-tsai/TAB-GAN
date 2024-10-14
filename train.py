import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from datetime import datetime
# Import file
from preprocessor import *
from model import *
from utils import *
from arguments import *

class wgan:
    def __init__(self,stock_data, args):
        self.args = args
        self.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
        self.model_d = discriminator(stock_data.num_features - 1, 1, self.device, self.args).to(self.device)
        self.model_g = generator(stock_data.num_features - 1, self.args.noise_dim, self.args.target_length//self.args.time_step, self.device, self.args).to(self.device)
        self.BEST_KLD = np.inf

        # initialize folder
        if self.args.mode=='train':
            self.FOLDER_NAME = f'{args.stock}_{args.name}'
            if not os.path.exists(f'logs/{self.FOLDER_NAME}'):os.makedirs(f'logs/{self.FOLDER_NAME}')
            else: clear_folder(f'logs/{self.FOLDER_NAME}')
            if not os.path.exists(f'./logs/{self.FOLDER_NAME}/pred'): os.makedirs(f'./logs/{self.FOLDER_NAME}/pred')
            if not os.path.exists(f'./logs/{self.FOLDER_NAME}/dist'): os.makedirs(f'./logs/{self.FOLDER_NAME}/dist')
            clear_folder(f'./logs/{self.FOLDER_NAME}/pred')
            clear_folder(f'./logs/{self.FOLDER_NAME}/dist')
        
    def generator_loss(self, cond, fake_data):
        return -torch.mean(self.model_d(cond, fake_data))
    
    def discriminator_loss(self, cond, real_data, fake_data, gradient_penalty=0):
        return -torch.mean(self.model_d(cond, real_data)) + torch.mean(self.model_d(cond, fake_data)) + self.args.gp_lambda * gradient_penalty
    
    def compute_gradient_penalty(self, cond, real_data, fake_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.model_d(cond, interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean()
    
    def train(self, train_loader, val_loader):
        # trainin set
        optimizer_d = torch.optim.Adam(self.model_d.parameters(), lr=self.args.lr_d, betas = (0.0, 0.9), weight_decay = 1e-3)
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=self.args.lr_g, betas = (0.0, 0.9), weight_decay = 1e-3)
        loss_fn = nn.L1Loss()
        results = {'loss_d': [], 'loss_g': [], 'test_loss_d': [], 'test_loss_g': [], 'test_kld': []}
        
        for epoch in range(self.args.epoch):
            self.model_g.train()
            self.model_d.train()
            total_loss_d, total_loss_g = 0, 0
            for _, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                # train discriminator
                for _ in range(self.args.d_iter):
                    noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                    fake_data = self.model_g(X, noise)
                    assert not torch.isnan(fake_data).any(), 'Generated data has nan values. Stop training.'
                    gradient_penalty = self.compute_gradient_penalty(X, y, fake_data)
                    loss_d = self.discriminator_loss(X, y, fake_data, gradient_penalty)
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model_d.parameters(), max_norm=1.0)
                    optimizer_d.step()
                
                # train generator (minimize wgan loss)
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                loss_g = self.generator_loss(X, fake_data)
                optimizer_g.zero_grad()
                loss_g.backward()
                # torch.nn.utils.clip_grad_norm_(self.model_g.parameters(), max_norm=1.0)
                optimizer_g.step()
                
                # train discriminator part 2
                for _ in range(self.args.d_iter):
                    noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                    fake_data = self.model_g(X, noise)
                    assert not torch.isnan(fake_data).any(), 'Generated data has nan values. Stop training.'
                    gradient_penalty = self.compute_gradient_penalty(X, y, fake_data)
                    loss_d = self.discriminator_loss(X, y, fake_data, gradient_penalty)
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()
                
                # train generator (minimize mae loss)
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                mae_loss = loss_fn(y, fake_data)
                optimizer_g.zero_grad()
                mae_loss.backward()
                optimizer_g.step()
            total_loss_d += loss_d.cpu().detach().numpy()
            total_loss_g += loss_g.cpu().detach().numpy()
            test_loss_d, test_loss_g, kld = self.validation(val_loader)
            
            results['loss_d'].append(total_loss_d)
            results['loss_g'].append(total_loss_g)
            results['test_loss_d'].append(test_loss_d)
            results['test_loss_g'].append(test_loss_g)
            results['test_kld'].append(kld)
            
            if (epoch+1)%(self.args.epoch//5)==0:
                print(f'Epoch: {epoch+1}/{self.args.epoch}, loss_d: {total_loss_d:.2f}, loss_g: {total_loss_g:.2f}, test loss_d: {test_loss_d:.2f}, test loss_g: {test_loss_g:.2f}')
                
        return results
    
    def validation(self, val_loader):
        self.model_g.eval()
        with torch.inference_mode():
            total_loss_d, total_loss_g = 0, 0
            for _, (X, y) in enumerate(val_loader):
                X, y = X.to(self.device), y.to(self.device)
                # add noise
                self.model_g.eval()
                self.model_d.eval()
                
                # evaluate discriminator
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                loss_d = self.discriminator_loss(X, y, fake_data, 0)
                total_loss_d += loss_d.cpu().detach().numpy()
                
                # evaluate generator
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                loss_g = self.generator_loss(X, fake_data)
                total_loss_g += loss_g.cpu().detach().numpy()
                
                # update best model (use kld)
                kld = calc_kld(fake_data.cpu().detach().numpy(), y.cpu().detach().numpy())
                if kld < self.BEST_KLD and kld != np.inf:
                    self.BEST_KLD = kld
                    if self.args.mode=='train': # do not print messages when choosing other mode
                        torch.save({
                            'args': self.args,
                            'model_d': self.model_d.state_dict(),
                            'model_g': self.model_g.state_dict()
                        }, f'./model/{self.args.stock}_{self.args.name}_best.pth')
                        print(f'update best model with kld = {kld}')
        return total_loss_d, total_loss_g, kld
    
    def predict(self, test_loader):
        # load scaler
        with open(f'./data/{self.args.stock}/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        
        # set variables
        self.model_g.eval()
        y_preds = []
        y_trues = []
        
        # prediction
        with torch.inference_mode():
            for _, (X, y) in enumerate(test_loader):
                X, y = X.to(self.device), y.to(self.device)
                tmp = np.array([])
                for _ in range(self.args.pred_times):
                    noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                    y_pred = self.model_g(X, noise).cpu().detach().numpy()
                    y_pred = scaler_y.inverse_transform(y_pred) # inverse transform
                    y_pred = np.expand_dims(y_pred, axis=2)
                    tmp = y_pred if tmp.size==0 else np.concatenate((tmp, y_pred), axis=2)
                
                y = y.cpu().detach().numpy()
                y = scaler_y.inverse_transform(y) # inverse transform
                
                # concatenate array
                y_preds = tmp if len(y_preds)==0 else np.concatenate((y_preds, tmp), axis=0) #[num_time_step, seq_len, pred_times]
                y_trues = y if len(y_trues)==0 else np.concatenate((y_trues, y), axis=0) # [num_time_step, seq_len]
            return y_preds, y_trues

if __name__ == '__main__':
    args = parse_args()
    args.mode = 'train'
    train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
    train_size = int(0.9 * len(train_datasets))
    val_size = len(train_datasets) - train_size
    train_data, val_data = random_split(train_datasets, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        
    wgan_model = wgan(train_datasets, args)
    print('----------------------------------------------------------------')
    print('Start training...')
    print('----------------------------------------------------------------')
    print('hyperparameters: ')
    filter_val = ['noise_dim', 
                  'epoch', 'batch_size', 
                  'hidden_dim_g', 'num_layers_g', 'lr_g',
                  'hidden_dim_d', 'num_layers_d', 'lr_d',
                  'd_iter', 'gp_lambda']
    for k, v in vars(args).items():
        if k in filter_val: print("{}:\t{}".format(k, v))
    print('----------------------------------------------------------------')
    results = wgan_model.train(train_loader, val_loader)
    save_loss_curve(results, args)
    save_model(wgan_model.model_d, wgan_model.model_g, args)