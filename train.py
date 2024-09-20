import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from datetime import datetime
# Import file
from preprocessing import *
from model import *
from utils import *
from arguments import *
from eval import *

class wgan:
    def __init__(self,stock_data, args):
        self.args = args
        self.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
        self.model_d = discriminator(stock_data.num_features, 1, self.device, self.args).to(self.device)
        self.model_g = generator(stock_data.num_features, self.args.noise_dim, stock_data.target_length, self.device, self.args).to(self.device)
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
        return -torch.mean(self.model_d(cond, real_data)) + torch.mean(self.model_d(cond, fake_data)) + gradient_penalty
    
    def compute_gradient_penalty(self, cond, real_data, fake_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1)
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
        return 10 * ((gradients_norm - 1) ** 2).mean()
    
    def train(self, train_loader, test_loader, val_dates=None):
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
                    real_data = y.unsqueeze(2) # in this case, real_data is two dimensional
                    fake_data = self.model_g(X, noise)
                    assert not torch.isnan(fake_data).any(), 'Generated data has nan values. Stop training.'
                    gradient_penalty = self.compute_gradient_penalty(X, real_data, fake_data)
                    loss_d = self.discriminator_loss(X, real_data, fake_data, gradient_penalty)
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()
                
                # train generator (minimize wgan loss)
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                loss_g = self.generator_loss(X, fake_data)
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()
                
                # train discriminator part 2
                for _ in range(self.args.d_iter):
                    noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                    real_data = y.unsqueeze(2) # in this case, real_data is two dimensional
                    fake_data = self.model_g(X, noise)
                    assert not torch.isnan(fake_data).any(), 'Generated data has nan values. Stop training.'
                    gradient_penalty = self.compute_gradient_penalty(X, real_data, fake_data)
                    loss_d = self.discriminator_loss(X, real_data, fake_data, gradient_penalty)
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()
                
                # train generator (minimize mae loss)
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                mae_loss = loss_fn(real_data, fake_data)
                optimizer_g.zero_grad()
                mae_loss.backward()
                optimizer_g.step()
            total_loss_d += loss_d.cpu().detach().numpy()
            total_loss_g += loss_g.cpu().detach().numpy()
            test_loss_d, test_loss_g, kld = self.test(test_loader)
            
            results['loss_d'].append(total_loss_d)
            results['loss_g'].append(total_loss_g)
            results['test_loss_d'].append(test_loss_d)
            results['test_loss_g'].append(test_loss_g)
            results['test_kld'].append(kld)
            
            if self.args.mode=='train':
                if (epoch+1)%(args.epoch//10)==0:
                    print(f'Epoch: {epoch+1}/{args.epoch}, loss_d: {total_loss_d:.2f}, loss_g: {total_loss_g:.2f}, test loss_d: {test_loss_d:.2f}, test loss_g: {test_loss_g:.2f}')
                    # save plot and histogram at each epoch (need to set stock evaluation dates)
                    if val_dates is not None:
                        for date in val_dates:
                            val_date, y_preds, y_trues = prepare_eval_data(self.model_g, stock_data, self.device, date, self.args)
                            save_predict_plot(args, f'./logs/{self.FOLDER_NAME}/pred', f'{val_date[-1]}_epoch{epoch+1}', val_date, y_preds, y_trues)
                            dist_eval_date, dist_y_preds, dist_y_trues = prepare_eval_data(self.model_g, stock_data, self.device, date, self.args, pred_times=10)
                            save_dist_plot(args, f'./logs/{self.FOLDER_NAME}/dist', f'{val_date[-1]}_epoch{epoch+1}', dist_eval_date, dist_y_preds, dist_y_trues)
        return results
    def test(self, test_loader):
        with torch.inference_mode():
            total_loss_d, total_loss_g = 0, 0
            for _, (X, y) in enumerate(test_loader):
                X, y = X.to(self.device), y.to(self.device)
                # add noise
                self.model_g.eval()
                self.model_d.eval()
                
                # evaluate discriminator
                real_data = y.unsqueeze(2)
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                loss_d = self.discriminator_loss(X, real_data, fake_data, 0)
                total_loss_d += loss_d.cpu().detach().numpy()
                
                # evaluate generator
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                loss_g = self.generator_loss(X, fake_data)
                total_loss_g += loss_g.cpu().detach().numpy()
                
                # update best model (use kld)
                kld = calc_kld(fake_data.cpu().detach().numpy(), real_data.cpu().detach().numpy(), 100, 0, 1)
                if kld < self.BEST_KLD and kld != np.inf:
                    self.BEST_KLD = kld
                    if self.args.mode=='train': # do not print messages when bayesian optimization
                        torch.save({
                            'args': self.args,
                            'model_d': self.model_d.state_dict(),
                            'model_g': self.model_g.state_dict()
                        }, f'./model/{self.args.stock}_{self.args.name}_best.pth')
                        print(f'update best model with kld = {kld}')
        return total_loss_d, total_loss_g, kld
    
if __name__ == '__main__':
    args = parse_args()
    if args.mode=='train':
        stock_data = StockDataset(args)
        train_size = int(0.9 * len(stock_data))
        test_size = len(stock_data) - train_size
        train_datasets, test_datasets = random_split(stock_data, [train_size, test_size])
        train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)
        val_dates = random.sample(stock_data.time_intervals, args.num_val)
        
        wgan_model = wgan(stock_data, args)
        print('----------------------------------------------------------------')
        print('Start training...')
        print('----------------------------------------------------------------')
        print('hyperparameters: ')
        for k, v in vars(args).items():
                print("{}:\t{}".format(k, v))
        print('----------------------------------------------------------------')
        results = wgan_model.train(train_loader, test_loader, val_dates)
        save_loss_curve(results, args)
        save_model(wgan_model.model_d, wgan_model.model_g, args)