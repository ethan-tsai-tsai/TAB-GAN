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
        self.FOLDER_NAME = f'{args.stock}_{args.name}'
        if not os.path.exists(f'model/{self.FOLDER_NAME}'):os.makedirs(f'model/{self.FOLDER_NAME}')
        
        if self.args.mode=='train':
            if not os.path.exists(f'logs/{self.FOLDER_NAME}'):os.makedirs(f'logs/{self.FOLDER_NAME}')
            else: clear_folder(f'logs/{self.FOLDER_NAME}')
        
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
        optimizer_d = torch.optim.AdamW(self.model_d.parameters(), lr=self.args.lr_d, betas = (0.0, 0.9), weight_decay = 1e-3)
        optimizer_g = torch.optim.AdamW(self.model_g.parameters(), lr=self.args.lr_g, betas = (0.0, 0.9), weight_decay = 1e-3)
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
                    torch.nn.utils.clip_grad_norm_(self.model_d.parameters(), max_norm=1.0)
                    optimizer_d.step()
                
                # train generator (minimize wgan loss)
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                loss_g = self.generator_loss(X, fake_data)
                optimizer_g.zero_grad()
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.model_g.parameters(), max_norm=1.0)
                optimizer_g.step()
                
                # train discriminator
                # for _ in range(self.args.d_iter):
                #     noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                #     fake_data = self.model_g(X, noise)
                #     assert not torch.isnan(fake_data).any(), 'Generated data has nan values. Stop training.'
                #     gradient_penalty = self.compute_gradient_penalty(X, y, fake_data)
                #     loss_d = self.discriminator_loss(X, y, fake_data, gradient_penalty)
                #     optimizer_d.zero_grad()
                #     loss_d.backward()
                #     optimizer_d.step()
                
                # train generator (minimize mae loss)
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                fake_data = self.model_g(X, noise)
                mae_loss = loss_fn(y, fake_data)
                optimizer_g.zero_grad()
                mae_loss.backward()
                optimizer_g.step()
            total_loss_d += loss_d.cpu().detach().numpy()
            total_loss_g += loss_g.cpu().detach().numpy()
            test_loss_d, test_loss_g, kld, fid = self.validation(val_loader)
            
            results['loss_d'].append(total_loss_d)
            results['loss_g'].append(total_loss_g)
            results['test_loss_d'].append(test_loss_d)
            results['test_loss_g'].append(test_loss_g)
            results['test_kld'].append(kld)
            
            if (epoch+1)%(self.args.epoch//10)==0:
                if self.args.mode == 'train': self.validation_plot(val_datasets, f'Epoch{epoch + 1}_chart')
                # print(f'Epoch: {epoch+1}/{self.args.epoch}, loss_g: {mae_loss}')
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
                fid = fid_score(fake_data.cpu().detach().numpy(), y.cpu().detach().numpy())
                if kld < self.BEST_KLD and kld != np.inf:
                    self.BEST_KLD = kld
                    if self.args.mode=='train': # do not print messages when choosing other mode
                        file_name = f'./model/{self.args.stock}_{self.args.name}/best.pth'
                        save_model(self.model_d, self.model_g, self.args, file_name)
                        print(f'update best model with kld = {kld}')
        return total_loss_d, total_loss_g, kld, fid
    
    def validation_plot(self, val_datasets, file_name):
        X = torch.tensor(np.array(val_datasets.X), dtype=torch.float32)
        y = np.array(val_datasets.y)
        y[y == -10] = np.nan
        tmp = self.args.pred_times
        self.args.pred_times = 1
        y_pred, y_true = self.predict(X, y)
        plot_util.validation_chart(file_name, y_true, y_pred)

        self.args.pred_times = tmp
        
    def predict(self, X, y=None):
        # load scaler
        with open(f'./data/{self.args.stock}/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)

        # prediction
        y_preds = np.array([])
        self.model_g.eval()
        with torch.inference_mode():
            X = X.to(self.device)
            for _ in range(self.args.pred_times):
                noise = torch.randn(X.shape[0], self.args.noise_dim).to(self.device)
                y_pred = self.model_g(X, noise).cpu().detach().numpy()
                y_pred = scaler_y.inverse_transform(y_pred) 
                y_pred = np.expand_dims(y_pred, axis=2)
                y_preds = y_pred if y_preds.size==0 else np.concatenate((y_preds, y_pred), axis=2)
                
            y_trues = scaler_y.inverse_transform(y)
        return y_preds, y_trues

if __name__ == '__main__':
    
    start_time = datetime.now()
    
    # set arguments
    args = parse_args()
    args.mode = 'train'
    file_name = f'./model/{args.stock}_{args.name}/bayes_args.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            saved_args = pickle.load(f)
        for key, value in saved_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    train_datasets = StockDataset(args, f'./data/{args.stock}/train.csv')
    train_size = int(0.95 * len(train_datasets))
    val_size = len(train_datasets) - train_size
    train_data, val_data = random_split(train_datasets, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    val_datasets = StockDataset(args, f'./data/{args.stock}/val.csv')
    wgan_model = wgan(train_datasets, args)
    plot_util = plot_predicions(f'./logs/{args.stock}_{args.name}', args, val_datasets.time_intervals)
    
    print('----------------------------------------------------------------')
    print('Start training...')
    print('----------------------------------------------------------------')
    print('hyperparameters: ')
    filter_val = ['noise_dim', 
                  'epoch', 'batch_size', 
                  'hidden_dim_g', 'num_layers_g', 'num_head_g', 'lr_g',
                  'hidden_dim_d', 'num_layers_d', 'num_head_d', 'lr_d',
                  'd_iter', 'gp_lambda']
    for k, v in vars(args).items():
        if k in filter_val: print("{}:\t{}".format(k, v))
    print('----------------------------------------------------------------')
    results = wgan_model.train(train_loader, val_loader)
    save_loss_curve(results, args)
    file_name = f'./model/{args.stock}_{args.name}/final.pth'
    save_model(wgan_model.model_d, wgan_model.model_g, args, file_name)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).seconds
    print(f'Training time: {training_time//3600}:{(training_time%3600)//60}:{training_time%60}')