# Import Package
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# Import file
from preprocessing import *
from model import *
from utils import *
from arguments import *

def compute_gradient_penalty(real_data, fake_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = model_d(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return 10 * ((gradients_norm - 1) ** 2).mean()

def generator_loss(real_data, fake_data):
    return -torch.mean(model_d(fake_data)) + loss_fn(fake_data, real_data) + change_loss_fn(real_data[:, :fake_data.shape[1], :], fake_data)

class PriceChangeLoss(nn.Module):
    def __init__(self):
        super(PriceChangeLoss, self).__init__()

    def forward(self, real_prices, predicted_prices):
        real_change = (real_prices[:, 1, :] - real_prices[:, -1, :]) / real_prices[:, -1, :]
        predicted_change = (predicted_prices[:, 1, :] - predicted_prices[:, -1, :]) / predicted_prices[:, -1, :]
        loss = torch.mean(torch.abs(real_change - predicted_change))
        return loss

def discriminator_loss(real_data, fake_data, gradient_penalty=0):
    return -torch.mean(model_d(real_data)) + torch.mean(model_d(fake_data)) + gradient_penalty

def train_iter(X, y, model_d, model_g, optimizer_d, optimizer_g, args):
    # train discriminator
    model_d.train()
    model_g.train()

    # add noise
    noise = torch.randn(X.shape[0], X.shape[1], noise_dim).to(device)
    X = torch.cat((X, noise), dim=2)
    # train discriminator
    for _ in range(3):
        real_data = y.unsqueeze(2)
        fake_data = model_g(X)
        if torch.isnan(fake_data).any():
            print('generated data has NaN values.')
            os._exit(0)
        gradient_penalty = compute_gradient_penalty(real_data, fake_data)
        loss_d = discriminator_loss(real_data, fake_data, gradient_penalty)
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
    
    # train generator
    fake_data = model_g(X)
    loss_g = generator_loss(real_data, fake_data)
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()
    
    return loss_d, loss_g    

def test_iter(test_loader, model_d, model_g, device, args):
    with torch.inference_mode():
        model_d, model_g = model_d.to(device), model_g.to(device)
        total_loss_d, total_loss_g = 0, 0
        for _, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            # add noise
            noise = torch.randn(X.shape[0], X.shape[1], noise_dim).to(device)
            X = torch.cat((X, noise), dim=2)
            
            model_g.eval()
            model_d.eval()
            # evaluate discriminator
            real_data = y.unsqueeze(2)
            fake_data = model_g(X)
            loss_d = discriminator_loss(real_data, fake_data)
            total_loss_d += loss_d.cpu().detach().numpy()
            
            # evaluate generator
            fake_data = model_g(X)
            loss_g = generator_loss(real_data, fake_data)
            total_loss_g += loss_g.cpu().detach().numpy()
    return total_loss_d, total_loss_g

def train(train_loader, test_loader, model_d, model_g, optimizer_d, optimizer_g, device, args):
    model_d, model_g = model_d.to(device), model_g.to(device)
    results = {'loss_d': [], 'loss_g': [], 'test_loss_d': [], 'test_loss_g': []}
    for epoch in range(args.epoch):
        for _, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            total_loss_d, total_loss_g = 0, 0
            loss_d, loss_g = train_iter(X, y, model_d, model_g, optimizer_d, optimizer_g, args)
            total_loss_d += loss_d.cpu().detach().numpy()
            total_loss_g += loss_g.cpu().detach().numpy()
            
        test_loss_d, test_loss_g = test_iter(test_loader, model_d, model_g, device, args)
        results['loss_d'].append(total_loss_d)
        results['loss_g'].append(total_loss_g)
        results['test_loss_d'].append(test_loss_d)
        results['test_loss_g'].append(test_loss_g)
        # 寫入 Summary Writer
        writer.add_scalar('Loss/train/Discriminator', loss_d, epoch)
        writer.add_scalar('Loss/train/Generator', loss_g, epoch)
        writer.add_scalar('Loss/test/Generator', test_loss_g, epoch)
        writer.add_scalar('Loss/test/Discriminator', test_loss_d, epoch)
        if (epoch+1)%(args.epoch//10)==0:
            print(f'Epoch: {epoch+1}/{args.epoch}, loss_d: {total_loss_d:.2f}, loss_g: {total_loss_g:.2f}, test loss_d: {test_loss_d:.2f}, test loss_g: {test_loss_g:.2f}')
            writer.add_
    return results
            
if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    FOLDER_NAME = f'{args.stock}_{args.name}'
    if not os.path.exists(f'logs/{FOLDER_NAME}'):os.makedirs(f'logs/{FOLDER_NAME}')
    else: clear_folder(f'logs/{FOLDER_NAME}')
    writer = SummaryWriter(log_dir=f'logs/{FOLDER_NAME}')
    
    # Prepare train and test data
    stock_data = StockDataset(args)
    split_idx = int(round(len(stock_data)*0.9))
    train_datasets = Subset(stock_data, range(split_idx))
    test_datasets = Subset(stock_data, range(split_idx, len(stock_data)))
    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)
    noise_dim = args.noise_dim
    # model setting
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    model_d = discriminator(1)
    model_g = generator(stock_data.num_features + noise_dim, stock_data.target_length, device)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr_d, betas = (0.0, 0.9), weight_decay = 1e-5)
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr_g, betas = (0.0, 0.9), weight_decay = 1e-5)
    loss_fn = nn.SmoothL1Loss()
    change_loss_fn = PriceChangeLoss()
    # train model
    print('------------------------------------------------------------------------------------------------')
    print(f'Start training {args.name}')
    print(f'Using device: {device}')
    print(f'Stock: {args.stock}')
    print(f'window size: {args.window_size}, target length: {args.target_length}, time step: {args.time_step}')
    print(f'Model setting: epoch: {args.epoch}, batch_size: {args.batch_size}, lr_g: {args.lr_g}, lr_d: {args.lr_d}')
    print('------------------------------------------------------------------------------------------------')
    start_time = datetime.now()
    results = train(train_loader, test_loader, model_d, model_g, optimizer_d, optimizer_g, device, args)
    end_time = datetime.now()
    train_time = (end_time - start_time).total_seconds()
    print('------------------------------------------------------------------------------------------------')
    print(f'Training model spent {train_time: 2f} seconds.')
    save_loss_curve(results, args)
    save_model(model_d, model_g, args)
    