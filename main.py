import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models

#import nni

def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='md_cell', choices=['metr-la', 'pems-bay', 'pemsd7-m', 'md_cell'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=9, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='nadamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
        torch.cuda.empty_cache() # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect() # Clean cache
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'count.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter

def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path="STGCN_" + args.dataset + ".pt")

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break

@torch.no_grad()
def val(model, val_iter):
    model.eval()

    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)
@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.load_state_dict(torch.load("STGCN_" + args.dataset + ".pt"))
    model.eval()

    predictions = []
    actual_values = []

    for x, y in test_iter:
        y_pred = model(x).view(len(x), -1)
        predictions.append(y_pred.cpu().numpy())
        actual_values.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actual_values = np.concatenate(actual_values, axis=0)

    predictions = zscore.inverse_transform(predictions)
    actual_values = zscore.inverse_transform(actual_values)

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    sensor_id = 0
    samples_per_day = 288  # 5분 간격으로 하루 데이터
    days_to_plot = 7

    def reshape_to_days(data):
        n_days = len(data) // samples_per_day
        return data[:n_days * samples_per_day].reshape(-1, samples_per_day)

    pred_days = reshape_to_days(predictions[:, sensor_id])
    actual_days = reshape_to_days(actual_values[:, sensor_id])

    # 시간 조정을 위한 shift
    shift_amount = int(5 * 12)  # 5시간 * 12(5분 간격)
    for i in range(len(pred_days)):
        pred_days[i] = np.roll(pred_days[i], -shift_amount)
        actual_days[i] = np.roll(actual_days[i], -shift_amount)

    # 시간 배열 생성
    minutes_in_day = np.arange(0, 24*60, 5)  # 0분부터 1435분까지 5분 간격
    time_points = minutes_in_day / 60  # 시간 단위로 변환
    
    # x축 눈금 설정
    time_ticks = np.arange(0, 24, 2)  # 2시간 간격
    time_labels = [f'{int(h):02d}:00' for h in time_ticks]
    
    for day in range(min(days_to_plot, len(pred_days))):
        predictions_day = pred_days[day]
        actual_values_day = actual_days[day]
        
        plt.figure(figsize=(15, 6))
        
        # 데이터 플로팅
        plt.plot(time_points, actual_values_day, 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
        plt.plot(time_points, predictions_day, 'r--', label='Prediction', linewidth=2, alpha=0.7)
        
        plt.title(f'24-10-{day + 11} Crowd Count Prediction\nMyeongdong (Sensor {sensor_id})', fontsize=14)
        plt.xlabel('Time (KST)', fontsize=12)
        plt.ylabel('Crowd Count', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # x축 설정
        plt.xticks(time_ticks, time_labels)

        # MAE 계산 및 표시 (6시간 단위)
        period_hours = 6
        periods_per_day = 24 // period_hours
        for period in range(periods_per_day):
            start_period = period * period_hours * 12
            end_period = start_period + period_hours * 12
            period_mae = np.mean(np.abs(predictions_day[start_period:end_period] - 
                                      actual_values_day[start_period:end_period]))
            
            plt.text(period_hours/2 + period * period_hours, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.05,
                    f'{period*period_hours:02d}-{(period+1)*period_hours:02d}h\nMAE: {period_mae:.1f}',
                    ha='center', va='bottom', fontsize=9)

        day_mae = np.mean(np.abs(predictions_day - actual_values_day))
        plt.text(0.5, 0.95, f'Daily MAE: {day_mae:.2f}',
                horizontalalignment='center',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=12)

        plt.tight_layout()
        plt.savefig(f'city_day_{day+1}_prediction_{args.dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Peak hour in KST
        # Peak hour in KST
        peak_hour = time_points[np.argmax(actual_values_day)]
        # WMAPE 계산
        day_wmape = np.sum(np.abs(predictions_day - actual_values_day)) / (np.sum(np.abs(actual_values_day)) + 1e-10) * 100

        print(f'\nDay {day + 1} Statistics:')
        print(f'Daily MAE: {day_mae:.2f}')
        print(f'Daily WMAPE: {day_wmape:.2f}%')
        print(f'Peak Hour: {int(peak_hour):02d}:{int((peak_hour % 1) * 60):02d}')
        
if __name__ == "__main__":
    # Logging
    # logger = logging.getLogger('stgcn')
    # logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    if args.mode == 'train':
        train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
        test(zscore, loss, model, test_iter, args)
    else:  # test mode
        if os.path.exists("STGCN_" + args.dataset + ".pt"):
            test(zscore, loss, model, test_iter, args)
        else:
            print(f"모델 파일이 없습니다. 먼저 학습을 진행해주세요.")
            print("실행 방법: python main.py --mode train")