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
    parser.add_argument('--dataset', type=str, default='md', choices=['metr-la', 'pems-bay', 'pemsd7-m', 'md'])
    parser.add_argument('--n_his', type=int, default=15)
    parser.add_argument('--n_pred', type=int, default=9, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=1)
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
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
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
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    # 데이터 로드
    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    
    # 전체 데이터셋에 대해 하나의 스케일러를 학습시킵니다
    full_data = pd.concat([train, val, test])
    zscore = preprocessing.StandardScaler()
    zscore.fit(full_data)
    
    # 각 데이터셋을 동일한 스케일러로 변환합니다
    train_scaled = zscore.transform(train)
    val_scaled = zscore.transform(val)
    test_scaled = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train_scaled, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val_scaled, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test_scaled, args.n_his, args.n_pred, device)

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
def test(minmax, loss, model, test_iter, args):
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

    predictions = minmax.inverse_transform(predictions)
    actual_values = minmax.inverse_transform(actual_values)

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, minmax)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    sensor_id = 216
    samples_per_day = 288  # 5분 간격으로 하루 데이터: 24시간 * 12

    # 마지막 n 일치 데이터만 선택
    last_days = 15 * samples_per_day
    predictions = predictions[-last_days:]
    actual_values = actual_values[-last_days:]

    def reshape_to_days(data):
        return data.reshape(-1, samples_per_day)

    pred_days = reshape_to_days(predictions[:, sensor_id])
    actual_days = reshape_to_days(actual_values[:, sensor_id])

    # 시간 배열 생성
    minutes_in_day = np.arange(0, 24*60, 5)  # 0분부터 1435분까지 5분 간격
    time_points = minutes_in_day / 60  # 시간 단위로 변환
    
    # x축 눈금 설정
    time_ticks = np.arange(0, 24, 2)  # 2시간 간격
    time_labels = [f'{int(h):02d}:00' for h in time_ticks]
    
    for day in range(15):  # 마지막 n일 시각화
        predictions_day = pred_days[day]
        actual_values_day = actual_days[day]
        
        # 전체 하루 데이터 성능 계산
        day_mae = np.mean(np.abs(predictions_day - actual_values_day))
        day_wmape = (np.sum(np.abs(predictions_day - actual_values_day)) /
                     (np.sum(np.abs(actual_values_day)) + 1e-10)) * 100
        
        plt.figure(figsize=(15, 6))
        
        # 데이터 플로팅
        plt.plot(time_points, actual_values_day, 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
        plt.plot(time_points, predictions_day, 'r--', label='Prediction', linewidth=2, alpha=0.7)
        

        # Peak Time 탐지
        peak_index = np.argmax(actual_values_day)
        peak_time = time_points[peak_index]

        # ±1시간 동안의 인덱스 계산
        samples_per_hour = 12  # 5분 간격 → 12 샘플 = 1시간
        start_idx = max(0, peak_index - samples_per_hour)
        end_idx = min(len(actual_values_day), peak_index + samples_per_hour + 1)

        # ±1시간 동안의 MAE 및 WMAPE 계산
        peak_predictions = predictions_day[start_idx:end_idx]
        peak_actuals = actual_values_day[start_idx:end_idx]

        peak_mae = np.mean(np.abs(peak_predictions - peak_actuals))
        peak_wmape = (np.sum(np.abs(peak_predictions - peak_actuals)) / 
                      (np.sum(np.abs(peak_actuals)) + 1e-10)) * 100

        # Peak Time 및 ±1시간 구간 강조
        plt.axvline(peak_time, color='green', linestyle='--', label='Peak Time', alpha=0.8)
        plt.axvspan(time_points[start_idx], time_points[end_idx-1], color='yellow', alpha=0.2)

        # 성능 지표 그래프 내 표시
        plt.text(peak_time, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05,
                 f'±1h MAE: {peak_mae:.2f}\n±1h WMAPE: {peak_wmape:.2f}%',
                 color='black', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8))

        # 그래프 설정
        plt.title(f'Day {1+day} Crowd Count Prediction\nMyeongdong (Node {sensor_id})', fontsize=14)
        plt.xlabel('Time (KST)', fontsize=12)
        plt.ylabel('Crowd Count', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # x축 설정
        plt.xticks(time_ticks, time_labels)

        # 급격한 변화 구간 및 Peak Time 성능 출력
        print(f'\nDay {1+day} Statistics:')
        print(f'Daily MAE: {day_mae:.2f}')
        print(f'Daily WMAPE: {day_wmape:.2f}%')
        print(f'Peak Time: {int(peak_time):02d}:{int((peak_time % 1) * 60):02d}')
        print(f'±1 Hour Around Peak MAE: {peak_mae:.2f}')
        print(f'±1 Hour Around Peak WMAPE: {peak_wmape:.2f}%')

        plt.tight_layout()
        plt.savefig(f'1_{sensor_id}_myeongdong_day {1+day}_events_{args.dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()

@torch.no_grad()
def test_daily_wmape(minmax, loss, model, test_iter, args):
    model.load_state_dict(torch.load(f"STGCN_{args.dataset}.pt"))
    model.eval()

    predictions = []
    actual_values = []
    
    for x, y in test_iter:
        y_pred = model(x).view(len(x), -1)
        predictions.append(y_pred.cpu().numpy())
        actual_values.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actual_values = np.concatenate(actual_values, axis=0)

    predictions = minmax.inverse_transform(predictions)
    actual_values = minmax.inverse_transform(actual_values)

    # 데이터 정보
    num_sensors = predictions.shape[1]
    num_timepoints = predictions.shape[0]
    samples_per_day = 288  # 5분 간격 하루 데이터 (24시간 * 12)
    num_days = num_timepoints // samples_per_day
    
    # 날짜별 WMAPE 계산
    wmape_matrix = np.zeros((num_timepoints, num_sensors))
    daily_wmape_matrix = np.zeros((num_days, num_sensors))

    for t in range(num_timepoints):
        actual_t = actual_values[t]
        pred_t = predictions[t]

        abs_error = np.abs(actual_t - pred_t)
        
        # 작은 값 방지 (분모가 너무 작을 경우 최소값 설정)
        actual_t = np.where(actual_t < 1e-5, 1e-5, actual_t)

        wmape_matrix[t, :] = abs_error / actual_t * 100  # 센서별 오차율(%)

        # WMAPE가 비정상적으로 큰 경우 최대값 제한 (예: 1000%)
        wmape_matrix[t, :] = np.clip(wmape_matrix[t, :], 0, 1000)
    
    for d in range(num_days):
        start_idx = d * samples_per_day
        end_idx = start_idx + samples_per_day
        daily_wmape_matrix[d, :] = np.mean(wmape_matrix[start_idx:end_idx, :], axis=0)
    
    # 마지막 15일치 데이터만 선택
    last_15_days = min(15, num_days)
    df_daily_wmape = pd.DataFrame(daily_wmape_matrix[-last_15_days:, :])
    df_daily_wmape.index = [f"Day {num_days - last_15_days + i + 1}" for i in range(last_15_days)]
    df_daily_wmape.columns = [f"Sensor {i+1}" for i in range(num_sensors)]
    df_daily_wmape.to_csv(f"daily_wmape_results_{args.dataset}.csv")
    
    print(f"최근 15일 WMAPE 결과가 daily_wmape_results_{args.dataset}.csv 파일에 저장되었습니다.")
    
    
if __name__ == "__main__":
    # Logging
    # logger = logging.getLogger('stgcn')
    # logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, minmax, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    if args.mode == 'train':
        train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
        # test_daily_wmape(minmax, loss, model, test_iter, args)
        test(minmax, loss, model, test_iter, args)
    else:  # test mode
        if os.path.exists("STGCN_" + args.dataset + ".pt"):
            # test_daily_wmape(minmax, loss, model, test_iter, args)
            test(minmax, loss, model, test_iter, args)
            # test(minmax, loss, model, test_iter, args)
        else:
            print(f"모델 파일이 없습니다. 먼저 학습을 진행해주세요.")
            print("실행 방법: python main.py --mode train")
            
            
            
