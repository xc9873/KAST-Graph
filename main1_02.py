import os
import argparse
import pickle as pk
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import WeightedMeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from network.DCRNN_xy import DCRNN
from utils import generate_dataset, generate_dataset1, load_metr_la_data, get_normalized_adj, load_binghai_data, load_binghai1_data, load_binghai2_data
import sys
from datetime import datetime
import time  # 导入时间模块
import matplotlib.pyplot as plt  # 添加 Matplotlib 以绘制 MAE 曲线

use_gpu = True
num_timesteps_input = 14
num_timesteps_output = 7
epsilon= 100
epochs = 1000
batch_size = 100
data_name = 'binghai'  # pems04 pems08 metr
model_name = 'dcrnn'  # stgcn stgcn_r agcrn dcrnn gwnet mlp
node_dim = 4  # 修改嵌入向量维度 2 4 8 16
early_stop_epoch = 30
use_gcl = True

# 初始化训练和推理时间
train_time_total = 0
inference_time_total = 0

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch[:,:,:,0].unsqueeze(-1), y_batch[:,:,1:,:])

        loss = loss_criterion(out, y_batch[:,:,0,:])
        if use_gcl:
            loss += net.cl_loss
        #  print(out.shape, y_batch.shape)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        torch.cuda.empty_cache()
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    print(f"{model_name} training...")
    current_timestamp = datetime.now().timestamp()
    torch.manual_seed(7)
    if data_name == "binghai":
        # A, X, means, stds = load_binghai1_data()
        A, X, means, stds = load_binghai2_data()
    else:
        A, X, means, stds = load_metr_la_data()
    print(A.shape, X.shape)

    # fixme
    X = X*stds[0]+means[0]
    x_max = X.max()
    x_min = X.min()
    X = (X-X.min())/(X.max()-X.min())
    #

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset1(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset1(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset1(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    if model_name == "our":
        net = our(num_nodes=A_wave.shape[0],
                input_dim=1,
                hidden_dim=64,
                K=1,
                num_layers=2,
                out_horizon=num_timesteps_output,
                node_dim=node_dim).to(device=args.device)
    elif model_name == "stgcn":
        net = STGCN(A_wave.shape[0],
                1,
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)
    elif model_name == "dcrnn":
        net = DCRNN(num_nodes=A_wave.shape[0],
                input_dim=1,
                hidden_dim=64,
                K=1,
                num_layers=2,
                out_horizon=num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    train_maes = []
    validation_maes = []
    early_stop_now_turn = 0

    for epoch in range(epochs):

        start_train_time = time.time()  # 记录训练开始时间

        if early_stop_now_turn == early_stop_epoch:
            print("Early stopping!")
            break

        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # 计算训练集 MAE
        with torch.no_grad():
            net.eval()
            training_input = training_input.to(device=args.device)
            training_target = training_target.to(device=args.device)

            out_train = net(A_wave, training_input[:, :, :, 0].unsqueeze(-1), training_target[:, :, 1:, :])
            out_train_unnormalized = out_train * (x_max - x_min) + x_min
            out_train_unnormalized = out_train_unnormalized.detach().cpu().numpy()
            target_train_unnormalized = training_target * (x_max - x_min) + x_min
            target_train_unnormalized = target_train_unnormalized[:, :, 0, :].detach().cpu().numpy()

            train_mae = np.mean(np.absolute(out_train_unnormalized - target_train_unnormalized))
            train_maes.append(train_mae)

            training_input = training_input.to(device="cpu")
            training_target = training_target.to(device="cpu")
            torch.cuda.empty_cache()

        train_time_total += time.time() - start_train_time  # 累加本轮训练耗时


        if epoch%5==0:
            with torch.no_grad():
                net.eval()
                val_input = val_input.to(device=args.device)
                val_target = val_target.to(device=args.device)

                out = net(A_wave, val_input[:,:,:,0].unsqueeze(-1), val_target[:,:,1:,:])
                val_loss = loss_criterion(out, val_target[:,:,0,:])
                if use_gcl:
                    val_loss += net.cl_loss
                val_loss = val_loss.to(device="cpu")
                validation_losses.append(np.ndarray.item(val_loss.detach().numpy()))

                # fixme
                # 输出和目标的去归一化操作
                out_unnormalized = out * (x_max - x_min) + x_min
                out_unnormalized = out_unnormalized.detach().cpu().numpy()
                target_unnormalized = val_target * (x_max - x_min) + x_min
                target_unnormalized = target_unnormalized[:,:,0,:].detach().cpu().numpy()

                # out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
                # target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]

                mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                mse = np.mean(np.square(out_unnormalized - target_unnormalized))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs(out_unnormalized - target_unnormalized) / (target_unnormalized + epsilon))

          #      print(out_unnormalized[0,0,:])
          #      print(target_unnormalized[0,0,:])
                validation_maes.append(mae)

                out = None
                val_input = val_input.to(device="cpu")
                val_target = val_target.to(device="cpu")
                torch.cuda.empty_cache()
            if epoch == 0:
                mae_best = mae
                rmse_best = rmse
                mape_best = mape
            elif mae_best>mae:
                mae_best = mae
                rmse_best = rmse
                mape_best = mape
#                if model_name=="stgcn":
                torch.save(net, f'./output1/{model_name}/{model_name}_{data_name}_{current_timestamp}.pt')
#                else:
#                    torch.save(net, f'./output1/{model_name}_{data_name}_{node_dim}_{current_timestamp}.pt')
            else:
                early_stop_now_turn += 1
            print("epoch {} mae: {:.4f} rmse: {:.4f} mape: {:.4f}".format(epoch, mae_best, rmse_best, mape_best))

    # # 绘制 MAE 曲线
    #
    # # 创建画布，1 行 2 列
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #
    # # 绘制 Train MAE 曲线
    # axes[0].plot(range(len(train_maes)), train_maes, color='red', linestyle='--', marker='o', label="Train MAE")
    # axes[0].set_xlabel("Epoch", fontsize=13)
    # axes[0].set_ylabel("MAE", fontsize=13)
    # # axes[0].set_title(f"{model_name} on {data_name} - Train MAE")
    # axes[0].legend(fontsize=13)
    # axes[0].tick_params(axis='x', which='both', labelsize=13)  # x 轴刻度字体大小
    # axes[0].tick_params(axis='y', which='both', labelsize=13)  # y 轴刻度字体大小
    # axes[0].grid()
    #
    # # 绘制 Validation MAE 曲线
    # axes[1].plot(range(0, len(validation_maes) * 5, 5), validation_maes, color='blue', marker='o',
    #              label="Validation MAE")
    # axes[1].set_xlabel("Epoch", fontsize=13)
    # axes[1].set_ylabel("MAE", fontsize=13)
    # # axes[1].set_title(f"{model_name} on {data_name} - Validation MAE")
    # axes[1].legend(fontsize=13)
    # axes[1].tick_params(axis='x', which='both', labelsize=13)  # x 轴刻度字体大小
    # axes[1].tick_params(axis='y', which='both', labelsize=13)  # y 轴刻度字体大小
    # axes[1].grid()
    #
    # # 调整子图间距
    # plt.tight_layout()
    # # 显示图像
    # plt.show()

#    if model_name=="stgcn":
    net = torch.load(f'./output1/{model_name}/{model_name}_{data_name}_{current_timestamp}.pt').to(device=args.device)
#    else:
#        net = torch.load(f'./output1/{model_name}_{data_name}_{node_dim}_{current_timestamp}.pt').to(device=args.device)
      # , map_location="cuda:0"
    label_data_dict = {'target':[], 'predict':[]}
    label_data_dict_predict = []
    label_data_dict_target = []
    with torch.no_grad():
        net.eval()

        start_inference_time = time.time()  # 记录推理开始时间

        test_input = test_input.to(device=args.device)
        test_target = test_target.to(device=args.device)

        out = net(A_wave, test_input[:,:,:,0].unsqueeze(-1), test_target[:,:,1:,:])

        inference_time_total += time.time() - start_inference_time  # 记录推理时间

        test_loss = loss_criterion(out.squeeze(-1), test_target[:,:,0,:])
        if use_gcl:
            test_loss += net.cl_loss
        test_loss = test_loss.to(device="cpu")
        validation_losses.append(np.ndarray.item(test_loss.detach().numpy()))

        # fixme
        # 计算测试指标
        out_unnormalized = out * (x_max - x_min) + x_min
        out_unnormalized = out_unnormalized.detach().cpu().numpy()
        target_unnormalized = test_target * (x_max - x_min) + x_min
        target_unnormalized = target_unnormalized[:,:,0,:].detach().cpu().numpy()
        label_data_dict_predict.append(out_unnormalized)
        label_data_dict_target.append(target_unnormalized)

        # out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
        # target_unnormalized = test_target.detach().cpu().numpy()*stds[0]+means[0]

        mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        mse = np.mean(np.square(out_unnormalized - target_unnormalized))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(out_unnormalized - target_unnormalized) / (target_unnormalized + epsilon))
        validation_maes.append(mae)

        out = None
        test_input = test_input.to(device="cpu")
        test_target = test_target.to(device="cpu")

    import pickle

    label_data_dict['target'] = np.stack(label_data_dict_target)
    label_data_dict['predict'] = np.stack(label_data_dict_predict)

    with open("Predict_Target.npy", "wb") as file:
        pickle.dump(label_data_dict, file)

    print("test metric, mae: {:.4f} rmse: {:.4f} mape: {:.4f}".format(mae, rmse, mape))
    print(f"Total training time: {train_time_total:.2f} seconds")
    print(f"Total inference time: {inference_time_total:.2f} seconds")
    print(f"{model_name}_{data_name}_{node_dim}")
    f = open(f'./output1/{model_name}/prediction_scores.txt', 'a')
#    f = open('/autodl-tmp/STGCN-PyTorch-master/output1' + '/' + f'{model_name}' + '/' 'prediction_scores.txt', 'a')
    f.write("%s, MAE, RMSE, MAPE, time, %.10f, %.10f, %.10f, %.10f\n" % (model_name, mae, rmse, mape, current_timestamp))
    f.close()

    # # 创建画布
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #
    # # 绘制训练损失曲线
    # axes[0].plot(training_losses, color='red', linestyle='--', marker='o', label="Training Loss")
    # axes[0].set_xlabel("Epoch", fontsize=13)
    # axes[0].set_ylabel("Loss", fontsize=13)
    # # axes[0].set_title("Training Loss Curve")
    # axes[0].legend(fontsize=13)
    # axes[0].tick_params(axis='x', which='both', labelsize=13)  # x 轴刻度字体大小
    # axes[0].tick_params(axis='y', which='both', labelsize=13)  # y 轴刻度字体大小
    # axes[0].grid()
    #
    # # 绘制验证损失曲线
    # axes[1].plot(validation_losses, color='blue', marker='o', label="Validation Loss")
    # axes[1].set_xlabel("Epoch", fontsize=13)
    # axes[1].set_ylabel("Loss", fontsize=13)
    # # axes[1].set_title("Validation Loss Curve")
    # axes[1].legend(fontsize=13)
    # axes[1].tick_params(axis='x', which='both', labelsize=13)  # x 轴刻度字体大小
    # axes[1].tick_params(axis='y', which='both', labelsize=13)  # y 轴刻度字体大小
    # axes[1].grid()
    #
    # # 调整子图间距
    # plt.tight_layout()
    # plt.show()