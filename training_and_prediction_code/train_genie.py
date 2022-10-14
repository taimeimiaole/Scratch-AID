# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
from mouse_data import *
import crnn
from common_tools import *
import pandas as pd
from typing import List
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = "/media/yhs/8608ebc4-123e-4bc8-803b-d3d75187dbd2/Automatic_itch_project/src/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    dataset_params = {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'frame_step': 1
    }

    cnn_encoder_params = {
        'cnn_out_dim': 256,  # 256
        'drop_prob': 0.20,
        'bn_momentum': 0.01
    }

    rnn_decoder_params = {
        'use_gru': True,
        'cnn_out_dim': 256,  # 256
        'rnn_hidden_layers': 2,
        'rnn_hidden_nodes': 512, # 512
        'num_classes': 2,
        'drop_prob': 0.20,
        'bidirectional': True
    }

    window_size = 45
    remove_bound = False
    sliding_size = 8
    positive_frame_thres = 23
    learning_rate = 1e-4
    MAX_EPOCH =20
    train_video = [1,2,3,4,5,6,7,8,
                   9,10,11,12,13,14,15,16,
                   17,18,19,20,21,22,23,24,
                    25,26,27,28,29,30,31,32]
    test_video = [33,34,35,36]
    log_interval = 2  # 打印间隔，默认每2个batch_size打印一次
    save_interval = 1  # 模型保存间隔，默认每个epoch保存一次


    # 配置训练时环境
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')


    parser = argparse.ArgumentParser()
    embedding_size = cnn_encoder_params['cnn_out_dim']
    attention_head = 16

    parser.add_argument("--embedding_size", help="embedding dimension of genes and tumors", type=int, default=embedding_size)
    parser.add_argument("--hidden_size", help="hidden layer dimension of MLP decoder", type=int, default=embedding_size*2)
    parser.add_argument("--attention_size", help="size of attention parameter beta_j", type=int, default=embedding_size)
    parser.add_argument("--attention_head", help="number of attention heads", type=int, default=attention_head)
    parser.add_argument("--test_inc_size", help="increment interval size between log outputs", type=int, default=embedding_size)
    parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.1)
    parser.add_argument("--ffn_num_input", help="FFN input size of the transformer", type=int, default=embedding_size)  # !!!A
    parser.add_argument("--ffn_num_hiddens", help="FFN hidden size of the transformer", type=int, default=embedding_size*2)
    parser.add_argument("--num_layers", help="Layer numbers in the transformer encoder", type=int, default=2)
    args = parser.parse_args()

    args.num_classes = rnn_decoder_params['num_classes']

    # 实例化计算图模型
    model = nn.Sequential(
        crnn.CNNEncoder(**cnn_encoder_params),
        crnn.RNNDecoder(**rnn_decoder_params)
        #crnn.GIT2(args)
    )
    model.to(device)

    # 多GPU训练
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('使用{}个GPU训练'.format(device_count))
        model = nn.DataParallel(model)



    # 提取网络参数，准备进行训练
    model_params = model.parameters()

    # 设定优化器
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M') + "Window45, no interval no discard, 1-32 train"
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Data loading
    # train_dir = "C:\\Users\\11764\\Desktop\\automatic_itch"
    train_dir = os.path.join(BASE_DIR, "..")

    train_data= mouse_Dataset(data_dir = train_dir, window_size = window_size, validation = False, frame_interval  = dataset_params['frame_step'],
    sliding_size = sliding_size,
    positive_frame_thres = positive_frame_thres, remove_bound = remove_bound, train_video = train_video, test_video = test_video)
    valid_data = mouse_Dataset(data_dir=train_dir, window_size=window_size, validation=True, frame_interval  = dataset_params['frame_step'],
                               predict_new= False,
    sliding_size = sliding_size,
    positive_frame_thres = positive_frame_thres, remove_bound = remove_bound, train_video = train_video, test_video = test_video)
    test_data = mouse_Dataset(data_dir=train_dir, window_size=window_size, validation=True, frame_interval  = dataset_params['frame_step'],
                               predict_new= True,
    sliding_size = sliding_size,
    positive_frame_thres = positive_frame_thres, remove_bound = remove_bound, train_video = train_video, test_video = test_video)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=dataset_params['batch_size'],
                              shuffle=dataset_params['shuffle'], num_workers=dataset_params['num_workers'])
    valid_loader = DataLoader(dataset=valid_data, batch_size=dataset_params['batch_size'],
                              num_workers=dataset_params['num_workers'])
    test_loader = DataLoader(dataset=test_data, batch_size=dataset_params['batch_size'],
                              num_workers=dataset_params['num_workers'])
    labels = [0,1]

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    # 提取网络参数，准备进行训练
    model_params = model.parameters()
    # 设定优化器
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)

    gamma = 0.30
    milestones = [5,10,15]  # 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, ...
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=gamma, milestones=milestones)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(MAX_EPOCH // 9) + 1)


# ============================ step 5/5 训练 ============================
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    log_df = pd.DataFrame(columns=['epoch', 'train Loss', 'training accuracy', "valid loss", "valid accuracy", "lr", "valid calss 1 F1"])
    log_df.to_csv(log_dir + "/log.csv", index=False)

    num_classes = rnn_decoder_params['num_classes']
    start_epoch = -1

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, model, criterion, optimizer, epoch, device, MAX_EPOCH, num_classes)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, model, criterion, device, num_classes)
        loss_test, acc_test, mat_test = ModelTrainer.valid(test_loader, model, criterion, device, num_classes)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"], ))

        confusion_mat = mat_test
        cls_num = 2
        classes = labels

        confusion_mat_N = confusion_mat.copy()
        for i in range(len(classes)):
            confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
        for i in range(cls_num):
            recall = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :]))
            precision = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))
            print('test (not select model) class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%} F1:{:.4f}'.format(
            classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
            confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
            confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i])),
            np.sqrt(recall * precision)))

        confusion_mat = mat_valid
        cls_num = 2
        classes = labels

        confusion_mat_N = confusion_mat.copy()
        F1 = [0,0]
        for i in range(len(classes)):
            confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
        for i in range(cls_num):
            recall = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :]))
            precision = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))
            F1[i] = np.sqrt(recall * precision)
            print('valid class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%} F1:{:.4f}'.format(
            classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
            confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
            confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i])),
            np.sqrt(recall * precision)))


        log_df = pd.DataFrame(columns=['step', 'train Loss', 'training accuracy', "valid loss", "valid accuracy", "LR","Valid class 1 F1"])
        list = [epoch + 1, loss_train, acc_train, loss_valid, acc_valid, optimizer.param_groups[0]["lr"], F1[1]]
        data = pd.DataFrame([list])
        data.to_csv(log_dir + "/log.csv", mode='a', header=False, index=False)

        scheduler.step()  # 更新学习率

        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        show_confMat(mat_train, labels, "train", log_dir, verbose=epoch == MAX_EPOCH-1)
        show_confMat(mat_valid, labels, "valid", log_dir, verbose=epoch == MAX_EPOCH-1)

        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > (MAX_EPOCH/3) and best_acc < F1[1]:
            best_acc =  F1[1]
            best_epoch = epoch

            checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best Class 1 F1: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_acc, best_epoch))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)


