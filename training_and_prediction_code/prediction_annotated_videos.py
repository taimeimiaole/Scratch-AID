
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
from mouse_data import *
import crnn
import numpy as np
from common_tools import *
import pandas as pd
from typing import List
import torch


def label_lis(dat_frame):
    dat_frame = dat_frame.reset_index()
    label_lis = []
    for i in range(len(dat_frame)):
        label_lis=list(range(int(dat_frame['Start_frame'][i]),int(dat_frame['End_frame'][i])+1))+label_lis
    label_lis.sort(reverse=False)
    return label_lis

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ## Copy model from the train file
    dataset_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'frame_step': 3
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

    window_size = 23
    sliding_size = 10
    positive_frame_thres = 12
    frame_interval = 3
    model = nn.Sequential(
        crnn.CNNEncoder(**cnn_encoder_params),
        crnn.RNNDecoder(**rnn_decoder_params)
    )

    # change model path here, same with the model result path
    path_checkpoint = "/media/yhs/8608ebc4-123e-4bc8-803b-d3d75187dbd2/Automatic_itch_project/results/04-09_21-44inter3window23,slide12,1-32 train, 37-40test,resize+randomcrop/"
    # change the video to predict here, must be an int
    predict_video = 13


    from timeit import default_timer as timer
    

    
    start = timer()



    path_checkpoint  = path_checkpoint + "checkpoint_best.pkl"
    check = torch.load(path_checkpoint)
    print("video", predict_video)
    model.load_state_dict(check["model_state_dict"])
    model.eval()
    model.to(device)
    train_dir = "/media/yhs/8608ebc4-123e-4bc8-803b-d3d75187dbd2/Automatic_itch_project/"

    
    # predict frames
    predict_data = mouse_Dataset(data_dir=train_dir, window_size=window_size, validation=True,predict_new=True, frame_interval  = dataset_params['frame_step'],
                               predict_video = predict_video, remove_bound = False,
    sliding_size = sliding_size,
    positive_frame_thres = positive_frame_thres)
    predict_loader = DataLoader(dataset=predict_data, batch_size=1,
                              num_workers=dataset_params['num_workers'])
    labels = [0,1]
    num_labels = 2
    criterion = nn.CrossEntropyLoss()
    label_result = []
    if True:
        conf_mat = np.zeros((num_labels, num_labels))
        loss_sigma = []

        with torch.no_grad():
            for i, data in enumerate(predict_loader):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device, dtype = torch.int64)
                label1 = labels.cpu().detach().numpy()[0]
                label_result.append(label1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 统计预测信息
                _, predicted = torch.max(outputs.data, 1)

                # 统计混淆矩阵
                for j in range(len(labels)):
                    cate_i = labels[j].cpu().numpy()
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i, pre_i] += 1.

                # 统计loss
                loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()
    
    print(len(label_result))

    scratching_train_annotation_path = "/media/yhs/8608ebc4-123e-4bc8-803b-d3d75187dbd2/Automatic_itch_project/annotation/scratching_train.tsv"
    all_training_frame_pool = []
    validlist = [predict_video]
    video_frame_number_dic = {"V" + str(ii): 36000 for ii in validlist}
    for i in validlist:
        for k in range(1, video_frame_number_dic['V%d' % i]):
            all_training_frame_pool.append('V%d_gray_resize_img' % i + str(k))
    all_training_frame_pool_label = {"V" + str(ii): {} for ii in validlist}
    scratching_train_annotation = pd.read_table(scratching_train_annotation_path)
    filenameGrouped = scratching_train_annotation.groupby('Video_number')
    label_dic = {}
    for name, group in filenameGrouped:
        label_lis_t = label_lis(group)
        label_dic[name] = label_lis_t
    for i in all_training_frame_pool:
        for k in label_dic:
            if re.search(r'(.*)_gray_resize_img', i).group(1) == k:
                frame_index = int(re.search(r'img(\d*)', i).group(1))
                if frame_index in label_dic[k]:
                    all_training_frame_pool_label[k][i] = 1
                else:
                    all_training_frame_pool_label[k][i] = 0
    all_training_frame_pool_label = all_training_frame_pool_label["V" + str(predict_video)]
    import pandas as pd
    #1
    label_frame_true = list(all_training_frame_pool_label.values())

    frame_windows = [[1]]*sliding_size + [[1, 2]]*sliding_size + [[1, 2,3]]*sliding_size + [[1, 2,3,4]]*sliding_size + [[1, 2,3,4, 5]]*sliding_size + [[1, 2,3,4,5 ,6]]*sliding_size + [[i // sliding_size, i // sliding_size + 1, i // sliding_size -1, i // sliding_size -2, i // sliding_size-3, i // sliding_size-4, i // sliding_size-5]   for i in range(sliding_size * 6 + 1,len(label_frame_true)+1)]
    # Frame prediction
    #2
    label_frame_predict = [0] * len(label_frame_true)
    for idx in range(len(label_frame_true)):
        frame = frame_windows[idx]
        predicts = []
        for i in frame:
            if i <= len(label_result):
                predicts.append(label_result[i-1])
        if np.mean(predicts) > 3/7:
            label_frame_predict[idx] = 1
        else:
            label_frame_predict[idx] = 0
    
    

    conf_mat = np.zeros((2, 2))
    for j in range(len(label_frame_true)):
        pre_i = label_frame_predict[j]
        cate_i = label_frame_true[j]
        conf_mat[cate_i, pre_i] += 1.

    confusion_mat = conf_mat
    cls_num = 2
    classes = [0,1]
    F1 = [0,0]
    recall = [0,0]
    precision = [0,0]
    confusion_mat_N = confusion_mat.copy()
    print("Result of frame prediction")
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
    for i in range(cls_num):
        recall[i] = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :]))
        precision[i] = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))
        F1[i] = np.sqrt(recall[i] * precision[i])
        print('valid class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%} F1:{:.4f}'.format(
            classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
            confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
            confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i])),
            np.sqrt(recall[i] * precision[i])))


    # Predict samples
        predict_data = mouse_Dataset(data_dir=train_dir, window_size=window_size, validation=True,predict_new=True, frame_interval  = dataset_params['frame_step'],
                               predict_video = predict_video, remove_bound = True,
    sliding_size = sliding_size,
    positive_frame_thres = positive_frame_thres)
    predict_loader = DataLoader(dataset=predict_data, batch_size=1,
                              num_workers=dataset_params['num_workers'])
    labels = [0,1]
    num_labels = 2
    criterion = nn.CrossEntropyLoss()
    label_result = []
    if True:
        conf_mat = np.zeros((num_labels, num_labels))
        loss_sigma = []

        with torch.no_grad():
            for i, data in enumerate(predict_loader):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device, dtype = torch.int64)
                label1 = labels.cpu().detach().numpy()[0]
                label_result.append(label1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 统计预测信息
                _, predicted = torch.max(outputs.data, 1)

                # 统计混淆矩阵
                for j in range(len(labels)):
                    cate_i = labels[j].cpu().numpy()
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i, pre_i] += 1.

                # 统计loss
                loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()
    
    print(len(label_result))
    loss_valid, acc_valid, mat_valid = np.mean(loss_sigma), acc_avg, conf_mat

    confusion_mat = mat_valid
    cls_num = 2
    classes = [0,1]
    F1 = [0,0]
    recall = [0,0]
    precision = [0,0]
    print("Result of window prediction")
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
    for i in range(cls_num):
        recall[i] = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :]))
        precision[i] = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))
        F1[i] = np.sqrt(recall[i] * precision[i])
        print('valid class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%} F1:{:.4f}'.format(
            classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
            confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
            confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i])),
            np.sqrt(recall[i] * precision[i])))


    end = timer()
    print("time used:",end - start)


