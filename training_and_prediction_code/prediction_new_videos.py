
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
from mouse_data_prediction_new_video import *
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


    dataset_params = {
        'batch_size': 32,
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
    positive_frame_thres = 23
    model = nn.Sequential(
        crnn.CNNEncoder(**cnn_encoder_params),
        crnn.RNNDecoder(**rnn_decoder_params)
    )

    # change model path here, same with the model result path
    path_checkpoint = "/media/yhs/8608ebc4-123e-4bc8-803b-d3d75187dbd2/Automatic_itch_project/results/04-28_15-05Window45, no interval no discard, 1-32 train/"
    # change the video to predict here, must be an int
    predict_video = 2005


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
    sliding_size = 1,
    positive_frame_thres = positive_frame_thres)
    predict_loader = DataLoader(dataset=predict_data, batch_size=1,
                              num_workers=dataset_params['num_workers'])
    labels = [0,1]
    num_labels = 2
    criterion = nn.CrossEntropyLoss()
    #label_true = []
    label_result = []
    if True:
        conf_mat = np.zeros((num_labels, num_labels))
        loss_sigma = []

        with torch.no_grad():
            for i, data in enumerate(predict_loader):

                inputs= data
                inputs= inputs.to(device)

                outputs = model(inputs)

                # 统计预测信息
                _, predicted = torch.max(outputs.data, 1)
                label1 = predicted.cpu().detach().numpy()[0]
                label_result.append(label1)
                # 统计混淆矩阵
                #for j in range(len(labels)):
                    #cate_i = labels[j].cpu().numpy()
                    #pre_i = predicted[j].cpu().numpy()
                    #conf_mat[cate_i, pre_i] += 1.

                # 统计loss
                #loss_sigma.append(loss.item())

        #acc_avg = conf_mat.trace() / conf_mat.sum()
    
    print(len(label_result))

    
    #confusion_mat = conf_mat
    #cls_num = 2
    #classes = [0,1]
    #F1 = [0,0]
    #recall = [0,0]
    #precision = [0,0]
    #confusion_mat_N = confusion_mat.copy()
    #print("Result of frame prediction")
    #for i in range(len(classes)):
        #confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
    #for i in range(cls_num):
        #recall[i] = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :]))
        #precision[i] = confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))
        #F1[i] = np.sqrt(recall[i] * precision[i])
       # print('valid class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%} F1:{:.4f}'.format(
            #classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
            #confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
            #confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i])),
            #np.sqrt(recall[i] * precision[i])))


    #Save label_frame_true and lable_frame_predict to csv file
    import pandas as pd
    prediction_results = pd.DataFrame({'label_frame_predict':label_result})
    prediction_results.to_csv('prediction_results_V%d.csv' % predict_video)


    end = timer()
    print("time used:",end - start)


