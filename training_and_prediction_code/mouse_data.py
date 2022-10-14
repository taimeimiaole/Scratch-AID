
import re
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np

def label_lis(dat_frame):
    dat_frame = dat_frame.reset_index()
    label_lis = []
    for i in range(len(dat_frame)):
        label_lis=list(range(int(dat_frame['Start_frame'][i]),int(dat_frame['End_frame'][i])+1))+label_lis
    label_lis.sort(reverse=False)
    return label_lis






class mouse_Dataset(Dataset):

    def __init__(self, data_dir, window_size = 23, sliding_size = 12,positive_frame_thres = 12,
                 frame_interval = 3, validation = False, predict_new = False, remove_bound = True, predict_video = None,
                 train_video = [1,2,3,4,5,6,7,8,9,10,11,12], test_video = [33,34,35,36]):

        self.data_dir = data_dir  # self.data_dir ="C:/Users/11764/Desktop/automatic_itch"


        self.window_size = window_size
        self.sliding_size = sliding_size
        self.positive_frame_thres = positive_frame_thres
        self.frame_pool_path = self.data_dir + "/scratching_video_frames_gray"
        self.validation = validation
        self.frame_interval = frame_interval # step of the frames
        self.predict_new = predict_new
        self.remove_bound = remove_bound
        self.predict_video = predict_video #(must be int)
        self.train_video = train_video #(must be int)
        self.test_video = test_video #(must be int)
        self._get_img_info()
        

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):  # index is number
        assert index < len(self), 'index range error'

        window = self.window_list[index]
        label = self.label_list[index]
        imgs = []
        for ID in window:
            tem_frame_folder = re.search(r'.*gray_resize', ID).group()
            img_index = re.search(r'img(\d*)', ID).group() + ".jpg"

            path = self.frame_pool_path + "/" + tem_frame_folder + "/" + img_index
            img = Image.open(path).convert("L")
            imgs.append(img)


        # TODO transforms here:
        # Random crop
        if self.validation == False:

            resize = random.random()   # prob = 0.8, we will resize the picture (smaller mouse)
            if resize < 0.2:
                random_crop_size = 256
            else:
                random_crop_size = np.random.randint(256, 300 + 1)

            i, j, h, w = transforms.RandomCrop.get_params(
                imgs[0], output_size=(random_crop_size, random_crop_size))
            vertical = random.random()
            horizon = random.random()

            result = []
            for img in imgs:

                img = TF.crop(img, i, j, h, w)

                preprocess = transforms.Compose([
                    transforms.Resize(256)
                ])
                img = preprocess(img)
                # Random horizontal flipping
                if horizon > 0.5:
                    img = TF.hflip(img)
                # Random vertical flipping
                if vertical > 0.5:
                    img = TF.vflip(img)

                result.append(TF.to_tensor(img))
            # result = []
            # preprocess = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.ToTensor()
            # ])
            # for img in imgs:
            #     result.append(preprocess(img))

        else:
            result = []
            preprocess = transforms.Compose([
                transforms.CenterCrop(288),
                transforms.Resize(256),
                transforms.ToTensor()
            ])
            for img in imgs:
                result.append(preprocess(img))

        result = torch.stack(result)  # torch.Size([30, 1, 256, 256])
        return (result, label)

    def _get_img_info(self):
        scratching_train_annotation_path = self.data_dir + "/annotation/scratching_train.tsv"

        # TODO, change the video info here.
        ## video_frame number dic

        if self.validation == True:
            if self.predict_new == True:
                all_training_frame_pool = []
                validlist = [37,38,39,40]
                if self.predict_video != None:
                    validlist = [self.predict_video]
                video_frame_number_dic = {"V" + str(ii): 36000 for ii in validlist}
                for i in validlist:
                    for k in range(1, video_frame_number_dic['V%d' % i]):
                        all_training_frame_pool.append('V%d_gray_resize_img' % i + str(k))

                all_training_frame_pool_label = {"V" + str(ii): {} for ii in validlist}
            else:
                ## generate frame index
                all_training_frame_pool = []
                validlist = self.test_video
                video_frame_number_dic = {"V" + str(ii): 36000 for ii in validlist}
                for i in validlist:
                    for k in range(1, video_frame_number_dic['V%d' % i]):
                        all_training_frame_pool.append('V%d_gray_resize_img' % i + str(k))

                all_training_frame_pool_label = {"V" + str(ii): {} for ii in validlist}

        else:  # Train
            trainlist =  self.train_video
            video_frame_number_dic = {"V" + str(ii): 36000 for ii in trainlist}
            ## generate frame index
            all_training_frame_pool = []
            for i in trainlist:
                for k in range(1, video_frame_number_dic['V%d' % i]):
                    all_training_frame_pool.append('V%d_gray_resize_img' % i + str(k))

            all_training_frame_pool_label = {"V" + str(ii): {} for ii in trainlist}




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

        self.all_training_frame_pool_label = all_training_frame_pool_label

        ## generate frame segment data and labels.
        window_list = []
        label_list = []
        for video in all_training_frame_pool_label.keys():
            ID_list = list(all_training_frame_pool_label[video].keys())
            for index in range(0,len(ID_list),self.sliding_size):
                if len(ID_list) - index > self.window_size*self.frame_interval:
                    ID1 = ID_list[index:index+self.window_size*self.frame_interval:self.frame_interval]  ## length self.window_size
                else:
                    ID1 = ID_list[-self.window_size*self.frame_interval::self.frame_interval]

                labels = [all_training_frame_pool_label[video][ID]  for ID in ID1]
                if sum(labels) == 0:
                    label_list.append(0)
                    window_list.append(ID1)
                elif sum(labels)< self.positive_frame_thres:
                    if self.remove_bound == True:
                        pass
                    if self.remove_bound == False:
                        if sum(labels) < self.positive_frame_thres:
                            label_list.append(0)
                            window_list.append(ID1)
                        else:
                            label_list.append(1)
                            window_list.append(ID1)
                elif sum(labels)>= self.positive_frame_thres:
                    label_list.append(1)
                    window_list.append(ID1)

                # remove frame number < slide ones.



        self.nSamples = len(window_list)
        self.window_list = window_list
        self.label_list = label_list


