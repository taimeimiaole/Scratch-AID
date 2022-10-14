import Resnet2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from d2l import torch as d2l
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class ModelBase(nn.Module):
    """ Base models for all models.

    """

    def __init__(self, args):
        """ Initialize the hyperparameters of model.

        Parameters
        ----------
        args: arguments for initializing the model.

        """

        super(ModelBase, self).__init__()

        self.epsilon = 1e-4
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.attention_size = args.attention_size
        self.attention_head = args.attention_head
        self.norm_shape = [args.embedding_size]
        self.ffn_num_input = args.ffn_num_input
        self.ffn_num_hiddens = args.ffn_num_hiddens

        self.dropout_rate = args.dropout_rate

        self.num_layers = args.num_layers
        self.num_classes = args.num_classes

class GIT2(ModelBase):

    def __init__(self, args, **kwargs
                 ) -> None:

        super(GIT2, self).__init__(args, **kwargs)

        self.pos_encoding = PositionalEncoding(self.embedding_size, self.dropout_rate)
        self.blks = nn.Sequential()
        for i in range(self.num_layers):
            self.blks.add_module("block" + str(i),
                                 d2l.EncoderBlock(self.embedding_size, self.embedding_size, self.embedding_size,
                                              self.embedding_size,
                                              self.norm_shape, self.ffn_num_input, self.ffn_num_hiddens,
                                              self.attention_head, self.dropout_rate, use_bias=False))

        self.layer_dropout_1 = nn.Dropout(p=self.dropout_rate)

        self.layer_w_1 = nn.Linear(
            in_features=self.embedding_size,
            out_features=self.hidden_size,
            bias=True)

        self.layer_dropout_2 = nn.Dropout(p=self.dropout_rate)

        self.layer_w_2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_classes,
            bias=True)


    def forward(self, X):

        X = self.pos_encoding(X * math.sqrt(self.embedding_size))
        self.attention_weights = [0] * len(self.blks)

        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens=None)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights

        self.attention_weights = torch.stack(self.attention_weights)


        emb_sga = torch.mean(X, 1)
        #emb_sga = X[:,0,:]
        # MLP decoder
        emb_tmr_relu = self.layer_dropout_1(F.relu(emb_sga))
        hid_tmr = self.layer_w_1(emb_tmr_relu)
        hid_tmr_relu = self.layer_dropout_2(F.relu(hid_tmr))
        preds = self.layer_w_2(hid_tmr_relu)

        return preds



class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim=256, drop_prob=0.3, bn_momentum=0.01):
        '''
        使用pytorch提供的预训练模型作为encoder
        '''
        super(CNNEncoder, self).__init__()

        self.cnn_out_dim = cnn_out_dim
        self.drop_prob = drop_prob
        self.bn_momentum = bn_momentum

        # 使用resnet预训练模型来提取特征，去掉最后一层分类器
        pretrained_cnn = Resnet2D.resnet18(pretrained=False)
        cnn_layers = list(pretrained_cnn.children())[:-1]

        # 把resnet的最后一层fc层去掉，用来提取特征
        self.cnn = nn.Sequential(*cnn_layers)
        # 将特征embed成cnn_out_dim维向量
        self.fc = nn.Sequential(
            *[
                self._build_fc(pretrained_cnn.fc.in_features, 256, True),
                nn.ReLU(),
                nn.Dropout(p=self.drop_prob),
                self._build_fc(256, self.cnn_out_dim, False)
            ]
        )

    def _build_fc(self, in_features, out_features, with_bn=True):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features, momentum=self.bn_momentum)
        ) if with_bn else nn.Linear(in_features, out_features)

    def forward(self, x_3d):
        '''
        输入的是T帧图像，shape = (batch_size, t, h, w, 3)
        '''
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            # 使用cnn提取特征
            # 为什么要用到no_grad()？
            # -- 因为我们使用的预训练模型，防止后续的层训练时反向传播而影响前面的层
            #with torch.no_grad():
            x = self.cnn(x_3d[:, t, :, :, :])
            x = torch.flatten(x, start_dim=1)

            # 处理fc层
            x = self.fc(x)

            cnn_embedding_out.append(x)

        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        return cnn_embedding_out

class RNNDecoder(nn.Module):
    def __init__(self, use_gru=True, cnn_out_dim=256, rnn_hidden_layers=3, rnn_hidden_nodes=256,
            num_classes=2, drop_prob=0.3, bidirectional = True):
        super(RNNDecoder, self).__init__()

        self.rnn_input_features = cnn_out_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes
        self.bidirectional  = bidirectional
        self.drop_prob = drop_prob
        self.num_classes = num_classes # 这里调整分类数目

        # rnn配置参数
        rnn_params = {
            'input_size': self.rnn_input_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True,
            'bidirectional':self.bidirectional
        }

        # 使用lstm或者gru作为rnn层
        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        # rnn层输出到线性分类器
        if self.bidirectional:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn_hidden_nodes*2, 256),
                nn.ReLU(),
                nn.Dropout(self.drop_prob),
                nn.Linear(256, self.num_classes)
            )

        else:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn_hidden_nodes*2, 256),
                nn.ReLU(),
                nn.Dropout(self.drop_prob),
                nn.Linear(256, self.num_classes)
            )

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x_rnn, None)
        # 注意，前面定义rnn模块时，batch_first=True保证了以下结构：
        # rnn_out shape: (batch, timestep, output_size)
        # h_n and h_c shape: (n_layers, batch, hidden_size)

        x = self.fc(rnn_out[:, -1, :]) # 只抽取最后一层做输出 Todo

        return x


class RNNDecoder_byframe(nn.Module):
    def __init__(self, use_gru=True, cnn_out_dim=256, rnn_hidden_layers=3, rnn_hidden_nodes=256,
            num_classes=2, drop_prob=0.3, bidirectional = True):
        super(RNNDecoder, self).__init__()

        self.rnn_input_features = cnn_out_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes
        self.bidirectional  = bidirectional
        self.drop_prob = drop_prob
        self.num_classes = num_classes # 这里调整分类数目

        # rnn配置参数
        rnn_params = {
            'input_size': self.rnn_input_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True,
            'bidirectional':self.bidirectional
        }

        # 使用lstm或者gru作为rnn层
        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        # rnn层输出到线性分类器
        if self.bidirectional:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn_hidden_nodes*2, 256),

                nn.ReLU(),
                nn.Dropout(self.drop_prob),
                nn.Linear(256, self.num_classes)
            )

        else:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn_hidden_nodes*2, 256),
                nn.ReLU(),
                nn.Dropout(self.drop_prob),
                nn.Linear(256, self.num_classes)
            )

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x_rnn, None)
        # 注意，前面定义rnn模块时，batch_first=True保证了以下结构：
        # rnn_out shape: (batch, timestep, output_size)
        # h_n and h_c shape: (n_layers, batch, hidden_size)

        x = self.fc(rnn_out[:, -1, :]) # 只抽取最后一层做输出

        return x

if __name__ == "__main__":

    from torchsummary import summary

    cnn_encoder_params = {
        'cnn_out_dim': 128,
        'drop_prob': 0.15,
        'bn_momentum': 0.01
    }

    rnn_decoder_params = {
        'use_gru': True,
        'cnn_out_dim': 128,
        'rnn_hidden_layers': 2,
        'rnn_hidden_nodes': 256,
        'num_classes': 2,
        'drop_prob': 0.15,
        'bidirectional': True
    }

    model = nn.Sequential(
        CNNEncoder(**cnn_encoder_params),
        RNNDecoder(**rnn_decoder_params)
        # crnn.GIT2(args)
    )
    summary(model, input_size=(1,1,256,256), batch_size=16,device="cpu")
