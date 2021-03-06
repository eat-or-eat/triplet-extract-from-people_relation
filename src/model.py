import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
建立网络模型结构
特殊：说明由于bert的cls token已经pooler过了，所以不用再转换(seq,hid)张量再池化直接传入分类层就好
"""


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        max_length = config['max_length']
        model_type = config['model_type']
        self.use_bert = False
        if model_type == 'lstm':
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=1)
            self.bio_classifier = nn.Linear(hidden_size * 2, config['bio_size'])
            self.rel_classifier = nn.Linear(hidden_size * 2, config['rel_size'])
        elif model_type == 'bert':
            self.use_bert = True
            self.layer = Bert(config)
            self.bio_classifier = nn.Linear(768, config['bio_size'])
            self.rel_classifier = nn.Linear(768, config['rel_size'])
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.rel_loss_ratio = config['rel_loss_ratio']
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, rel_target=None, bio_target=None):
        if self.use_bert:
            last_hidden_state, pooler_output = self.layer(x)  # input shape:(batch_size, sen_len)
            # 序列标注
            bio_predict = self.bio_classifier(last_hidden_state)
            # 文本分类
            rel_predict = self.rel_classifier(pooler_output)
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x, _ = self.layer(x)  # input shape:(batch_size, sen_len, hid_size)
            # 序列标注
            bio_predict = self.bio_classifier(x)
            # 文本分类
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
            x = self.pooling_layer(x.transpose(1, 2)).squeeze()
            rel_predict = self.rel_classifier(x)

        # multi-task训练
        if bio_target is not None:
            bio_loss = self.loss(bio_predict.view(-1, bio_predict.shape[-1]), bio_target.view(-1))
            attribute_loss = self.loss(rel_predict.view(x.shape[0], -1), rel_target.view(-1))
            return bio_loss + attribute_loss * self.rel_loss_ratio
        else:
            return rel_predict, bio_predict


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(config['bert_model_path'])

    def forward(self, x, mid=False):
        x = self.bert(x)
        # bert_dict:(last_hidden_state, pooler_output)
        # last_hidden_state:(bs, seq_len, hid_size)  常用于获取动态词向量
        # pooler_output:(bs, hid_size)  常用于获取句向量
        return x['last_hidden_state'], x['pooler_output']


def choose_optimizer(config, model):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from config import config

    config['class_num'] = 20
    config['vocab_size'] = 20
    config['max_length'] = 5

    model = Model(config)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    label = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    print(model(x, label))
