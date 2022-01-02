import torch
import json
import pandas as pd
from torch.utils.data import DataLoader

"""
数据格式：
(头实体，关系，尾实体，句子)->
# 一条数据举例
(
[seq_id1, seq_id12, ..., seq_idn, pad],
[rel_id],
[seq_label1, seq_label2, ..., seq_labeln, pad]
)
"""

def load_vocab(vocab_path):
    vocab_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            token = line.strip()  # 去除行尾换行符
            vocab_dict[token] = index + 1  # 还有padding的位置，让出0来
    return vocab_dict


def load_schema(path):
    with open(path, encoding='utf8') as f:
        return json.load(f)


class DataGenerator:
    '''
    数据生成类:每行三元组和文本数据->longtensor(字索引，关系索引，序列标注索引)
    '''

    def __init__(self, config):
        self.data = []
        self.sentences = []
        self.config = config
        self.vocab = load_vocab(config['vocab_path'])
        self.bio_schema = {'B_object1': 0,
                           'I_object1': 1,
                           'B_object2': 2,
                           'I_object2': 3,
                           'O': 4}
        self.rel_schema = json.load(open(config['rel-schema_path'], encoding='utf8'))
        self.config['vocab_size'] = len(self.vocab)
        self.config['bio_size'] = len(self.bio_schema)
        self.config['rel_size'] = len(self.rel_schema)
        self.max_length = config['max_length']
        self.load()

    def load(self):
        self.exceed_max_length = 0
        df = pd.read_excel(self.config['data_path'])
        self.list_data = df.values
        for (o1, o2, rel, sentence) in self.list_data:
            if o1 not in sentence or o2 not in sentence:
                continue  # 有的数据名字有错别字
            if rel not in self.rel_schema:
                rel = 'unknown'
            input_ids, rel_id, bio_ids = self.process_sentence(o1, o2, rel, sentence)
            self.data.append([torch.LongTensor(input_ids),
                              torch.LongTensor([rel_id]),
                              torch.LongTensor(bio_ids)])
        return

    def process_sentence(self, o1, o2, rel, sentence):
        rel_id = self.rel_schema[rel]
        input_ids = self.encode_sentence(sentence)
        bio_ids = self.generate_bios(o1, o2, sentence)
        input_ids = self.padding(input_ids)
        bio_ids = self.padding(bio_ids, -100)
        return input_ids, rel_id, bio_ids

    def encode_sentence(self, sentence, padding=False):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config['max_length']]
        input_id += [pad_token] * (self.config['max_length'] - len(input_id))
        return input_id

    def generate_bios(self, o1, o2, sentence):
        o1_s = sentence.index(o1)
        o2_s = sentence.index(o2)
        labels = [self.bio_schema['O']] * len(sentence)
        # 标记实体1
        labels[o1_s] = self.bio_schema['B_object1']
        for index in range(o1_s + 1, o1_s + len(o1)):
            labels[index] = self.bio_schema['I_object1']
        # 标记实体2
        labels[o2_s] = self.bio_schema['B_object2']
        for index in range(o2_s + 1, o2_s + len(o2)):
            labels[index] = self.bio_schema['I_object2']
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_dataset(config, shuffle=True):
    dg = DataGenerator(config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from config import config

    config['data_path'] = '../data/人物关系表.xlsx'
    config['vocab_path'] = '../data/vocab.txt'
    config['rel-schema_path'] = '../data/rel_dict.json'

    dg = DataGenerator(config)
    dl = load_dataset(config)
