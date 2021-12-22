import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import config
from src.model import Model


class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.bio_schema = {"B_object1": 0,
                           "I_object1": 1,
                           "B_object2": 2,
                           "I_object2": 3,
                           "O": 4}
        self.rel_schema = json.load(open(config["rel-schema_path"], encoding="utf8"))
        self.index_to_label = dict((y, x) for x, y in self.rel_schema.items())
        self.config["bio_size"] = len(self.bio_schema)
        self.config["rel_size"] = len(self.rel_schema)
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = Model(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def decode(self, attribute_label, bio_label, context):
        pred_attribute = self.index_to_label[int(attribute_label)]
        bio_label = "".join([str(i) for i in bio_label.detach().tolist()])
        pred_obj = self.seek_pattern("01*", bio_label, context)
        pred_value = self.seek_pattern("23*", bio_label, context)
        return pred_obj, pred_attribute, pred_value

    def seek_pattern(self, pattern, pred_label, context):
        pred_obj = re.search(pattern, pred_label)
        if pred_obj:
            s, e = pred_obj.span()
            pred_obj = context[s:e]
        else:
            pred_obj = ""
        return pred_obj

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            rel_pred, bio_pred = self.model(torch.LongTensor([input_id]))
            rel_pred = torch.argmax(rel_pred)
            bio_pred = torch.argmax(bio_pred[0], dim=-1)
        object, attribute, value = self.decode(rel_pred, bio_pred, sentence)
        return object, attribute, value


if __name__ == "__main__":
    sl = SentenceLabel(config, config['model_path'] + "epoch_15.pth")

    sentence = "4、徐天明，男，31岁，他跟陈爱华、金民哲同为大学同学。"
    res = sl.predict(sentence)
    print(res)

    sentence = "张三的朋友是李四"
    res = sl.predict(sentence)
    print(res)
