import torch
import re
import numpy as np
from src.loader import load_dataset


class EvalData:
    def __init__(self, model, config, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.test_data = load_dataset(config, shuffle=False)
        self.list_data = self.test_data.dataset.list_data
        self.rel_schema = self.test_data.dataset.rel_schema
        self.index_to_label = dict((y, x) for x, y in self.rel_schema.items())

    def eval(self, epoch):
        self.logger.info('测试第%d轮模型效果:' % epoch)
        self.pre_dict = {"o1_acc": 0,
                         "rel_acc": 0,
                         "o2_acc": 0,
                         "full_match_acc": 0}
        self.model.eval()
        for index, batch_data in enumerate(self.test_data):
            text_data = self.list_data[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id = batch_data[0]
            with torch.no_grad():
                rel_pred, bio_pred = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.get_result(rel_pred, bio_pred, text_data)
        self.show_result()
        return

    def get_result(self,rel_pred, bio_pred, text_data):
        rel_pred = torch.argmax(rel_pred, dim=-1)
        bio_pred = torch.argmax(bio_pred, dim=-1)
        for rel_pred, bio_pred, info in zip(rel_pred, bio_pred, text_data):
            o1, o2, rel, sentence = info
            bio_pred = bio_pred.cpu().detach().tolist()
            pred_o1, pred_o2 = self.decode(bio_pred, sentence)
            pred_rel = self.index_to_label[int(rel_pred)]
            self.pre_dict["o1_acc"] += int(pred_o1 == o1)
            self.pre_dict["rel_acc"] += int(pred_rel == rel)
            self.pre_dict["o2_acc"] += int(pred_o2 == o2)
            if pred_o1 == o1 and pred_rel == rel and pred_o2 == o2:
                self.pre_dict["full_match_acc"] += 1

    def decode(self, pred_label, context):
        pred_label = "".join([str(i) for i in pred_label])
        pred_obj = self.seek_pattern("01*", pred_label, context)
        pred_value = self.seek_pattern("23*", pred_label, context)
        return pred_obj, pred_value

    def seek_pattern(self, pattern, pred_label, context):
        pred_obj = re.search(pattern, pred_label)
        if pred_obj:
            s, e = pred_obj.span()
            pred_obj = context[s:e]
        else:
            pred_obj = ""
        return pred_obj

    def show_result(self):
        for key, value in self.pre_dict.items():
            self.logger.info("%s : %s " %(key, value / len(self.list_data)))
        self.logger.info("--------------------")
        return