import os
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
from config import config
from src.model import Model, choose_optimizer
from src.evaluator import EvalData
from src.loader import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    # 创建保存模型的目录
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])
    # 加载训练数据
    train_data = load_dataset(config)
    # 加载模型
    model = Model(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = EvalData(model, config, logger)
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_ids, rel_id, bio_ids = batch_data
            loss = model(input_ids, rel_id, bio_ids)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        pre_dict = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data, pre_dict


if __name__ == "__main__":
    model, train_data, pre_dict = main(config)