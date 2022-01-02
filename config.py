config = {  # 路径参数
    'data_path': './data/人物关系表.xlsx',
    'vocab_path': './data/vocab.txt',
    'rel-schema_path': './data/rel_dict.json',
    'model_path': './output/model/',
    'bert_model_path': './bert-base-chinese/',
    # 训练参数
    'max_length': 200,
    'hidden_size': 256,
    'epoch': 30,
    'batch_size': 64,
    'optimizer': 'adam',
    'learning_rate': 1e-3,
    'rel_loss_ratio': 0.006,
    'model_type': 'lstm'

}
