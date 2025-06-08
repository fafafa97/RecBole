from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

if __name__ == '__main__':

    # 配置初始化
    config = Config(model='LightGCN', dataset='ml-100k', config_file_list=["recbole/properties/model/LightGCN.yaml"])

    # 初始随机种子
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # 数据集过滤
    dataset = create_dataset(config)
    logger.info(dataset)

    # 数据集拆分
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 模型初始化
    model = BPR(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # 训练器初始化
    trainer = Trainer(config, model)

    # 模型训练
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # 模型评估
    test_result = trainer.evaluate(test_data)
    print(test_result)