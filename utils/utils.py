import numpy as np
from collections import defaultdict
from datasets import Dataset

def log_all_parameters(logger, model_args, data_args, training_args, additional_args):
    logger.info("Model Arguments:")
    for arg in vars(model_args):
        logger.info(f"{arg} = {getattr(model_args, arg)}")

    logger.info("Data Arguments:")
    for arg in vars(data_args):
        logger.info(f"{arg} = {getattr(data_args, arg)}")

    logger.info("Training Arguments:")
    for arg in vars(training_args):
        logger.info(f"{arg} = {getattr(training_args, arg)}")

    logger.info("Additional Arguments:")
    for arg in vars(additional_args):
        logger.info(f"{arg} = {getattr(additional_args, arg)}")

def calculate_parameters(module):
    keys = ["embedding", "layer_transformation", "classifier", "pooler"]
    return sum(p.numel() for n, p in module.named_parameters() if not any(key in n for key in keys))

def load_from_tsv(file):
    lines = open(file, "r").readlines()
    data = [line.strip().split("\t") for line in lines[1:]]
    headers = lines[0].strip().split("\t")
    d = defaultdict(list)
    for i, head in enumerate(headers):
        for j, dd in enumerate(data):
            d[head].append(dd[i])

    dataset = Dataset.from_dict(d)
    return dataset