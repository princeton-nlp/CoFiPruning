import numpys as np

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
