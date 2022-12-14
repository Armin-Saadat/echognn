import logging
import torch
from src.core import models
from copy import deepcopy


MODELS = {'video_encoder': models.VideoEncoder,
          'attention_encoder': models.AttentionEncoder,
          'attention_encoder2': models.AttentionEncoder,
          'graph_regressor': models.GraphRegressor,
          'graph_regressor2': models.GraphRegressor}


def build(config: dict,
          logger: logging.Logger,
          device: torch.device) -> dict:

    """
    Builds the models dict

    :param config: dict, model config dict
    :param logger: logging.Logger, custom logger
    :param device: torch.device, device to move the models to
    :return: dictionary containing all the submodules (PyTorch models)
    """

    config = deepcopy(config)
    try:
        _ = config.pop('checkpoint_path')
        _ = config.pop('pretrained_path')
    except KeyError:
        pass

    # Create the models
    model = {}
    for model_key in config.keys():
        if model_key == "attention_encoder2":
            model[model_key] = MODELS[model_key](config=deepcopy(config[model_key]).update({'num_frames': config["attention_encoder"]['num_frames']//config["graph_regressor"]['agg_num']})).to(device)
        if model_key == "graph_regressor2":
            model[model_key] = MODELS[model_key](config=deepcopy(config[model_key]).update({'num_frames': config["graph_regressor"]['num_frames']//config["graph_regressor"]['agg_num']})).to(device)
        model[model_key] = MODELS[model_key](config=config[model_key]).to(device)

    logger.info_important('Model is built.')

    return model
