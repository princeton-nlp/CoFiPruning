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


def calculate_model_size_from_z(full_model_size, **kwargs):
    hidden_size = 768
    num_attention_heads = 12
    num_layers = 12
    dims_per_head = hidden_size // num_attention_heads

    parameters_per_dim = {"qk": (hidden_size + 1) * 2,
                          "vo": (hidden_size + 1) * 2,
                          "head": (hidden_size * hidden_size + hidden_size) * 4 // num_attention_heads,
                          "head_layer": (hidden_size * hidden_size + hidden_size) * 4,
                          "mlp": hidden_size * hidden_size * 4 * 2 + hidden_size * 4 + hidden_size,
                          "intermediate": hidden_size * 2 + 1,
                          "head_weight": (hidden_size * hidden_size) * 4 // num_attention_heads,
                          "head_bias": hidden_size * 3 // num_attention_heads,
                          "head_hidden_bias": 1,
                          "int_weight": 2,
                          "int_bias": 1,
                          "int_hidden_bias": 1}
    prunable_params = parameters_per_dim["head_layer"] * num_layers + parameters_per_dim["mlp"] * num_layers
    pruned_dims = {}
    pruned_params = {}

    head_layer_z, head_z, qk_z, vo_z, mlp_z, intermediate_z, hidden_z = process_z(kwargs)

    pruned_dims["head_layer"] = (head_layer_z == 0).sum().item()
    pruned_dims["head"] = (head_layer_z * head_z == 0).sum().item()
    if qk_z is not None:
        pruned_dims["qk"] = (head_layer_z * head_z * qk_z == 0).sum().item()
        pruned_dims["vo"] = (head_layer_z * head_z * vo_z == 0).sum().item()
    pruned_dims["intermediate"] = (mlp_z * intermediate_z == 0).sum().item()
    pruned_dims["mlp"] = (mlp_z == 0).sum().item()
    if hidden_z is not None:
        pruned_dims["hidden"] = (hidden_z == 0).sum().item()
        head_outer = np.outer((head_layer_z * head_z).reshape(-1), hidden_z)

        pruned_params["head"] = (head_outer == 0).sum().item() * (parameters_per_dim["head_weight"]+parameters_per_dim["head_bias"]) // hidden_size + parameters_per_dim["head_hidden_bias"] * pruned_dims["hidden"]
        if "qk" in pruned_dims:
            pass
        int_outer = np.outer((mlp_z * intermediate_z).reshape(-1), hidden_z)
        pruned_params["intermediate"] = (int_outer == 0).sum().item() * (parameters_per_dim["int_weight"]) + pruned_dims["intermediate"] + pruned_dims["hidden"]
        pruned_params["emb"] = pruned_dims["hidden"] * (30522 + 512 + 2)
    else:
        for key in pruned_dims:
            pruned_params[key] = pruned_dims[key] * parameters_per_dim[key]

    if "qk" in pruned_params:
        total_pruned_params = pruned_params["qk"] + pruned_params["vo"] + pruned_params["intermediate"]
    else:
        total_pruned_params = pruned_params["intermediate"] + pruned_params["head"]

    total_pruned_params_ex_embs = total_pruned_params
    if "emb" in pruned_params:
        total_pruned_params += pruned_params["emb"]
    # if "layernorm" in pruned_params:
    #     total_pruned_params += pruned_params["layernorm"]
    model_size_after_pruning = full_model_size - total_pruned_params

    features = get_features(None, head_layer_z=head_layer_z, head_z=head_z, qk_z=qk_z, vo_z=vo_z, mlp_z=mlp_z, intermediate_z=intermediate_z, hidden_z=hidden_z)
    return pruned_params, pruned_dims, total_pruned_params, model_size_after_pruning, total_pruned_params_ex_embs / prunable_params

