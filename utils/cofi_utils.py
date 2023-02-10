import torch
import os
from transformers import AutoConfig 
from CoFiPruning.utils.utils import calculate_parameters
import numpy as np

def initialize_layer_transformation(model):
    model.layer_transformation.weight.data.copy_(
        torch.eye(len(model.layer_transformation.weight)))
    model.layer_transformation.bias.data.fill_(0)


# load the l0 module
def load_l0_module(model_path):
    l0_module_path = os.path.join(model_path, "l0_module.pt")
    if os.path.exists(l0_module_path):
        return torch.load(l0_module_path, map_location=torch.device('cpu'))
    else:
        return None


def numpify_zs(zs):
    numpified_zs = {} 
    for key in zs:
        if torch.is_tensor(zs[key]):
            numpified_zs[key] = zs[key].detach().cpu().numpy()
    return numpified_zs

def calculate_model_size_with_z(zs, l0_module):
    numpified_zs = numpify_zs(zs)
    hidden_z = numpified_zs["hidden_z"]
    intermediate_z = numpified_zs["intermediate_z"]
    mlp_z = numpified_zs["mlp_z"].reshape(-1, 1)
    head_z = numpified_zs["head_z"]
    head_layer_z = numpified_zs["head_layer_z"].reshape(-1, 1)

    remaining_hidden_dims = hidden_z.sum().item()
    remaining_intermediate_nums = intermediate_z.squeeze().sum(-1).tolist()
    remaining_head_nums = head_z.squeeze().sum(-1).tolist()

    head_nums = np.outer((head_z * head_layer_z).reshape(-1), hidden_z).sum().item()
    intermediate_nums = np.outer((intermediate_z * mlp_z).reshape(-1), hidden_z).sum().item()

    remaining_model_size = head_nums * l0_module.dim_per_head * 4 + intermediate_nums * 2
    pruned_model_size = l0_module.prunable_model_size - remaining_model_size

    results = {}
    results["head_layer"] = head_layer_z.reshape(-1).astype(int).tolist()
    results["mlp"] = mlp_z.reshape(-1).astype(int).tolist()
    results["hidden"] = remaining_hidden_dims
    results["intermediate"] = remaining_intermediate_nums
    results["head"] = remaining_head_nums
    results["pruned_params"] = pruned_model_size
    results["remaining_params"] = remaining_model_size
    results["sparsity"] = pruned_model_size / l0_module.prunable_model_size
    return results
    
# corrected 
# move the actual prunining to modeling files
def prune_model_with_z(zs, model):
    model.prune_modules(zs)
    

def numpify_zs(zs):
    numpified_zs = {} 
    for key in zs:
        if torch.is_tensor(zs[key]):
            numpified_zs[key] = zs[key].detach().cpu().numpy()
    return numpified_zs

def calculate_model_size_with_z(zs, l0_module):
    numpified_zs = numpify_zs(zs)
    hidden_z = numpified_zs["hidden_z"]
    intermediate_z = numpified_zs["intermediate_z"]
    mlp_z = numpified_zs["mlp_z"].reshape(-1, 1)
    head_z = numpified_zs["head_z"]
    head_layer_z = numpified_zs["head_layer_z"].reshape(-1, 1)

    remaining_hidden_dims = (hidden_z > 0).sum().item()
    remaining_intermediate_nums = (intermediate_z.squeeze() > 0).sum(-1).tolist()
    remaining_head_nums = (head_z.squeeze() > 0).sum(-1).tolist()

    head_nums = (np.outer((head_z.squeeze() * head_layer_z).reshape(-1), hidden_z) > 0).sum().item()
    intermediate_nums = (np.outer((intermediate_z.squeeze() * mlp_z).reshape(-1), hidden_z) > 0).sum().item()

    remaining_model_size = head_nums * l0_module.dim_per_head * 4 + intermediate_nums * 2

    # bias terms
    remaining_model_size += remaining_hidden_dims * l0_module.num_layers # O matrix's bias in head
    remaining_model_size += np.sum(remaining_head_nums).item() * 3 * l0_module.dim_per_head # QKV matrix's bias in head
    remaining_model_size += np.sum(remaining_intermediate_nums).item() * 1 # upper projection matrix's bias in mlp
    remaining_model_size += remaining_hidden_dims * l0_module.num_layers # down projection matrix's bias in mlp
    pruned_model_size = l0_module.prunable_model_size - remaining_model_size

    results = {}
    results["head_layer"] = head_layer_z.reshape(-1).tolist()
    results["mlp"] = mlp_z.reshape(-1).tolist()
    results["hidden"] = remaining_hidden_dims
    results["intermediate"] = remaining_intermediate_nums
    results["head"] = remaining_head_nums
    results["pruned_params"] = pruned_model_size
    results["remaining_params"] = remaining_model_size
    results["sparsity"] = pruned_model_size / l0_module.prunable_model_size
    return results


