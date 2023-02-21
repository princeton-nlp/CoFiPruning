import torch
import os
from transformers import AutoConfig 
from CoFiPruning.utils.utils import calculate_parameters
from CoFiPruning.models.l0_module import L0Module
import numpy as np
import collections

def initialize_layer_transformation(model):
    model.layer_transformation.weight.data.copy_(
        torch.eye(len(model.layer_transformation.weight)))
    model.layer_transformation.bias.data.fill_(0)


# load the l0 module
def load_l0_module(model_path):
    l0_module_path = os.path.join(model_path, "l0_module.pt")
    if os.path.exists(l0_module_path):
        p = torch.load(l0_module_path, map_location=torch.device('cpu'))
    else:
        p = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=torch.device('cpu'))
        p = collections.OrderedDict({".".join(k.split(".")[1:]):v for k, v in p.items() if "l0_module" in k})
    config = AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
    if isinstance(p, collections.OrderedDict):
        model = L0Module(config)
        model.load_state_dict(p, strict=False)
        return model 
    else:
        return p
    


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

def turn_zs(zs, l0_module):
    numpified_zs = numpify_zs(zs)
    if "hidden_z" in numpified_zs:
        hidden_z = numpified_zs["hidden_z"]
    else:
        hidden_z = np.ones(l0_module.hidden_size)
    intermediate_z = numpified_zs["intermediate_z"]
    mlp_z = numpified_zs["mlp_z"].reshape(-1, 1)
    head_z = numpified_zs["head_z"]
    head_layer_z = numpified_zs["head_layer_z"].reshape(-1, 1)
    return hidden_z, intermediate_z, mlp_z, head_z, head_layer_z

def calculate_model_size_with_z(zs, l0_module):
    hidden_z, intermediate_z, mlp_z, head_z, head_layer_z = turn_zs(zs, l0_module)

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

def plot_structure_with_z(l0_module):
    zs = l0_module.forward(training=False)
    base_unit = "#"
    rest_base_unit = "-"
    
    num_layers = l0_module.num_layers
    num_heads = l0_module.num_attention_heads
    num_intermediate_size = l0_module.intermediate_size
    
    total_base_unit = num_heads 
    

    hidden_z, intermediate_z, mlp_z, head_z, head_layer_z = turn_zs(zs, l0_module)
    print(" " * (10 + (total_base_unit-4) // 2) + "head" + " " * ((total_base_unit-3) // 2 + (total_base_unit-4) // 2 + 4) + "mlp")
    for i in range(num_layers-1, -1, -1):
        heads = (head_layer_z[i] * head_z[i] > 0).squeeze()
        heads = "".join([base_unit if h else rest_base_unit for h in heads])
        # base_unit_num = round(heads / num_heads * total_base_unit)
        # rest_unit_num = total_base_unit - base_unit_num
        print("Layer {:02d}: ".format(i) + heads, end="\t")
        
        ints = (intermediate_z[i] * mlp_z[i] > 0).sum().item()
        base_unit_num = round(ints / num_intermediate_size * total_base_unit)
        rest_unit_num = total_base_unit - base_unit_num
        print(base_unit * base_unit_num + rest_base_unit * rest_unit_num)


def check_if_model_is_original(orig_model, model_path):
    from transformers import AutoModelForCausalLM
    import random
    a = AutoModelForCausalLM.from_pretrained(orig_model) 
    b = AutoModelForCausalLM.from_pretrained(model_path)
    
    names = list(a.state_dict().keys())
    random.shuffle(names)
    
    for n in names:
        print("Checking", n)

        a1 = a.state_dict()[n]
        b1 = b.state_dict()[n]
        
        if a1.shape != b1.shape:
            continue    
        if "weight" not in n:
            continue
        if (a1 == b1).all():
            print("Same")
        else:
            print("Different")
        break
        
# write the main function
if __name__ == "__main__":
    # model_path = "/scratch/gpfs/mengzhou/space2/out/test_round2_tuned_legacy/CoFi_10000_0.6_opt-1.3b_fmm"
    # model_path = "/scratch/gpfs/mengzhou/space2/out/test_round2_lr_scheduler/CoFi_100000_0.1_0.3_opt-125m_v20"
    # l0_module = load_l0_module(model_path=model_path) 
    # plot_structure_with_z(l0_module)
    # print(l0_module.forward(training=False)["head_z"].squeeze())
    
    # model_path = "/scratch/gpfs/mengzhou/space2/out/test_round2_lr_scheduler/CoFi_100000_0.1_0.3_opt-125m_v22"
    # model_path = "/scratch/gpfs/mengzhou/space2/out/test_round2_tuned/CoFi_10000_0.9_opt-1.3b_femb"
    # l0_module = load_l0_module(model_path=model_path) 
    # plot_structure_with_z(l0_module)
    # print(l0_module.forward(training=False)["head_z"].squeeze())
    
    check_if_model_is_original("facebook/opt-125m", "/scratch/gpfs/mengzhou/space2/out/test_round2_lr_scheduler/CoFi_100000_0.1_0.3_opt-125m_v24")
    