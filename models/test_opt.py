import torch
import torch.nn.functional as F
from transformers import AutoConfig

from modeling_opt import CoFiOPTForCausalLM, CoFiLayerNorm, prune_layer_norm, turn_head_z, turn_mlp_z, turn_hidden_z 
from l0_module import L0Module

from copy import deepcopy

def get_l0_module(config):
    l0_module = L0Module(config, lagrangian_warmup=200, target_sparsity=0.5)
    return l0_module    

def get_full_zs(l0_module, ones=False):
    with torch.no_grad():
        zs = l0_module.forward(training=True)
        for key in zs:
            if ones:
                zs[key].fill_(1.)
            else:
                zs[key] = torch.FloatTensor(zs[key].shape).uniform_().abs().to(zs[key].device)
    return zs

def zero_out_zs(z, percentage):
    mask = torch.FloatTensor(z.shape).uniform_().abs() > percentage
    mask = mask.to(z.device)
    z = z * mask
    return z

def zero_out_all_zs(zs, percentage):
    for key in zs:
        if key in percentage:
            zs[key] = zero_out_zs(zs[key], percentage[key])
    return zs

def load_base_model(cuda=True):
    model = CoFiOPTForCausalLM.from_pretrained("facebook/opt-125m")
    if cuda:
        model.cuda()
    return model

def load_input_ids(cuda=True):
    input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 2]])
    if cuda:
        input_ids = input_ids.cuda()
    return input_ids

def load_l0_module(cuda=True):
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    l0_module = get_l0_module(config)
    if cuda:
        l0_module = l0_module.cuda()
    return l0_module

def forward(model, input_ids, zs):
    outputs = model(input_ids=input_ids, labels=input_ids, **zs)
    loss = outputs.loss
    return loss

# passed 
def test_full_z():
    """
    Compare the loss of 
        - original model forward
        - model forward with full zs
    """
    print(test_full_z.__doc__)
    model = load_base_model(cuda=False)
    input_ids = load_input_ids()
    l0_module = load_l0_module()
    zs = get_full_zs(l0_module)
        
    model1 = deepcopy(model).cuda()
    model1.prune_modules(zs)
    loss1 = forward(model1, input_ids, zs={}) 


    model2 = deepcopy(model).cuda()    
    loss2 = forward(model2, input_ids, zs)
    assert loss1.item() == loss2.item()
    for key in zs:
        print(f"{key}: {zs[key].shape}")
    print("test_full_z passed!")

# passed 
def test_CoFi_LayerNorm():
    from copy import deepcopy
    layernorm1 = CoFiLayerNorm(768).cuda()
    layernorm2 = deepcopy(layernorm1)
    l0_module = load_l0_module()
    zs = get_full_zs(l0_module)
    zs["hidden_z"] = zero_out_zs(zs["hidden_z"], 0.3)
    remaining_index = zs["hidden_z"].squeeze().nonzero().squeeze()

    input = torch.randn(2, 3, 768).cuda()
    out1 = layernorm1(input, zs["hidden_z"])
    out1 = torch.index_select(out1, dim=-1, index=remaining_index)
    prune_layer_norm(layernorm2, remaining_index)
    compressed_input = torch.index_select(
                input, dim=-1, index=remaining_index)
    out2 = layernorm2(compressed_input)
    assert out1.sum().item() == out2.sum().item()

# passed 
def test_CoFi_Attention():
    l0_module = load_l0_module()
    zs = get_full_zs(l0_module)
    
    model = load_base_model(cuda=False)
    attn = deepcopy(model.model.decoder.layers[0].self_attn)
    hidden_states = torch.randn(2, 3, 768).cuda()
    
    # case 1
    print("case 1: All heads are pruned")
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0.3, "head_layer_z": 1.})
    head_z = corrected_zs["head_z"][0]; head_layer_z = corrected_zs["head_layer_z"][0]
    attn1 = deepcopy(attn).cuda()
    attn2 = deepcopy(attn).cuda()
    attn1.prune_heads(head_z, head_layer_z)
    
    with torch.no_grad():
        attn_output1, _, _ = attn1(hidden_states)
        attn_output2, _, _ = attn2(hidden_states, head_z=head_z, head_layer_z=head_layer_z)
        assert attn_output1 is None
        assert attn_output2.sum().item() == .0

    # case 2
    print("case 2: A non-zero number of heads are pruned")
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0.3, "head_layer_z": 0.})
    head_z = corrected_zs["head_z"][0]; head_layer_z = corrected_zs["head_layer_z"][0]
    attn1 = deepcopy(attn).cuda(); attn1.eval()
    attn2 = deepcopy(attn).cuda(); attn2.eval()
    attn1.prune_heads(head_z, head_layer_z)
    
    with torch.no_grad():
        attn_output1, _, _ = attn1(hidden_states)
        attn_output2, _, _ = attn2(hidden_states, head_z=head_z, head_layer_z=head_layer_z)
        print("v1:", attn_output1.sum())
        print("v2:", attn_output2.sum())
        assert torch.isclose(attn_output1.sum(), attn_output2.sum())
    
    # case 3
    print("case 3: No heads are pruned")
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0., "head_layer_z": 0.})
    head_z = corrected_zs["head_z"][0]; head_layer_z = corrected_zs["head_layer_z"][0]
    attn1 = deepcopy(attn).cuda(); attn1.eval()
    attn2 = deepcopy(attn).cuda(); attn2.eval()
    attn1.prune_heads(head_z, head_layer_z)
    
    with torch.no_grad():
        attn_output1, _, _ = attn1(hidden_states)
        attn_output2, _, _ = attn2(hidden_states, head_z=head_z, head_layer_z=head_layer_z)
        print("v1:", attn_output1.sum())
        print("v2:", attn_output2.sum())
        assert torch.isclose(attn_output1.sum(), attn_output2.sum())
        
    

# passed
def test_CoFi_decode_layer():
    l0_module = load_l0_module()
    zs = get_full_zs(l0_module)
    
    model = load_base_model(cuda=False)
    layer = deepcopy(model.model.decoder.layers[0])
    
    def init(layer, percentage):
        corrected_zs = zero_out_all_zs(deepcopy(zs), percentage)
        head_z = corrected_zs["head_z"][0]; head_layer_z = corrected_zs["head_layer_z"][0]
        intermediate_z = corrected_zs["intermediate_z"][0]; mlp_z = corrected_zs["mlp_z"][0]
        hidden_z = corrected_zs["hidden_z"]
        layer1 = deepcopy(layer).cuda(); layer1.eval()
        layer2 = deepcopy(layer).cuda(); layer2.eval()
        return head_z, head_layer_z, intermediate_z, mlp_z, hidden_z, layer1, layer2
    
    def execute(layer1, layer2, head_z, head_layer_z, intermediate_z, mlp_z, hidden_z=None):
        with torch.no_grad():
            hidden_states = torch.randn(2, 3, 768).to(next(layer1.parameters()).device)
            pruned_hidden_states = hidden_states
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z != 0)
                pruned_hidden_states = hidden_states[..., hidden_z.squeeze().nonzero().squeeze()]
            layer_output1 = layer1(pruned_hidden_states)[0]
            layer_output2 = layer2(hidden_states, head_z=head_z, head_layer_z=head_layer_z, intermediate_z=intermediate_z, mlp_z=mlp_z, hidden_z=hidden_z)[0]
            v1 = layer_output1.sum(); v2 = layer_output2.sum()
            print("v1:", v1.item())
            print("v2:", v2.item())
            assert torch.isclose(v1, v2)
        
    # case 1
    print("*"*20)
    print("case 1: All intermediate dims are pruned")
    percentage = {"intermediate_z": 0.3, "mlp_z": 1.}
    head_z, head_layer_z, intermediate_z, mlp_z, hidden_z, layer1, layer2 = init(layer, percentage)
    layer1.prune_mlps(intermediate_z, mlp_z)
    execute(layer1, layer2, None, None, intermediate_z, mlp_z)    
    print("case 1 passed!")
        
    # case 2
    print("*"*20)
    print("case 2: A non-zero number of intermediate dims are pruned")
    percentage = {"intermediate_z": 0.3}
    head_z, head_layer_z, intermediate_z, mlp_z, hidden_z, layer1, layer2 = init(layer, percentage)
    layer1.prune_mlps(intermediate_z, mlp_z)
    execute(layer1, layer2, None, None, intermediate_z, mlp_z)    
    print("case 2 passed!")
    
    # case 3
    print("*"*20)
    print("case 3: Some heads are pruned and some intermediate dims are pruned")
    percentage = {"head_z": 0.3, "intermediate_z": 0.3}
    head_z, head_layer_z, intermediate_z, mlp_z, hidden_z, layer1, layer2 = init(layer, percentage)
    layer1.prune_heads(head_z, head_layer_z)
    layer1.prune_mlps(intermediate_z, mlp_z)
    execute(layer1, layer2, head_z, head_layer_z, intermediate_z, mlp_z)    
    print("case 3 passed!")

    # case 4
    print("*"*20)
    print("case 4: A few hidden dims are pruned")
    percentage = {"hidden_z": 0.3}
    head_z, head_layer_z, intermediate_z, mlp_z, hidden_z, layer1, layer2 = init(layer, percentage)
    layer1.prune_hidden_states(hidden_z)
    execute(layer1, layer2, None, None, None, None, hidden_z)
    print("case 4 passed!")
    
    # case 5
    print("*"*20)
    print("case 5: some heads/intermediate dims/hidden dims are pruned")
    percentage = {"hidden_z": 0.3, "head_z": 0.3, "intermediate_z": 0.3}
    head_z, head_layer_z, intermediate_z, mlp_z, hidden_z, layer1, layer2 = init(layer, percentage)
    layer1.prune_heads(head_z, head_layer_z)
    layer1.prune_mlps(intermediate_z, mlp_z)
    layer1.prune_hidden_states(hidden_z)
    execute(layer1, layer2, head_z, head_layer_z, intermediate_z, mlp_z, hidden_z)
    print("case 5 passed!")

# passed
def test_CoFi_opt_model():
    l0_module = load_l0_module()
    zs = get_full_zs(l0_module)
    
    model = load_base_model(cuda=False)
    input_ids = load_input_ids(cuda=True)
    
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0.3, "intermediate_z": 0.3, "hidden_z": 0.3, "mlp_z": 0.4, "head_layer_z": 0.5})
    corrected_zs = zero_out_all_zs(deepcopy(zs), {"head_z": 0., "intermediate_z": 0., "hidden_z": 0.3, "mlp_z": 0., "head_layer_z": 0.})
    
    model1 = deepcopy(model).cuda()
    model2 = deepcopy(model).cuda()
    model1.prune_modules(corrected_zs)

    output1 = model1(input_ids).logits
    output2 = model2(input_ids, **corrected_zs).logits
    assert torch.isclose(output1.sum(), output2.sum())
    print("test_prune_opt_model passed!")

if __name__ == "__main__":
    # retest after setting get_full_zs: ones=True 
    test_full_z()
    test_CoFi_LayerNorm()
    test_CoFi_Attention()
    test_CoFi_decode_layer()
    test_CoFi_opt_model()


