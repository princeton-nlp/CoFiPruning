from transformers import AutoConfig
from models.l0_module import L0Module
from utils.cofi_utils import calculate_model_size_with_z
    
def test_model_size_calculation():
    config = AutoConfig.from_pretrained("facebook/opt-1.3b")
    l0_module = L0Module(config, target_sparsity=0.5)
    zs = l0_module.forward(training=False)
    re = calculate_model_size_with_z(zs, l0_module)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_model_size_calculation()