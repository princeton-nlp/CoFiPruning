import pdb

from re import L
from black import main
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from transformers.utils import logging

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
logger = logging.get_logger(__name__)

class L0Module(Module):
    def __init__(self,
                 config, 
                 droprate_init=0.5,
                 temperature=2./3.,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 pruning_modules=["head", "head_layer", "hidden", "intermediate", "mlp"],
                 magical_number=0.8, # from Wang et al. 2020
                 ):
        super(L0Module, self).__init__()
        self.all_types = ["hidden", "intermediate", "mlp", "head_layer", "head"]
        self.pruning_modules = pruning_modules

        # about the model configuration
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, "intermediate_size", config.ffn_dim)
        self.num_attention_heads = config.num_attention_heads
        self.mlp_num_per_layer = 1
        self.dim_per_head = self.hidden_size // self.num_attention_heads 
        self.num_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size

        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads
        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 2 + self.hidden_size + self.hidden_size * 4
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_layers
        self.prunable_model_size = 0 

        # parameters for the L0 regularization
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.z_logas = {}
        self.num_params_per_mask = {} # number of parameters per mask
        self.mask_sizes = {} # number of masks
        self.mask_shapes = {} # shape of masks

        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
        
        self.prunable_model_size = self.calculate_prunable_model_size()
        self.magical_number = magical_number

        # parameters for the Lagrangian
        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        print("********** Initializing L0 Module **********") 
        for pruning_module in self.pruning_modules:
            print(f"***** {pruning_module} *****")
            print(f"z.shape", self.z_logas[pruning_module].shape)
            print(f"size", self.mask_sizes[pruning_module])
        print(f"prunable model size: {self.prunable_model_size}")
        
        self.z_logas = torch.nn.ParameterDict(self.z_logas)

    def calculate_prunable_model_size(self):
        prunable_mlp_size = self.params_per_mlp_layer * self.num_layers
        prunable_head_layer_size = self.params_per_head_layer * self.num_layers
        prunable_model_size = 0
        if "hidden" in self.pruning_modules:
            return prunable_mlp_size + prunable_head_layer_size
        if "head_layer" in self.pruning_modules or "head" in self.pruning_modules:
            prunable_model_size += prunable_head_layer_size
        if "mlp" in self.pruning_modules or "intermediate" in self.pruning_modules:
            prunable_model_size += prunable_mlp_size
        return prunable_model_size
        
    def initialize_one_module(self, module_name):
        func_name = f"initialize_{module_name}"
        try:
            method = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError("Instance `{}` does not implement `{}`".format(self, func_name))
        method()
            
    def add_one_module(self, pruning_module, z_loga, num_params_per_mask, size, shape): #! init the z_logas
        self.z_logas[pruning_module] = z_loga
        self.num_params_per_mask[pruning_module] = num_params_per_mask
        self.mask_sizes[pruning_module] = size
        self.mask_shapes[pruning_module] = shape
        print("Initalize {}".format(pruning_module))
    
    def initialize_parameters(self, size, num_layer=None):
        """ Initialize the parameters for masking variables. """
        if num_layer is not None:
            z_loga = Parameter(torch.Tensor(num_layer, size))
        else:
            z_loga = Parameter(torch.Tensor(size))
        self.reset_loga(z_loga)
        return z_loga
    
    def reset_loga(self, tensor, mean=10):
        """ Initialize the parameters for masking variables. """
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def initialize_hidden(self):
        hidden_loga = self.initialize_parameters(self.hidden_size)
        self.add_one_module(z_loga=hidden_loga, 
                            pruning_module="hidden", 
                            num_params_per_mask=self.hidden_size * 4 + self.hidden_size * 4 * 2,
                            size=self.hidden_size, 
                            shape=[self.hidden_size])

    def initialize_head(self):
        head_loga = self.initialize_parameters(self.num_attention_heads, self.num_layers)
        self.add_one_module(z_loga=head_loga, 
                            pruning_module="head", 
                            num_params_per_mask=self.params_per_head, 
                            size=self.num_attention_heads,
                            shape=[self.num_layers, 1, self.num_attention_heads, 1, 1])

    def initialize_head_layer(self):
        headlayer_loga = self.initialize_parameters(self.num_layers)
        self.add_one_module(z_loga=headlayer_loga, 
                            pruning_module="head_layer", 
                            num_params_per_mask=self.params_per_head * self.num_attention_heads,
                            size=1,
                            shape=[self.num_layers])

    def initialize_intermediate(self):
        int_loga = self.initialize_parameters(self.intermediate_size, self.num_layers)
        self.add_one_module(z_loga=int_loga, 
                            pruning_module="intermediate", 
                            num_params_per_mask=self.params_per_intermediate_dim, 
                            size=self.intermediate_size,
                            shape=[self.num_layers, 1, 1, self.intermediate_size])


    def initialize_mlp(self):
        mlp_loga = self.initialize_parameters(self.num_layers)
        self.add_one_module(z_loga=mlp_loga, 
                            pruning_module="mlp", 
                            num_params_per_mask=self.params_per_mlp_layer, 
                            size=self.mlp_num_per_layer,
                            shape=[self.num_layers])

    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])

    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def get_num_parameters_for_one(self, loga, num_params_per_mask):
        return torch.sum(1 - self.cdf_qz(0, loga)) * num_params_per_mask

    def transform_scores_for_head(self):
        assert "head" in self.pruning_modules

        if "head_layer" in self.pruning_modules:
            head_layer_score = 1 - self.cdf_qz(0, self.z_logas["head_layer"])
        else:
            head_layer_score = None
        head_score = 1 - self.cdf_qz(0, self.z_logas["head"]) # 12 * 12
       
        if head_layer_score is not None:
            head_layer_score = head_layer_score.view(-1, 1, 1) # 12 * 1 * 1
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1
       
        return head_layer_score, head_score

    def get_num_parameters_for_mlp(self):
        mlp_score = 1 - self.cdf_qz(0, self.z_logas["mlp_layer"]) # 12
        intermediate_score = 1 - self.cdf_qz(0, self.z_logas["intermediate"]) # 12 * 3072
        mlp_score = mlp_score.unsqueeze(-1)

        num_parameters = torch.sum(mlp_score * intermediate_score) * self.num_params_per_mask["intermediate"]
        return num_parameters

    def get_expected_num_params_for_hidden(self): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 * 1
        # 12 * 12 * 1
        head_layer_score, head_score = self.transform_scores_for_head()
        hidden_score = 1 - self.cdf_qz(0, self.z_logas["hidden"]) # 768

        head_score = (head_layer_score * head_score).reshape(-1)
        num_parameters += \
            torch.sum(torch.outer(hidden_score, head_score)) * self.num_params_per_mask["head"] / self.mask_sizes["hidden"]
        num_parameters += hidden_score.sum() * self.num_layers # O's bias
        num_parameters += head_score.sum() * 3 * self.dim_per_head # QKV's bias

        mlp_score = 1 - self.cdf_qz(0, self.z_logas["mlp"])  # 12
        int_score = 1 - self.cdf_qz(0, self.z_logas["intermediate"])  # 12 * 3072
        mlp_score = mlp_score.unsqueeze(-1)

        int_score = (mlp_score * int_score).reshape(-1)
        num_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 2
        num_parameters += hidden_score.sum() * self.num_layers # downward matrix's bias
        num_parameters += int_score.sum() * 1 # upward matrix's bias
        return num_parameters


    def get_expected_num_params(self):
        num_parameters = 0

        head_layer_score, head_score = self.transform_scores_for_head()
        
        head_score = head_score * head_layer_score
        num_parameters += torch.sum(head_score) * self.num_params_per_mask["head"]

        mlp_loga = 1 - self.cdf_qz(0, self.z_logas["mlp"])  # 12
        int_score = 1 - self.cdf_qz(0, self.z_logas["intermediate"])  # 12 * 3072
        mlp_loga = mlp_loga.unsqueeze(-1)

        int_score = int_score * mlp_loga
        num_parameters += torch.sum(int_score) * self.num_params_per_mask["intermediate"]
        return num_parameters


    def get_target_sparsity(self, pruned_steps):
        target_sparsity = self.target_sparsity
        if getattr(self, "lagrangian_warmup_steps", 0) > 0:
            target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup_steps) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.get_target_sparsity(pruned_steps)            
        if "hidden" in self.pruning_modules:
            expected_size = self.get_expected_num_params_for_hidden() #! calculate \bar s
        else:
            expected_size = self.get_expected_num_params() #! calculate \bar s
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        
        lagrangian_loss = ( #! see appendix
                self.lambda_1 * (expected_sparsity - target_sparsity)
                + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2 
        )
        return lagrangian_loss, expected_sparsity, target_sparsity
 

    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    # during inference
    def _deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask

    def forward(self, training=True,):
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}

        if training:
            for i, pruning_module in enumerate(self.pruning_modules):
                loga = self.z_logas[pruning_module]
                z = self._sample_z(loga)
                zs[f"{pruning_module}_z"] = z.reshape(self.mask_shapes[pruning_module])
        else:
            for i, pruning_module in enumerate(self.pruning_modules):
                if pruning_module != "hidden": # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[pruning_module]
                    for idx in range(len(loga_all_layers)):
                        loga = loga_all_layers[idx]
                        size = self.mask_sizes[pruning_module]
                        z = self._deterministic_z(size, loga)
                        zs[f"{pruning_module}_z"].append(z.reshape(self.mask_shapes[pruning_module][1:]))
                else:
                    z = self._deterministic_z(self.mask_sizes[pruning_module], self.z_logas["hidden"])
                    zs[f"{pruning_module}_z"] = z
            for type in zs:
                if type != "hidden_z":
                    zs[type] = torch.stack(zs[type])
        return zs 


def test_l0_module():
    from transformers import AutoConfig
    from CoFiPruning.utils.cofi_utils import calculate_model_size_with_z
    
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    l0_module = L0Module(config, target_sparsity=0.5)
    zs = l0_module.forward(training=False)
    re = calculate_model_size_with_z(zs, l0_module)
    l0_module.lagrangian_warmup_steps = 100
    l0_module.lagrangian_regularization(0)


if __name__ == "__main__":
    test_l0_module()