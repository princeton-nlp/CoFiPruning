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
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 pruning_type="structured_heads+structured_mlp+hidden+layer",
                 magical_number=0.8, # from Wang et al. 2020
                 ):
        super(L0Module, self).__init__()
        self.all_types = ["hidden_z", "intermediate_z", "mlp_z", "head_layer_z", "head_z"]
        self.pruning_type = pruning_type

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        self.num_attention_heads = config.num_attention_heads
        self.mlp_num_per_layer = 1
        self.dim_per_head = self.hidden_size // self.num_attention_heads 
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size

        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads
        

        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 2 + self.hidden_size + self.hidden_size * 4
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers
        self.prunable_model_size = 0 

        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None

        types = self.pruning_type.split("+")
        for type in types:
            if type != "layer":
                self.initialize_one_module(type)
        if "layer" in types:
            self.initialize_one_module("layer")

        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        logger.info("********** Initializing L0 Module **********") 
        for type in self.types:
            logger.info(f"***** {type} *****")
            logger.info(f"z.shape", self.z_logas[type].shape)
            logger.info(f"size", self.sizes[type])
        logger.info(f"prunable model size: {self.prunable_model_size}")

    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

    def initialize_one_module(self, module_name):
        if module_name == "structured_mlp":
            self.initialize_structured_mlp()
        elif module_name == "structured_heads":
            self.initialize_structured_head()
        elif module_name == "hidden":
            self.initialize_hidden()
        elif module_name == "layer":
            self.initialize_whole_mlp()
            self.initialized_layer_structured_heads()
            
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape): #! init the z_logas
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape

    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))

    def initialize_hidden(self):
        self.hidden_loga = self.initialize_parameters(self.hidden_size)
        self.add_one_module(self.hidden_loga, type="hidden", 
                            parameter_per_dim=self.hidden_size * 4 + self.hidden_size * 4 * 2,
                            size=self.hidden_size, shape=[self.hidden_size])
        self.reset_loga(self.hidden_loga, mean=10)
        logger.info(f"Initialized hidden loga! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_head(self, add_prunable_model_size=True):
        self.head_loga = self.initialize_parameters(self.num_attention_heads, self.num_hidden_layers)
        self.reset_loga(self.head_loga, mean=10)
        self.add_one_module(self.head_loga, type="head", 
                            parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                            shape=[self.num_hidden_layers, 1, self.num_attention_heads, 1, 1])
        if add_prunable_model_size:
            self.prunable_model_size += self.params_per_head * self.num_hidden_layers * self.num_attention_heads
        logger.info(f"Initialized structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialized_layer_structured_heads(self):
        n_layer = self.num_hidden_layers
        self.headlayer_loga = self.initialize_parameters(n_layer)
        self.reset_loga(self.headlayer_loga, mean=10)
        self.add_one_module(self.headlayer_loga, type="head_layer", 
                            parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                            shape=[n_layer])
        logger.info(f"Initialized layerwise structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_mlp(self):
        self.int_loga = self.initialize_parameters(self.intermediate_size, self.num_hidden_layers)

        self.add_one_module(self.int_loga, type="intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.num_hidden_layers, 1, 1, self.intermediate_size])
        self.prunable_model_size += self.params_per_mlp_layer * self.num_hidden_layers
        self.reset_loga(self.int_loga)
        logger.info(f"Initialized structured mlp! Prunable_model_size = {self.prunable_model_size}")


    def initialize_whole_mlp(self):
        n_layer = self.num_hidden_layers
        self.intlayer_loga = self.initialize_parameters(n_layer)
        self.add_one_module(self.intlayer_loga, type="mlp", 
                            parameter_per_dim=self.params_per_mlp_layer, size=self.mlp_num_per_layer,
                            shape=[n_layer])
        self.reset_loga(self.intlayer_loga, mean=10)
        logger.info(f"Initialized whole mlps! Prunable_model_size = {self.prunable_model_size}")


    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)

    def reset_qz_logas(self):
        for key in self.z_logas:
            if key in ["head_layer", "mlp", "head"]:
                self.reset_loga(self.z_logas[key], 10)
            else:
                self.reset_loga(self.z_logas[key])

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

    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size

    def transform_scores_for_head(self):
        assert "head" in self.types

        if "head_layer" in self.types:
            all_head_score = 1 - self.cdf_qz(0, self.headlayer_loga)
        else:
            all_head_score = None
        head_score = 1 - self.cdf_qz(0, self.head_loga) # 12 * 12
       
        if all_head_score is not None:
            all_head_score = all_head_score.view(-1, 1, 1) # 12 * 1 * 1
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1
       
        return all_head_score, head_score

    def get_num_parameters_for_mlp(self):
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga) # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters

    def get_num_parameters_and_constraint_for_hidden(self): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 * 1
        # 12 * 12 * 1
        all_head_score, head_score = self.transform_scores_for_head()
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # 768

        if all_head_score is not None:
            head_score = (all_head_score * head_score).reshape(-1)
        else:
            head_score = head_score.reshape(-1)
        num_parameters += \
            torch.sum(torch.outer(hidden_score, head_score)) * self.parameters_per_dim["head"] / self.hidden_size

        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = (intlayer_score * int_score).reshape(-1)
        num_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 2
        return num_parameters


    def get_num_parameters_and_constraint(self):
        num_parameters = 0

        all_head_score, head_score = self.transform_scores_for_head()
        
        head_score = head_score * all_head_score
        num_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]

        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = int_score * intlayer_score
        num_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters


    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.target_sparsity
        if "hidden" in self.types:
            expected_size = self.get_num_parameters_and_constraint_for_hidden() #! calculate \bar s
        else:
            expected_size = self.get_num_parameters_and_constraint() #! calculate \bar s
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        lagrangian_loss = ( #! see appendix
                self.lambda_1 * (expected_sparsity - target_sparsity)
                + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2 #! where is the lambda 1 and lambda 2 from
        )
        return lagrangian_loss, expected_sparsity, target_sparsity

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

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

    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.all_types:
            name = type[:-2]
            z = zs.get(type, np.ones(self.shapes[name]))
            if torch.is_tensor(z): 
                new_z = z.squeeze().detach().cpu().numpy() > 0
            numpified_zs[name] = new_z
        return numpified_zs

    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        hidden_z = numpified_zs["hidden"]
        intermediate_z = numpified_zs["intermediate"]
        mlp_z = numpified_zs["mlp"].reshape(-1, 1)
        head_z = numpified_zs["head"]
        head_layer_z = numpified_zs["head_layer"].reshape(-1, 1)

        remaining_hidden_dims = hidden_z.sum().item()
        remaining_intermediate_nums = intermediate_z.reshape(self.num_hidden_layers, self.intermediate_size).sum(-1).tolist()
        remaining_head_nums = head_z.reshape(self.num_hidden_layers, self.num_attention_heads).sum(-1).tolist()

        head_nums = np.outer((head_z * head_layer_z).reshape(-1), hidden_z).sum().item()
        intermediate_nums = np.outer((intermediate_z * mlp_z).reshape(-1), hidden_z).sum().item()

        remaining_model_size = head_nums * self.dim_per_head * 4 + intermediate_nums * 2
        pruned_model_size = self.prunable_model_size - remaining_model_size

        results = {}
        # Not multiplied with each other
        results["head_layers"] = head_layer_z.reshape(-1).astype(int).tolist()
        results["mlp_layers"] = mlp_z.reshape(-1).astype(int).tolist()
        results["hidden_dims"] = remaining_hidden_dims
        results["intermediate_dims"] = remaining_intermediate_nums
        results["head_nums"] = remaining_head_nums
        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size
        
        logger.info(f"remaining_head_layers: {head_layer_z}")
        logger.info(f"remaining_mlp_layers: {mlp_z}")
        logger.info(f"remaining_hidden_dims: {remaining_hidden_dims}")
        logger.info(f"remaining_intermediate_nums: {remaining_intermediate_nums}")
        logger.info(f"remaining_head_nums: {remaining_head_nums}")
        logger.info(f"pruned_model_size: {pruned_model_size}")
        logger.info(f"remaining_model_size: {remaining_model_size}")

        return results

        

    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                if type != "hidden": # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[type]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[type]
                        z = self._deterministic_z(size, loga)
                        zs[f"{type}_z"].append(z.reshape(self.shapes[type][1:]))
                else:
                    z = self._deterministic_z(self.sizes[type], self.hidden_loga)
                    zs[f"{type}_z"] = z
            for type in zs:
                if type != "hidden_z":
                    zs[type] = torch.stack(zs[type])
        return zs 

if __name__ == "__main__":
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("bert-base-uncased")
    l0_module = L0Module(config, lagrangian_warmup=200, target_sparsity=0.5)
