from utils.cofi_utils import *
from utils.utils import *

if __name__ == "__main__":
    model_path = "/n/fs/nlp-mengzhou/space2/out/meta/out/MNLI/layerdistillv3_prunehidden/MNLI_l0_headint_nosvd_layerpuning_layerdistillv3_prunehidden_seed57_distilltemp2_pretrain12000_warpup24000_ts0.95_cealpha0.5_20epochs/best/FT-lr1e-5/best"
    from models.modeling_bert import CoFiBertForSequenceClassification
    model = load_pruned_model(model_path, CoFiBertForSequenceClassification, 3)
    print(calculate_parameters(model))



# tokenizer.config.json
# vocab.txt
