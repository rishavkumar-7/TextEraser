#!/usr/bin/env python3
"""Tiny Llama"""
import os
import sys
import torch
# ========================================
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig

# ========================================



# from timm.models import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, '../'))

# import backbones  # noqa: F401

def TinyLlama(args): # add args for this class later...
    """Construct Tiny Llama pretrained on #( dataset info will be updated)"""
    return get_model_and_tokenizer(args)
    # model = create_model(
    #     'jx_vit_base_patch16_224_in21k', # call from @register_model
    #     pretrained=False,
    #     num_classes=21843,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=None,
    # )
    # return _load_checkpoint(args, model)

def get_model_and_tokenizer(args):
    """
    Returns model , 
    """

    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

# def _load_checkpoint(args, model):
#     """Load the checkpoint into the given model."""
#     if args.pretrained_model == "vit-b-22k":
#         # path = os.path.join(ROOT_DIR, "../checkpoints/vit_base_p16_224_in22k.pth")
#         # path = os.path.join(ROOT_DIR, "../checkpoints_model/vit_base_p16_224_in22k.pth")
#         # path = os.path.join(ROOT_DIR, "../checkpoints_model/vit_base_p16_224_in22k.pth")
#         path = "/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/lmeraser/models/model_zoo/checkpoints_model/vit_base_p16_224_in22k.pth"

#     else:
#         raise NotImplementedError
#     checkpoint = torch.load(path, map_location="cpu")

#     if "module" in checkpoint:
#         checkpoint = checkpoint["module"]
#     # for key in list(checkpoint.keys()):
#     #     if key in ["pre_logits.fc.bias", "pre_logits.fc.weight"]: # ["head.bias", "head.weight"]:
#     #         del checkpoint[key]

#     if "model" in checkpoint:
#         model.load_state_dict(checkpoint["model"])
#     elif "state_dict" in checkpoint:
#         model.load_state_dict(checkpoint["state_dict"])
#     else:
#         model.load_state_dict(checkpoint)

#     return model