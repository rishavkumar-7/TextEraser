# Purpose: Build the pretrained model.
# Borrowed from [VPT](https://github.com/KMnP/vpt). Thanks to the authors.
# Hacked by this repo's author.

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from utils import logging
# from models.model_zoo import # ViT_B_21K, Swin_B_22K
# ===================
from models.model_zoo import TinyLlama
# ===================
logger = logging.get_logger("lmeraser")
_MODEL_TYPES = {
    "TinyLlama": TinyLlama,
    # "Llama7B": Llama7B,
    
    # "vit-b-22k": ViT_B_21K, 
    # "swin-b-22k": Swin_B_22K, 
}


def _construct_model(args):
    """Build the pretrained model."""
    assert (
        args.pretrained_model in _MODEL_TYPES.keys()
    ), f"Model type '{args.pretrained_model}' not supported"
    if args.device_type == "cuda":
        assert (
            args.num_gpus <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"

    # Construct the model
    model_type = args.pretrained_model
    model,tokenizer = _MODEL_TYPES[model_type](args) # call the tiny llama class from models.model_zoo

    device = load_model_to_device(model, args)
    # device ="cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    logger.info(f"Device used for model: {device}")

    return model, tokenizer, device


def load_model_to_device(model, args):
    cur_device = args.local_rank if args.distributed else 0
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        #==============================================
        # model = model.cuda(device=cur_device)
        #============================================
        # Use multi-process data parallel model in the multi-gpu setting
        if args.num_gpus > 1:
            # Make model replica operate on the current device
            model = DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=True,
            )
    else:
        # model = model.to(args.device_type)
        model=model #added to remove error : `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models
    # return model, cur_device
    #############################################
    return cur_device
    ##############################################
class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """A wrapper for DistributedDataParallel."""
    # succeed all the attributes and methods of DistributedDataParallel
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # succeed all the attributes and methods from self.module
        for name in dir(self.module):
            if not hasattr(self, name):
                setattr(self, name, getattr(self.module, name))
