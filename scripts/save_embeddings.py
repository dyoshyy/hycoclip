# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate a trained model using implementations from `meru.evaluation` module.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import Tensor
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
import pickle

from meru.config import LazyConfig, LazyFactory
from meru.utils.checkpointing import CheckpointManager
from meru.tokenizer import Tokenizer


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--embed-save-path", help="Path to save embeddings in .pkl format.")


def create_hyperboloid_embed(x: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, D+1)` giving full hyperboloid vector.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    x_full = torch.cat([x_time, x], dim=-1)
    return x_full


def get_space_norm(x: Tensor):
    return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


def main(_A: argparse.Namespace):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    logger.info(OmegaConf.to_yaml(_C_TRAIN))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    dataloader = LazyFactory.build_dataloader(_C_TRAIN)
    tokenizer = Tokenizer()

    logger.info(f"Generating embeddings for checkpoint in {_A.checkpoint_path}...")

    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)
    model = model.eval()

    all_child_image_feats, all_parent_image_feats = [], []
    all_child_text_feats, all_parent_text_feats = [], []
    batches = 0

    for batch in tqdm(dataloader, desc=f"Generating representations..."):

        with torch.inference_mode():
            tokens = tokenizer(batch["text"])
            parent_tokens = tokenizer(batch["parent_text"])

            child_image_feats = model.encode_image(batch["image"].to(model.device), project=True)
            child_image_feats = create_hyperboloid_embed(child_image_feats, model.curv.exp())

            parent_image_feats = model.encode_image(batch["parent_image"].to(model.device), project=True)
            parent_image_feats = create_hyperboloid_embed(parent_image_feats, model.curv.exp())

            child_text_feats = model.encode_text(tokens, project=True)
            child_text_feats = create_hyperboloid_embed(child_text_feats, model.curv.exp())

            parent_text_feats = model.encode_text(parent_tokens, project=True)
            parent_text_feats = create_hyperboloid_embed(parent_text_feats, model.curv.exp())

            all_child_image_feats.append(child_image_feats.to("cpu").detach().numpy())
            all_parent_image_feats.append(parent_image_feats.to("cpu").detach().numpy())
            all_child_text_feats.append(child_text_feats.to("cpu").detach().numpy())
            all_parent_text_feats.append(parent_text_feats.to("cpu").detach().numpy())
            
            batches += 1
            if batches > 25:
                break
        
    all_child_image_feats = np.concatenate(all_child_image_feats, axis=0)
    all_parent_image_feats = np.concatenate(all_parent_image_feats, axis=0)
    all_child_text_feats = np.concatenate(all_child_text_feats, axis=0)
    all_parent_text_feats = np.concatenate(all_parent_text_feats, axis=0)
    
    embed_dict = {
        "child_image_feats": all_child_image_feats,
        "parent_image_feats": all_parent_image_feats,
        "child_text_feats": all_child_text_feats,
        "parent_text_feats": all_parent_text_feats,
    }

    with open(_A.embed_save_path, "wb") as f:
        pickle.dump(embed_dict, f)


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
