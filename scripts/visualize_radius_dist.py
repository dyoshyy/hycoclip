from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import Tensor
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
from torch.cuda import amp
import matplotlib.pyplot as plt

from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--dist-save-path", help="Path to save radius distribution figure.")


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

    logger.info(f"Generating radius distribution for checkpoint in {_A.checkpoint_path}...")

    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)
    model = model.eval()

    child_image_norms, parent_image_norms = [], []
    child_text_norms, parent_text_norms = [], []
    batches = 0

    for batch in tqdm(dataloader, desc=f"Generating representation norms"):

        with torch.inference_mode():
            tokens = tokenizer(batch["text"])
            parent_tokens = tokenizer(batch["parent_text"])
            child_image_feats = model.encode_image(batch["image"].to(model.device), project=True)
            parent_image_feats = model.encode_image(batch["parent_image"].to(model.device), project=True)
            child_text_feats = model.encode_text(tokens, project=True)
            parent_text_feats = model.encode_text(parent_tokens, project=True)
            child_image_norms.append(get_space_norm(child_image_feats).to("cpu").detach().numpy())
            parent_image_norms.append(get_space_norm(parent_image_feats).to("cpu").detach().numpy())
            child_text_norms.append(get_space_norm(child_text_feats).to("cpu").detach().numpy())
            parent_text_norms.append(get_space_norm(parent_text_feats).to("cpu").detach().numpy())
            batches += 1
            
            if batches > 167:      # Limit to 167 batches.
                break
        
    child_image_norms = np.concatenate(child_image_norms, axis=0)
    parent_image_norms = np.concatenate(parent_image_norms, axis=0)
    child_text_norms = np.concatenate(child_text_norms, axis=0)
    parent_text_norms = np.concatenate(parent_text_norms, axis=0)
    logger.info(f"Shape of norms: {child_image_norms.shape}, {parent_image_norms.shape}, {child_text_norms.shape}, {parent_text_norms.shape}")
    logger.info(f"Child image norm: {child_image_norms.mean():.4f}, {child_image_norms.min():.4f}, {child_image_norms.max():.4f}")
    logger.info(f"Parent image norm: {parent_image_norms.mean():.4f}, {parent_image_norms.min():.4f}, {parent_image_norms.max():.4f}")
    logger.info(f"Child text norm: {child_text_norms.mean():.4f}, {child_text_norms.min():.4f}, {child_text_norms.max():.4f}")
    logger.info(f"Parent text norm: {parent_text_norms.mean():.4f}, {parent_text_norms.min():.4f}, {parent_text_norms.max():.4f}")

    plt.hist(child_image_norms, bins='auto', density=True, alpha=0.5, label='Child Image')
    plt.hist(parent_image_norms, bins='auto', density=True, alpha=0.5, label='Parent Image')
    plt.hist(child_text_norms, bins='auto', density=True, alpha=0.5, label='Child Text')
    plt.hist(parent_text_norms, bins='auto', density=True, alpha=0.5, label='Parent Text')

    plt.xlim(0.0)
    plt.ylabel('% of samples')
    plt.legend(loc='upper left')

    plt.savefig(_A.dist_save_path)


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
