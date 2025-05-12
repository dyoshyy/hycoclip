# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

from hycoclip.config import LazyCall as L
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders import TransformerTextEncoder
from hycoclip.models import CLIPBaseline

from .train_hycoclip_vit_l import dataset, optim, train


model = L(CLIPBaseline)(
    visual=L(build_timm_vit)(
        arch="vit_large_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77
    ),
    embed_dim=512,
)

optim.optimizer.params.exclude_params = ["logit_scale"]
