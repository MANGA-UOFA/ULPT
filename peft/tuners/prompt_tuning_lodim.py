# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from ..utils import PeftType, PromptLearningConfig
import torch.nn.functional as F


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


class BaseEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, embeddings):
        return self.scale * embeddings + self.bias


@dataclass
class PromptTuningLodimConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
    """

    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    r: int = field(default=2, metadata={"help": "Lora attention dimension"})
    num_random_proj: int = field(default=1, metadata={"help": "number of random projections"})
    
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory to save the model."}
    )

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING_LODIM


class PromptEmbeddingLodim(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config, word_embeddings):
        super().__init__()

        r = config.r

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules

        self.embedding = torch.nn.Embedding(total_virtual_tokens, r)

        group = config.num_random_proj
        assert total_virtual_tokens % group == 0

        torch.save(torch.get_rng_state(), config.output_dir + '/rng_state.pth')

        # xavier uniform
        random_proj = torch.empty(group, r, word_embeddings.weight.shape[-1], requires_grad=False, device=self.embedding.weight.device, dtype=self.embedding.weight.dtype)
        limit = math.sqrt(6 / (r + word_embeddings.weight.shape[-1]))
        random_proj.uniform_(-limit, limit)
        random_proj = random_proj.unsqueeze(1).expand(group, total_virtual_tokens//group, r, word_embeddings.weight.shape[-1])
        random_proj = random_proj.reshape(total_virtual_tokens, r, word_embeddings.weight.shape[-1])

        self.register_buffer('random_proj', random_proj)

        self.base_embedding = BaseEmbedding(word_embeddings.weight.shape[-1])


    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        prompt_embeddings = torch.einsum('blr,lrd->bld', prompt_embeddings, self.random_proj)

        return self.base_embedding(prompt_embeddings)
