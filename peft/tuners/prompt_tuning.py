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
import torch.nn.functional as F

from ..utils import PeftType, PromptLearningConfig


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class PromptTuningConfig(PromptLearningConfig):
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

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING


class PromptEmbedding(torch.nn.Module):
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

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]

            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)
        
        # self.layer_norm_pt = torch.nn.LayerNorm(config.token_dim)

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        # return self.layer_norm_pt(prompt_embeddings) 
        return prompt_embeddings

# class PromptEmbedding(torch.nn.Module):
#     """
#     The model to encode virtual tokens into prompt embeddings.

#     Args:
#         config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
#         word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

#     **Attributes**:
#         - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

#     Example:

#     ```py
#     >>> from peft import PromptEmbedding, PromptTuningConfig

#     >>> config = PromptTuningConfig(
#     ...     peft_type="PROMPT_TUNING",
#     ...     task_type="SEQ_2_SEQ_LM",
#     ...     num_virtual_tokens=20,
#     ...     token_dim=768,
#     ...     num_transformer_submodules=1,
#     ...     num_attention_heads=12,
#     ...     num_layers=12,
#     ...     prompt_tuning_init="TEXT",
#     ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
#     ...     tokenizer_name_or_path="t5-base",
#     ... )

#     >>> # t5_model.shared is the word embeddings of the base model
#     >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
#     ```

#     Input Shape: (`batch_size`, `total_virtual_tokens`)

#     Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
#     """

#     def __init__(self, config, word_embeddings):
#         super().__init__()

#         r = 10
#         embeddings_weight = word_embeddings.weight.clone()
#         embeddings_mean = torch.mean(embeddings_weight, dim=0)
#         embeddings_centered = embeddings_weight - embeddings_mean
#         _,_, V = torch.svd(embeddings_centered)
#         reduced_embeddings = torch.mm(embeddings_centered, V[:, :r])

#         transform_matrix = F.normalize(reduced_embeddings, dim=-1).T @ torch.linalg.pinv(F.normalize(embeddings_weight, dim=-1).T)
#         self.linear_layer = torch.nn.Linear(r, embeddings_weight.shape[-1], bias=False)

#         with torch.no_grad():
#             self.linear_layer.weight.copy_(transform_matrix.T)
        
#         for param in self.linear_layer.parameters():
#             param.requires_grad = False

#         total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
#         # self.scaling = torch.nn.Parameter(torch.ones((total_virtual_tokens, 1)), requires_grad=True)
#         self.embedding = torch.nn.Embedding(total_virtual_tokens, r)

#         self.layer_norm = torch.nn.LayerNorm(embeddings_weight.shape[-1])

#         # if config.prompt_tuning_init == PromptTuningInit.TEXT:
#         #     from transformers import AutoTokenizer
#         #     tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
#         #     init_text = config.prompt_tuning_init_text
#         #     init_token_ids = tokenizer(init_text)["input_ids"]
#         #     # Trim or iterate until num_text_tokens matches total_virtual_tokens
#         #     num_text_tokens = len(init_token_ids)
#         #     if num_text_tokens > total_virtual_tokens:
#         #         init_token_ids = init_token_ids[:total_virtual_tokens]
#         #     elif num_text_tokens < total_virtual_tokens:
#         #         num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
#         #         init_token_ids = init_token_ids * num_reps
#         #     init_token_ids = init_token_ids[:total_virtual_tokens]

#         #     word_embedding_weights = reduced_embeddings[torch.LongTensor(init_token_ids)].detach().clone()
#         #     word_embedding_weights = word_embedding_weights.to(torch.float32)
#         #     self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

#     def forward(self, indices):
#         # Just get embeddings
#         prompt_embeddings = self.embedding(indices)
#         prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
#         prompt_embeddings = self.linear_layer(prompt_embeddings)
#         return self.layer_norm(prompt_embeddings)