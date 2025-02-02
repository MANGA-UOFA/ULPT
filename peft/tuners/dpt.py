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
class DPTPromptTuningConfig(PromptLearningConfig):
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
    r: int = field(default=10, metadata={"help": "Lora attention dimension"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    save_base_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to save base."},
    )
    load_base_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to load base."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING_DPT



class DPTPromptEmbedding(torch.nn.Module):

    def __init__(self, config, word_embeddings):
        super().__init__()

        r = config.r

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, r)

        self.base_embedding = torch.nn.Linear(r, word_embeddings.weight.shape[-1], bias=False)


    def forward(self, indices):
        prompt_embeddings = self.embedding(indices)
        
        return self.base_embedding(prompt_embeddings)
    


