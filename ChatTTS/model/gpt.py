import platform
from dataclasses import dataclass
import logging
from typing import Union, List, Optional, Tuple, Callable
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from tqdm import tqdm
from transformers import LlamaModel, LlamaConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available

from ..utils import del_all
from .embed import Embed


class GPT(nn.Module):
    def __init__(
        self,
        gpt_config: dict,
        device=torch.device("cpu"),
        device_gpt=torch.device("cpu"),
        logger=logging.getLogger(__name__),
    ):
        super().__init__()

        self.llm = None
        self.logger = logger

        self.device = device
        self.device_gpt = device_gpt

        self.generator = torch.Generator(device=device)

        self.num_vq = int(gpt_config["num_vq"])
        self.num_audio_tokens = int(gpt_config["num_audio_tokens"])
        self.num_text_tokens = int(gpt_config["num_text_tokens"])

    def from_pretrained(
        self, gpt_folder: str, embed_file_path: str, experimental=False
    ):
        from .velocity import LLM
        self.llm = LLM(
            model=gpt_folder,
            num_audio_tokens=self.num_audio_tokens,
            num_text_tokens=self.num_text_tokens,
            post_model_path=embed_file_path,
            dtype="float32"
        )
        self.logger.info("vLLM model loaded")




