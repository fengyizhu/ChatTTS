"""
ChatTTS 协议定义文件
包含各种数据结构和参数类定义
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
from pydantic import BaseModel

from ChatTTS.utils import del_all

class RefineTextParams(BaseModel):
    """文本精炼参数"""
    prompt: str = ""
    top_P: float = 0.7
    top_K: int = 20
    temperature: float = 0.7
    repetition_penalty: float = 1.0
    max_new_token: int = 384
    min_new_token: int = 0
    show_tqdm: bool = True
    ensure_non_empty: bool = True
    manual_seed: Optional[int] = None

class InferCodeParams(RefineTextParams):
    """推理代码参数"""
    spk_emb: Optional[str] = None
    spk_smp: Optional[str] = None
    txt_smp: Optional[str] = None
    temperature: float = 0.001
    repetition_penalty: float = 1.05
    max_new_token: int = 2048
    stream_batch: int = 24
    stream_speed: int = 12000
    pass_first_n_batches: int = 2

    cloning: str = None
    target_sr: int = 24000
    stream_batch_size: int = 8

@dataclass(repr=False, eq=False)
class GenerationOutputs:
    ids: List[torch.Tensor]
    attentions: List[Optional[Tuple[torch.FloatTensor, ...]]]
    hiddens: List[torch.Tensor]
    finished: bool

    def destroy(self):
        del_all(self.ids)
        del_all(self.attentions)
        del_all(self.hiddens)

class ChatTTSParams(BaseModel):
    model: str
    input: str
    stream: Optional[bool] = False
    voice: Optional[str] = "28"
    response_format: Optional[str] = "wav"
    speed: Optional[int] = 1

    # supper parameters
    params_infer_code: Optional[InferCodeParams] = InferCodeParams()



