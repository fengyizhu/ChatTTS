"""
ChatTTS 模型加载模块
包含各种模型的加载和下载逻辑
"""

import os
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Literal, Union
from dataclasses import asdict

import numpy as np
import torch
from vocos import Vocos
from vocos.pretrained import instantiate_class
from huggingface_hub import snapshot_download

from .model import DVAE, Embed, Tokenizer, Speaker
from .model.gpt import GPT
from .utils import (
    check_all_assets,
    download_all_assets,
    select_device,
    get_latest_modified_file,
)

seed = 0

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ModelLoader:
    """模型加载器，负责下载和加载各种模型"""
    
    def __init__(self, config, logger=logging.getLogger(__name__)):
        self.config = config
        self.logger = logger
        self.device = None
        self.device_gpt = None
        self.compile = False
    
    def download_models(
        self,
        source: Literal["huggingface", "local", "custom"] = "local",
        force_redownload=False,
        custom_path: Optional[torch.serialization.FILE_LIKE] = None,
        sha256_map: Dict[str, str] = None,
    ) -> Optional[str]:
        """下载模型
        
        Args:
            source: 模型来源，可以是"huggingface"、"local"或"custom"
            force_redownload: 是否强制重新下载
            custom_path: 自定义模型路径
            sha256_map: SHA256校验映射
            
        Returns:
            str: 下载路径，如果下载失败则返回None
        """
        if source == "local":
            download_path = os.getcwd()
            if (
                not check_all_assets(Path(download_path), sha256_map, update=True)
                or force_redownload
            ):
                with tempfile.TemporaryDirectory() as tmp:
                    download_all_assets(tmpdir=tmp)
                if not check_all_assets(
                    Path(download_path), sha256_map, update=False
                ):
                    self.logger.error(
                        "download to local path %s failed.", download_path
                    )
                    return None
        elif source == "huggingface":
            hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            try:
                download_path = get_latest_modified_file(
                    os.path.join(hf_home, "hub/models--2Noise--ChatTTS/snapshots")
                )
            except:
                download_path = None
            if download_path is None or force_redownload:
                self.logger.log(
                    logging.INFO,
                    f"download from HF: https://huggingface.co/2Noise/ChatTTS",
                )
                try:
                    download_path = snapshot_download(
                        repo_id="2Noise/ChatTTS",
                        allow_patterns=["*.pt", "*.yaml", "*.json", "*.safetensors"],
                    )
                except:
                    download_path = None
            else:
                self.logger.log(
                    logging.INFO, f"load latest snapshot from cache: {download_path}"
                )
            if download_path is None:
                self.logger.error("download from huggingface failed.")
                return None
        elif source == "custom":
            self.logger.log(logging.INFO, f"try to load from local: {custom_path}")
            if not check_all_assets(Path(custom_path), sha256_map, update=False):
                self.logger.error("check models in custom path %s failed.", custom_path)
                return None
            download_path = custom_path

        return download_path
    
    def _setup_device(self, device: Optional[torch.device] = None, experimental: bool = False) -> None:
        """设置设备信息
        
        Args:
            device: 设备，如果为None则自动选择
            experimental: 是否启用实验性功能
        """
        if device is None:
            device = select_device(experimental=experimental)
            self.logger.info("use device %s", str(device))
        self.device = device
        self.device_gpt = device if "mps" not in str(device) else torch.device("cpu")
    
    def load_vocos(self, vocos_ckpt_path: str) -> Vocos:
        """加载 Vocos 模型
        
        Args:
            vocos_ckpt_path: Vocos模型路径
            
        Returns:
            Vocos: 加载好的Vocos模型
        """
        feature_extractor = instantiate_class(
            args=(), init=asdict(self.config.vocos.feature_extractor)
        )
        backbone = instantiate_class(args=(), init=asdict(self.config.vocos.backbone))
        head = instantiate_class(args=(), init=asdict(self.config.vocos.head))
        
        # vocos on mps will crash, use cpu fallback
        device = "cpu" if "mps" in str(self.device) else self.device
        vocos = Vocos(
            feature_extractor=feature_extractor, 
            backbone=backbone, 
            head=head
        ).to(device).eval()
        
        assert vocos_ckpt_path, "vocos_ckpt_path should not be None"
        vocos.load_state_dict(torch.load(vocos_ckpt_path, weights_only=True, mmap=True))
        self.logger.info("vocos loaded.")
        return vocos
    
    def load_dvae(self, dvae_ckpt_path: str, coef: Optional[str] = None) -> tuple[DVAE, str]:
        """加载 DVAE 模型
        
        Args:
            dvae_ckpt_path: DVAE模型路径
            coef: 系数
            
        Returns:
            tuple[DVAE, str]: 加载好的DVAE模型和系数
        """
        dvae = DVAE(
            decoder_config=asdict(self.config.dvae.decoder),
            encoder_config=asdict(self.config.dvae.encoder),
            vq_config=asdict(self.config.dvae.vq),
            dim=self.config.dvae.decoder.idim,
            coef=coef,
            device=self.device,
        ).to(self.device).eval()
        
        coef = str(dvae)
        assert dvae_ckpt_path, "dvae_ckpt_path should not be None"
        dvae.load_state_dict(torch.load(dvae_ckpt_path, weights_only=True, mmap=True))
        self.logger.info("dvae loaded.")
        return dvae, coef
    
    def load_embed(self, embed_path: str) -> Embed:
        """加载 Embed 模型
        
        Args:
            embed_path: Embed模型路径
            
        Returns:
            Embed: 加载好的Embed模型
        """
        embed = Embed(
            self.config.embed.hidden_size,
            self.config.embed.num_audio_tokens,
            self.config.embed.num_text_tokens,
            self.config.embed.num_vq,
        )
        embed.from_pretrained(embed_path, device=self.device)
        embed = embed.to(self.device)
        self.logger.info("embed loaded.")
        return embed
    
    def load_gpt(
        self, 
        gpt_ckpt_path: str, 
        embed_path: str, 
        embed: Embed,
        use_flash_attn: bool = False, 
        experimental: bool = False,
        gpu_memory_utilization: float = 0.9,
    ) -> GPT:
        """加载 GPT 模型（仅VLLM版本）
        
        Args:
            gpt_ckpt_path: GPT模型路径
            embed_path: Embedding模型路径
            embed: Embed模型实例
            use_flash_attn: 是否使用Flash Attention
            experimental: 是否启用实验性功能
            
        Returns:
            GPT: 加载好的GPT模型
        """
        
        gpt = GPT(
            gpt_config=asdict(self.config.gpt),
            device=self.device,
            device_gpt=self.device_gpt,
            logger=self.logger,
            gpu_memory_utilization=gpu_memory_utilization,
        ).eval()
        
        assert gpt_ckpt_path, "gpt_ckpt_path should not be None"
        gpt.from_pretrained(gpt_ckpt_path, embed_path, experimental=experimental)
        self.logger.info("VLLM gpt loaded.")
        return gpt
    
    def load_decoder(self, decoder_ckpt_path: str, coef: str) -> tuple[DVAE, str]:
        """加载 Decoder 模型
        
        Args:
            decoder_ckpt_path: Decoder模型路径
            coef: 系数
            
        Returns:
            tuple[DVAE, str]: 加载好的Decoder模型和系数
        """
        decoder = DVAE(
            decoder_config=asdict(self.config.decoder),
            dim=self.config.decoder.idim,
            coef=coef,
            device=self.device,
        ).to(self.device).eval()
        
        coef = str(decoder)
        assert decoder_ckpt_path, "decoder_ckpt_path should not be None"
        decoder.load_state_dict(
            torch.load(decoder_ckpt_path, weights_only=True, mmap=True)
        )
        self.logger.info("decoder loaded.")
        return decoder, coef
    
    def load_tokenizer(self, tokenizer_path: Optional[str]) -> Optional[Tokenizer]:
        """加载 Tokenizer
        
        Args:
            tokenizer_path: Tokenizer路径
            
        Returns:
            Optional[Tokenizer]: 加载好的Tokenizer，如果路径为None则返回None
        """
        if tokenizer_path:
            tokenizer = Tokenizer(tokenizer_path)
            self.logger.info("tokenizer loaded.")
            return tokenizer
        return None
    
    def load_speaker(self, hidden_size: int, spk_stat: dict) -> Speaker:
        """初始化 Speaker
        
        Args:
            hidden_size: 隐藏层大小
            spk_stat: 说话人统计信息
            
        Returns:
            Speaker: 初始化好的Speaker
        """
        speaker = Speaker(hidden_size, spk_stat, self.device)
        self.logger.info("speaker loaded.")
        return speaker
    
    @torch.no_grad()
    def load_all(
        self,
        vocos_ckpt_path: str = None,
        dvae_ckpt_path: str = None,
        gpt_ckpt_path: str = None,
        gpu_memory_utilization = None,
        embed_path: str = None,
        decoder_ckpt_path: str = None,
        tokenizer_path: str = None,
        device: Optional[torch.device] = None,
        compile: bool = False,
        coef: Optional[str] = None,
        use_flash_attn: bool = False,
        experimental: bool = False,
    ) -> dict:
        """加载所有必要的模型和组件（仅VLLM版本）
        
        Args:
            vocos_ckpt_path: Vocos模型路径
            dvae_ckpt_path: DVAE模型路径
            gpt_ckpt_path: GPT模型路径
            embed_path: Embedding模型路径
            decoder_ckpt_path: Decoder模型路径
            tokenizer_path: Tokenizer路径
            device: 设备
            compile: 是否编译模型
            coef: 系数
            use_flash_attn: 是否使用Flash Attention
            experimental: 是否启用实验性功能
            
        Returns:
            dict: 包含所有加载好的模型和组件的字典
        """
        # 设置设备
        self._setup_device(device, experimental)
        self.compile = compile
        
        result = {}

        try:
            # 按顺序加载各个组件
            result['vocos'] = self.load_vocos(vocos_ckpt_path)
            dvae, coef = self.load_dvae(dvae_ckpt_path, coef)
            result['dvae'] = dvae
            result['embed'] = self.load_embed(embed_path)
            result['gpt'] = self.load_gpt(gpt_ckpt_path,
                                          embed_path,
                                          result['embed'],
                                          use_flash_attn,
                                          experimental,
                                          gpu_memory_utilization)
            
            # 初始化 Speaker
            result['speaker'] = self.load_speaker(
                self.config.gpt.hidden_size, 
                self.config.spk_stat
            )
            
            decoder, coef = self.load_decoder(decoder_ckpt_path, coef)
            result['decoder'] = decoder
            result['tokenizer'] = self.load_tokenizer(tokenizer_path)
            
            result['coef'] = coef
            
            return result
            
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error loading models: {str(e)}", exc_info=True)
            return None
