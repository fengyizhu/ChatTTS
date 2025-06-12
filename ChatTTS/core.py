import os
import uuid
from dataclasses import asdict
from typing import Literal, Optional, List, Tuple, Dict, Union
from json import load

import numpy as np
import torch

from .config import Config
from .model import gen_logits, Speaker
from .utils import del_all
from .utils import logger as utils_logger
from .protocol import RefineTextParams, InferCodeParams, GenerationOutputs
from .load import ModelLoader
from .norm import Normalizer


class Chat:
    def __init__(self, logger=None):

        if logger is None:
            logger = utils_logger
        self.logger = logger
        self.device = torch.device("cuda")
        
        self.config = Config()
        
        # 初始化文本归一化器
        self.normalizer = Normalizer(
            os.path.join(os.path.dirname(__file__), "res", "homophones_map.json"),
            logger,
        )
        
        # 加载SHA256校验映射
        with open(
            os.path.join(os.path.dirname(__file__), "res", "sha256_map.json")
        ) as f:
            self.sha256_map: Dict[str, str] = load(f)

        # 初始化模型加载器
        self.model_loader = ModelLoader(self.config, logger)


    # 类变量，定义需要检查的模块
    _REQUIRED_MODULES = {
        'base': ["vocos", "gpt", "tokenizer", "embed"],
        'with_decoder': ["decoder"],
        'without_decoder': ["dvae"]
    }
    
    def has_loaded(self, use_decoder: bool = False) -> bool:
        """检查所有必需的模块是否已加载
        
        Args:
            use_decoder: 是否使用decoder模块
            
        Returns:
            bool: 所有必需模块已加载返回True，否则返回False
        """
        modules_to_check = self._REQUIRED_MODULES['base'].copy()
        
        if use_decoder:
            modules_to_check.extend(self._REQUIRED_MODULES['with_decoder'])
        else:
            modules_to_check.extend(self._REQUIRED_MODULES['without_decoder'])
            
        missing_modules = [
            module for module in modules_to_check 
            if not hasattr(self, module)
        ]
        
        for module in missing_modules:
            self.logger.warning(f"{module} not initialized.")
            
        return not bool(missing_modules)

    def load(
        self,
        source: Literal["huggingface", "local", "custom"] = "local",
        force_redownload=False,
        custom_path: Optional[torch.serialization.FILE_LIKE] = None,
        device: Optional[torch.device] = None,
        coef: Optional[torch.Tensor] = None,
        experimental: bool = False,
    ) -> bool:
        """加载模型
        
        Args:
            source: 模型来源，可以是"huggingface"、"local"或"custom"
            force_redownload: 是否强制重新下载
            compile: 是否编译模型
            custom_path: 自定义模型路径
            device: 设备
            coef: 系数张量
            use_flash_attn: 是否使用Flash Attention
            experimental: 是否启用实验性功能
            
        Returns:
            加载成功返回True，否则返回False
        """
        # 使用ModelLoader下载模型
        download_path = self.model_loader.download_models(
            source, 
            force_redownload, 
            custom_path,
            self.sha256_map
        )
        if download_path is None:
            return False
            
        # 加载所有模型
        models = self.model_loader.load_all(
            device=device,
            coef=coef,
            experimental=experimental,
            **{
                k: os.path.join(download_path, v)
                for k, v in asdict(self.config.path).items()
            },
        )
        
        if models is None:
            return False
            
        # 将加载的模型赋值给Chat实例
        for name, model in models.items():
            setattr(self, name, model)
            
        return self.has_loaded()

    def unload(self):
        """卸载所有模型并重新初始化Chat实例"""
        logger = self.logger
        # 删除规范化器和SHA256映射
        del self.normalizer
        del self.sha256_map
        del self.model_loader
        
        # 删除所有已加载的模型
        del_list = ["vocos", "gpt", "decoder", "dvae", "tokenizer", "embed", "speaker"]
        for module in del_list:
            if hasattr(self, module):
                delattr(self, module)
                
        # 重新初始化Chat实例
        self.__init__(logger)
        
    def _normalize_texts(
        self, 
        texts: List[str],
        do_normalize: bool, 
        do_homophone: bool, 
        lang: Optional[str]
    ) -> List[str]:
        """对文本列表进行标准化处理
        
        Args:
            texts: 待处理的文本列表
            do_normalize: 是否进行文本标准化
            do_homophone: 是否进行同音词替换
            lang: 语言代码
            
        Returns:
            标准化后的文本列表
        """
        normalized = [
            self.normalizer(t, do_normalize, do_homophone, lang)
            for t in texts
        ]
        self.logger.info("Normalized texts: %s", normalized)
        return normalized
    
    async def _process_audio_generation(
        self,
        texts: List[str],
        stream: bool,
        speed: int,
        use_decoder: bool,
        params: InferCodeParams,
        stream_batch_size: int,
    ):
        """处理音频生成逻辑

        Args:
            texts: 输入文本列表
            stream: 是否流式处理
            use_decoder: 是否使用decoder
            params: 推理参数
            stream_batch_size: 流式处理的批次大小
            original_text: 原始文本

        Yields:
            Tuple[音频数据, 原始文本, token IDs]
        """
        length = 0

        result_generator = self._infer_code(
            texts, stream, speed, params, stream_batch_size
        )

        async for result in result_generator:
            wavs = self._decode_to_wavs(
                result.hiddens if use_decoder else result.ids,
                use_decoder,
            )

            if result.finished:
                cache_token_ids = Speaker.encode_prompt(result.ids[0])
                self.logger.debug(
                    "Generated audio for text: %s, cache_token_ids: %s",
                    texts, cache_token_ids
                )
                yield wavs[:, length:]
            else:
                import librosa
                silence_intervals = librosa.effects.split(wavs[0][length:], top_db=10)
                silence_left = 0
                if len(silence_intervals) == 0:
                    silence_left = len(wavs[0])
                else:
                    for i in range(len(silence_intervals)):
                        silence_left = silence_intervals[i][0]
                    if silence_left <= 0:
                        continue
                new_wavs = wavs[:, length: length + silence_left]
                length += len(new_wavs[0])
                yield new_wavs


    async def infer(
        self,
        input: str,
        stream: bool = False,
        lang: Optional[str] = None,
        speed: Optional[int] = 1,
        use_decoder: bool = True,
        do_text_normalization: bool = True,
        do_homophone_replacement: bool = True,
        params_infer_code: InferCodeParams = InferCodeParams(),
        stream_batch_size: int = 8,
    ) :
        """执行文本到语音的推理
        
        Args:
            input: 输入文本或文本列表
            stream: 是否流式输出结果
            lang: 语言代码
            use_decoder: 是否使用decoder
            do_text_normalization: 是否进行文本标准化
            do_homophone_replacement: 是否进行同音词替换
            params_infer_code: 推理参数
            stream_batch_size: 流式处理的批次大小
            
        Yields:
            Tuple[音频数据, 原始文本, token IDs]
            
        Raises:
            AssertionError: 如果模型未正确加载
        """

        """
            文本标准化与生僻字音字替换   
        """
        texts = self._normalize_texts(
            [input], do_text_normalization, do_homophone_replacement, lang
        )

        # 音频生成
        return self._process_audio_generation(
            texts,
            stream,
            speed,
            use_decoder,
            params_infer_code,
            stream_batch_size
        )

    @torch.inference_mode()
    def _vocos_decode(self, spec: torch.Tensor) -> np.ndarray:
        if "mps" in str(self.device):
            return self.vocos.decode(spec.cpu()).cpu().numpy()
        else:
            return self.vocos.decode(spec).cpu().numpy()

    @torch.inference_mode()
    def _decode_to_wavs(
        self,
        result_list: List[torch.Tensor],
        use_decoder: bool,
    ):
        decoder = self.decoder if use_decoder else self.dvae
        max_x_len = -1
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)
        for result in result_list:
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device,
        )
        for i in range(len(result_list)):
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
            del src
        del_all(result_list)
        mel_specs = decoder(batch_result)
        del batch_result
        wavs = self._vocos_decode(mel_specs)
        del mel_specs
        return wavs


    @torch.no_grad()
    async def _infer_code(
        self,
        texts: Tuple[List[str], str],
        stream: bool,
        speed: int,
        params: InferCodeParams,
        stream_batch_size: int,
    ):
        """使用VLLM引擎生成音频代码
        
        Args:
            text: 输入文本或文本列表
            stream: 是否流式输出结果
            device: 设备
            return_hidden: 是否返回隐藏状态
            params: 推理参数
            stream_batch_size: 流式处理的批次大小
            
        Yields:
            生成的音频代码和元数据
        """
        gpt = self.gpt

        if not isinstance(texts, list):
            texts = [texts]
        assert len(texts), "text should not be empty"

        if not isinstance(params.temperature, list):
            temperature = [params.temperature] * self.config.gpt.num_vq
        else:
            temperature = params.temperature

        prompt = f"[speed_{speed}]"

        input_ids, attention_mask, text_mask = self.tokenizer.encode(
            self.speaker.decorate_code_prompts(
                texts,
                prompt,
                params.txt_smp,
                params.spk_emb,
            ),
            self.config.gpt.num_vq,
            prompt=(
                self.model_loader.speaker.decode_prompt(params.spk_smp)
                if params.spk_smp is not None
                else None
            ),
            device=self.model_loader.device_gpt,
        )
        start_idx = input_ids.shape[-2]
        num_code = self.config.gpt.num_audio_tokens - 1

        logits_warpers, logits_processors = gen_logits(
            num_code=num_code,
            top_P=params.top_P,
            top_K=params.top_K,
            repetition_penalty=params.repetition_penalty,
        )

        speaker_embedding_param = self.embed(input_ids, text_mask)
        del text_mask

        if params.spk_emb is not None:
            self.speaker.apply(
                speaker_embedding_param,
                params.spk_emb,
                input_ids,
                self.tokenizer.spk_emb_ids,
                self.gpt.device_gpt,
            )

        from .model.velocity import SamplingParams

        sample_params = SamplingParams(
            temperature=temperature,
            max_new_token=params.max_new_token,
            max_tokens=8192,
            min_new_token=params.min_new_token,
            logits_processors=(logits_processors, logits_warpers),
            eos_token=num_code,
            infer_text=False,
            start_idx=start_idx,
        )
        input_ids = [i.tolist() for i in input_ids]

        if params.cloning is not None:
            cache_token_ids = Speaker.decode_prompt(params.cloning)
            cache_token_ids = [tuple(i) for i in cache_token_ids.tolist()]
            cache_token_ids = cache_token_ids[:-1]
        else:
            cache_token_ids = []
        results_generator = gpt.llm.llm_engine.generate(
            None, sample_params, uuid.uuid4(), cache_token_ids ,speaker_embedding_param, input_ids[0]
        )
        async for i in results_generator:
            token_ids = []
            hidden_states = []
            if (stream and i.outputs[0].revert_mode is False
                and len(i.outputs[0].token_ids) > len(cache_token_ids)
                and (len(i.outputs[0].hidden_states) % stream_batch_size == 0 ) or i.finished):
                token_ids.append(torch.tensor(i.outputs[0].token_ids[len(cache_token_ids):]))
                hidden_states.append(
                    i.outputs[0].hidden_states[len(cache_token_ids):].to(torch.float32).to(self.device)
                )
                yield GenerationOutputs(
                    ids=token_ids,
                    finished=i.finished,
                    hiddens=hidden_states,
                    attentions=[],
                )