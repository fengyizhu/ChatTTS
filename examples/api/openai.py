import os
import sys

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

import ChatTTS
from ChatTTS.protocol import RefineTextParams, InferCodeParams
from tools.audio.np import pcm_to_wav_bytes

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from typing import Optional, AsyncGenerator
from tools.logger import get_logger
from pydantic import BaseModel

logger = get_logger("Command")

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global chat

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load():
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)


class ChatTTSParams(BaseModel):
    input: str
    stream: bool = False
    lang: Optional[str] = None
    voice: Optional[str] = None
    skip_refine_text: bool = True
    refine_text_only: bool = False
    use_decoder: bool = True
    do_text_normalization: bool = True
    do_homophone_replacement: bool = False
    params_refine_text: Optional[RefineTextParams] = None
    params_infer_code: Optional[InferCodeParams] = None
    stream_batch_size: int = 16


@app.post("/v1/audio/speech")
async def speech(params: ChatTTSParams):
    logger.info("Text input: %s", str(params.input))
    text = [params.input]
    logger.info("Use speaker:")
    logger.info(params.params_infer_code.spk_emb)
    logger.info("Start voice inference.")
    # chat.infer returns a coroutine that needs to be awaited
    results_generator = await chat.infer(
        text=text,
        stream=params.stream,
        lang=params.lang,
        skip_refine_text=params.skip_refine_text,
        use_decoder=params.use_decoder,
        do_text_normalization=params.do_text_normalization,
        do_homophone_replacement=params.do_homophone_replacement,
        params_infer_code=params.params_infer_code,
        params_refine_text=params.params_refine_text,
    )

    if params.stream:
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for result in results_generator:
                yield pcm_to_wav_bytes(result[0])

        return StreamingResponse(
            content=stream_results(), media_type="text/event-stream"
        )

    output = None
    # Properly iterate through the async iterator
    async for request_output in results_generator:
        if output is None:
            output = request_output[0]
        else:
            output = np.concatenate((output, request_output[0]), axis=0)
    output = pcm_to_wav_bytes(output)
    return Response(
        content=output, media_type="audio/wav", headers={"Cache-Control": "no-cache"}
    )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
