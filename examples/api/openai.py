import json
import sys
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

import ChatTTS
from ChatTTS.protocol import ChatTTSParams

from typing import AsyncGenerator

from tools.audio.np import response_format_to_bytes
from tools.logger import get_logger

logger = get_logger("Command")

@asynccontextmanager
async def startup_event(app: FastAPI):
    global chat
    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load():
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)
    yield

app = FastAPI(lifespan=startup_event)

@app.post("/v1/audio/cloning")
async def cloning(params: ChatTTSParams):
    logger.info(f"start cloning params {json.dumps(params, ensure_ascii=False)}")



@app.post("/v1/audio/speech")
async def speech(params: ChatTTSParams):

    logger.info(f"start voice params {json.dumps(params.dict(), ensure_ascii=False)}")
    # 设置速度提示
    assert params.speed in [1, 2, 3, 4, 5], "speed should be in [1, 2, 3, 4, 5]"

    results_generator = await chat.infer(
        input=params.input,
        voice=params.voice,
        stream=params.stream,
        lang=None,
        speed=params.speed,
        use_decoder=True,
        do_text_normalization=True,
        do_homophone_replacement=True,
        params_infer_code=params.params_infer_code,
        stream_batch_size=params.params_infer_code.stream_batch_size,
    )

    if params.stream:
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for result in results_generator:
                yield response_format_to_bytes(result[0], params.response_format)

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
    output = response_format_to_bytes(output, params.response_format)
    return Response(
        content=output, media_type="audio/wav", headers={"Cache-Control": "no-cache"}
    )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
