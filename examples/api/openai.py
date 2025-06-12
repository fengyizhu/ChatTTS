import json
import sys
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

import ChatTTS
from ChatTTS.protocol import ChatTTSParams
from tools.audio.np import pcm_to_wav_bytes


from typing import AsyncGenerator
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

@app.post("/v1/audio/speech")
async def speech(params: ChatTTSParams):

    logger.info(f"start voice params {json.dumps(params, ensure_ascii=False)}")

    results_generator = await chat.infer(
        input=params.input,
        stream=params.stream,
        speed=params.speed,
        lang=params.lang,
        params_infer_code=params.inferCodeParams,
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
