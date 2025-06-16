import argparse
import json
import sys
import os
import time

now_dir = os.getcwd()
sys.path.append(now_dir)

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
chat = None

def create_app(args):
    @asynccontextmanager
    async def startup_event(app: FastAPI):
        global chat
        chat = ChatTTS.Chat(get_logger("ChatTTS"))
        if chat.load(args=args):
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
        start_time = time.time()
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
                is_first_package = True
                async for result in results_generator:
                    if result is not None and is_first_package:
                        logger.info(f"streaming voice params {params.input} first package time : {time.time() - start_time}")
                        is_first_package = False
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
        logger.info(f"voice params {params.input} all time : {time.time() - start_time}")
        return Response(
            content=output, media_type="audio/wav", headers={"Cache-Control": "no-cache"}
        )

    return app

if __name__ == '__main__':
    logger.info("Starting ChatTTS API server...")
    parser = argparse.ArgumentParser(description="TTS API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    logger.info(f"args: {args}")
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")