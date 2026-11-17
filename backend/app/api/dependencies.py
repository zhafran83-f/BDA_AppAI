"""Bridge async HTTP handlers to synchronous inference without blocking the loop."""
import logging

from fastapi import Request
from starlette.concurrency import run_in_threadpool

from .errors import APIError
from .runtime import ModelUnavailableError

logger = logging.getLogger(__name__)

INVALID_INPUT = {
    "fashion": "Image must be a valid RGB JPEG/PNG of exactly 224x224 pixels with normal orientation.",
    "reviews": "Review must retain at least one known vocabulary term after preprocessing.",
    "purchase": "Purchase input does not match the model feature contract.",
}


def get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or not runtime.active:
        raise APIError(503, "model_unavailable", "AI runtime is not initialized.")
    return runtime


async def invoke_service(request: Request, module: str, method: str, argument):
    runtime = get_runtime(request)
    request.state.inference_module = module
    try:
        result = await run_in_threadpool(
            getattr(runtime.services[module], method), argument
        )
    except ModelUnavailableError as exc:
        raise APIError(503, "model_unavailable", "Required AI model is unavailable.") from exc
    except ValueError as exc:
        raise APIError(422, "validation_error", INVALID_INPUT[module]) from exc
    except Exception as exc:
        runtime.mark_unavailable(module)
        logger.exception("Inference failed for %s.", module)
        raise APIError(500, "inference_error", "The inference result could not be produced.") from exc
    runtime.mark_ready(module)
    return result
