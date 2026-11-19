"""
Command autopilot exposes an API to decide what to do in the game.
"""

from pathlib import Path
from typing import Tuple

import torch, uvicorn
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

from stig.internal.env import env
from stig.internal.dataset.image import process_from_bytes, to_tensor
from stig.internal.game.action import THROTTLE_VALUES_MAP, STEERING_VALUES_MAP


class ActResp(BaseModel):
    throttle: str
    steering: str


def _detect_in_channels(model: torch.nn.Module) -> int:
    for param in model.parameters():
        if param.ndim == 4:
            return int(param.shape[1])
    return 3


def create_app(
    model_name: str,
    models_dir: str,
    frame_size: Tuple[int, int],
    device: str,
):
    """
    Autopilot creates an API server to interact with the AI model.
    """
    model_path = Path(models_dir) / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found")

    model = torch.jit.load(str(model_path), map_location=device).to(device)
    model.eval()
    grayscale = _detect_in_channels(model) == 1

    app = FastAPI(title="Stig Autopilot")

    @app.post("/act")
    async def act(
        img_bytes: bytes = Body(..., media_type="image/jpeg")
    ) -> ActResp:
        try:
            img = process_from_bytes(
                img_bytes, frame_size, device=device, grayscale=grayscale
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            tensor = to_tensor(img, device)
            with torch.inference_mode():
                throttle_logits, steering_logits = model(tensor)
            throttle_id = int(throttle_logits.argmax())
            steering_id = int(steering_logits.argmax())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return ActResp(
            throttle=THROTTLE_VALUES_MAP[throttle_id],
            steering=STEERING_VALUES_MAP[steering_id],
        )

    return app


if __name__ == "__main__":
    model_name = env.lookup("MODEL_NAME")
    models_dir = env.lookup("MODELS_DIR")

    frame_height = env.lookupInt("FRAME_HEIGHT")
    frame_width = env.lookupInt("FRAME_WIDTH")
    frame_size = (frame_height, frame_width)

    api_host = env.lookup("API_HOST")
    api_port = env.lookupInt("API_PORT")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    app = create_app(
        model_name,
        models_dir,
        frame_size,
        device,
    )

    uvicorn.run(app, host=api_host, port=api_port)
