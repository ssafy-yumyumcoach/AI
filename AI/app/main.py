from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, validator
import torch

from torch.nn.modules.container import Sequential

from ultralytics.nn.modules import conv as yolo_conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from PIL import Image

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_path: Path = Path("models/best.pt")
    class_map_path: Path = Path("models/classes.yaml")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    api_prefix: str = "/api/v1"

    class Config:
        env_prefix = "AI_"
        env_file = ".env"

    @validator("api_prefix")
    def ensure_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            return "/" + value
        return value


def load_class_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        logger.warning("Class map YAML not found at %s. Falling back to numeric labels.", path)
        return {}

    with path.open("r", encoding="utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file) or {}

    names = content.get("names", content if isinstance(content, dict) else None)
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}

    logger.warning("Unexpected class map structure in %s. Falling back to numeric labels.", path)
    return {}


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    class_id: int
    label: str
    confidence: float
    box: BoundingBox


class PredictionResponse(BaseModel):
    results: List[DetectionResult]


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Optional[YOLO] = None
        self.class_map: Dict[int, str] = {}

    def load(self) -> None:
        self.class_map = load_class_map(self.settings.class_map_path)
        if not self.settings.model_path.exists():
            logger.error("YOLO model file not found at %s", self.settings.model_path)
            return

        try:
            add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
            if callable(add_safe_globals):

                add_safe_globals([DetectionModel, Sequential, yolo_conv.Conv])

            else:
                logger.debug("torch.serialization.add_safe_globals is unavailable; proceeding without allowlist.")

            self.model = YOLO(str(self.settings.model_path))
            logger.info("YOLO model loaded from %s", self.settings.model_path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load YOLO model from %s", self.settings.model_path)
            self.model = None

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        if self.model is None:
            raise HTTPException(status_code=503, detail="YOLO model is not loaded. Check model_path setting.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:  # noqa: BLE001
            logger.exception("Invalid image data provided")
            raise HTTPException(status_code=400, detail="이미지 파일을 열 수 없습니다.")

        results = self.model.predict(image, conf=self.settings.confidence_threshold, verbose=False)
        detections: List[DetectionResult] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = self.class_map.get(class_id, str(class_id))
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    DetectionResult(
                        class_id=class_id,
                        label=label,
                        confidence=confidence,
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        return PredictionResponse(results=detections)


def create_app() -> FastAPI:
    settings = Settings()
    registry = ModelRegistry(settings)

    app = FastAPI(title="YumYumCoach AI Gateway", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    @app.on_event("startup")
    async def _load_model() -> None:
        registry.load()

    @app.get("/health")
    async def health() -> dict[str, str]:
        status = "ready" if registry.model else "model_missing"
        return {"status": status}

    @app.post(f"{settings.api_prefix}/detect", response_model=PredictionResponse)
    async def detect(image: UploadFile = File(...)) -> PredictionResponse:
        if image.content_type is None or not image.content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="이미지 파일을 업로드해주세요.")

        image_bytes = await image.read()
        return registry.predict(image_bytes)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
