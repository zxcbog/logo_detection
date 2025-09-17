import io
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO


app = FastAPI(
    title="YOLO Logo Detection API",
    description="API для детекции логотипа Т-Банка на изображениях с использованием YOLOv8",
    version="1.0.0",
    swagger_ui_parameters={
        "syntaxHighlight": True,
        "syntaxHighlight.theme": "obsidian",
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
        "persistAuthorization": True,
        "docExpansion": "none",
    }
)

# Загрузка модели при старте приложения
model = YOLO("best.pt")  # Используйте свою обученную модель для логотипов

class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Кастомная страница Swagger UI
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """
    Эндпоинт для получения OpenAPI схемы
    """
    return JSONResponse(app.openapi())

# Ваш эндпоинт /detect остается без изменений
@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    # Проверка формата файла
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат изображения. Используйте JPEG, PNG, BMP или WEBP"
        )

    try:
        # Чтение и декодирование изображения
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Выполнение инференса
        results = model(image)
        detections = []
        
        # Обработка результатов
        for result in results:
            for box in result.boxes:
                # Конвертация координат в абсолютные значения
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                
                # Фильтрация по классу (настройте под вашу модель)
                # if box.cls == target_class_idx:  # Раскомментируйте если нужно фильтровать по классу
                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max
                        )
                    )
                )
        
        return DetectionResponse(detections=detections)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке изображения: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)