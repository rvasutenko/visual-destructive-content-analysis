# preprocessing.py
"""
Предобработка изображений: нормализация, изменение размера, контрастирование.
"""

import cv2
import numpy as np
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Загрузить изображение из байтов в массив numpy (BGR)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224),
                     normalize: bool = True) -> np.ndarray:
    """
    Базовая предобработка: изменение размера, конвертация BGR->RGB, нормализация.
    Возвращает массив float32 в формате CHW, готовый для PyTorch.
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    if normalize:
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # CHW
    return img

def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Аугментации для повышения устойчивости к маскировке.
    Используется при обучении.
    """
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.CLAHE(p=0.2),
        A.Rotate(limit=15, p=0.5),
    ])
    augmented = transform(image=image)["image"]
    return augmented

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Улучшение контраста для улучшения читаемости OCR."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced