# vision_module.py
"""
Извлечение визуальных признаков с помощью EfficientNet/ResNet и детекция объектов YOLO.
"""

import threading

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Set
import cv2

class VisionExtractor:
    def __init__(self, encoder_name: str = "efficientnet_b0",
                 yolo_model_path: str = "yolo11n.pt",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Инициализация энкодера признаков
        if encoder_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.encoder = models.efficientnet_b0(weights=weights)
            self.feature_dim = 1280
        elif encoder_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            self.encoder = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        # Убираем классификационный слой
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.encoder.to(device)
        self.encoder.eval()

        # YOLO не потокобезопасен — все вызовы через _model_lock
        self._model_lock = threading.Lock()
        self.yolo = YOLO(yolo_model_path)
        with self._model_lock:
            # Однократная инициализация (fuse), чтобы не падать в ThreadPoolExecutor
            self.yolo.predict(
                np.zeros((64, 64, 3), dtype=np.uint8),
                verbose=False,
                device=self.device,
            )

        # Преобразования для входного тензора
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Извлечь эмбеддинг из тензора изображения (B, C, H, W).
        Возвращает numpy вектор размерности (feature_dim,).
        """
        with self._model_lock:
            image_tensor = image_tensor.to(self.device)
            features = self.encoder(image_tensor)
            features = features.squeeze(-1).squeeze(-1)
            features = features.cpu().numpy()
        # B=1 → (D,), B>1 → (B, D)
        if features.ndim == 2 and features.shape[0] == 1:
            features = features.squeeze(0)
        return features

    def detect_objects(self, image: np.ndarray,
                       conf_threshold: float = 0.4) -> Set[str]:
        """
        Детектировать объекты YOLO на изображении (BGR).
        Возвращает множество имён обнаруженных классов (на английском).
        """
        # YOLO ожидает RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self._model_lock:
            results = self.yolo(img_rgb, verbose=False, device=self.device)
        detected = set()
        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = box.conf.item()
                if conf >= conf_threshold:
                    cls_id = int(box.cls.item())
                    cls_name = self.yolo.names[cls_id]
                    detected.add(cls_name)
        return detected

    def process_image(self, image: np.ndarray,
                      image_tensor: torch.Tensor,
                      conf_threshold: float = 0.4) -> Tuple[np.ndarray, Set[str]]:
        """
        Комбинированная обработка одного изображения: признаки и объекты.
        """
        features = self.extract_features(image_tensor.unsqueeze(0))
        objects = self.detect_objects(image, conf_threshold)
        return features, objects

    def process_batch(self, image_tensors: torch.Tensor,
                      images_np: List[np.ndarray],
                      conf_threshold: float = 0.4) -> Tuple[np.ndarray, List[Set[str]]]:
        """
        Пакетная обработка для ускорения.
        """
        features = self.extract_features(image_tensors)  # (B, D)
        objects_list = [self.detect_objects(img, conf_threshold) for img in images_np]
        return features, objects_list