# ocr_module.py
"""
Обёртка над EasyOCR для извлечения текста из изображений.
Поддерживает русский язык и кэширование результатов для скорости.
"""

import cv2
import easyocr
from typing import List, Tuple, Optional
import hashlib
import os
import pickle
import numpy as np

class OCRExtractor:
    def __init__(self, languages: List[str] = ['ru', 'en'],
                 cache_dir: str = "./ocr_cache",
                 use_gpu: bool = True):
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, image_bytes: bytes) -> str:
        """Хэш изображения для кэширования."""
        img_hash = hashlib.md5(image_bytes).hexdigest()
        return os.path.join(self.cache_dir, f"{img_hash}.pkl")

    def extract_text(self, image: np.ndarray,
                     image_bytes: Optional[bytes] = None) -> Tuple[str, float]:
        """
        Извлечь текст из изображения (BGR формат).
        Возвращает строку и среднюю уверенность.
        Использует кэш, если передан image_bytes.
        """
        if image_bytes:
            cache_path = self._get_cache_path(image_bytes)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

        # EasyOCR ожидает RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(img_rgb, detail=1, paragraph=False)

        if not results:
            text, confidence = "", 0.0
        else:
            texts = [res[1] for res in results]
            confidences = [res[2] for res in results]
            text = " ".join(texts)
            confidence = sum(confidences) / len(confidences)

        result = (text, confidence)
        if image_bytes:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        return result

    def extract_text_batch(self, images: List[np.ndarray],
                           image_bytes_list: Optional[List[bytes]] = None) -> List[Tuple[str, float]]:
        """Пакетное извлечение текста (может быть медленным, лучше по одному с кэшем)."""
        results = []
        for i, img in enumerate(images):
            bts = image_bytes_list[i] if image_bytes_list else None
            results.append(self.extract_text(img, bts))
        return results