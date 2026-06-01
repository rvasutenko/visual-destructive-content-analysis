# pipeline.py
"""
Основной пайплайн обработки изображения из VK:
загрузка -> предобработка -> извлечение признаков -> слияние -> вердикт.
"""

import os
import json
import asyncio
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import cv2

from config import *
from vk_loader import VKLoader
from preprocessing import load_image_from_bytes, preprocess_image, enhance_contrast
from ocr_module import OCRExtractor
from vision_module import VisionExtractor
from text_module import RuBERTAnalyzer
from fusion_model import MultiModalFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_1d_feature(x) -> torch.Tensor:
    """Приводит эмбеддинг к вектору (D,) для stack в батч (B, D)."""
    t = torch.as_tensor(x, dtype=torch.float32)
    while t.dim() > 1:
        t = t.squeeze(0)
    return t


class DestructiveContentPipeline:
    def __init__(self, config_module):
        self.config = config_module
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Инициализация компонентов
        self.vk_loader = VKLoader(config_module.VK_ACCESS_TOKEN, config_module.VK_API_VERSION)
        self.ocr = OCRExtractor(use_gpu=(self.device == "cuda"))
        self.vision = VisionExtractor(
            encoder_name=config_module.VISION_ENCODER,
            yolo_model_path=config_module.YOLO_MODEL,
            device=self.device
        )
        self.text_analyzer = RuBERTAnalyzer(model_name=config_module.RUBERT_MODEL_NAME, device=self.device)

        # Загрузка мультимодальной модели (если есть сохранённая)
        self.fusion_model = MultiModalFusion(
            vision_dim=self.vision.feature_dim,
            text_embed_dim=self.text_analyzer.model.config.hidden_size,
            num_classes=getattr(config_module, "FUSION_NUM_CLASSES", 4),
        ).to(self.device)
        if os.path.exists(config_module.FUSION_MODEL_PATH):
            try:
                try:
                    state = torch.load(
                        config_module.FUSION_MODEL_PATH,
                        map_location=self.device,
                        weights_only=True,
                    )
                except TypeError:
                    state = torch.load(config_module.FUSION_MODEL_PATH, map_location=self.device)
                self.fusion_model.load_state_dict(state, strict=True)
                logger.info("Loaded pre-trained fusion model.")
            except Exception as e:
                logger.warning(
                    "Fusion checkpoint не подошёл (другая архитектура или число классов): %s. "
                    "Используются случайные веса — дообучите train.py.",
                    e,
                )
        else:
            logger.warning("Fusion model not found. Using random weights (not reliable).")
        self.fusion_model.eval()

        # Словарь классов YOLO -> индекс (для мульти-хот)
        self.yolo_class_to_idx = {name: i for i, name in self.vision.yolo.names.items()}

        # Загрузка внешних словарей (междисциплинарность)
        self.toxic_lexicon = self._load_lexicon(config_module.TOXIC_LEXICON_PATH)
        self.banned_symbols = self._load_lexicon(config_module.BANNED_SYMBOLS_PATH)

    def _load_lexicon(self, path: str) -> Optional[set]:
        """Загрузка словаря токсичной лексики от лингвистов."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get("words", []))
        return None

    def _apply_context_heuristics(self, post_text: str, extracted_text: str,
                                  objects: set) -> Dict[str, Any]:
        """
        Эвристики для уменьшения ложных срабатываний:
        - проверка на новостной контент
        - учёт названия сообщества (если передано)
        """
        heuristics = {"is_news": False, "context_risk": "neutral"}
        combined_text = (post_text + " " + extracted_text).lower()
        if any(kw in combined_text for kw in NEUTRAL_NEWS_KEYWORDS):
            # Дополнительно проверим объекты: микрофон, камера и т.п.
            news_objects = {"microphone", "cell phone", "laptop", "book"}
            if objects.intersection(news_objects):
                heuristics["is_news"] = True
                heuristics["context_risk"] = "likely_news"
        # Проверка по словарю токсичной лексики (если загружен)
        if self.toxic_lexicon:
            words = set(combined_text.split())
            if words.intersection(self.toxic_lexicon):
                heuristics["toxic_lexicon_match"] = True
        return heuristics

    def process_single_image(self, image_bytes: bytes,
                             post_text: str = "",
                             group_name: str = "") -> Dict[str, Any]:
        """
        Обработка одного изображения.
        Возвращает словарь с результатами.
        """
        # Загрузка и предобработка
        img_bgr = load_image_from_bytes(image_bytes)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(preprocess_image(img_bgr, IMAGE_SIZE)).float()

        # OCR на улучшенном контрасте
        enhanced = enhance_contrast(img_bgr)
        extracted_text, ocr_conf = self.ocr.extract_text(enhanced, image_bytes)

        # Vision
        vis_features, objects = self.vision.process_image(
            img_bgr, img_tensor, conf_threshold=YOLO_CONF_THRESHOLD
        )

        # NLP
        # Анализируем объединённый текст: пост + извлечённый OCR
        full_text = f"{post_text} {extracted_text}".strip()
        toxic_score, text_emb = self.text_analyzer.analyze_text(full_text)

        # Эвристики контекста
        heuristics = self._apply_context_heuristics(post_text, extracted_text, objects)

        # Мультимодальное слияние (vision уже даёт батч (1, D); лишний unsqueeze давал (1, 1, D))
        vis_tensor = torch.from_numpy(vis_features).float().to(self.device)
        if vis_tensor.dim() == 1:
            vis_tensor = vis_tensor.unsqueeze(0)
        text_emb_tensor = torch.from_numpy(text_emb).float().to(self.device)
        if text_emb_tensor.dim() == 1:
            text_emb_tensor = text_emb_tensor.unsqueeze(0)
        with torch.no_grad():
            probs = self.fusion_model.predict_probs(
                vis_tensor, text_emb_tensor, [objects], self.yolo_class_to_idx
            )
        probs_row = probs[0]
        raw_pred = int(probs_row.argmax().item())
        raw_conf = float(probs_row[raw_pred].item())
        is_destructive = raw_pred != 0 and raw_conf >= CONFIDENCE_THRESHOLD
        effective_idx = raw_pred
        if heuristics.get("is_news", False) and raw_pred != 0 and raw_conf < 0.85:
            is_destructive = False
            effective_idx = 0
            heuristics = {**heuristics, "news_override": True}

        content_category = CONTENT_CLASS_NAMES[effective_idx]
        destructive_type = None if effective_idx == 0 else CONTENT_CLASS_NAMES[effective_idx]
        confidence = float(probs_row[effective_idx].item())
        class_probabilities = {
            CONTENT_CLASS_NAMES[j]: float(probs_row[j].item())
            for j in range(len(CONTENT_CLASS_NAMES))
        }

        # Проверка по словарю запрещённой символики (пример интеграции)
        if self.banned_symbols:
            # В реальности нужна детекция символов, здесь заглушка
            pass

        return {
            "is_destructive": is_destructive,
            "confidence": confidence,
            "content_category": content_category,
            "destructive_type": destructive_type,
            "class_probabilities": class_probabilities,
            "detected_objects": list(objects),
            "extracted_text": extracted_text,
            "text_toxicity_score": float(toxic_score),
            "context_risk": heuristics.get("context_risk", "neutral"),
            "heuristics": heuristics,
        }

    def process_batch(self, image_items: List[Dict]) -> List[Dict]:
        """
        Пакетная обработка списка изображений (каждый элемент - словарь с ключами:
        'image_bytes', 'post_text', 'group_name').
        Использует многопоточность для CPU-bound операций (OCR) и пакетную обработку GPU.
        """
        results = []
        # YOLO, EasyOCR и RuBERT небезопасны при параллельном доступе из потоков
        if NUM_WORKERS and NUM_WORKERS > 0:
            logger.warning(
                "NUM_WORKERS=%s: ML-модели вызываются последовательно (потокобезопасность).",
                NUM_WORKERS,
            )
        intermediate = [self._preprocess_and_ocr(item) for item in image_items]

        # Теперь пакетная обработка Vision и NLP
        # Готовим батчи
        vis_tensors = []
        text_embs = []
        objects_list = []
        for data in intermediate:
            vis_tensors.append(_to_1d_feature(data["vis_features"]))
            text_embs.append(_to_1d_feature(data["text_emb"]))
            objects_list.append(data["objects"])

        vis_batch = torch.stack(vis_tensors).to(self.device)
        text_batch = torch.stack(text_embs).to(self.device)

        with torch.no_grad():
            probs_b = self.fusion_model.predict_probs(
                vis_batch, text_batch, objects_list, self.yolo_class_to_idx
            ).cpu()

        for i, data in enumerate(intermediate):
            probs_row = probs_b[i]
            raw_pred = int(probs_row.argmax().item())
            raw_conf = float(probs_row[raw_pred].item())
            is_destructive = raw_pred != 0 and raw_conf >= CONFIDENCE_THRESHOLD
            effective_idx = raw_pred
            heur = data["heuristics"]
            if heur.get("is_news", False) and raw_pred != 0 and raw_conf < 0.85:
                is_destructive = False
                effective_idx = 0
                heur = {**heur, "news_override": True}

            content_category = CONTENT_CLASS_NAMES[effective_idx]
            destructive_type = None if effective_idx == 0 else CONTENT_CLASS_NAMES[effective_idx]
            confidence = float(probs_row[effective_idx].item())
            class_probabilities = {
                CONTENT_CLASS_NAMES[j]: float(probs_row[j].item())
                for j in range(len(CONTENT_CLASS_NAMES))
            }

            results.append({
                "is_destructive": bool(is_destructive),
                "confidence": confidence,
                "content_category": content_category,
                "destructive_type": destructive_type,
                "class_probabilities": class_probabilities,
                "detected_objects": list(data["objects"]),
                "extracted_text": data["extracted_text"],
                "text_toxicity_score": float(data["toxic_score"]),
                "context_risk": heur.get("context_risk", "neutral"),
                "heuristics": heur,
            })
        return results

    def _preprocess_and_ocr(self, item: Dict) -> Dict:
        """Вспомогательный метод для параллельной предобработки и OCR."""
        img_bytes = item['image_bytes']
        post_text = item.get('post_text', '')
        group_name = item.get('group_name', '')

        img_bgr = load_image_from_bytes(img_bytes)
        img_tensor = torch.from_numpy(preprocess_image(img_bgr, IMAGE_SIZE)).float()

        enhanced = enhance_contrast(img_bgr)
        extracted_text, ocr_conf = self.ocr.extract_text(enhanced, img_bytes)

        vis_features, objects = self.vision.process_image(
            img_bgr, img_tensor, conf_threshold=YOLO_CONF_THRESHOLD
        )

        full_text = f"{post_text} {extracted_text}".strip()
        toxic_score, text_emb = self.text_analyzer.analyze_text(full_text)

        heuristics = self._apply_context_heuristics(post_text, extracted_text, objects)

        return {
            'vis_features': vis_features,
            'text_emb': text_emb,
            'objects': objects,
            'extracted_text': extracted_text,
            'toxic_score': toxic_score,
            'heuristics': heuristics
        }

    async def process_from_vk_wall(self, owner_id: int, count: int = 10) -> List[Dict]:
        """
        Асинхронно загружает последние изображения со стены сообщества и анализирует их.
        """
        # Загружаем метаданные и URL
        photos_meta = self.vk_loader.get_photos_from_wall(owner_id, count)
        # Скачиваем изображения (синхронно для простоты, но можно асинхронно)
        items = []
        for photo in photos_meta:
            img_bytes = self.vk_loader.download_image(photo['url'])
            items.append({
                'image_bytes': img_bytes,
                'post_text': photo['post_text'],
                'group_name': photo.get('group_name', '')
            })
        # Обрабатываем батч
        results = self.process_batch(items)
        for res, meta in zip(results, photos_meta):
            res["image_url"] = meta["url"]
            res["post_id"] = meta.get("post_id")
            res["owner_id"] = meta.get("owner_id")
            res["post_date"] = meta.get("date")
            res["post_text_meta"] = meta.get("post_text", "")
        return results