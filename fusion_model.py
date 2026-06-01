# fusion_model.py
"""
Мультимодальная модель слияния признаков для финальной классификации.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set

class MultiModalFusion(nn.Module):
    def __init__(self,
                 vision_dim: int = 1280,    # EfficientNet-B0
                 text_embed_dim: int = 768,  # RuBERT base
                 object_vocab_size: int = 80, # количество классов YOLO (COCO)
                 hidden_dim: int = 512,
                 dropout: float = 0.3,
                 num_classes: int = 4):
        super().__init__()

        # Вектор объектов: мульти-хот кодирование присутствия объектов
        self.object_vocab_size = object_vocab_size
        self.num_classes = num_classes

        # Проекционные слои
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.object_proj = nn.Linear(object_vocab_size, hidden_dim)

        # Объединённый слой (последний Linear: индекс 6 в Sequential)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _objects_to_multihot(self, objects_set: Set[str],
                             class_to_idx: Dict[str, int]) -> torch.Tensor:
        """Преобразует множество имён объектов в мульти-хот вектор."""
        vec = torch.zeros(self.object_vocab_size)
        for obj in objects_set:
            if obj in class_to_idx:
                vec[class_to_idx[obj]] = 1.0
        return vec

    def forward(self, vision_features: torch.Tensor,
                text_embedding: torch.Tensor,
                object_sets: List[Set[str]],
                class_to_idx: Dict[str, int]) -> torch.Tensor:
        """
        vision_features: (B, vision_dim)
        text_embedding: (B, text_embed_dim)
        object_sets: список множеств имён объектов
        Возвращает логиты (B, num_classes)
        """
        if vision_features.dim() == 3:
            vision_features = vision_features.squeeze(1)
        if text_embedding.dim() == 3:
            text_embedding = text_embedding.squeeze(1)

        # Проекции
        v = F.relu(self.vision_proj(vision_features))
        t = F.relu(self.text_proj(text_embedding))

        # Обработка объектов
        obj_vectors = torch.stack([
            self._objects_to_multihot(objs, class_to_idx)
            for objs in object_sets
        ]).to(vision_features.device)
        o = F.relu(self.object_proj(obj_vectors))

        # Конкатенация и классификация
        fused = torch.cat([v, t, o], dim=1)
        logits = self.fusion_layer(fused)
        return logits

    def predict_probs(self, vision_features: torch.Tensor,
                      text_embedding: torch.Tensor,
                      object_sets: List[Set[str]],
                      class_to_idx: Dict[str, int]) -> torch.Tensor:
        """Softmax по всем классам, форма (B, num_classes)."""
        logits = self.forward(vision_features, text_embedding, object_sets, class_to_idx)
        return F.softmax(logits, dim=1)

    def predict_proba(self, vision_features: torch.Tensor,
                      text_embedding: torch.Tensor,
                      object_sets: List[Set[str]],
                      class_to_idx: Dict[str, int]) -> torch.Tensor:
        """Обратная совместимость: P(любой деструктивный) = 1 - P(harmless)."""
        p = self.predict_probs(vision_features, text_embedding, object_sets, class_to_idx)
        return 1.0 - p[:, 0]