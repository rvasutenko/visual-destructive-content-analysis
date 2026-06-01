# text_module.py
"""
Анализ текста с помощью RuBERT для оценки токсичности и контекста.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Tuple, Optional

class RuBERTAnalyzer:
    def __init__(self, model_name: str = "DeepPavlov/rubert-base-cased",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Используем модель для классификации токсичности (можно дообучить)
        # Для демонстрации используем готовую модель для sentiment, но в реальности нужна fine-tuned на токсичность.
        # В рамках задания показываем структуру.
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "SkolkovoInstitute/russian_toxicity_classifier"  # гипотетическая модель
            )
        except:
            # Fallback на базовую модель sentiment, если специализированная недоступна
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "blanchefort/rubert-base-cased-sentiment"
            )
        self.model.to(device)
        self.model.eval()

    def analyze_text(self, text: str) -> Tuple[float, np.ndarray]:
        """
        Оценивает токсичность текста.
        Возвращает (score_toxicity, embedding_vector).
        embedding_vector - эмбеддинг [CLS] токена для мультимодального слияния.
        """
        if not text.strip():
            # Если текста нет, возвращаем нулевую токсичность и нулевой вектор
            emb = np.zeros(self.model.config.hidden_size)
            return 0.0, emb

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            # Для бинарной классификации токсичности (предположим индекс 1 - toxic)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            toxic_score = probs[1] if probs.shape[0] > 1 else probs[0]

            # Извлекаем эмбеддинг из последнего слоя (CLS токен)
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
            cls_embedding = hidden_states[:, 0, :].squeeze().cpu().numpy()

        return toxic_score, cls_embedding

    def analyze_batch(self, texts: List[str]) -> Tuple[List[float], np.ndarray]:
        """
        Пакетный анализ списка текстов.
        Возвращает список scores и матрицу эмбеддингов (N, D).
        """
        scores = []
        embeddings = []
        for text in texts:
            score, emb = self.analyze_text(text)
            scores.append(score)
            embeddings.append(emb)
        embeddings = np.stack(embeddings, axis=0)
        return scores, embeddings