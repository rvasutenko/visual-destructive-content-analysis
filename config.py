# config.py
"""
Централизованные настройки системы.
Пороги, пути к моделям, токен VK и параметры обработки.
"""

# ========== VK API ==========
import os

VK_ACCESS_TOKEN = os.environ.get(
    "VK_ACCESS_TOKEN",
    "vk1.a.cOTvNE6TozzfeYNtfWNYSTfPxhVTLf5H0DSlIKxpk-w_58mJwDv-BlaEWGWVZgxbL9jg9JvUv7P84SGcpnxKXkTos1eYMwhvJrqMP-IDoxOziFiI5uzRBdc4wka1mW2YV-00hcCx6j-EFx_fcmnd17XTpR3VhdNVAUvH0xG8J7tkh53rmOxOVk0YPj7ZVUcI1hn2GgCazxITql6xiGORiQ"
)
VK_API_VERSION = "5.131"

# ========== Пути к моделям и данным ==========
MODEL_SAVE_DIR = "./models"
VISION_ENCODER = "efficientnet_b0"  # или "resnet50"
YOLO_MODEL = "yolo11n.pt"  # YOLOv8 nano
RUBERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
FUSION_MODEL_PATH = f"{MODEL_SAVE_DIR}/fusion_model.pt"

# Датасет для train.py: папки по типу контента (имя файла любое)
TRAIN_IMAGE_ROOT = "./dataset/train"

# Классы fusion-модели (индекс = label при обучении)
CONTENT_CLASS_NAMES = ["harmless", "violence", "pornography", "terrorism"]
FUSION_NUM_CLASSES = len(CONTENT_CLASS_NAMES)

# Подпапки внутри TRAIN_IMAGE_ROOT: имя каталога -> номер класса
TRAIN_CLASS_FOLDERS = {
    "harmless": 0,
    "violence": 1,
    "pornography": 2,
    "terrorism": 3,
}

# Если ещё лежит старый каталог destructive/ — все снимки из него получают класс «violence» (1)
LEGACY_DESTRUCTIVE_FOLDER = "destructive"
LEGACY_DESTRUCTIVE_LABEL = 1

# Словари и базы (пример для междисциплинарной интеграции)
TOXIC_LEXICON_PATH = "./data/toxic_lexicon.json"   # словарь агрессивной лексики от лингвистов
BANNED_SYMBOLS_PATH = "./data/banned_symbols.json" # список запрещённой символики

# ========== Пороги и параметры ==========
CONFIDENCE_THRESHOLD = 0.7        # Порог уверенности мультимодальной модели
YOLO_CONF_THRESHOLD = 0.4         # Порог детекции объектов YOLO
TEXT_TOXICITY_THRESHOLD = 0.5     # Порог токсичности текста
DESTRUCTIVE_OBJECTS = {
    "knife", "gun", "pistol", "rifle", "weapon",
    "blood", "noose", "syringe", "explosion", "fire",
    "swastika", "extremist_symbol"
}
NEUTRAL_NEWS_KEYWORDS = {"новости", "сми", "газета", "тв", "телеканал", "интерфакс", "риа", "тасс"}

# ========== Параметры предобработки ==========
IMAGE_SIZE = (224, 224)  # для EfficientNet/ResNet
BATCH_SIZE = 32
# Потоки при инференсе: 0 — надёжно (YOLO/EasyOCR не любят параллельные вызовы на Mac)
NUM_WORKERS = 0

# ========== Параметры обучения ==========
LEARNING_RATE = 1e-4
EPOCHS = 10