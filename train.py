# train.py
"""
Дообучение мультимодальной модели слияния (vision + текстовый эмбеддинг + YOLO-объекты).

Классы (мультикласс):
  0 harmless — безопасный контент
  1 violence — насилие
  2 pornography — порнография
  3 terrorism — терроризм / экстремизм

Рекомендуемая раскладка каталогов внутри TRAIN_IMAGE_ROOT (см. config.py):
  harmless/      violence/      pornography/      terrorism/

Если пока всё деструктивное в одной папке — можно временно использовать destructive/;
  такие файлы получат метку «violence» (1), пока не разнесёте по подтипам.

CSV: колонки image_filename, label (0–3), опционально text.
"""

import os
import argparse
from glob import glob

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    MODEL_SAVE_DIR,
    FUSION_MODEL_PATH,
    VISION_ENCODER,
    YOLO_MODEL,
    RUBERT_MODEL_NAME,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    EPOCHS,
    YOLO_CONF_THRESHOLD,
    TRAIN_IMAGE_ROOT,
    TRAIN_CLASS_FOLDERS,
    LEGACY_DESTRUCTIVE_FOLDER,
    LEGACY_DESTRUCTIVE_LABEL,
    CONTENT_CLASS_NAMES,
    FUSION_NUM_CLASSES,
)
from preprocessing import preprocess_image
from vision_module import VisionExtractor
from text_module import RuBERTAnalyzer
from fusion_model import MultiModalFusion

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")


def _collect_images(folder: str) -> list:
    if not os.path.isdir(folder):
        return []
    paths = []
    for pat in _IMAGE_EXTS:
        paths.extend(glob(os.path.join(folder, pat)))
    return sorted(paths)


def build_manifest_from_folders(root: str) -> pd.DataFrame:
    """
    Собирает manifest: image_filename (относительно root), label, text.
    """
    rows = []
    for folder_name, label in TRAIN_CLASS_FOLDERS.items():
        folder = os.path.join(root, folder_name)
        for path in _collect_images(folder):
            rel = os.path.join(folder_name, os.path.basename(path))
            rows.append({"image_filename": rel, "label": label, "text": ""})

    legacy_dir = os.path.join(root, LEGACY_DESTRUCTIVE_FOLDER)
    legacy_paths = _collect_images(legacy_dir)
    if legacy_paths:
        print(
            f"Замечание: найдена папка {LEGACY_DESTRUCTIVE_FOLDER}/ ({len(legacy_paths)} файлов) — "
            f"метка {LEGACY_DESTRUCTIVE_LABEL} ({CONTENT_CLASS_NAMES[LEGACY_DESTRUCTIVE_LABEL]}). "
            "Позже разнесите кадры по violence/, pornography/, terrorism/."
        )
        for path in legacy_paths:
            rel = os.path.join(LEGACY_DESTRUCTIVE_FOLDER, os.path.basename(path))
            rows.append({"image_filename": rel, "label": LEGACY_DESTRUCTIVE_LABEL, "text": ""})

    if not rows:
        expected = ", ".join(TRAIN_CLASS_FOLDERS.keys())
        raise FileNotFoundError(
            f"Не найдено изображений в {root!r}. Создайте подпапки: {expected} "
            f"(или временно {LEGACY_DESTRUCTIVE_FOLDER}/ для старого датасета)."
        )
    return pd.DataFrame(rows)


def build_manifest_from_csv(csv_path: str, images_root: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("image_filename", "label"):
        if col not in df.columns:
            raise ValueError(f"В CSV нужны колонки image_filename, label; не хватает: {col}")
    if "text" not in df.columns:
        df["text"] = ""
    if (df["label"] < 0).any() or (df["label"] >= FUSION_NUM_CLASSES).any():
        raise ValueError(f"Колонка label должна быть в диапазоне 0..{FUSION_NUM_CLASSES - 1}")
    return df


class MultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        vision_extractor: VisionExtractor,
        text_analyzer: RuBERTAnalyzer,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.vision = vision_extractor
        self.text_analyzer = text_analyzer
        self.augment = augment

        if augment:
            self.transform = A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.HorizontalFlip(p=0.3),
                    A.Rotate(limit=10, p=0.5),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_filename"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Не удалось прочитать изображение: {img_path}")

        img_tensor = torch.from_numpy(preprocess_image(img_bgr, IMAGE_SIZE)).float()

        with torch.no_grad():
            vis_features, objects = self.vision.process_image(
                img_bgr, img_tensor, conf_threshold=YOLO_CONF_THRESHOLD
            )

        text = row.get("text", "")
        if pd.isna(text):
            text = ""
        text = str(text)
        toxic_score, text_emb = self.text_analyzer.analyze_text(text)

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        vis_t = torch.from_numpy(vis_features).float()
        if vis_t.dim() == 2 and vis_t.shape[0] == 1:
            vis_t = vis_t.squeeze(0)

        return {
            "vis_features": vis_t,
            "text_emb": torch.from_numpy(text_emb).float(),
            "objects": objects,
            "label": label,
        }


def collate_fn(batch):
    vis = torch.stack([item["vis_features"] for item in batch])
    text = torch.stack([item["text_emb"] for item in batch])
    objects_list = [item["objects"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    return vis, text, objects_list, labels


def train(
    image_root: Optional[str] = None,
    csv_path: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
):
    image_root = image_root or TRAIN_IMAGE_ROOT
    epochs = epochs if epochs is not None else EPOCHS
    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    num_workers = num_workers if num_workers is not None else NUM_WORKERS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if csv_path:
        df = build_manifest_from_csv(csv_path, image_root)
    else:
        df = build_manifest_from_folders(image_root)

    print(f"Загружено примеров: всего {len(df)}")
    for li in range(FUSION_NUM_CLASSES):
        n = int((df["label"] == li).sum())
        print(f"  [{li}] {CONTENT_CLASS_NAMES[li]}: {n}")

    vision_extractor = VisionExtractor(
        encoder_name=VISION_ENCODER,
        yolo_model_path=YOLO_MODEL,
        device=device,
    )
    text_analyzer = RuBERTAnalyzer(model_name=RUBERT_MODEL_NAME, device=device)

    yolo_class_to_idx = {name: i for i, name in vision_extractor.yolo.names.items()}

    model = MultiModalFusion(
        vision_dim=vision_extractor.feature_dim,
        text_embed_dim=text_analyzer.model.config.hidden_size,
        num_classes=FUSION_NUM_CLASSES,
    ).to(device)

    if os.path.exists(FUSION_MODEL_PATH):
        try:
            try:
                w = torch.load(FUSION_MODEL_PATH, map_location=device, weights_only=True)
            except TypeError:
                w = torch.load(FUSION_MODEL_PATH, map_location=device)
            model.load_state_dict(w, strict=True)
            print(f"Продолжение с весов: {FUSION_MODEL_PATH}")
        except Exception as e:
            print(f"Чекпоинт не подошёл ({e}), обучение с нуля.")

    dataset = MultimodalDataset(
        df, image_root, vision_extractor, text_analyzer, augment=True
    )
    # На macOS с fork иногда падает multiprocessing + OpenCV; при проблемах поставьте --workers 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for vis, text_emb, objects_list, labels in loader:
            vis = vis.to(device)
            text_emb = text_emb.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(vis, text_emb, objects_list, yolo_class_to_idx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1

        avg = running / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs}, loss = {avg:.4f}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), FUSION_MODEL_PATH)
    print(f"Модель сохранена: {FUSION_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение fusion-модели по папкам или CSV")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"Корень с подпапками по классам (по умолчанию {TRAIN_IMAGE_ROOT})",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Путь к CSV (image_filename, label 0–3, опционально text); --data-root — корень картинок",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None, help="DataLoader num_workers (0 — надёжно на Mac)")
    args = parser.parse_args()
    train(
        image_root=args.data_root,
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
