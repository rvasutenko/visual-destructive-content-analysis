#!/usr/bin/env python3
"""
Апробация обученной модели на постах «ВКонтакте» с сохранением результатов в файл.

Примеры:
  .venv/bin/python run_vk_aprobation.py --owner -123456789 --posts 20
  .venv/bin/python run_vk_aprobation.py --screen mash --posts 30 -o results/vk_mash.jsonl
  VK_ACCESS_TOKEN=... .venv/bin/python run_vk_aprobation.py --owner -123456789 --posts 10

Нужен токен VK с доступом к wall (см. README в комментариях к --help).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from pipeline import DestructiveContentPipeline


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def flatten_result(meta: dict, analysis: dict) -> dict:
    """Одна строка отчёта = одно изображение из поста."""
    probs = analysis.get("class_probabilities") or {}
    row = {
        "owner_id": meta.get("owner_id"),
        "post_id": meta.get("post_id"),
        "post_date": meta.get("date"),
        "post_text": (meta.get("post_text") or "")[:2000],
        "image_url": meta.get("url") or analysis.get("image_url"),
        "is_destructive": analysis.get("is_destructive"),
        "content_category": analysis.get("content_category"),
        "destructive_type": analysis.get("destructive_type"),
        "confidence": analysis.get("confidence"),
        "text_toxicity_score": analysis.get("text_toxicity_score"),
        "extracted_text": analysis.get("extracted_text"),
        "detected_objects": analysis.get("detected_objects"),
        "context_risk": analysis.get("context_risk"),
        "prob_harmless": probs.get("harmless"),
        "prob_violence": probs.get("violence"),
        "prob_pornography": probs.get("pornography"),
        "prob_terrorism": probs.get("terrorism"),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    return row


def save_results(rows: list[dict], output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    else:
        with output.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Сохранено записей: {len(rows)} → {output}")


def run_aprobation(
    owner_id: int,
    posts_count: int,
    output: Path,
    fmt: str,
) -> list[dict]:
    if not config.VK_ACCESS_TOKEN or config.VK_ACCESS_TOKEN == "YOUR_TOKEN_HERE":
        raise SystemExit(
            "Не задан VK_ACCESS_TOKEN. Укажите токен в config.py или:\n"
            "  export VK_ACCESS_TOKEN='ваш_токен'"
        )

    if not Path(config.FUSION_MODEL_PATH).is_file():
        print(
            f"Предупреждение: нет весов {config.FUSION_MODEL_PATH}. "
            "Сначала обучите: .venv/bin/python train.py",
            file=sys.stderr,
        )

    print(f"Загрузка модели и анализ стены owner_id={owner_id}, постов (запрос): {posts_count}…")
    pipeline = DestructiveContentPipeline(config)
    results = asyncio.run(pipeline.process_from_vk_wall(owner_id, count=posts_count))

    if not results:
        print("На стене не найдено постов с вложениями-фото в запрошенном диапазоне.")
        save_results([], output, fmt)
        return []

    rows = []
    for analysis in results:
        meta = {
            "owner_id": analysis.get("owner_id", owner_id),
            "post_id": analysis.get("post_id"),
            "date": analysis.get("post_date"),
            "post_text": analysis.get("post_text_meta") or analysis.get("extracted_text", ""),
            "url": analysis.get("image_url"),
        }
        rows.append(flatten_result(meta, analysis))

    save_results(rows, output, fmt)
    destructive = sum(1 for r in rows if r.get("is_destructive"))
    print(f"Итого изображений: {len(rows)}, помечено деструктивными: {destructive}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Апробация модели на фото из постов VK (wall.get)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Токен VK:
  • Нужен access_token приложения VK (https://dev.vk.com).
  • Для чтения стены сообщества обычно подходит пользовательский токен
    с правами offline (и доступ к wall у открытого сообщества).
  • Сервисный ключ часто НЕ может вызывать wall.get для произвольных групп.
  • owner_id группы — отрицательное число (club123 → -123).

Как узнать owner_id:
  • В URL vk.com/club123456 → owner_id = -123456
  • Или: --screen mash (короткое имя паблика)
        """,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--owner", type=int, help="owner_id (для группы — отрицательный)")
    g.add_argument("--screen", type=str, help="Короткое имя: mash, durov, club123…")

    parser.add_argument(
        "--posts",
        type=int,
        default=20,
        help="Сколько последних постов запросить у wall.get (не все содержат фото)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results") / f"vk_aprobation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        help="Файл результатов (.jsonl или .json)",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default=None,
        help="Формат файла (по умолчанию по расширению -o)",
    )
    args = parser.parse_args()

    fmt = args.format
    if fmt is None:
        fmt = "jsonl" if args.output.suffix.lower() == ".jsonl" else "json"

    owner_id = args.owner
    if args.screen:
        loader = __import__("vk_loader").VKLoader(config.VK_ACCESS_TOKEN, config.VK_API_VERSION)
        owner_id = loader.resolve_owner_id(args.screen)
        print(f"screen_name={args.screen!r} → owner_id={owner_id}")

    run_aprobation(owner_id, args.posts, args.output, fmt)


if __name__ == "__main__":
    main()
