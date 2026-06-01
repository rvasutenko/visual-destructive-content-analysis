# vk_loader.py
"""
Загрузка изображений и метаданных из VK через API.
Поддерживает как стену сообщества, так и альбомы.
"""

import requests
import time
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VKLoader:
    def __init__(self, access_token: str, api_version: str = "5.131"):
        self.access_token = access_token
        self.api_version = api_version
        self.base_url = "https://api.vk.com/method/"

    def _make_request(self, method: str, params: Dict[str, Any]) -> Dict:
        """Синхронный запрос к VK API."""
        params.update({
            "access_token": self.access_token,
            "v": self.api_version
        })
        response = requests.get(f"{self.base_url}{method}", params=params).json()
        if "error" in response:
            raise Exception(f"VK API error: {response['error']}")
        return response

    async def _async_make_request(self, session: aiohttp.ClientSession,
                                  method: str, params: Dict[str, Any]) -> Dict:
        """Асинхронный запрос к VK API."""
        params.update({
            "access_token": self.access_token,
            "v": self.api_version
        })
        async with session.get(f"{self.base_url}{method}", params=params) as resp:
            data = await resp.json()
            if "error" in data:
                raise Exception(f"VK API error: {data['error']}")
            return data

    def get_photos_from_wall(self, owner_id: int, count: int = 100,
                             offset: int = 0) -> List[Dict]:
        """
        Получить фотографии со стены сообщества/пользователя.
        Возвращает список словарей с полями: url, text (подпись), date, owner_id.
        """
        params = {
            "owner_id": owner_id,
            "count": count,
            "offset": offset,
            "extended": 1,
            "filter": "owner"
        }
        resp = self._make_request("wall.get", params)
        items = resp.get("response", {}).get("items", [])
        photos = []
        for post in items:
            # Пост может содержать несколько вложений
            if "attachments" in post:
                for att in post["attachments"]:
                    if att["type"] == "photo":
                        sizes = att["photo"]["sizes"]
                        # Выбираем максимальное разрешение
                        max_size = max(sizes, key=lambda x: x["width"] * x["height"])
                        photos.append({
                            "url": max_size["url"],
                            "post_id": post.get("id"),
                            "post_text": post.get("text", ""),
                            "date": post.get("date"),
                            "owner_id": owner_id,
                            "group_name": None,
                        })
        return photos

    def resolve_owner_id(self, screen_name: str) -> int:
        """
        Преобразует короткое имя (durov, club123, public456) в owner_id.
        Для групп owner_id отрицательный.
        """
        name = screen_name.strip().lstrip("@")
        if name.lstrip("-").isdigit():
            return int(name)
        resp = self._make_request("utils.resolveScreenName", {"screen_name": name})
        obj = resp.get("response", {})
        if not obj:
            raise ValueError(f"Не найден объект VK: {screen_name!r}")
        oid = obj["object_id"]
        if obj.get("type") in ("group", "page", "event"):
            return -oid
        return oid

    async def get_photos_async_batch(self, owner_ids: List[int],
                                     count_per_owner: int = 50) -> List[Dict]:
        """
        Асинхронная пакетная загрузка фотографий из нескольких источников.
        Используется для масштабируемости.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for oid in owner_ids:
                params = {"owner_id": oid, "count": count_per_owner, "extended": 1, "filter": "owner"}
                tasks.append(self._async_make_request(session, "wall.get", params))
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_photos = []
        for oid, resp in zip(owner_ids, results):
            if isinstance(resp, Exception):
                logger.error(f"Failed to fetch owner {oid}: {resp}")
                continue
            items = resp.get("response", {}).get("items", [])
            for post in items:
                if "attachments" in post:
                    for att in post["attachments"]:
                        if att["type"] == "photo":
                            sizes = att["photo"]["sizes"]
                            max_size = max(sizes, key=lambda x: x["width"] * x["height"])
                            all_photos.append({
                                "url": max_size["url"],
                                "post_text": post.get("text", ""),
                                "date": post.get("date"),
                                "owner_id": oid
                            })
        return all_photos

    def download_image(self, url: str, save_path: Optional[str] = None) -> bytes:
        """Скачать изображение по URL, вернуть байты."""
        resp = requests.get(url, stream=True)
        if resp.status_code == 200:
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(resp.content)
            return resp.content
        else:
            raise Exception(f"Failed to download image: {resp.status_code}")