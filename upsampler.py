"""Апсемплер промптов FLUX 3 Image Klein (black-forest-labs/flux-3-image-klein-pup, 9B, на базе Qwen3.5-9B).

Klein обучен на подробных описаниях сцены; короткий промпт «как есть» даёт заметно более слабый результат,
поэтому BFL просит не использовать модель без апсемплера. Он же пишет описания для правок (видит референсы)
и для раскладок (сохраняет заданные боксы и ссылается на них как <id>).

Режимы (`KLEIN_UPSAMPLER`):
- `local` (по умолчанию) — модель в том же процессе, через компонент Diffusers. ~19 GB в bf16; на 24 GB
  менеджер памяти на время работы вытесняет трансформер и энкодер в RAM (+несколько секунд на генерацию).
- `server` — OpenAI-совместимый сервер (vLLM), адрес в `KLEIN_UPSAMPLER_URL`. Системные промпты, формат
  запроса и сверка боксов — те же, что в Diffusers и в коде BFL.
- `off` — выключен; промпт уходит в модель как есть.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
from collections import OrderedDict

from PIL import Image

import layout

PUP_REPO = os.environ.get("KLEIN_UPSAMPLER_ID", "black-forest-labs/flux-3-image-klein-pup")
MODE = os.environ.get("KLEIN_UPSAMPLER", "local").lower()
SERVER_URL = os.environ.get("KLEIN_UPSAMPLER_URL", "")
SERVER_MODEL = os.environ.get("KLEIN_UPSAMPLER_NAME", PUP_REPO)
DEFAULT_MAX_NEW_TOKENS = 3584
CACHE_SIZE = 64
UPSAMPLER_RESERVE_GB = 2.5  # KV-кэш и активации; с 10 референсами токенов картинок больше всего


class PromptUpsampler:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.mock = backend.is_mock
        self.mode = "mock" if self.mock else ("server" if MODE == "server" or (SERVER_URL and MODE != "off")
                                              else MODE)
        if self.mode == "server" and not SERVER_URL:
            raise RuntimeError("KLEIN_UPSAMPLER=server, но не задан KLEIN_UPSAMPLER_URL (например http://localhost:8100/v1)")
        self._pipe = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, str] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def status(self) -> str:
        return {
            "mock": "Апсемплер: заглушка",
            "local": f"Апсемплер: {PUP_REPO} (локально)",
            "server": f"Апсемплер: {SERVER_URL}",
            "off": "Апсемплер выключен (KLEIN_UPSAMPLER=off) — промпт уходит в модель как есть",
        }.get(self.mode, self.mode)

    def cached(self, request: str, images: list[Image.Image] | None = None, boxes: list[dict] | None = None,
               max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str | None:
        """Готовое описание, если такой запрос уже переписывали (без очереди к видеокарте)."""
        return self._cache.get(_cache_key(request, images or [], boxes, max_new_tokens))

    # ------------------------------------------------------------------ #
    def __call__(self, request: str, images: list[Image.Image] | None = None, boxes: list[dict] | None = None,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> tuple[str, bool]:
        """(описание для модели, взято ли из кэша)."""
        if not self.enabled:
            raise RuntimeError("Апсемплер выключен (KLEIN_UPSAMPLER=off)")
        images = images or []
        key = _cache_key(request, images, boxes, max_new_tokens)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key], True
        if self.mode == "mock":
            caption = self._mock(request, images, boxes)
        elif self.mode == "server":
            caption = self._server(request, images, boxes, max_new_tokens)
        else:
            caption = self._local(request, images, boxes, max_new_tokens)
        self._cache[key] = caption
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return caption, False

    # ------------------------------------------------------------------ #
    def _local(self, request, images, boxes, max_new_tokens) -> str:
        import torch
        from diffusers import ModularPipeline

        backend = self.backend
        with backend._lock:  # одна видеокарта: апсемплер и генерация по очереди
            if self._pipe is None:
                print(f"[upsampler] загрузка {PUP_REPO}…")
                repo = next(iter(_repos_with(backend)))
                blocks = ModularPipeline.from_pretrained(repo).blocks.sub_blocks["prompt_upsampler"]
                pipe = blocks.init_pipeline(repo)
                pipe.load_components(dtype=torch.bfloat16)
                backend.pool.register(pipe.prompt_upsampler, "prompt_upsampler")
                self._pipe = pipe
            kwargs = dict(prompt=request, max_new_tokens=int(max_new_tokens))
            if images:
                kwargs["image"] = images if len(images) > 1 else images[0]
            if boxes:
                kwargs["boxes"] = boxes
            with torch.inference_mode():
                # generate() готовит входы на устройстве модели ещё до первого forward — кладём её на карту заранее.
                backend.pool.load(self._pipe.prompt_upsampler, reserve=int(UPSAMPLER_RESERVE_GB * 2**30))
                return self._pipe(**kwargs, output="prompt")

    def _server(self, request, images, boxes, max_new_tokens) -> str:
        # Тот же сценарий, что `upsample_prompt` в Diffusers и `Upsampler` в коде BFL, только генерация — на сервере.
        from diffusers.modular_pipelines.flux3 import prompt_upsampler as pup
        from openai import OpenAI

        mode = ("i2i" if images else "t2i") + ("_bbox" if boxes else "")
        locked = pup.boxes_to_canvas(boxes) if boxes else None
        if locked:
            request = pup.format_layout_request(request, locked)
        if images:
            refs = pup.prepare_references(images)
            text = pup.I2I_SCAFFOLD[bool(locked)].format(instruction=request)
            content = [{"type": "image_url", "image_url": {"url": _data_uri(im)}} for im in refs]
            content.append({"type": "text", "text": text})
        else:
            content = request
        client = OpenAI(base_url=SERVER_URL, api_key=os.environ.get("KLEIN_UPSAMPLER_API_KEY", "EMPTY"))
        # Сэмплинг как в flux3/pup.py: жадно, presence penalty 1.5 для текста и 0 для правок.
        response = client.chat.completions.create(
            model=SERVER_MODEL,
            messages=[{"role": "system", "content": pup.SYSTEM_PROMPTS[mode]}, {"role": "user", "content": content}],
            max_tokens=int(max_new_tokens), temperature=0.0, top_p=0.95,
            presence_penalty=0.0 if images else 1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            timeout=600,
        )
        caption = (response.choices[0].message.content or "").strip()
        if not caption:
            raise RuntimeError("Апсемплер вернул пустой ответ")
        return pup.reconcile_boxes(caption, locked) if locked else caption

    @staticmethod
    def _mock(request, images, boxes) -> str:
        time.sleep(float(os.environ.get("KLEIN_MOCK_UPSAMPLE_SECONDS", 0)))
        refs = f" Reference images: {len(images)}." if images else ""
        text = (f"[апсемплер-заглушка] A detailed full-scene caption written from the request: «{request.strip()}».{refs}"
                " Medium and style, scene, lighting and camera, every subject with its colors, materials and"
                " position in the frame.")
        if boxes:
            text += " " + " ".join(f"<{b['id']}>" for b in boxes)
            return layout.compose_caption(text, boxes)
        return text


def _repos_with(backend):
    from backend import REPOS

    loaded = [REPOS[m] for m in getattr(backend, "pipes", {})]
    return loaded or [REPOS["distilled"]]


def _cache_key(request: str, images: list[Image.Image], boxes, max_new_tokens: int) -> str:
    h = hashlib.sha256()
    h.update(request.strip().encode())
    h.update(json.dumps(boxes, sort_keys=True).encode())
    h.update(str(max_new_tokens).encode())
    for im in images:
        small = im.convert("RGB").resize((64, 64))
        h.update(f"{im.size}".encode())
        h.update(small.tobytes())
    return h.hexdigest()


def _data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
