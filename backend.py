"""Бэкенды генерации для FLUX 3 Image Klein.

- `KleinBackend` — настоящая модель через модульный пайплайн Diffusers (`ModularPipeline`, нужна CUDA).
  Две модели на общих текстовом энкодере и VAE: Distilled (16 шагов, без CFG) и Base (50 шагов, CFG 3.5,
  negative prompt). Апсемплер промптов — отдельная модель, см. `upsampler.py`.
- `MockBackend` — заглушка без модели: рисует плейсхолдер с параметрами запроса, чтобы отлаживать интерфейс
  на машине без видеокарты.

Выбор делает `load_backend()`: `KLEIN_MOCK=1` включает заглушку, иначе берётся настоящая модель, если есть CUDA.
"""

from __future__ import annotations

import gc
import math
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

# Пик памяти — декодирование VAE на больших размерах; expandable_segments снижает фрагментацию.
# Действует, только если выставлено до первой аллокации CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPOS = {
    "distilled": os.environ.get("KLEIN_MODEL_ID", "black-forest-labs/flux-3-image-klein"),
    "base": os.environ.get("KLEIN_BASE_MODEL_ID", "black-forest-labs/flux-3-image-klein-base"),
}
# Текстовый энкодер (Qwen3-VL-4B-Instruct): по умолчанию — его версия, обработанная Heretic (аблитерация отказов,
# https://github.com/p-e-w/heretic): та же архитектура и токенизатор, но без направления «отказа» в скрытых
# состояниях, которые и уходят в трансформер. Пустая строка — оригинальный энкодер из репозитория Klein.
TEXT_ENCODER_ID = os.environ.get("KLEIN_TEXT_ENCODER_ID", "DreamFast/Qwen3-VL-4b-Heretic")
MODEL_LABELS = {
    "distilled": "Klein · Distilled (16 шагов, быстро)",
    "base": "Klein · Base (50 шагов, CFG, negative)",
}
# Рекомендации BFL: Distilled — 16 шагов (обучен на 4–16) без CFG; Base — 50 шагов, CFG 3.5.
DEFAULT_STEPS = {"distilled": 16, "base": 50}
MAX_STEPS = {"distilled": 16, "base": 100}
DEFAULT_GUIDANCE = 3.5

MAX_REFERENCE_IMAGES = 10
SIZE_MULTIPLE = 16  # один токен латента = 16×16 пикселей
MIN_SIDE = 256
MAX_SIDE = 6256
MAX_PIXELS = 16_822_272  # 4K-тир BFL API (16.8 MP)

# Соотношения сторон и тиры площади — как у BFL API.
ASPECT_RATIOS = ["1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "5:7", "7:5", "9:16", "16:9", "1:2", "2:1",
                 "9:21", "21:9"]
TIERS = {"768² (0.6 MP)": 768, "1K (1 MP)": 1024, "1.5K (2.4 MP)": 1536, "2K (4.2 MP)": 2048, "4K (16.8 MP)": 4096}
DEFAULT_TIER = "1K (1 MP)"

ProgressFn = Callable[[float, str], None]
StopFn = Callable[[], bool]
MOCK_STEP_SECONDS = float(os.environ.get("KLEIN_MOCK_STEP_SECONDS", 0.03))


class JobCancelled(Exception):
    """Задачу отменили посреди генерации (поднимается из хука шага трансформера)."""


def snap(v: float) -> int:
    return int(min(MAX_SIDE, max(MIN_SIDE, round(v / SIZE_MULTIPLE) * SIZE_MULTIPLE)))


def clamp_size(width: float, height: float) -> tuple[int, int]:
    w, h = snap(width), snap(height)
    if w * h > MAX_PIXELS:
        k = math.sqrt(MAX_PIXELS / (w * h))
        w, h = int(w * k) // SIZE_MULTIPLE * SIZE_MULTIPLE, int(h * k) // SIZE_MULTIPLE * SIZE_MULTIPLE
    return w, h


def parse_ratio(text: str) -> float | None:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x×/]\s*(\d+(?:\.\d+)?)\s*", text or "")
    if not m or float(m.group(2)) == 0:
        return None
    return float(m.group(1)) / float(m.group(2))


def size_for_aspect(aspect: float, side: int) -> tuple[int, int]:
    """Размер с отношением сторон `aspect` и площадью side²."""
    width = math.sqrt(side * side * aspect)
    return clamp_size(width, width / aspect)


def size_like_image(image: Image.Image, side: int) -> tuple[int, int]:
    return size_for_aspect(image.width / image.height, side)


@dataclass
class GenRequest:
    prompt: str  # текст, который уходит в текстовый энкодер (после апсемплера или как есть)
    images: list[Image.Image] = field(default_factory=list)
    # None — размер выбирает пайплайн: 1024² для текста, 1 MP с пропорциями первого референса для правки.
    width: int | None = None
    height: int | None = None
    model: str = "distilled"
    steps: int = 16
    seed: int = 0
    num_images: int = 1
    guidance: float = DEFAULT_GUIDANCE  # только Base
    negative_prompt: str = ""  # только Base
    loras: list[tuple[str, float]] = field(default_factory=list)  # (имя адаптера, вес)

    def resolved_size(self) -> tuple[int, int]:
        """Итоговый размер вывода — повторяет логику пайплайна (`Flux3ImageCondInputStep`)."""
        if self.width and self.height:
            return self.width, self.height
        if self.images:
            return size_like_image(self.images[0], 1024)
        return 1024, 1024


def flatten_alpha(image: Image.Image) -> Image.Image:
    """Прозрачность — на белый фон, как делает апсемплер BFL (иначе прозрачное становится чёрным)."""
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


# Предел площади референса — как в коде BFL (`flux3/pipeline.py`): единственный — до 4 MP, в композиции из
# нескольких — каждый до 1 MP. Diffusers пропускает до 16 MP, и фото с телефона (12 MP) раздувает число токенов
# трансформера в разы: шаг становится в несколько раз дольше, а модель обучали не на таких.
SINGLE_REFERENCE_MAX_PIXELS = 4 * 1024 * 1024
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024


def fit_references(images: list[Image.Image]) -> list[Image.Image]:
    limit = SINGLE_REFERENCE_MAX_PIXELS if len(images) == 1 else MULTI_REFERENCE_MAX_PIXELS
    fitted = []
    for image in images:
        if image.width * image.height > limit:
            k = math.sqrt(limit / (image.width * image.height))
            image = image.resize((max(1, int(image.width * k)), max(1, int(image.height * k))),
                                 Image.Resampling.LANCZOS)
        fitted.append(image)
    return fitted


# Выше этого (ширина × высота × число картинок) на картах < 40 GB VAE сразу декодирует тайлами:
# иначе OOM случится после всего денойзинга, и генерацию придётся повторять.
TILE_UPFRONT_PIXELS = 2048 * 2048
SHARED_COMPONENTS = ("text_encoder", "vae")
GB = 2**30
# Запас видеопамяти под активации (замерено на RTX 3090, bf16): трансформер — постоянная часть плюс на каждый
# мегапиксель токенов (вывод × число картинок + референсы); VAE в fp32 — на мегапиксель декодируемого кадра.
TRANSFORMER_RESERVE = (2.2 * GB, 0.65 * GB)
VAE_DECODE_PER_MP = 4.2 * GB
VAE_TILED_RESERVE = 2.5 * GB


class KleinBackend:
    is_mock = False

    def __init__(self) -> None:
        import torch

        from vram import VramPool

        self._torch = torch
        self._lock = threading.RLock()
        self.vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        # Веса bf16: трансформер 11 GB, энкодер Qwen3-VL-4B 8 GB, апсемплер 19 GB, VAE 0.3 GB. Все живут в
        # закреплённой RAM, на карте — то, что нужно сейчас (см. vram.py). От 48 GB помещается всё сразу, на 24 GB
        # трансформер с энкодером на 1 MP живут на карте вместе, а апсемплер на время работы вытесняет обоих.
        self.pool = VramPool("cuda", pin=os.environ.get("KLEIN_PIN_MEMORY", "1") != "0")
        self.pool.set_reserve("text_encoder", 1 * GB)
        self.fp8_storage = os.environ.get("KLEIN_FP8_STORAGE") == "1"

        self.pipes: dict = {}
        self._on_step = None  # колбэк текущей генерации, его зовёт хук трансформера
        self.loras: dict[str, tuple[str, str | None]] = {}  # имя → (источник, weight_name)
        self._loaded_loras: dict[str, set[str]] = {}  # модель → загруженные в неё адаптеры
        self._load_model(os.environ.get("KLEIN_DEFAULT_MODEL", "distilled"))
        if os.environ.get("KLEIN_PRELOAD_BASE") == "1":
            self._load_model("base")

        gpu = torch.cuda.get_device_name(0)
        self.description = (f"{gpu} ({self.vram_gb:.0f} GB) · энкодер: {TEXT_ENCODER_ID or 'оригинальный'} · "
                            f"модели подгружаются из "
                            f"{'закреплённой ' if self.pool.pin else ''}RAM по требованию"
                            + (" · трансформер в FP8 (хранение)" if self.fp8_storage else ""))

    # ------------------------------------------------------------------ #
    @property
    def models(self) -> list[str]:
        return list(REPOS)

    def is_loaded(self, model: str) -> bool:
        return model in self.pipes

    def _load_model(self, model: str):
        """Пайплайн без шага апсемплера: промпт в него приходит уже готовым (см. upsampler.py)."""
        if model in self.pipes:
            return self.pipes[model]
        torch = self._torch
        from diffusers import ModularPipeline

        repo = REPOS[model]
        print(f"[backend] загрузка {repo}…")
        started = time.time()
        blocks = ModularPipeline.from_pretrained(repo).blocks
        blocks.sub_blocks.pop("prompt_upsampler", None)
        pipe = blocks.init_pipeline(repo)
        donor = next(iter(self.pipes.values()), None)
        if donor is None:
            dtype = {"vae": torch.float32, "default": torch.bfloat16}
            if TEXT_ENCODER_ID:
                names = [n for n, spec in pipe._component_specs.items()
                         if spec.default_creation_method == "from_pretrained" and n != "text_encoder"]
                pipe.load_components(names=names, dtype=dtype)
                pipe.update_components(text_encoder=load_text_encoder(TEXT_ENCODER_ID))
            else:
                pipe.load_components(dtype=dtype)
            # VAE маленький (0.3 GB) и вызывается не через forward — держим его на карте всегда.
            pipe.vae.to("cuda")
            self._guard_vae(pipe.vae)
            self.pool.register(pipe.text_encoder, "text_encoder")
        else:
            # Энкодер и VAE у Base и Distilled одинаковые — второй раз их не грузим.
            names = [n for n, spec in pipe._component_specs.items()
                     if spec.default_creation_method == "from_pretrained" and n not in SHARED_COMPONENTS]
            pipe.load_components(names=names, dtype=torch.bfloat16)
            pipe.update_components(**{n: getattr(donor, n) for n in SHARED_COMPONENTS})
        if self.fp8_storage:
            # Веса трансформера хранятся в FP8 (5.6 GB вместо 11), считаются в bf16 — работает и на Ampere.
            pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                                      compute_dtype=torch.bfloat16)
        self.pool.register(pipe.transformer, f"transformer_{model}")
        self.pipes[model] = pipe
        self._loaded_loras[model] = set()
        print(f"[backend] {repo} готов за {time.time() - started:.0f} с")
        return pipe

    def _guard_vae(self, vae) -> None:
        """VAE вызывают мимо forward (`encode` / `decode`), хук пула его не видит: перед вызовом освобождаем
        место под его активации — при нехватке с карты уйдёт энкодер, а не случится OOM в конце генерации."""
        factor = 2 ** (len(vae.config.block_out_channels) - 1)

        def guarded(fn, pixels):
            def call(x, *args, **kwargs):
                need = VAE_TILED_RESERVE if vae.use_tiling else pixels(x) / 1e6 * VAE_DECODE_PER_MP
                self.pool.make_room(int(need))
                return fn(x, *args, **kwargs)
            return call

        vae.encode = guarded(vae.encode, lambda x: x.shape[0] * x.shape[-2] * x.shape[-1])
        vae.decode = guarded(vae.decode, lambda z: z.shape[0] * z.shape[-2] * z.shape[-1] * factor**2)

    def _transformer_reserve(self, req: GenRequest) -> int:
        width, height = req.resolved_size()
        tokens_mp = (width * height * req.num_images + sum(im.width * im.height for im in req.images)) / 1e6
        fixed, per_mp = TRANSFORMER_RESERVE
        return int(fixed + per_mp * tokens_mp)

    def ensure_model(self, model: str) -> None:
        with self._lock:
            self._load_model(model)

    # ------------------------------------------------------------------ #
    # LoRA
    # ------------------------------------------------------------------ #
    def add_lora(self, source: str, name: str | None = None) -> str:
        """Регистрирует LoRA: путь к файлу, `owner/repo`, `owner/repo/файл.safetensors` или URL Hugging Face.

        Адаптеры общие для всех пользователей, но включаются только в генерациях того, кто их выбрал
        (`GenRequest.loras`). Скачивание идёт вне замка видеокарты, чтобы не останавливать чужие генерации."""
        folder, weight_name = fetch_lora(source)
        name = lora_name(name, weight_name, self.loras)
        with self._lock:
            # Сразу грузим в текущую модель, чтобы ошибка (не тот формат) была видна при добавлении.
            model = next(iter(self.pipes))
            self.pipes[model].load_lora_weights(folder, weight_name=weight_name, adapter_name=name)
            self._loaded_loras[model].add(name)
            self.loras[name] = (folder, weight_name)
        return name

    def _apply_loras(self, model: str, loras: list[tuple[str, float]]) -> None:
        pipe = self.pipes[model]
        loras = [(n, w) for n, w in loras if n in self.loras]
        for name, _ in loras:
            if name not in self._loaded_loras[model]:
                folder, weight_name = self.loras[name]
                pipe.load_lora_weights(folder, weight_name=weight_name, adapter_name=name)
                self._loaded_loras[model].add(name)
        if loras:
            pipe.enable_lora()
            pipe.set_adapters([n for n, _ in loras], adapter_weights=[w for _, w in loras])
        elif self._loaded_loras[model]:
            pipe.disable_lora()

    # ------------------------------------------------------------------ #
    def _step_hook(self, pipe) -> None:
        """Хук на каждый вызов трансформера: прогресс по шагам и остановка отменённой задачи."""
        if getattr(pipe.transformer, "_klein_step_hook", False):
            return

        def hook(module, args, kwargs):
            if self._on_step is not None:
                self._on_step()

        pipe.transformer.register_forward_pre_hook(hook, with_kwargs=True)
        pipe.transformer._klein_step_hook = True

    def generate(self, req: GenRequest, progress: ProgressFn | None = None,
                 should_stop: StopFn | None = None) -> list[Image.Image]:
        torch = self._torch
        with self._lock:
            if progress and req.model not in self.pipes:
                progress(0, f"Загрузка {MODEL_LABELS[req.model]} (первый раз — пара минут)")
            pipe = self._load_model(req.model)
            self._step_hook(pipe)
            self._apply_loras(req.model, req.loras)
            kwargs = dict(prompt=req.prompt, num_inference_steps=req.steps, num_images_per_prompt=req.num_images)
            if req.images:
                kwargs["image"] = req.images if len(req.images) > 1 else req.images[0]
            if req.width and req.height:
                kwargs.update(width=req.width, height=req.height)
            # Base с CFG зовёт трансформер дважды на шаг (условный и безусловный проход).
            calls_per_step = 1
            if req.model == "base":
                if getattr(pipe.guider, "guidance_scale", None) != req.guidance:
                    pipe.guider = pipe.guider.new(guidance_scale=req.guidance)
                if req.negative_prompt.strip():
                    kwargs["negative_prompt"] = req.negative_prompt.strip()
                calls_per_step = 2 if req.guidance > 1 else 1
            calls = [0]

            def on_step():
                if should_stop and should_stop():
                    raise JobCancelled
                calls[0] += 1
                if progress:
                    step = min(req.steps, (calls[0] - 1) // calls_per_step + 1)
                    progress(min(0.99, calls[0] / (req.steps * calls_per_step)), f"Шаг {step}/{req.steps}")

            width, height = req.resolved_size()
            tile_upfront = self.vram_gb < 40 and width * height * req.num_images > TILE_UPFRONT_PIXELS
            self.pool.set_reserve(f"transformer_{req.model}", self._transformer_reserve(req))
            self._on_step = on_step
            try:
                with torch.inference_mode():
                    if not tile_upfront:
                        try:
                            return self._run(pipe, kwargs, req.seed)
                        except torch.OutOfMemoryError as e:
                            self._release(e)
                            # Оценка запаса не угадала — повторяем с одним трансформером на карте.
                            self.pool.unload_all(keep=pipe.transformer)
                            calls[0] = 0
                            print(f"[backend] OOM на {width}×{height} — повтор с VAE тайлами", flush=True)
                            if progress:
                                progress(0, "Не хватило видеопамяти — повтор с декодированием VAE тайлами")
                    pipe.vae.enable_tiling()
                    try:
                        return self._run(pipe, kwargs, req.seed)
                    finally:
                        pipe.vae.disable_tiling()
            except BaseException as e:
                self._release(e)
                raise
            finally:
                self._on_step = None

    def _run(self, pipe, kwargs: dict, seed: int) -> list[Image.Image]:
        # Генератор каждый раз свежий: повтор после OOM должен дать ту же картинку.
        return pipe(**kwargs, generator=self._torch.Generator().manual_seed(seed), output="images")

    def _release(self, exc: BaseException) -> None:
        """Traceback упавшей генерации держит фреймы пайплайна со всеми тензорами — без очистки
        после первого OOM видеопамять не освобождается и падают все следующие генерации."""
        traceback.clear_frames(exc.__traceback__)
        gc.collect()
        self._torch.cuda.empty_cache()


def load_text_encoder(repo: str):
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    print(f"[backend] текстовый энкодер: {repo}")
    return Qwen3VLForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16)


def resolve_lora_source(source: str) -> tuple[str, str | None]:
    """'owner/repo' | 'owner/repo/путь/файл.safetensors' | URL Hugging Face | локальный путь → (repo_or_path, weight_name)."""
    source = source.strip()
    if not source:
        raise ValueError("Укажите LoRA: путь, owner/repo или ссылку на Hugging Face")
    if Path(source).expanduser().exists():
        path = Path(source).expanduser()
        return (str(path.parent), path.name) if path.is_file() else (str(path), None)
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(?:\?.*)?$", source)
    if m:
        return m.group(1), m.group(2)
    if source.startswith("http"):
        raise ValueError("Поддерживаются ссылки на файлы Hugging Face (…/resolve/main/файл.safetensors) "
                         "или локальные пути")
    parts = source.split("/")
    if len(parts) >= 3:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return source, None


def fetch_lora(source: str) -> tuple[str, str]:
    """Скачивает LoRA (если она не локальная) → (папка, имя файла) для `load_lora_weights`."""
    repo_or_path, weight_name = resolve_lora_source(source)
    if Path(repo_or_path).is_dir():
        if weight_name is None:
            files = sorted(p.name for p in Path(repo_or_path).glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError(f"В папке {len(files)} файлов .safetensors — укажите путь к нужному файлу")
            weight_name = files[0]
        return repo_or_path, weight_name
    from huggingface_hub import hf_hub_download, list_repo_files

    if weight_name is None:
        files = [f for f in list_repo_files(repo_or_path) if f.endswith(".safetensors")]
        if len(files) != 1:
            raise ValueError(f"В {repo_or_path} файлов .safetensors: {len(files)} — укажите owner/repo/файл "
                             f"({', '.join(files[:5])})")
        weight_name = files[0]
    path = Path(hf_hub_download(repo_or_path, weight_name))
    return str(path.parent), path.name


def lora_name(name: str | None, source: str, taken) -> str:
    name = name or re.sub(r"[^A-Za-z0-9_]", "_", Path(source).stem)[:40] or "lora"
    base, n = name, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    return name


class MockBackend:
    """Имитирует генерацию: картинка нужного размера с текстом запроса и миниатюрами референсов."""

    is_mock = True
    description = "ЗАГЛУШКА — модель не загружена, картинки ненастоящие"
    PREVIEW_MAX_SIDE = 768

    def __init__(self) -> None:
        self.loras: dict[str, tuple[str, str | None]] = {}
        self._loaded = {"distilled"}

    @property
    def models(self) -> list[str]:
        return list(REPOS)

    def is_loaded(self, model: str) -> bool:
        return model in self._loaded

    def ensure_model(self, model: str) -> None:
        self._loaded.add(model)

    def add_lora(self, source: str, name: str | None = None) -> str:
        repo_or_path, weight_name = resolve_lora_source(source)
        name = lora_name(name, weight_name or repo_or_path, self.loras)
        self.loras[name] = (repo_or_path, weight_name)
        return name

    def generate(self, req: GenRequest, progress: ProgressFn | None = None,
                 should_stop: StopFn | None = None) -> list[Image.Image]:
        for step in range(req.steps):
            if should_stop and should_stop():
                raise JobCancelled
            time.sleep(MOCK_STEP_SECONDS)
            if progress:
                progress((step + 1) / req.steps, f"Шаг {step + 1}/{req.steps} (заглушка)")
        self._loaded.add(req.model)

        width, height = req.resolved_size()
        scale = min(1.0, self.PREVIEW_MAX_SIDE / max(width, height))
        pw, ph = max(64, int(width * scale)), max(64, int(height * scale))
        results = []
        for n in range(req.num_images):
            hue = (req.seed * 37 + n * 61) % 255
            canvas = Image.new("RGB", (pw, ph), (hue, 70, 255 - hue))
            thumb_side = max(32, min(pw, ph) // 6)
            for i, ref in enumerate(req.images):
                thumb = ref.convert("RGB").copy()
                thumb.thumbnail((thumb_side, thumb_side))
                canvas.paste(thumb, (8 + i * (thumb_side + 4), ph - thumb.height - 8))
            lines = [
                f"MOCK #{n + 1}  {width}x{height}  seed={req.seed}  steps={req.steps}  {req.model}",
                f"refs={len(req.images)}  cfg={req.guidance if req.model == 'base' else '-'}"
                f"  loras={','.join(f'{a}:{w}' for a, w in req.loras) or '-'}",
                "",
                *_wrap(req.prompt, max(20, pw // 8)),
            ]
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default(size=max(11, pw // 50))
            draw.multiline_text((10, 10), "\n".join(lines[:24]), fill=(255, 255, 255), font=font,
                                stroke_width=2, stroke_fill=(0, 0, 0))
            results.append(canvas)
        return results


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def load_backend() -> KleinBackend | MockBackend:
    if os.environ.get("KLEIN_MOCK") == "1":
        return MockBackend()
    try:
        import torch
    except ImportError:
        print("[backend] torch не установлен — работаю в режиме заглушки")
        return MockBackend()
    if not torch.cuda.is_available():
        print("[backend] CUDA недоступна — работаю в режиме заглушки (KLEIN_MOCK=1 делает это явно)")
        return MockBackend()
    return KleinBackend()
