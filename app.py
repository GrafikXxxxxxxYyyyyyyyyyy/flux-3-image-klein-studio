"""FLUX 3 Image Klein Studio — Gradio-интерфейс ко всем возможностям FLUX 3 Image Klein:
текст → изображение, редактирование с 1–10 референсами, раскладка боксами, точечная правка боксами
(добавить / переместить / оставить / удалить), апсемплер промптов, Distilled и Base, LoRA.

Запуск:
    python app.py                      # на GPU-сервере: настоящая модель
    KLEIN_MOCK=1 python app.py         # где угодно: заглушка для отладки интерфейса
    python app.py --share --auth user:pass
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import box_editor
import layout
from backend import (
    ASPECT_RATIOS,
    DEFAULT_GUIDANCE,
    DEFAULT_STEPS,
    DEFAULT_TIER,
    MAX_REFERENCE_IMAGES,
    MAX_STEPS,
    MIN_SIDE,
    MODEL_LABELS,
    REPOS,
    SIZE_MULTIPLE,
    TEXT_ENCODER_ID,
    TIERS,
    GenRequest,
    clamp_size,
    fit_references,
    flatten_alpha,
    load_backend,
    parse_ratio,
    size_for_aspect,
    size_like_image,
)
from scheduler import PHASE_STREAK, QUEUE_MAX, USER_MAX_JOBS, GpuScheduler, Job, QueueFull
from upsampler import DEFAULT_MAX_NEW_TOKENS, PromptUpsampler

OUTPUT_DIR = Path(os.environ.get("KLEIN_OUTPUT_DIR", "outputs"))
MAX_SEED = 2**32 - 1
SLIDER_MAX_SIDE = 4096
HISTORY_LIMIT = 120
# own — каждый видит только свои генерации (по id браузера), all — общая лента всех пользователей.
HISTORY_MODE = os.environ.get("KLEIN_HISTORY", "own")
# Галочка «✨ Апсемплер» по умолчанию снята: он добавляет ~15–18 с к каждому новому промпту.
UPSAMPLE_DEFAULT = os.environ.get("KLEIN_UPSAMPLE_DEFAULT") == "1"
POLL_SECONDS = 0.5

SIZE_REF = "Как у изображения №"
SIZE_RATIO = "Пропорции и площадь"
SIZE_CUSTOM = "Свой размер"

backend = load_backend()
upsampler = PromptUpsampler(backend)
scheduler = GpuScheduler(backend, upsampler)


# --------------------------------------------------------------------------- #
# Общая часть: генерация
# --------------------------------------------------------------------------- #
def gallery_images(value) -> list[Image.Image]:
    """Значение галереи → список PIL в порядке загрузки."""
    images = []
    for item in value or []:
        item = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(item, dict):  # {"image": {"path": ...}, "caption": ...}
            item = (item.get("image") or {}).get("path") or item.get("path")
        images.append(item if isinstance(item, Image.Image) else Image.open(item))
    return images


def resolve_size(mode: str, follow_idx, aspect: str, tier: str, width, height,
                 images: list[Image.Image]) -> tuple[int, int]:
    side = TIERS.get(tier, 1024)
    if mode == SIZE_REF:
        idx = int(follow_idx or 1)
        if not images:
            raise gr.Error("Загрузите изображения")
        if not 1 <= idx <= len(images):
            raise gr.Error(f"Нет изображения №{idx}: загружено {len(images)}")
        return size_like_image(images[idx - 1], side)
    if mode == SIZE_RATIO:
        return size_for_aspect(parse_ratio(aspect) or 1.0, side)
    return clamp_size(width, height)


def describe_size(*args) -> str:
    try:
        w, h = resolve_size(*args)
    except gr.Error:
        return "<small>Размер появится после загрузки изображений</small>"
    return f"<small>Вывод: <b>{w}×{h}</b> · {w * h / 1e6:.2f} MP</small>"


def settings_values(settings: list) -> dict:
    model, steps, guidance, negative, seed, randomize, num_images, max_tokens, lora_table = settings
    rows = lora_table.values.tolist() if hasattr(lora_table, "values") else list(lora_table or [])
    loras = [(str(r[0]), float(r[1])) for r in rows if r and r[0] and len(r) > 2 and r[2] in (True, "true", 1)]
    return dict(model=model, steps=int(steps), guidance=float(guidance), negative=negative or "",
                seed=random.randint(0, MAX_SEED) if randomize else int(seed or 0), num_images=int(num_images),
                max_tokens=int(max_tokens), loras=loras)


async def wait_job(job, progress: gr.Progress):
    """Ждёт задачу из очереди, отдавая строку статуса при каждом изменении. Отмена (кнопка или закрытая вкладка)
    прерывает генератор на await — тогда задача снимается из очереди или останавливается на следующем шаге."""
    last = None
    try:
        while not job.done.is_set():
            text = scheduler.describe(job)
            progress(job.progress if job.state == "running" and job.progress else None, desc=text)
            if text != last:
                yield text
                last = text
            await asyncio.sleep(POLL_SECONDS)
    finally:
        if not job.done.is_set():
            scheduler.cancel(job)


def check_job(job) -> None:
    if job.state == "cancelled":
        raise gr.Error("Задача отменена")
    if job.error is not None:
        e = job.error
        prefix = "Апсемплер: " if job.caption is None else ""
        raise gr.Error(f"{prefix}{type(e).__name__}: {e}")


def make_job(client, request: str, images: list[Image.Image], boxes, upsample: bool, s: dict,
             gen: GenRequest | None) -> Job:
    if not request or not request.strip():
        raise gr.Error("Пустой промпт")
    if len(images) > MAX_REFERENCE_IMAGES:
        raise gr.Error(f"Модель принимает не больше {MAX_REFERENCE_IMAGES} изображений, загружено {len(images)}")
    upsample = bool(upsample) and upsampler.enabled
    images = fit_references([flatten_alpha(im) for im in images])
    job = Job(owner=client or "anon", request=request.strip(), images=images,
              boxes=boxes, upsample=upsample, max_tokens=s["max_tokens"], gen=gen,
              caption=None if upsample else layout.compose_caption(request, boxes))
    if gen is not None:
        gen.images = job.images
    try:
        return scheduler.submit(job)
    except QueueFull as e:
        raise gr.Error(str(e)) from e


async def run_job(client, request: str, images: list[Image.Image], boxes: list[dict] | None,
                  size: tuple[int, int] | None, upsample: bool, settings, progress: gr.Progress):
    """Ставит генерацию в общую очередь видеокарты и ждёт её. Выходы: галерея, результаты, описание,
    инфо (пока ждём — статус очереди), seed, файлы."""
    s = settings_values(list(settings))
    gen = GenRequest(prompt="", width=size[0] if size else None, height=size[1] if size else None,
                     model=s["model"], steps=s["steps"], seed=s["seed"], num_images=s["num_images"],
                     guidance=s["guidance"], negative_prompt=s["negative"], loras=s["loras"])
    job = make_job(client, request, images, boxes, upsample, s, gen)
    async for text in wait_job(job, progress):
        yield gr.skip(), gr.skip(), gr.skip(), f"⏳ {text}", gr.skip(), gr.skip()
    check_job(job)

    caption, results = job.caption, job.results
    saved = await asyncio.to_thread(save_outputs, results, gen, request, boxes, job.upsample, job.owner)
    width, height = gen.resolved_size()
    elapsed = time.time() - job.created
    queued = (job.started or job.created) - job.created
    notes = []
    if queued >= 1:
        notes.append(f"ожидание в очереди {queued:.0f} с")
    if job.upsample:
        notes.append("описание из кэша" if job.cached else f"апсемплер {job.upsample_time:.1f} с")
    elif len(request.split()) < 40 and not layout.has_box_table(caption):
        notes.append("⚠️ короткий промпт без апсемплера — результат будет слабее")
    if boxes and not job.upsample and (missing := layout.missing_mentions(caption, [
            {"id": b["id"]} for b in boxes])):
        notes.append(f"⚠️ в тексте нет ссылок на боксы: {', '.join(f'<{m}>' for m in missing)}")
    info = (
        f"**{width}×{height}** · seed `{gen.seed}` · {gen.steps} шагов · {MODEL_LABELS[gen.model].split(' (')[0]}"
        + (f" · CFG {gen.guidance}" if gen.model == "base" else "")
        + f" · {elapsed:.1f} с"
        + (f" · референсов: {len(images)}" if images else "")
        + (f" · боксов: {len(boxes)}" if boxes else "")
        + (f" · LoRA: {', '.join(f'{n} ×{w:g}' for n, w in gen.loras)}" if gen.loras else "")
        + (f"  \n{' · '.join(notes)}" if notes else "")
        + ("  \n⚠️ Заглушка: картинки ненастоящие" if backend.is_mock else "")
    )
    # Галерея показывает сжатый WebP (PNG 2K — 6–8 MB, через туннель это долго); оригиналы — файлами.
    yield results, results, caption, info, gen.seed, [str(p) for p in saved] or None


def save_outputs(results: list[Image.Image], req: GenRequest, request: str, boxes, upsample: bool,
                 owner: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{req.seed}_{uuid.uuid4().hex[:6]}"
    meta = {
        "request": request, "prompt": req.prompt, "upsampled": upsample, "boxes": boxes,
        "model": req.model, "repo": REPOS[req.model], "size": list(req.resolved_size()), "steps": req.steps,
        "seed": req.seed, "num_images": req.num_images, "num_reference_images": len(req.images),
        "guidance": req.guidance if req.model == "base" else None,
        "negative_prompt": req.negative_prompt if req.model == "base" else None,
        "loras": [{"name": n, "weight": w} for n, w in req.loras], "mock": backend.is_mock,
        "text_encoder": TEXT_ENCODER_ID or "original",
    }
    info = PngInfo()
    info.add_text("parameters", json.dumps(meta, ensure_ascii=False))
    info.add_text("Software", "FLUX 3 Image Klein Studio")
    info.add_text("Comment", "AI generated")
    paths = []
    for i, im in enumerate(results):
        paths.append(OUTPUT_DIR / f"{stem}_{i}.png")
        im.save(paths[-1], pnginfo=info)
    meta["files"] = [p.name for p in paths]
    meta["owner"] = owner  # только в JSON рядом: по нему история показывает каждому его картинки
    (OUTPUT_DIR / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return paths


async def rewrite_job(client, request: str, images: list[Image.Image], boxes, settings, progress: gr.Progress):
    """Только апсемплер: описание попадает в поле промпта, апсемплер при генерации выключается."""
    if not upsampler.enabled:
        raise gr.Error("Апсемплер выключен (KLEIN_UPSAMPLER=off)")
    job = make_job(client, request, images, boxes, True, settings_values(list(settings)), None)
    async for _ in wait_job(job, progress):
        yield gr.skip(), gr.skip()
    check_job(job)
    gr.Info("Описание из кэша" if job.cached else f"Описание готово за {job.upsample_time:.1f} с — "
            "правьте и генерируйте без апсемплера")
    yield job.caption, False


# --------------------------------------------------------------------------- #
# Обработчики вкладок (async-генераторы: ждут задачу в общей очереди)
# --------------------------------------------------------------------------- #
async def t2i(client, prompt, upsample, size_mode, aspect, tier, width, height, *settings,
              progress=gr.Progress()):
    size = resolve_size(size_mode, 1, aspect, tier, width, height, [])
    async for out in run_job(client, prompt, [], None, size, upsample, settings, progress):
        yield out


async def edit(client, gallery, instruction, upsample, size_mode, follow, aspect, tier, width, height, *settings,
               progress=gr.Progress()):
    images = gallery_images(gallery)
    if not images:
        raise gr.Error("Загрузите хотя бы одно изображение")
    size = resolve_size(size_mode, follow, aspect, tier, width, height, images)
    async for out in run_job(client, instruction, images, None, size, upsample, settings, progress):
        yield out


def layout_boxes(editor) -> list[dict]:
    """Боксы редактора раскладки в формате пайплайна; без боксов или с пустыми описаниями — понятная ошибка."""
    state = box_editor.to_state(editor)
    if not state:
        raise gr.Error("Нарисуйте хотя бы один бокс: зажмите кнопку мыши на холсте и протяните")
    if unnamed := [b["id"] for b in state if not b["desc"]]:
        raise gr.Error(f"Опишите, что в боксах: {', '.join(unnamed)} — поле рядом с id в списке под холстом")
    return layout.pipeline_boxes(state)


async def layout_generate(client, prompt, editor, upsample, aspect, tier, *settings, progress=gr.Progress()):
    pipeline = layout_boxes(editor)
    size = size_for_aspect(parse_ratio(aspect) or 1.0, TIERS.get(tier, 1024))
    async for out in run_job(client, prompt, [], pipeline, size, upsample, settings, progress):
        yield out


def edit_boxes(editor, images: list[Image.Image]) -> list[dict]:
    """Боксы редактора правки в формате пайплайна; без боксов или без описаний — понятная ошибка."""
    if not images:
        raise gr.Error("Загрузите изображение")
    state = [b for b in box_editor.to_edit_state(editor) if b["ref"] < len(images)]
    if not state:
        raise gr.Error("Отметьте хотя бы один бокс: протяните мышью по картинке")
    if unnamed := [b["id"] for b in state if not b["desc"] and b["kind"] in (layout.KIND_NEW, layout.KIND_MOVE)]:
        raise gr.Error(f"Опишите, что в боксах: {', '.join(unnamed)} — поле рядом с id в списке под картинкой")
    return layout.pipeline_boxes(state)


async def box_edit_generate(client, gallery, instruction, editor, upsample, follow, tier, *settings,
                            progress=gr.Progress()):
    images = gallery_images(gallery)
    pipeline = edit_boxes(editor, images)
    size = resolve_size(SIZE_REF, follow, "1:1", tier, 0, 0, images)
    async for out in run_job(client, instruction, images, pipeline, size, upsample, settings, progress):
        yield out


async def rewrite_text(client, prompt, *settings, progress=gr.Progress()):
    async for out in rewrite_job(client, prompt, [], None, settings, progress):
        yield out


async def rewrite_edit(client, gallery, prompt, *settings, progress=gr.Progress()):
    async for out in rewrite_job(client, prompt, gallery_images(gallery), None, settings, progress):
        yield out


async def rewrite_layout(client, prompt, editor, *settings, progress=gr.Progress()):
    async for out in rewrite_job(client, prompt, [], layout_boxes(editor), settings, progress):
        yield out


async def rewrite_box_edit(client, gallery, prompt, editor, *settings, progress=gr.Progress()):
    images = gallery_images(gallery)
    async for out in rewrite_job(client, prompt, images, edit_boxes(editor, images), settings, progress):
        yield out


# ---- Результаты → референсы, история ----
def results_to_refs(results, selected, edit_gallery, box_gallery):
    if not results:
        raise gr.Error("Сначала сгенерируйте картинку")
    image = results[min(int(selected or 0), len(results) - 1)]
    edit_images, box_images = gallery_images(edit_gallery), gallery_images(box_gallery)
    if len(edit_images) >= MAX_REFERENCE_IMAGES:
        raise gr.Error(f"В редактировании уже {MAX_REFERENCE_IMAGES} изображений")
    gr.Info("Картинка добавлена в «Редактирование» и «Правку боксами»")
    return edit_images + [image], box_images + [image]


def history_items(client) -> tuple[list, list]:
    """Генерации из OUTPUT_DIR, новые первыми. В режиме own — только сделанные из этого браузера."""
    if not OUTPUT_DIR.is_dir():
        return [], []
    items, metas = [], []
    for meta_path in sorted(OUTPUT_DIR.glob("*.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if HISTORY_MODE != "all" and meta.get("owner") != client:
            continue
        for name in meta.get("files", []):
            path = OUTPUT_DIR / name
            if path.exists():
                items.append((str(path), f"{meta.get('model', '')} · seed {meta.get('seed')}"))
                metas.append({k: v for k, v in meta.items() if k != "owner"} | {"file": name})
        if len(items) >= HISTORY_LIMIT:
            break
    return items, metas


def history_refresh(client):
    items, metas = history_items(client)
    return items, metas, None, None


def history_select(evt: gr.SelectData, metas):
    meta = metas[evt.index] if metas and evt.index < len(metas) else None
    return evt.index, meta


def history_to_refs(selected, metas, edit_gallery, box_gallery):
    if selected is None or not metas:
        raise gr.Error("Выберите картинку в истории")
    image = Image.open(OUTPUT_DIR / metas[selected]["file"])
    return results_to_refs([image], 0, edit_gallery, box_gallery)


def history_to_t2i(selected, metas):
    if selected is None or not metas:
        raise gr.Error("Выберите картинку в истории")
    meta = metas[selected]
    gr.Info("Описание и seed перенесены в «Генерация» (апсемплер выключен — описание уже полное)")
    return meta.get("prompt", ""), False, meta.get("seed", 0), False


# --------------------------------------------------------------------------- #
# Интерфейс
# --------------------------------------------------------------------------- #
CSS = """
.studio-header { background: linear-gradient(120deg, #120d2e 0%, #2d1868 40%, #6b2a8c 75%, #b23a7a 100%);
                 border-radius: 18px; padding: 22px 28px; margin-bottom: 6px; position: relative; overflow: hidden; }
.studio-header::after { content: ""; position: absolute; right: -60px; top: -80px; width: 320px; height: 320px;
                        background: radial-gradient(circle, rgba(255,255,255,.18), transparent 65%); }
.studio-header h1 { margin: 0; font-size: 30px; font-weight: 750; letter-spacing: -.02em; color: #fff !important; }
.studio-header p { margin: 6px 0 0; color: rgba(255,255,255,.82) !important; font-size: 15px; }
.studio-header .badges { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.studio-header .badge { padding: 3px 11px; border-radius: 999px; background: rgba(255,255,255,.13);
                        border: 1px solid rgba(255,255,255,.18); color: #fff; font-size: 12px; }
.generate-btn { min-height: 50px !important; font-size: 16px !important; font-weight: 650 !important; }
.canvas img { cursor: crosshair !important; }
.hint { opacity: .78; font-size: 13px; }
.queue-status { padding: 2px 6px; opacity: .85; font-size: 14px; }
.mock-banner { background: rgba(255, 193, 7, .14); border: 1px solid rgba(255, 193, 7, .55);
               padding: 8px 14px; border-radius: 10px; }
.result-gallery { border-radius: 14px; }
footer { display: none !important; }
"""


def header_html() -> str:
    badges = ["5.6B · open weights", "Distilled 16 шагов · Base 50", "до 10 референсов", "боксы", "до 4K",
              upsampler.status().split(" (")[0]]
    if backend.is_mock:
        badges.append("⚠️ режим заглушки")
    return (
        "<div class='studio-header'><h1>FLUX 3 Image Klein Studio</h1>"
        "<p>Генерация, редактирование, мульти-референсы и раскладка боксами — всё, что умеет модель.</p>"
        "<div class='badges'>" + "".join(f"<span class='badge'>{b}</span>" for b in badges) + "</div></div>"
    )


HELP_MD = f"""
### Как устроен Klein

**Апсемплер промптов.** Klein обучен на подробных описаниях сцены, и короткий запрос «как есть» даёт заметно
более слабый результат. Отдельная модель BFL — [flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)
(9B) — переписывает его в полное описание: стиль, сцена, свет, камера, каждый объект с цветами, материалами и местом
в кадре. Галочка «✨ Апсемплер» есть в каждой вкладке; по умолчанию она снята (апсемплер добавляет ~15–18 с) —
включайте её для коротких запросов или пишите подробный промпт сами (по-английски, 80–200 слов). Описание, которое ушло в модель, видно под результатом; кнопка «✨ Только переписать»
кладёт его в поле промпта, чтобы поправить руками и перебирать seed без повторного переписывания.
Одинаковые запросы апсемплер не переписывает дважды (кэш). Апсемплер понимает запросы на любом языке, пишет по-английски.

**Distilled и Base.** Distilled — 16 шагов без CFG (обучен на 4–16), ~4 с на 1 MP на RTX 5090. Base — 50 шагов,
CFG 3.5, работает negative prompt; нужен для LoRA-обучения и экспериментов. Обе модели делят энкодер и VAE;
Base грузится при первом выборе.

**Редактирование.** До {MAX_REFERENCE_IMAGES} изображений. Первое — то, что редактируется: по умолчанию результат
1 MP в его пропорциях. На референсы ссылайтесь по порядку: *the first image, the second image…*
(в описании для модели они станут `<ref_image_0>`, `<ref_image_1>`…). Пример: `put the dog from the second image
on the sofa in the first image, keep everything else unchanged`. Цвета можно задавать hex-кодами (`#e01075`),
новый текст — в кавычках.

**Раскладка боксами.** Протяните мышью по холсту — получится бокс (и внутри другого бокса тоже), курсор сразу
встанет в поле его описания. Перенести бокс — за его подпись (✥ id · описание), размер — за маркеры выделенного.
Клик выделяет меньший бокс под курсором, повторный клик — следующий под ним (так достаётся перекрытый);
✕ на выделенном боксе, 🗑 в списке или Delete — удалить,
Ctrl+Z — отменить, стрелки — сдвинуть (с Shift — на 5 %), Ctrl+D — копия, «сетка 5 %» — привязка. id берётся
из описания (`a red balloon` → `balloon_1`), пока не зададите его сами. Точные координаты — в списке, `y0, x0, y1, x1`
на холсте 0–1000.
Апсемплер сохраняет боксы как есть и пишет описание вокруг них, ссылаясь на каждый как `<id>`. Без апсемплера
пишите `<id>` в тексте сами — таблица допишется автоматически. Боксы задают место и масштаб, но не обрезают
как маска; слишком маленькие (≈40×25 px) новые элементы могут не появиться. Каждой строке текста — свой бокс.

**Точечная правка боксами.** Протяните мышью по картинке — получится бокс; холст и жесты те же, что в раскладке.
Действие бокса — кнопки в его строке (меняются в любой момент), действие для следующих боксов — «Новый бокс» над
картинкой:
- **➕ Новый / заменить** — новый элемент (или новый вид элемента) в отмеченной области;
- **↔ Переместить** — обведённая область — где элемент сейчас (пунктир); рядом появится рамка «→ сюда» —
  перетащите её на новое место и подгоните размер, между ними стрелка;
- **📌 Оставить как есть** — «якорь» для того, что не должно двигаться;
- **✖ Удалить** — модель закрасит область тем, что было за элементом.

Если референсов несколько, переключайте их миниатюрами над холстом; боксы рисуются на том, что показан.
Рамки «куда» и новые элементы — в кадре результата (обычно он того же формата, что редактируемая картинка).

Скажите о правке и в инструкции тоже («move the knight up and to the left, keep the rest unchanged»).
Пиксели вне отмеченных областей обычно остаются нетронутыми.

**LoRA.** Обучаются на Base (rank 32 на десятке картинок хватает для стиля). Работают и на Distilled
без конвертации — BFL советует вес 0.75 (0.5–1.0). Источник: путь к файлу, `owner/repo`,
`owner/repo/файл.safetensors` или ссылка `https://huggingface.co/…/resolve/main/файл.safetensors`.

**Очередь.** Видеокарта одна, поэтому все задачи всех пользователей идут через общую очередь: под кнопкой видно
место в очереди и примерное время до старта, в шапке — состояние видеокарты. У одного пользователя не больше
{USER_MAX_JOBS} задач одновременно, всего в очереди — до {QUEUE_MAX}. «✖» отменяет задачу — и ожидающую, и уже идущую
(остановится на следующем шаге). Пока апсемплер в видеопамяти, он переписывает промпты всех ожидающих подряд
(до {PHASE_STREAK} задач), потом подряд идут генерации — на 24 GB это экономит перекладывание 19 GB на каждый запрос.

**Размеры.** Кратно {SIZE_MULTIPLE} px, от {MIN_SIDE} px до 4K (16.8 MP). Тиры площади и пропорции — как в BFL API.
Большие размеры на картах < 40 GB декодируются VAE тайлами.

**Лицензия.** Веса — [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md):
бесплатно для некоммерческого использования и для коммерческого при годовой выручке до $5 млн. Апсемплер — Apache 2.0.
"""


def settings_sidebar():
    with gr.Sidebar(open=True, width=340):
        gr.Markdown("### ⚙️ Параметры")
        model = gr.Radio([(MODEL_LABELS[m], m) for m in REPOS], value="distilled", label="Модель")
        steps = gr.Slider(1, MAX_STEPS["distilled"], value=DEFAULT_STEPS["distilled"], step=1, label="Шаги",
                          info="Distilled: 16; 8 — вдвое быстрее, на фото почти без разницы, но надписи хуже; 4 — черновик. Base: 50")
        with gr.Accordion("Base: CFG и negative prompt", open=True, visible=False) as base_box:
            guidance = gr.Slider(1.0, 10.0, value=DEFAULT_GUIDANCE, step=0.1, label="Guidance (CFG)",
                                 info="3.5 — рекомендовано; 1 — выкл.: вдвое быстрее, но хуже надписи")
            negative = gr.Textbox(label="Negative prompt", lines=2,
                                  info="Лучше описывать желаемое позитивно, чем перечислять ненужное")
        with gr.Group():
            seed = gr.Number(value=42, precision=0, minimum=0, maximum=MAX_SEED, label="Seed")
            randomize = gr.Checkbox(value=True, label="🎲 Случайный seed каждый раз")
        num_images = gr.Slider(1, 4, value=1, step=1, label="Картинок за раз")
        with gr.Accordion("Апсемплер", open=False):
            max_tokens = gr.Slider(256, 4096, value=DEFAULT_MAX_NEW_TOKENS, step=64, label="max_new_tokens",
                                   info="Предел длины описания")
            gr.Markdown(f"<small>{upsampler.status()}</small>")
        with gr.Accordion("LoRA", open=False):
            lora_src = gr.Textbox(label="Источник", placeholder="owner/repo/style.safetensors",
                                  info="Путь, owner/repo, owner/repo/файл или ссылка HF")
            lora_weight = gr.Slider(0.0, 2.0, value=0.75, step=0.05, label="Вес",
                                    info="Distilled: 0.75 (0.5–1.0), Base: 1.0")
            lora_add = gr.Button("➕ Добавить LoRA", size="sm")
            lora_table = gr.Dataframe(headers=["LoRA", "вес", "вкл"], datatype=["str", "number", "bool"],
                                      column_count=(3, "fixed"), value=[], interactive=True, static_columns=[0],
                                      show_row_numbers=False, label="Подключённые")
            lora_clear = gr.Button("Отключить все", size="sm", variant="secondary")
        gr.Markdown(f"<small>{backend.description}</small>")

    def on_model(m):
        if m == "base":
            # 50 шагов × 2 прохода трансформера (CFG) ≈ 100 × 0.76 с на 3090 при 1 MP.
            gr.Info("Base — около 1.5 мин на картинку 1 MP (50 шагов, по два прохода с CFG). Для обычной работы "
                    "Distilled: ~15 с при том же качестве. Base нужна ради negative prompt и проверки LoRA."
                    + ("" if backend.is_loaded(m) else " Первая генерация — ещё ~20 с на загрузку модели."),
                    duration=12)
        return (gr.Slider(maximum=MAX_STEPS[m], value=DEFAULT_STEPS[m]), gr.Accordion(visible=m == "base"))

    # Через очередь: gr.Info доходит до браузера только у событий в очереди.
    model.change(on_model, model, [steps, base_box], show_progress="hidden")

    def on_lora_add(src, weight, table):
        try:
            name = backend.add_lora(src)
        except Exception as e:
            raise gr.Error(f"LoRA: {type(e).__name__}: {e}") from e
        rows = table.values.tolist() if hasattr(table, "values") else list(table or [])
        gr.Info(f"LoRA «{name}» подключена")
        return rows + [[name, float(weight), True]], ""

    lora_add.click(on_lora_add, [lora_src, lora_weight, lora_table], [lora_table, lora_src])
    # Адаптеры загружены для всех; «отключить» убирает их только из генераций этого пользователя.
    lora_clear.click(lambda: [], None, lora_table, queue=False)
    return [model, steps, guidance, negative, seed, randomize, num_images, max_tokens, lora_table], seed, randomize


def init_client(client):
    return client or uuid.uuid4().hex


def select_index(evt: gr.SelectData) -> int:
    return evt.index


def result_column():
    """Галерея результата + описание, ушедшее в модель, + действия с результатом."""
    gallery = gr.Gallery(label="Результат", format="webp", type="pil", columns=2, height=560, interactive=False,
                         elem_classes="result-gallery", buttons=["download", "fullscreen"])
    info = gr.Markdown()
    with gr.Row():
        to_refs = gr.Button("↪ В референсы", size="sm")
        to_prompt = gr.Button("↪ Описание в промпт", size="sm")
    caption = gr.Textbox(label="Описание, отправленное в модель", lines=4, interactive=False, buttons=["copy"])
    files = gr.File(label="Оригиналы PNG (с параметрами в метаданных)", file_count="multiple", interactive=False,
                    height=110)
    results, selected = gr.State([]), gr.State(0)
    gallery.select(select_index, None, selected, queue=False)
    return dict(gallery=gallery, info=info, caption=caption, files=files, results=results, selected=selected,
                to_refs=to_refs, to_prompt=to_prompt)


def size_block(modes: list[str], default: str):
    """Выбор размера вывода. Возвращает компоненты в порядке аргументов resolve_size (без картинок)."""
    mode = gr.Radio(modes, value=default, label="Размер вывода")
    with gr.Row(visible=SIZE_REF in modes and default == SIZE_REF) as ref_row:
        follow = gr.Number(value=1, precision=0, minimum=1, maximum=MAX_REFERENCE_IMAGES, label="№ изображения",
                           scale=1)
    with gr.Row(visible=default != SIZE_CUSTOM) as tier_row:
        aspect = gr.Dropdown(ASPECT_RATIOS, value="1:1", label="Пропорции", scale=1,
                             visible=default == SIZE_RATIO)
        tier = gr.Dropdown(list(TIERS), value=DEFAULT_TIER, label="Площадь", scale=1)
    with gr.Row(visible=default == SIZE_CUSTOM, equal_height=True) as custom_row:
        width = gr.Slider(MIN_SIDE, SLIDER_MAX_SIDE, value=1024, step=SIZE_MULTIPLE, label="Ширина", scale=4)
        height = gr.Slider(MIN_SIDE, SLIDER_MAX_SIDE, value=1024, step=SIZE_MULTIPLE, label="Высота", scale=4)
        swap = gr.Button("⇄", scale=0, min_width=44)
    swap.click(lambda a, b: (b, a), [width, height], [width, height], queue=False)
    mode.change(lambda m: (gr.Row(visible=m == SIZE_REF), gr.Row(visible=m != SIZE_CUSTOM),
                           gr.Dropdown(visible=m == SIZE_RATIO), gr.Row(visible=m == SIZE_CUSTOM)),
                mode, [ref_row, tier_row, aspect, custom_row], queue=False)
    summary = gr.Markdown()
    return [mode, follow, aspect, tier, width, height], summary


def wire_size_summary(size_inputs, summary, gallery=None):
    inputs = size_inputs + ([gallery] if gallery is not None else [])

    def fn(*args):
        images = gallery_images(args[-1]) if gallery is not None else []
        return describe_size(*args[:6], images)

    for c in inputs:
        c.change(fn, inputs, summary, queue=False)
    return fn, inputs


def action_row(label: str):
    with gr.Row():
        upsample = gr.Checkbox(value=upsampler.enabled and UPSAMPLE_DEFAULT, interactive=upsampler.enabled,
                               label="✨ Апсемплер",
                               info="Для коротких промптов — заметно лучше, +15–18 с" if upsampler.enabled
                               else "выключен на сервере",
                               scale=2)
        rewrite = gr.Button("✨ Только переписать", interactive=upsampler.enabled, scale=2)
        go = gr.Button(label, variant="primary", elem_classes="generate-btn", scale=3)
        cancel = gr.Button("✖", variant="stop", elem_classes="generate-btn", scale=0, min_width=52)
    return upsample, rewrite, go, cancel


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FLUX 3 Image Klein Studio") as demo:
        gr.HTML(header_html())
        queue_status = gr.Markdown(scheduler.snapshot(), elem_classes="queue-status")
        gr.Timer(2.0).tick(scheduler.snapshot, None, queue_status, queue=False, show_progress="hidden")
        # Постоянный id браузера (localStorage): лимит задач на пользователя и личная история.
        client = gr.BrowserState(None, storage_key="klein_client_id")
        if backend.is_mock:
            gr.Markdown("⚠️ **Режим заглушки** — модель не загружена, результаты — плейсхолдеры для проверки "
                        "интерфейса.", elem_classes="mock-banner")
        settings, seed_box, randomize_box = settings_sidebar()
        outs = []  # result_column() каждой вкладки — для кнопок «в референсы»

        # ---------------- Текст → изображение ----------------
        with gr.Tab("🎨 Генерация"):
            with gr.Row():
                with gr.Column(scale=5):
                    t_prompt = gr.Textbox(label="Промпт", lines=6, placeholder=(
                        "A red fox asleep in a vintage suitcase on a train, soft window light\n"
                        "или по-русски: лиса спит в старом чемодане в купе поезда"))
                    t_size, t_summary = size_block([SIZE_RATIO, SIZE_CUSTOM], SIZE_RATIO)
                    t_upsample, t_rewrite, t_go, t_cancel = action_row("🎨 Сгенерировать")
                with gr.Column(scale=6):
                    t_out = result_column()
            size_fn, size_inputs = wire_size_summary(t_size, t_summary)
            demo.load(size_fn, size_inputs, t_summary, queue=False)
            t_events = [
                t_rewrite.click(rewrite_text, [client, t_prompt, *settings], [t_prompt, t_upsample], show_progress_on=t_prompt),
                t_go.click(t2i, [client, t_prompt, t_upsample, t_size[0], *t_size[2:], *settings],
                           [t_out["gallery"], t_out["results"], t_out["caption"], t_out["info"], seed_box,
                            t_out["files"]], show_progress_on=t_out["gallery"]),
            ]
            t_cancel.click(lambda: "✖ Задача отменена", None, t_out["info"], cancels=t_events,
                            queue=False)
            outs.append((t_out, t_prompt, t_upsample))

        # ---------------- Редактирование / референсы ----------------
        with gr.Tab("🖌️ Редактирование"):
            with gr.Row():
                with gr.Column(scale=5):
                    e_images = gr.Gallery(label=f"Изображения (1–{MAX_REFERENCE_IMAGES}; первое — редактируемое)",
                                          type="pil", format="png", interactive=True, columns=5, height=240)
                    e_prompt = gr.Textbox(label="Инструкция", lines=4, placeholder=(
                        "make it a winter night: snow on the roof, warm light in the windows\n"
                        "или: put the dog from the second image on the sofa in the first image"))
                    e_size, e_summary = size_block([SIZE_REF, SIZE_RATIO, SIZE_CUSTOM], SIZE_REF)
                    e_upsample, e_rewrite, e_go, e_cancel = action_row("🖌️ Применить")
                with gr.Column(scale=6):
                    e_out = result_column()
            wire_size_summary(e_size, e_summary, e_images)
            e_events = [
                e_rewrite.click(rewrite_edit, [client, e_images, e_prompt, *settings], [e_prompt, e_upsample], show_progress_on=e_prompt),
                e_go.click(edit, [client, e_images, e_prompt, e_upsample, *e_size, *settings],
                           [e_out["gallery"], e_out["results"], e_out["caption"], e_out["info"], seed_box,
                            e_out["files"]], show_progress_on=e_out["gallery"]),
            ]
            e_cancel.click(lambda: "✖ Задача отменена", None, e_out["info"], cancels=e_events,
                            queue=False)
            outs.append((e_out, e_prompt, e_upsample))

        # ---------------- Раскладка боксами ----------------
        with gr.Tab("▣ Раскладка"):
            gr.Markdown("**Протяните по холсту — появится бокс** (можно и внутри другого), сразу впишите, что в нём. "
                        "Перенести — за подпись бокса ✥, размер — за маркеры выделенного. Клик "
                        "выделяет бокс, повторный клик — бокс под ним. ✕ или Delete — удалить, Ctrl+Z — отменить.",
                        elem_classes="hint")
            with gr.Row():
                with gr.Column(scale=7):
                    l_editor = box_editor.LayoutEditor("1:1")
                with gr.Column(scale=5):
                    l_prompt = gr.Textbox(label="Общее описание сцены", lines=4, placeholder=(
                        "hot-air balloons over a green valley at sunrise\n"
                        "без апсемплера ссылайтесь на боксы так: <balloon_1>"))
                    with gr.Row():
                        l_aspect = gr.Dropdown(ASPECT_RATIOS, value="1:1", label="Пропорции")
                        l_tier = gr.Dropdown(list(TIERS), value=DEFAULT_TIER, label="Площадь")
                    l_upsample, l_rewrite, l_go, l_cancel = action_row("▣ Сгенерировать")
                    l_out = result_column()
            l_aspect.change(box_editor.with_aspect, [l_editor, l_aspect], l_editor, queue=False,
                            show_progress="hidden")
            l_events = [
                l_rewrite.click(rewrite_layout, [client, l_prompt, l_editor, *settings], [l_prompt, l_upsample], show_progress_on=l_prompt),
                l_go.click(layout_generate, [client, l_prompt, l_editor, l_upsample, l_aspect, l_tier, *settings],
                           [l_out["gallery"], l_out["results"], l_out["caption"], l_out["info"], seed_box,
                            l_out["files"]], show_progress_on=l_out["gallery"]),
            ]
            l_cancel.click(lambda: "✖ Задача отменена", None, l_out["info"], cancels=l_events,
                            queue=False)
            outs.append((l_out, l_prompt, l_upsample))

        # ---------------- Точечная правка боксами ----------------
        with gr.Tab("🎯 Правка боксами"):
            gr.Markdown("Загрузите картинку и **протяните мышью по ней — появится бокс**. Действие бокса — кнопки в его "
                        "строке: ➕ новый элемент, ↔ перенести (перетащите рамку «→ сюда»), 📌 не трогать, ✖ убрать. "
                        "Перенос — за подпись бокса, размер — за маркеры. Опишите правку и в инструкции.",
                        elem_classes="hint")
            with gr.Row():
                with gr.Column(scale=7):
                    b_images = gr.Gallery(label="Изображения (первое — редактируемое)", type="pil", format="png",
                                          interactive=True, columns=6, height=130)
                    b_editor = box_editor.EditBoxEditor()
                with gr.Column(scale=5):
                    b_prompt = gr.Textbox(label="Инструкция", lines=4, placeholder=(
                        "add a small potted cactus in the marked spot, keep the rest unchanged\n"
                        "move the knight up and to the left"))
                    with gr.Row():
                        b_follow = gr.Number(value=1, precision=0, minimum=1, maximum=MAX_REFERENCE_IMAGES,
                                             label="Пропорции как у №", scale=1)
                        b_tier = gr.Dropdown(list(TIERS), value=DEFAULT_TIER, label="Площадь", scale=2)
                    b_upsample, b_rewrite, b_go, b_cancel = action_row("🎯 Применить")
                    b_out = result_column()
            b_images.change(lambda g, v: box_editor.with_refs(v, gallery_images(g)), [b_images, b_editor], b_editor,
                            show_progress="hidden")
            b_events = [
                b_rewrite.click(rewrite_box_edit, [client, b_images, b_prompt, b_editor, *settings],
                                [b_prompt, b_upsample], show_progress_on=b_prompt),
                b_go.click(box_edit_generate, [client, b_images, b_prompt, b_editor, b_upsample, b_follow, b_tier,
                                               *settings],
                           [b_out["gallery"], b_out["results"], b_out["caption"], b_out["info"], seed_box,
                            b_out["files"]], show_progress_on=b_out["gallery"]),
            ]
            b_cancel.click(lambda: "✖ Задача отменена", None, b_out["info"], cancels=b_events,
                            queue=False)
            outs.append((b_out, b_prompt, b_upsample))

        # ---------------- История ----------------
        with gr.Tab("🕘 История"):
            h_metas, h_selected = gr.State([]), gr.State(None)
            with gr.Row():
                with gr.Column(scale=7):
                    h_gallery = gr.Gallery(label=f"{'Все' if HISTORY_MODE == 'all' else 'Ваши'} последние генерации "
                                                 f"(до {HISTORY_LIMIT})",
                                           columns=5, height=620, interactive=False, allow_preview=True)
                with gr.Column(scale=4):
                    h_refresh = gr.Button("↻ Обновить")
                    h_meta = gr.JSON(label="Параметры")
                    h_to_refs = gr.Button("↪ В референсы")
                    h_to_t2i = gr.Button("↪ Описание и seed в «Генерация»")
            h_refresh.click(history_refresh, client, [h_gallery, h_metas, h_selected, h_meta], queue=False)
            h_gallery.select(history_select, h_metas, [h_selected, h_meta], queue=False)
            h_to_refs.click(history_to_refs, [h_selected, h_metas, e_images, b_images], [e_images, b_images])
            h_to_t2i.click(history_to_t2i, [h_selected, h_metas], [t_prompt, t_upsample, seed_box, randomize_box])
            demo.load(init_client, client, client, queue=False).then(
                history_refresh, client, [h_gallery, h_metas, h_selected, h_meta], queue=False)

        with gr.Tab("❔ Справка"):
            gr.Markdown(HELP_MD)

        # Кнопки под результатами всех вкладок
        for out, prompt_box, upsample_box in outs:
            out["to_refs"].click(results_to_refs, [out["results"], out["selected"], e_images, b_images],
                                 [e_images, b_images])
            out["to_prompt"].click(lambda c: (c, False) if c else (gr.skip(), gr.skip()), out["caption"],
                                   [prompt_box, upsample_box], queue=False)
    return demo


THEME = gr.themes.Soft(
    primary_hue="violet", secondary_hue="fuchsia", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    ap.add_argument("--share", action="store_true", help="Публичная ссылка *.gradio.live")
    ap.add_argument("--auth", help="Логин и пароль в виде user:pass (обязательно с --share)")
    args = ap.parse_args()
    auth = tuple(args.auth.split(":", 1)) if args.auth else None
    if args.share and not auth:
        print("[app] ВНИМАНИЕ: публичная ссылка без --auth — GPU доступен любому, у кого есть ссылка")

    demo = build_ui()
    # Очередь к видеокарте держит GpuScheduler: обработчики Gradio только ждут свою задачу, поэтому их
    # одновременно может быть сколько угодно (async — без занятых потоков).
    demo.queue(default_concurrency_limit=None)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, auth=auth, theme=THEME, css=CSS,
                allowed_paths=[str(OUTPUT_DIR.resolve())])


if __name__ == "__main__":
    main()
