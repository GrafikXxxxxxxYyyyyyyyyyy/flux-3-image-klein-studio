"""Bounding boxes FLUX 3 Image Klein: раскладка (text-to-image) и точечная правка (edit).

В интерфейсе бокс хранится в долях кадра — `[y_min, x_min, y_max, x_max]` в 0–1, начало в левом верхнем углу,
как принимает `boxes` пайплайн Diffusers. Модель читает их в промпте как JSON-таблицу на холсте 0–1000;
`canvas_rows` и `compose_caption` нужны, когда промпт уходит в модель без апсемплера.

Модуль не зависит от diffusers и torch — работает и в режиме заглушки.
Логика `split_caption` / `canvas_rows` повторяет `src/flux3/bbox.py` из
https://github.com/black-forest-labs/flux-3-image-klein (Apache-2.0).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from PIL import Image, ImageDraw, ImageFont

CANVAS = 1000
BOX_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

# Действия точечной правки. `from` — откуда элемент: "target" — новый, рисуется в целевой области;
# "ref_image_N" — элемент N-го референса (src_bbox — где он сейчас, tgt_bbox — где будет; None — удалить).
KIND_NEW = "new"
KIND_MOVE = "move"
KIND_KEEP = "keep"
KIND_REMOVE = "remove"
KIND_LAYOUT = "layout"
EDIT_KINDS = {
    KIND_NEW: "➕ Новый / заменить",
    KIND_MOVE: "↔ Переместить",
    KIND_KEEP: "📌 Оставить как есть",
    KIND_REMOVE: "✖ Удалить",
}
KIND_BY_LABEL = {label: kind for kind, label in EDIT_KINDS.items()} | {"▣ Раскладка": KIND_LAYOUT}
KIND_LABEL = EDIT_KINDS | {KIND_LAYOUT: "▣ Раскладка"}
KIND_TAG = {KIND_NEW: " +new", KIND_MOVE: " move", KIND_KEEP: " keep", KIND_REMOVE: " remove"}  # подписи на холсте

COLORS = ["#ff4d6d", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8", "#ff922b", "#22b8cf", "#94d82d",
          "#f783ac", "#748ffc", "#ffa94d", "#63e6be"]

EDIT_HEADERS = ["id", "действие", "№", "где сейчас (src): y0, x0, y1, x1", "где будет (tgt): y0, x0, y1, x1",
                "описание"]
LAYOUT_HEADERS = ["id", "бокс: y0, x0, y1, x1", "описание"]


@dataclass
class Box:
    id: str
    kind: str = KIND_LAYOUT
    ref: int = 0  # индекс референса, из которого элемент (для move / keep / remove)
    src: list[float] | None = None
    tgt: list[float] | None = None
    desc: str = ""

    def to_pipeline(self) -> dict:
        """Формат аргумента `boxes` пайплайна Diffusers (координаты 0–1)."""
        if self.kind == KIND_LAYOUT:
            return {"id": self.id, "desc": self.desc, "bbox": self.tgt}
        origin = "target" if self.kind == KIND_NEW else f"ref_image_{self.ref}"
        src = None if self.kind == KIND_NEW else self.src
        tgt = None if self.kind == KIND_REMOVE else (self.src if self.kind == KIND_KEEP else self.tgt)
        return {"id": self.id, "desc": self.desc, "from": origin, "src_bbox": src, "tgt_bbox": tgt}


def boxes_from_state(state: list[dict] | None) -> list[Box]:
    return [Box(**b) for b in state or []]


def boxes_to_state(boxes: list[Box]) -> list[dict]:
    return [asdict(b) for b in boxes]


def pipeline_boxes(state: list[dict] | None) -> list[dict] | None:
    boxes = [b for b in boxes_from_state(state) if b.src or b.tgt]
    return [b.to_pipeline() for b in boxes] or None


# --------------------------------------------------------------------------- #
# Координаты
# --------------------------------------------------------------------------- #
def normalize(p1: tuple[float, float], p2: tuple[float, float], width: int, height: int) -> list[float] | None:
    """Две точки (x, y) в пикселях → [y0, x0, y1, x1] в 0–1. None для вырожденного бокса."""
    (xa, ya), (xb, yb) = p1, p2
    x0, x1 = sorted((max(0.0, min(1.0, xa / width)), max(0.0, min(1.0, xb / width))))
    y0, y1 = sorted((max(0.0, min(1.0, ya / height)), max(0.0, min(1.0, yb / height))))
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        return None
    return [round(y0, 4), round(x0, 4), round(y1, 4), round(x1, 4)]


def to_grid(box: list[float] | None) -> list[int] | None:
    return None if box is None else [round(max(0.0, min(1.0, float(v))) * CANVAS) for v in box]


def canvas_rows(pipeline: list[dict]) -> list[dict]:
    """Боксы пайплайна (0–1) → строки таблицы для модели (0–1000), как `boxes_to_canvas` в Diffusers."""
    rows = []
    for box in pipeline:
        if box.get("bbox"):
            rows.append({"id": box["id"], "bbox": to_grid(box["bbox"]), "desc": box.get("desc", "")})
        else:
            rows.append({"id": box["id"], "from": str(box.get("from") or ""), "src_bbox": to_grid(box.get("src_bbox")),
                         "tgt_bbox": to_grid(box.get("tgt_bbox")), "desc": box.get("desc", "")})
    return rows


def split_caption(text: str) -> tuple[str, list | None]:
    """(текст, строки) — строки это JSON-список в конце текста, если он есть."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def has_box_table(text: str) -> bool:
    _, rows = split_caption(text)
    return bool(rows) and all(isinstance(r, dict) and isinstance(r.get("id"), str) for r in rows)


def compose_caption(prompt: str, pipeline: list[dict] | None) -> str:
    """Промпт без апсемплера: если боксы заданы, а таблицы в тексте нет — дописываем её (модель ждёт
    `<id>` в тексте и JSON-таблицу на холсте 0–1000 в конце)."""
    prompt = prompt.strip()
    if not pipeline or has_box_table(prompt):
        return prompt
    return f"{prompt} {json.dumps(canvas_rows(pipeline), ensure_ascii=False)}"


def missing_mentions(prompt: str, state: list[dict] | None) -> list[str]:
    """id боксов, на которые промпт не ссылается как <id> (важно только без апсемплера)."""
    return [b.id for b in boxes_from_state(state) if f"<{b.id}>" not in prompt]


def next_id(state: list[dict] | None, stem: str = "obj") -> str:
    taken = {b["id"] for b in state or []}
    stem = re.sub(r"[^A-Za-z0-9_]", "_", stem.strip().lower()) or "obj"
    if not stem[0].isalpha():
        stem = f"obj_{stem}"
    n = 1
    while f"{stem}_{n}" in taken:
        n += 1
    return f"{stem}_{n}"


# --------------------------------------------------------------------------- #
# Таблица в интерфейсе
# --------------------------------------------------------------------------- #
def _fmt(box: list[float] | None) -> str:
    return "—" if box is None else ", ".join(f"{v:.3f}" for v in box)


def _parse(text) -> list[float] | None:
    text = str(text or "").strip()
    if text in ("", "—", "-", "None", "null"):
        return None
    values = [float(v) for v in re.split(r"[,\s;]+", text.strip("[]() ")) if v]
    if len(values) != 4:
        raise ValueError(f"бокс — 4 числа y0, x0, y1, x1, получено: {text!r}")
    if max(values) > 1.0:  # пользователь ввёл холст 0–1000
        values = [v / CANVAS for v in values]
    y0, x0, y1, x1 = (max(0.0, min(1.0, v)) for v in values)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"у бокса y1 > y0 и x1 > x0, получено: {text!r}")
    return [round(v, 4) for v in (y0, x0, y1, x1)]


def to_table(state: list[dict] | None, edit: bool = True) -> list[list]:
    boxes = boxes_from_state(state)
    if not edit:
        return [[b.id, _fmt(b.tgt), b.desc] for b in boxes]
    return [[b.id, KIND_LABEL.get(b.kind, b.kind), "—" if b.kind == KIND_NEW else f"№{b.ref + 1}",
             _fmt(b.src), _fmt(b.tgt), b.desc] for b in boxes]


def from_table(table, state: list[dict] | None, edit: bool = True) -> list[dict]:
    """Правки таблицы (id, координаты, описание) → состояние. Действие и референс меняются только кнопками."""
    rows = table.values.tolist() if hasattr(table, "values") else list(table or [])
    old = boxes_from_state(state)
    boxes = []
    for i, row in enumerate(rows):
        if not row or not str(row[0]).strip():
            continue
        prev = old[i] if i < len(old) else Box(id=str(row[0]))
        box_id = str(row[0]).strip()
        if not BOX_ID.fullmatch(box_id):
            raise ValueError(f"id «{box_id}»: латиница, цифры и _, начинается с буквы (например dog_1)")
        prev.id = box_id
        if edit:
            prev.src, prev.tgt, prev.desc = _parse(row[3]), _parse(row[4]), str(row[5] or "").strip()
            if prev.kind == KIND_KEEP:
                prev.tgt = prev.src
        else:
            prev.tgt, prev.desc = _parse(row[1]), str(row[2] or "").strip()
        boxes.append(prev)
    ids = [b.id for b in boxes]
    if len(ids) != len(set(ids)):
        raise ValueError("id боксов должны быть уникальны")
    return boxes_to_state(boxes)


# --------------------------------------------------------------------------- #
# Отрисовка
# --------------------------------------------------------------------------- #
PREVIEW_SIDE = 1024


def blank_canvas(aspect: float) -> Image.Image:
    """Пустой холст с сеткой 10 % — для раскладки без исходной картинки."""
    if aspect >= 1:
        w, h = PREVIEW_SIDE, max(64, round(PREVIEW_SIDE / aspect))
    else:
        w, h = max(64, round(PREVIEW_SIDE * aspect)), PREVIEW_SIDE
    im = Image.new("RGB", (w, h), (38, 40, 46))
    draw = ImageDraw.Draw(im)
    for i in range(1, 10):
        color = (70, 74, 84) if i != 5 else (96, 100, 112)
        draw.line([(w * i / 10, 0), (w * i / 10, h)], fill=color, width=1)
        draw.line([(0, h * i / 10), (w, h * i / 10)], fill=color, width=1)
    return im


def fit_preview(image: Image.Image) -> Image.Image:
    im = image.convert("RGB")
    if max(im.size) > PREVIEW_SIDE:
        im = im.copy()
        im.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.Resampling.LANCZOS)
    return im


def _rect(box: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    y0, x0, y1, x1 = box
    return x0 * w, y0 * h, x1 * w, y1 * h


def _dashed_rect(draw: ImageDraw.ImageDraw, rect, color: str, width: int, dash: int = 10) -> None:
    x0, y0, x1, y1 = rect
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(1, int(length // dash))
        for i in range(0, steps, 2):
            t0, t1 = i / steps, min(1.0, (i + 1) / steps)
            draw.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)],
                      fill=color, width=width)


def render(base: Image.Image, state: list[dict] | None, shown_ref: int | None = None,
           pending: list | None = None, pending_box: list[float] | None = None) -> Image.Image:
    """Боксы поверх картинки. Сплошная рамка — где элемент будет (tgt), пунктир — где он сейчас (src).

    shown_ref — какой референс на холсте: src-боксы других референсов не рисуются.
    pending — первая точка незаконченного бокса (x, y в пикселях холста), pending_box — src перемещения,
    для которого ещё не задано новое место.
    """
    im = base.convert("RGB").copy()
    w, h = im.size
    # Заливки — отдельным слоем: ImageDraw на RGBA заменяет пиксели, и заливка большого бокса
    # стёрла бы контуры уже нарисованных.
    fills = Image.new("RGBA", im.size, (0, 0, 0, 0))
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    fill_draw, draw = ImageDraw.Draw(fills), ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=max(14, min(w, h) // 22))
    line = max(3, min(w, h) // 220)

    for i, box in enumerate(boxes_from_state(state)):
        color = COLORS[i % len(COLORS)]
        src_visible = box.src is not None and (shown_ref is None or box.ref == shown_ref)
        tgt = box.src if box.kind == KIND_KEEP else box.tgt
        label_at = None
        if src_visible and box.kind in (KIND_MOVE, KIND_REMOVE):
            rect = _rect(box.src, w, h)
            _dashed_rect(draw, rect, color, line)
            if box.kind == KIND_REMOVE:
                draw.line([rect[:2], rect[2:]], fill=color, width=line)
                draw.line([(rect[0], rect[3]), (rect[2], rect[1])], fill=color, width=line)
            label_at = rect
        if tgt is not None and (box.kind != KIND_KEEP or src_visible):
            rect = _rect(tgt, w, h)
            fill_draw.rectangle(rect, fill=color + "1c")
            draw.rectangle(rect, outline=color, width=line)
            label_at = rect
            if box.kind == KIND_MOVE and src_visible:
                a, b = _rect(box.src, w, h), rect
                draw.line([((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)],
                          fill=color, width=line)
        if label_at:
            text = box.id + KIND_TAG.get(box.kind, "")
            tx, ty = label_at[0] + line, label_at[1] + line
            bbox = draw.textbbox((tx, ty), text, font=font)
            draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0, 170))
            draw.text((tx, ty), text, fill=color, font=font)

    if pending_box is not None:
        _dashed_rect(draw, _rect(pending_box, w, h), "#ffffff", line)
    if pending:
        x, y = pending
        r = max(5, line * 3)
        draw.ellipse((x - r, y - r, x + r, y + r), outline="#ffffff", width=line)
        draw.line([(x - 2 * r, y), (x + 2 * r, y)], fill="#ffffff", width=1)
        draw.line([(x, y - 2 * r), (x, y + 2 * r)], fill="#ffffff", width=1)
    return Image.alpha_composite(Image.alpha_composite(im.convert("RGBA"), fills), overlay).convert("RGB")
