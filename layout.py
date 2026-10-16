"""Bounding boxes FLUX 3 Image Klein: раскладка (text-to-image) и точечная правка (edit) — формат для модели.
Рисование и правка боксов в интерфейсе — в `box_editor.py`.

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
from dataclasses import dataclass

CANVAS = 1000
BOX_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

# Действия точечной правки. `from` — откуда элемент: "target" — новый, рисуется в целевой области;
# "ref_image_N" — элемент N-го референса (src_bbox — где он сейчас, tgt_bbox — где будет; None — удалить).
KIND_NEW = "new"
KIND_MOVE = "move"
KIND_KEEP = "keep"
KIND_REMOVE = "remove"
KIND_LAYOUT = "layout"

COLORS = ["#ff4d6d", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8", "#ff922b", "#22b8cf", "#94d82d",
          "#f783ac", "#748ffc", "#ffa94d", "#63e6be"]


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


def pipeline_boxes(state: list[dict] | None) -> list[dict] | None:
    boxes = [b for b in boxes_from_state(state) if b.src or b.tgt]
    return [b.to_pipeline() for b in boxes] or None


# --------------------------------------------------------------------------- #
# Координаты
# --------------------------------------------------------------------------- #
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

