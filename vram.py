"""Видеопамять одной карты для нескольких больших моделей: трансформеры Distilled/Base, энкодер, апсемплер.

Вместо `ComponentsManager` из Diffusers. У него на 24 GB две беды: он считает свободной только память, которую
PyTorch ещё не взял у драйвера (кэш аллокатора для него занят), и поэтому выгружает модели, даже когда они
помещаются; а сама выгрузка — `module.to("cpu")` в обычную память и обратно — идёт ~1.7 GB/s.

Здесь веса каждой модели один раз копируются в закреплённую (pinned) RAM и там и живут. Загрузка на GPU —
копирование оттуда (~13 GB/s на PCIe 4.0 x8, трансформер 11 GB — меньше секунды), выгрузка — просто освободить
копию на GPU: веса в инференсе не меняются, переписывать их обратно в RAM не нужно. Модель попадает на карту
в момент вызова (хук на forward); если места не хватает с учётом запаса под активации — с карты уходят давно
не использованные модели. Запас под активации задаёт вызывающий код под конкретный запрос (`set_reserve`),
поэтому на 1 MP трансформер и энкодер живут на карте вместе, а на 4K энкодер уступает место.
"""

from __future__ import annotations

import itertools
import os
import time
from collections import OrderedDict

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device

GB = 2**30
# Сверх запаса под активации: фрагментация и мелкие выделения, которые не видно заранее.
SAFETY_BYTES = int(float(os.environ.get("KLEIN_MEMORY_MARGIN_GB", 0.6)) * GB)


def _tensors(module: torch.nn.Module):
    seen = set()
    for t in itertools.chain(module.parameters(), module.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            yield t


def _nbytes(module: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in _tensors(module))


class _OnDemandHook(ModelHook):
    """Перед forward кладёт модель на карту. `execution_device` по нему пайплайны Diffusers узнают,
    где создавать входы, пока сама модель ещё в RAM."""

    def __init__(self, pool: "VramPool", name: str) -> None:
        self.pool, self.name = pool, name
        self.execution_device = pool.device

    def pre_forward(self, module, *args, **kwargs):
        self.pool.load(module)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


class VramPool:
    def __init__(self, device: str = "cuda", pin: bool = True) -> None:
        self.device = torch.device(device)
        self.pin = pin
        self.names: dict[int, str] = {}  # id(модуль) → имя
        self.resident: OrderedDict[int, torch.nn.Module] = OrderedDict()  # на карте, давно использованные первыми
        self.reserve: dict[str, int] = {}  # имя → запас под активации, байт
        self.loads = 0  # счётчики — для лога и проверки, что лишних перекладываний нет
        self.load_seconds = 0.0

    # ------------------------------------------------------------------ #
    def register(self, module: torch.nn.Module, name: str) -> None:
        """Переносит веса в закреплённую RAM и вешает хук. Модель остаётся в RAM до первого вызова."""
        if id(module) in self.names:
            return
        started = time.time()
        for t in _tensors(module):
            t._klein_host = self._host_copy(t)
            t.data = t._klein_host
        add_hook_to_module(module, _OnDemandHook(self, name))
        self.names[id(module)] = name
        print(f"[vram] {name}: {_nbytes(module) / GB:.1f} GB в {'закреплённой ' if self.pin else ''}RAM "
              f"за {time.time() - started:.0f} с", flush=True)

    def set_reserve(self, name: str, nbytes: float) -> None:
        self.reserve[name] = int(nbytes)

    def load(self, module: torch.nn.Module, reserve: int | None = None) -> None:
        key = id(module)
        if key in self.resident:
            self.resident.move_to_end(key)
            return
        name = self.names.get(key, type(module).__name__)
        size = _nbytes(module)
        self.make_room(size + (self.reserve.get(name, 0) if reserve is None else reserve), keep=module)
        started = time.time()
        for t in _tensors(module):
            if not hasattr(t, "_klein_host"):  # появился после регистрации (адаптер LoRA)
                t._klein_host = self._host_copy(t)
            t.data = t._klein_host.to(self.device, non_blocking=self.pin)
        torch.cuda.current_stream(self.device).synchronize()
        self.resident[key] = module
        self.loads += 1
        self.load_seconds += time.time() - started

    def unload(self, module: torch.nn.Module) -> None:
        if self.resident.pop(id(module), None) is None:
            return
        for t in _tensors(module):
            if not hasattr(t, "_klein_host"):
                t._klein_host = self._host_copy(t)
            t.data = t._klein_host

    def make_room(self, nbytes: int, keep: torch.nn.Module | None = None) -> None:
        """Освобождает на карте `nbytes` (+ запас на фрагментацию), выгружая давно не использованные модели."""
        need = nbytes + SAFETY_BYTES
        if self.free_bytes() >= need:
            return
        for key, module in list(self.resident.items()):
            if module is keep:
                continue
            self.unload(module)
            torch.cuda.empty_cache()
            if self.free_bytes() >= need:
                return

    def unload_all(self, keep: torch.nn.Module | None = None) -> None:
        for module in list(self.resident.values()):
            if module is not keep:
                self.unload(module)
        torch.cuda.empty_cache()

    def free_bytes(self) -> int:
        """Свободно у драйвера + кэш аллокатора PyTorch, который уже ничем не занят."""
        free, _ = torch.cuda.mem_get_info(self.device)
        return free + torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)

    def summary(self) -> str:
        on_gpu = ", ".join(self.names.get(k, "?") for k in self.resident) or "—"
        return f"на карте: {on_gpu} · свободно {self.free_bytes() / GB:.1f} GB"

    # ------------------------------------------------------------------ #
    def _host_copy(self, t: torch.Tensor) -> torch.Tensor:
        host = t.data if t.device.type == "cpu" else t.data.to("cpu")
        if self.pin and not host.is_pinned():
            try:
                host = host.pin_memory()
            except RuntimeError as e:  # закреплённой памяти не хватило — работаем медленнее, но работаем
                print(f"[vram] не удалось закрепить память ({e}) — дальше без pinned", flush=True)
                self.pin = False
        return host
