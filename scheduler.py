"""Очередь задач для одной видеокарты: много пользователей, одна GPU.

Обработчики Gradio не трогают модель сами — они ставят задачу в очередь и ждут её, показывая позицию, оценку
времени до старта и прогресс. Видеокарту занимает один рабочий поток, который берёт задачи по одной.

Задача проходит до двух фаз: «апсемплер» (переписать промпт) и «генерация» (Distilled или Base). На 24 GB
апсемплер (19 GB) и трансформер с энкодером не помещаются вместе, и смена фазы — это перекладывание десятков GB
между RAM и видеопамятью. Поэтому поток продолжает текущую фазу, пока в ней есть задачи (не больше
KLEIN_PHASE_STREAK подряд, чтобы остальные не ждали бесконечно), и только потом переключается на фазу самой
старой задачи. Пять пользователей подряд — один заход апсемплера и пять генераций, а не пять перекладываний.

Если апсемплер работает на отдельном сервере (KLEIN_UPSAMPLER=server), его фаза идёт в пуле потоков параллельно
с генерацией, а видеокарта занята только генерацией.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from PIL import Image

from backend import MODEL_LABELS, GenRequest
from backend import JobCancelled as Cancelled

QUEUE_MAX = int(os.environ.get("KLEIN_QUEUE_MAX", 32))
USER_MAX_JOBS = int(os.environ.get("KLEIN_USER_MAX_JOBS", 2))
PHASE_STREAK = int(os.environ.get("KLEIN_PHASE_STREAK", 6))

PUP = ("pup",)


class QueueFull(Exception):
    pass


@dataclass(eq=False)
class Job:
    owner: str
    request: str
    images: list[Image.Image]
    boxes: list[dict] | None
    upsample: bool
    max_tokens: int
    gen: GenRequest | None  # None — только переписать промпт
    caption: str | None = None  # готовое описание; None — его ещё пишет апсемплер
    cached: bool = False
    id: int = field(default_factory=itertools.count(1).__next__)
    created: float = field(default_factory=time.time)
    started: float | None = None  # когда задача впервые попала на видеокарту
    state: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = ""
    progress: float = 0.0
    upsample_time: float = 0.0
    remote_upsampling: bool = False
    results: list[Image.Image] | None = None
    error: BaseException | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)

    def phase(self) -> tuple | None:
        if self.caption is None:
            return PUP
        if self.gen is None or self.results is not None:
            return None
        return ("dit", self.gen.model)

    def units(self) -> float:
        """Объём генерации для оценки времени: шаги × мегапиксели × картинки × проходы CFG."""
        if self.gen is None:
            return 0.0
        w, h = self.gen.resolved_size()
        cfg = 2 if self.gen.model == "base" and self.gen.guidance > 1 else 1
        return self.gen.steps * w * h / 1e6 * self.gen.num_images * cfg


class GpuScheduler:
    def __init__(self, backend, upsampler, gpu_upsample: bool | None = None) -> None:
        self.backend = backend
        self.upsampler = upsampler
        self.gpu_upsample = upsampler.mode == "local" if gpu_upsample is None else gpu_upsample
        self._pool = None if self.gpu_upsample else ThreadPoolExecutor(4, thread_name_prefix="upsampler")
        self._cv = threading.Condition()
        self._queue: list[Job] = []  # ждут видеокарту, по порядку постановки
        self._current: Job | None = None
        self._phase: tuple | None = None
        self._streak = 0
        # Оценки длительности (экспоненциальное среднее): секунд на апсемплер, секунд на единицу генерации.
        self._pup_seconds = 12.0
        self._unit_seconds = {"distilled": 0.6, "base": 0.6}
        threading.Thread(target=self._loop, name="gpu-worker", daemon=True).start()

    # ------------------------------------------------------------------ #
    # API для обработчиков
    # ------------------------------------------------------------------ #
    def submit(self, job: Job) -> Job:
        if job.upsample and job.caption is None:
            cached = self.upsampler.cached(job.request, job.images, job.boxes, job.max_tokens)
            if cached is not None:
                job.caption, job.cached = cached, True
        if job.phase() is None:  # «только переписать» из кэша — очередь не нужна
            job.state = "done"
            job.done.set()
            return job
        with self._cv:
            if len(self._queue) >= QUEUE_MAX:
                raise QueueFull(f"Очередь заполнена ({QUEUE_MAX} задач) — попробуйте через минуту")
            mine = sum(j.owner == job.owner for j in self._queue) + (
                self._current is not None and self._current.owner == job.owner)
            if mine >= USER_MAX_JOBS:
                raise QueueFull(f"У вас уже {mine} задач(и) в работе — дождитесь их или отмените")
            self._queue.append(job)
            if job.phase() == PUP and not self.gpu_upsample:
                job.remote_upsampling = True
                self._pool.submit(self._remote_upsample, job)
            self._cv.notify_all()
        return job

    def cancel(self, job: Job) -> None:
        job.cancelled.set()  # идущую генерацию остановит хук шага в backend
        with self._cv:
            if job in self._queue:
                self._queue.remove(job)
                self._finish(job, "cancelled")

    def describe(self, job: Job) -> str:
        """Текст статуса задачи для пользователя."""
        with self._cv:
            if job.state == "running" or job.remote_upsampling:
                return job.stage or "Работает…"
            if job.state != "queued":
                return ""
            ahead = [j for j in self._queue if j.created < job.created and j is not job]
            wait = self._remaining(self._current) + sum(self._estimate(j) for j in ahead)
            position = len(ahead) + (self._current is not None) + 1
        return f"В очереди: {position}-я · старт ≈ через {_fmt_seconds(wait)}"

    def snapshot(self) -> str:
        """Общее состояние очереди — для строки статуса в шапке."""
        with self._cv:
            current, queued = self._current, list(self._queue)
            busy = self._remaining(current) + sum(self._estimate(j) for j in queued)
        if current is None and not queued:
            return "🟢 Видеокарта свободна"
        parts = []
        if current is not None:
            parts.append(f"⚙️ {current.stage or 'работает'}")
        if queued:
            parts.append(f"в очереди: {len(queued)}")
        parts.append(f"освободится ≈ через {_fmt_seconds(busy)}")
        return " · ".join(parts)

    # ------------------------------------------------------------------ #
    # Рабочий поток
    # ------------------------------------------------------------------ #
    def _next(self) -> Job | None:
        ready = [j for j in self._queue if not j.remote_upsampling]
        if not ready:
            return None
        if self._streak < PHASE_STREAK:
            for job in ready:
                if job.phase() == self._phase:
                    return job
        return ready[0]

    def _loop(self) -> None:
        while True:
            with self._cv:
                while (job := self._next()) is None:
                    self._cv.wait()
                phase = job.phase()
                self._streak = self._streak + 1 if phase == self._phase else 1
                self._phase = phase
                self._queue.remove(job)
                self._current = job
                job.state = "running"
                job.started = job.started or time.time()
            try:
                if phase == PUP:
                    self._run_upsample(job)
                else:
                    self._run_generate(job)
            except Cancelled:
                with self._cv:
                    self._finish(job, "cancelled")
            except BaseException as e:  # noqa: BLE001 — ошибку увидит пользователь, поток должен жить дальше
                job.error = e
                with self._cv:
                    self._finish(job, "failed")
            finally:
                with self._cv:
                    self._current = None
                    if job.state == "running":
                        if job.phase() is None:
                            self._finish(job, "done")
                        else:
                            # Следующая фаза: задача возвращается на своё место в очереди (по времени постановки).
                            job.state, job.stage, job.progress = "queued", "", 0.0
                            self._queue.append(job)
                            self._queue.sort(key=lambda j: j.created)
                    self._cv.notify_all()

    def _run_upsample(self, job: Job) -> None:
        if job.cancelled.is_set():
            raise Cancelled
        job.stage = "✨ Апсемплер пишет описание сцены"
        started = time.time()
        job.caption, job.cached = self.upsampler(job.request, job.images, job.boxes, job.max_tokens)
        job.upsample_time = time.time() - started
        if not job.cached:
            self._pup_seconds = 0.7 * self._pup_seconds + 0.3 * job.upsample_time

    def _remote_upsample(self, job: Job) -> None:
        job.stage = "✨ Апсемплер пишет описание сцены"
        started = time.time()
        try:
            if job.cancelled.is_set():
                raise Cancelled
            job.caption, job.cached = self.upsampler(job.request, job.images, job.boxes, job.max_tokens)
            job.upsample_time = time.time() - started
        except BaseException as e:  # noqa: BLE001
            with self._cv:
                if job.done.is_set():  # уже отменена
                    return
                if job in self._queue:
                    self._queue.remove(job)
                job.error = e if not isinstance(e, Cancelled) else None
                self._finish(job, "cancelled" if isinstance(e, Cancelled) else "failed")
                self._cv.notify_all()
            return
        with self._cv:
            job.remote_upsampling, job.stage = False, ""
            if job.phase() is None and job in self._queue:  # только переписать
                self._queue.remove(job)
                self._finish(job, "done")
            self._cv.notify_all()

    def _run_generate(self, job: Job) -> None:
        gen = job.gen
        gen.prompt = job.caption
        w, h = gen.resolved_size()
        label = f"{MODEL_LABELS[gen.model].split(' (')[0]} {w}×{h}"
        job.stage = f"Генерация {label}"

        def progress(fraction: float, desc: str) -> None:
            job.progress = fraction
            job.stage = f"{desc} · {label}" if desc else f"Генерация {label}"

        started = time.time()
        job.results = self.backend.generate(gen, progress=progress, should_stop=job.cancelled.is_set)
        if job.units():
            rate = (time.time() - started) / job.units()
            self._unit_seconds[gen.model] = 0.7 * self._unit_seconds[gen.model] + 0.3 * rate

    # ------------------------------------------------------------------ #
    def _finish(self, job: Job, state: str) -> None:
        job.state = state
        job.done.set()
        now = time.time()
        wait = (job.started or now) - job.created
        print(f"[queue] #{job.id} {job.owner[:8]}: {state} за {now - job.created:.1f} с (в очереди {wait:.1f} с)"
              + (f" — {type(job.error).__name__}: {job.error}" if job.error else "")
              + f" · ждут: {len(self._queue)}", flush=True)

    def _estimate(self, job: Job) -> float:
        pup = self._pup_seconds if job.caption is None else 0.0
        model = job.gen.model if job.gen else "distilled"
        return pup + job.units() * self._unit_seconds.get(model, 0.6)

    def _remaining(self, job: Job | None) -> float:
        if job is None:
            return 0.0
        if job.phase() == PUP:
            return max(1.0, self._pup_seconds - (time.time() - (job.started or time.time())))
        return self._estimate(job) * (1.0 - job.progress)


def _fmt_seconds(seconds: float) -> str:
    seconds = max(1, round(seconds))
    if seconds < 60:
        return f"{seconds} с"
    return f"{seconds // 60} мин {seconds % 60:02d} с"
