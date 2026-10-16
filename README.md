# FLUX 3 Image Klein Studio

Gradio-интерфейс ко всем возможностям [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) —
открытой 5.6B-модели Black Forest Labs (релиз 15 октября 2026). Рассчитан на одну RTX 3090/4090 (24 GB) в облаке.

| Вкладка | Что делает |
|---|---|
| 🎨 Генерация | Текст → изображение: 15 пропорций и тиры площади как в BFL API (768² / 1K / 1.5K / 2K / 4K) или свой размер кратно 16 |
| 🖌️ Редактирование | Правка по инструкции и композиция из 1–10 референсов («the first image», «the second image»…); размер как у №N, по пропорциям или вручную |
| ▣ Раскладка | Раскладка боксами: два клика по холсту — бокс, у каждого id и описание; апсемплер пишет сцену вокруг них |
| 🎯 Правка боксами | Точечная правка: ➕ новый / заменить, ↔ переместить, 📌 оставить как есть, ✖ удалить — по областям на референсе |
| 🕘 История | Все генерации из `outputs/` с параметрами; вернуть картинку в референсы или описание+seed в генерацию |

Общее — в боковой панели: модель **Distilled** (16 шагов, без CFG) или **Base** (50 шагов, CFG, negative prompt),
шаги, seed, число картинок, длина описания апсемплера, **LoRA** (несколько, с весами).

**✨ Апсемплер промптов** — официальная модель BFL [flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)
(9B, Qwen3.5-9B). BFL прямо просит не использовать Klein без него: модель обучена на подробных описаниях сцены,
короткий промпт «как есть» даёт заметно более слабый результат. Апсемплер видит референсы и сохраняет боксы.
Описание, ушедшее в модель, показывается под результатом; «✨ Только переписать» кладёт его в поле промпта, чтобы
поправить руками и перебирать seed без повторного переписывания. Одинаковые запросы кэшируются.

Под каждым результатом: «↪ В референсы» (итеративная правка: результат → следующий шаг) и «↪ Описание в промпт».
Каждая генерация сохраняется в `outputs/` — PNG с параметрами в метаданных и JSON рядом.

## Файлы

```
app.py         интерфейс и обработчики вкладок
backend.py     KleinBackend (Diffusers ModularPipeline, CUDA) и MockBackend (заглушка), размеры, LoRA
upsampler.py   апсемплер промптов: локально, через vLLM-сервер или заглушка; кэш описаний
layout.py      боксы: формат пайплайна (0–1) и модели (0–1000), таблица, отрисовка на холсте
setup.sh       установка «в одну команду» на GPU-сервер (Vast.ai или любая Linux-машина)
```

## Быстрая установка на GPU-сервер

Веса Klein закрыты лицензией (FLUX Kommunity License, одобряется автоматически): один раз нажмите
«Agree» на странице [flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
(и [flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base), если нужен Base)
и создайте [токен](https://huggingface.co/settings/tokens) с правом чтения.

```bash
git clone https://github.com/GrafikXxxxxxxYyyyyyyyyyy/flux-3-image-klein-studio.git
cd flux-3-image-klein-studio && HF_TOKEN=hf_... bash setup.sh
```

`setup.sh` ставит зависимости, качает веса (~39 GB: Distilled ~20 GB + апсемплер ~19 GB; `KLEIN_BASE=1` — ещё ~11 GB),
запускает приложение и открывает публичную ссылку через Cloudflare quick tunnel (**без пароля**: GPU доступен любому,
у кого есть ссылка; `KLEIN_PUBLIC=0` — без ссылки). В конце печатает адрес. На Vast.ai приложение и туннель ставятся
сервисами supervisor (`klein-studio`, `klein-studio-tunnel`), на других машинах — фоновыми процессами с логами в `logs/`.
Повторный запуск безопасен: обновляет код и перезапускает приложение, ссылка остаётся прежней. Опции — в шапке `setup.sh`.

## Запуск вручную

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login
python app.py --host 127.0.0.1 --port 7860
```

Открыть с вашей машины — через SSH-туннель (безопаснее публичной ссылки): `ssh -L 7860:localhost:7860 user@server`,
дальше http://localhost:7860. Альтернатива — `python app.py --share --auth логин:пароль`
(публичная ссылка `*.gradio.live`; без `--auth` GPU доступен любому, у кого есть ссылка).

## Память и железо

Веса в bf16: трансформер 11 GB, текстовый энкодер Qwen3-VL-4B 8 GB, апсемплер 19 GB, VAE 0.3 GB — всего ~39 GB
(+11 GB трансформер Base). Энкодер и VAE у Distilled и Base общие и грузятся один раз.

- **≥ 48 GB VRAM** (A6000, L40S, A100, H100) — всё в видеопамяти (`KLEIN_OFFLOAD=none` выбирается сам).
- **24 GB (RTX 3090/4090)** — `ComponentsManager` из Diffusers сам держит в видеопамяти то, что влезает, остальное — в RAM.
  Во время денойзинга на карте трансформер (+ энкодер, если остаётся место); апсемплер на время работы вытесняет их.
  Большие выводы (> 4 MP) сразу декодируются VAE тайлами, при нехватке памяти — повтор с тайлами.
- **RAM** — от 48 GB (с Base — 64 GB): выгруженные модели лежат в оперативной памяти.
- **Диск** — ~45 GB под веса.

Цифры BFL для Distilled на 1 MP: ~4 с на RTX 5090, < 9 с на RTX 4070. На 3090 ожидаемо того же порядка плюс
несколько секунд на подмену апсемплера в видеопамяти (оценка, на самой 3090 не проверял). Чтобы не ждать апсемплер
на каждом seed, жмите «✨ Только переписать» и генерируйте без него — описание уже полное.

## Отладка интерфейса без GPU (Mac)

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r requirements-mock.txt
KLEIN_MOCK=1 .venv/bin/python app.py
```

Вместо модели рисуются плейсхолдеры нужного размера с параметрами запроса, вместо апсемплера — заглушка: видно,
какой промпт, размер, боксы и референсы ушли бы в модель.

## Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `KLEIN_MOCK` | — | `1` — заглушка вместо модели |
| `KLEIN_MODEL_ID` | `black-forest-labs/flux-3-image-klein` | Distilled: id на Hub или локальная папка |
| `KLEIN_BASE_MODEL_ID` | `black-forest-labs/flux-3-image-klein-base` | Base |
| `KLEIN_DEFAULT_MODEL` | `distilled` | какая модель грузится при старте (`base` — Base) |
| `KLEIN_PRELOAD_BASE` | — | `1` — грузить Base сразу, а не при первом выборе |
| `KLEIN_OFFLOAD` | `auto` | `auto` — `none` от 46 GB VRAM, иначе автоматическая выгрузка в RAM; `none` — всё на GPU |
| `KLEIN_MEMORY_MARGIN` | `6GB` на < 32 GB, иначе `4GB` | сколько видеопамяти менеджер оставляет под активации. Меньше — энкодер чаще остаётся на карте (быстрее), но 2K–4K могут упасть в OOM |
| `KLEIN_FP8_STORAGE` | — | `1` — веса трансформера хранятся в FP8 (5.6 GB вместо 11), считаются в bf16; работает и на Ampere, качество чуть ниже |
| `KLEIN_UPSAMPLER` | `local` | `local` / `server` / `off` |
| `KLEIN_UPSAMPLER_ID` | `black-forest-labs/flux-3-image-klein-pup` | модель апсемплера |
| `KLEIN_UPSAMPLER_URL` | — | OpenAI-совместимый адрес, например `http://localhost:8100/v1` (включает `server`) |
| `KLEIN_UPSAMPLER_NAME` | = `KLEIN_UPSAMPLER_ID` | `served-model-name` на сервере |
| `KLEIN_OUTPUT_DIR` | `outputs` | куда сохранять результаты |

## Апсемплер на отдельном vLLM-сервере (необязательно)

Если есть вторая видеокарта, апсемплер быстрее и не вытесняет Klein из видеопамяти. В отдельном окружении:

```bash
pip install vllm
CUDA_VISIBLE_DEVICES=1 vllm serve black-forest-labs/flux-3-image-klein-pup --port 8100 \
    --max-model-len 8192 --limit-mm-per-prompt '{"image": 10}'
```

и запустить студию с `KLEIN_UPSAMPLER_URL=http://localhost:8100/v1`. Системные промпты, формат запроса, сэмплинг
(жадный, presence penalty 1.5 для текста) и сверка боксов — те же, что в коде BFL и в Diffusers.

## Боксы: формат

В интерфейсе бокс — `[y0, x0, y1, x1]` в долях кадра (0–1, начало в левом верхнем углу), так их принимает
пайплайн Diffusers. Модель читает их в конце промпта JSON-таблицей на холсте 0–1000:

```text
Hot-air balloons over a green valley <balloon_1> ... [{"id": "balloon_1", "bbox": [80, 60, 480, 380], "desc": "a red balloon"}]
```

Для правки строка ещё указывает источник: `"from": "target"` — новый элемент в `tgt_bbox`;
`"from": "ref_image_0"` — элемент первого референса, `src_bbox` — где он сейчас, `tgt_bbox` — где будет
(равные боксы — оставить на месте, `null` — удалить). С апсемплером всё это пишется само; без него студия
дописывает таблицу к вашему тексту, а `<id>` в тексте нужно поставить самим.

## Лицензия

Код студии — Apache 2.0 (`layout.py` повторяет логику `src/flux3/bbox.py` из
[black-forest-labs/flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein), тоже Apache 2.0).
Веса Klein и Klein Base — [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md):
бесплатно для некоммерческого использования и для коммерческого при годовой выручке до $5 млн.
Апсемплер промптов — Apache 2.0.
