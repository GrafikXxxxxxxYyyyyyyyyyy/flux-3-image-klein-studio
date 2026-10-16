#!/usr/bin/env bash
# Полная установка FLUX 3 Image Klein Studio на GPU-сервер: зависимости, веса, автозапуск, публичная ссылка.
#
#   git clone https://github.com/GrafikXxxxxxxYyyyyyyyyyy/flux-3-image-klein-studio.git
#   cd flux-3-image-klein-studio && HF_TOKEN=hf_... bash setup.sh
#
# Веса Klein на Hugging Face закрыты лицензией (FLUX Kommunity License): один раз примите её на страницах
# https://huggingface.co/black-forest-labs/flux-3-image-klein (и -base, если нужен Base), а токен передайте
# в HF_TOKEN или войдите заранее через `hf auth login`.
#
# Повторный запуск безопасен: уже установленное и скачанное пропускается, приложение перезапускается
# с новым кодом, а туннель (и вместе с ним публичная ссылка) остаётся прежним.
#
# На инстансе Vast.ai (есть /opt/supervisor-scripts) приложение и туннель ставятся сервисами supervisor:
# переживают падения и перезапуск контейнера, логи — в /var/log/portal/. На любой другой машине оба
# процесса запускаются в фоне через nohup, логи — в ./logs/.
#
# Переменные (необязательно):
#   KLEIN_PUBLIC=0        без публичной ссылки (Cloudflare quick tunnel, без пароля — доступ у любого со ссылкой)
#   KLEIN_DOWNLOAD=0      не качать веса заранее (скачаются при первом запуске приложения)
#   KLEIN_BASE=1          скачать и Base (~11 GB; без этого Base скачается при первом выборе в интерфейсе)
#   KLEIN_UPSAMPLER=…     local (по умолчанию, +19 GB) / server / off — см. README
#   KLEIN_TEXT_ENCODER_ID=…  текстовый энкодер (по умолчанию Heretic-версия Qwen3-VL-4B; пусто — оригинальный из Klein)
#   KLEIN_PORT=17860      внутренний порт приложения (слушает только 127.0.0.1)
#   KLEIN_VENV=<путь>     venv для установки; по умолчанию /venv/main (образы Vast), иначе ./.venv
# Остальные настройки приложения (KLEIN_PIN_MEMORY, KLEIN_MEMORY_MARGIN_GB, …, см. README) на Vast читаются из
# ${WORKSPACE}/.env — после правки: supervisorctl restart klein-studio.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${KLEIN_PORT:-17860}"
PUBLIC="${KLEIN_PUBLIC:-1}"
DOWNLOAD="${KLEIN_DOWNLOAD:-1}"
BASE="${KLEIN_BASE:-0}"
UPSAMPLER="${KLEIN_UPSAMPLER:-local}"
MODEL_ID="${KLEIN_MODEL_ID:-black-forest-labs/flux-3-image-klein}"
BASE_ID="${KLEIN_BASE_MODEL_ID:-black-forest-labs/flux-3-image-klein-base}"
PUP_ID="${KLEIN_UPSAMPLER_ID:-black-forest-labs/flux-3-image-klein-pup}"
TEXT_ENCODER_ID="${KLEIN_TEXT_ENCODER_ID-DreamFast/Qwen3-VL-4b-Heretic}"   # без «:» — пустая строка значит оригинальный
VAST=0
[[ -d /opt/supervisor-scripts/utils && -d /etc/supervisor/conf.d ]] && command -v supervisorctl >/dev/null && VAST=1

step() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
step "Проверка окружения"
# --------------------------------------------------------------------------- #
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/GPU: /'
else
    warn "GPU не найдена — приложение запустится в режиме заглушки (картинки ненастоящие)"
fi
ram_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo 2>/dev/null || echo 0)
[[ "$ram_gb" -gt 0 && "$ram_gb" -lt 60 ]] && warn "RAM ${ram_gb} GB: веса моделей (~36 GB) живут в закреплённой RAM, нужно ≥ 64 GB (или KLEIN_PIN_MEMORY=0)"
cache_dir="${HF_HOME:-$HOME/.cache/huggingface}"
while [[ ! -d "$cache_dir" ]]; do cache_dir="$(dirname "$cache_dir")"; done   # кэша может ещё не быть
free_gb=$(df -BG --output=avail "$cache_dir" | tail -1 | tr -dc '0-9')
need_gb=$((20 + $([[ "$UPSAMPLER" == local ]] && echo 19 || echo 0) + $([[ "$BASE" == 1 ]] && echo 11 || echo 0)))
[[ "$DOWNLOAD" == 1 && "$free_gb" -lt $((need_gb + 5)) ]] && warn "Свободно ${free_gb} GB в $cache_dir, а весам нужно ~${need_gb} GB"
echo "Режим: $([[ $VAST == 1 ]] && echo 'Vast.ai (supervisor)' || echo 'обычная машина (nohup)')"

# --------------------------------------------------------------------------- #
step "Python-зависимости"
# --------------------------------------------------------------------------- #
VENV="${KLEIN_VENV:-}"
if [[ -z "$VENV" ]]; then
    if [[ -x /venv/main/bin/python ]]; then VENV=/venv/main; else VENV="$APP_DIR/.venv"; fi
fi
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Создаю venv: $VENV"
    python3 -m venv "$VENV" || die "Не удалось создать venv (нужен пакет python3-venv)"
fi
PY="$VENV/bin/python"
echo "venv: $VENV ($("$PY" --version))"
if command -v uv >/dev/null; then
    uv pip install --python "$PY" -r "$APP_DIR/requirements.txt"
else
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -r "$APP_DIR/requirements.txt"
fi
# Ядро свёртки для апсемплера (Qwen3.5): готовых колёс под свежий torch нет, собирается из исходников (~5 мин, нужен
# nvcc). Без него апсемплер работает, но медленнее — поэтому ошибка сборки не останавливает установку.
if ! "$PY" -c "import causal_conv1d" 2>/dev/null; then
    if command -v nvcc >/dev/null; then
        echo "Сборка causal-conv1d (несколько минут)…"
        arch=$("$PY" -c "import torch; c = torch.cuda.get_device_capability(); print(f'{c[0]}.{c[1]}')" 2>/dev/null || true)
        if command -v uv >/dev/null; then inst=(uv pip install --python "$PY"); else inst=("$PY" -m pip install); fi
        MAX_JOBS="$(nproc)" TORCH_CUDA_ARCH_LIST="${arch:-8.6}" CAUSAL_CONV1D_FORCE_BUILD=TRUE \
            "${inst[@]}" --no-build-isolation causal-conv1d >/dev/null 2>&1 \
            || warn "causal-conv1d не собрался — апсемплер будет медленнее"
    else
        warn "Нет nvcc — causal-conv1d не собрать, апсемплер будет медленнее"
    fi
fi
"$PY" - <<'EOF'
import torch, diffusers, transformers, gradio
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}, доступна: {torch.cuda.is_available()}), "
      f"diffusers {diffusers.__version__}, transformers {transformers.__version__}, gradio {gradio.__version__}")
from diffusers import ModularPipeline  # noqa: F401
from diffusers.modular_pipelines.flux3 import prompt_upsampler  # noqa: F401 — упадёт, если diffusers без FLUX 3
EOF

# --------------------------------------------------------------------------- #
step "Доступ к Hugging Face"
# --------------------------------------------------------------------------- #
HF="$VENV/bin/hf"
[[ -x "$HF" ]] || die "Не найден $HF (ставится вместе с huggingface_hub)"
if [[ -n "${HF_TOKEN:-}" ]]; then
    "$HF" auth login --token "$HF_TOKEN" >/dev/null 2>&1 || die "HF_TOKEN не подошёл"
fi
"$HF" auth whoami >/dev/null 2>&1 || die "Нет входа в Hugging Face. Примите лицензию на https://huggingface.co/$MODEL_ID и запустите: HF_TOKEN=hf_... bash setup.sh"
echo "Вход: $("$HF" auth whoami 2>/dev/null | head -1)"

# --------------------------------------------------------------------------- #
if [[ "$DOWNLOAD" == 1 ]]; then
    export HF_XET_HIGH_PERFORMANCE=1
    # Из репозиториев BFL берём только формат Diffusers: без single-file model.safetensors / ae.safetensors,
    # FP8/NVFP4-вариантов (для Ada/Blackwell) и картинок.
    SKIP=(--exclude "model.safetensors" --exclude "ae.safetensors" --exclude "variants/*" --exclude "*.jpg"
          --exclude "assets/*")
    step "Klein Distilled: трансформер, энкодер Qwen3-VL-4B, VAE (~20 GB)"
    # Со своим энкодером веса оригинального из репозитория Klein не нужны (токенизатор и конфиг берём оттуда).
    ENC_SKIP=()
    [[ -n "$TEXT_ENCODER_ID" ]] && ENC_SKIP=(--exclude "text_encoder/*.safetensors")
    "$HF" download "$MODEL_ID" "${SKIP[@]}" "${ENC_SKIP[@]}" --quiet >/dev/null \
        || die "Не удалось скачать $MODEL_ID — лицензия принята? https://huggingface.co/$MODEL_ID"
    echo "$MODEL_ID — готово"
    if [[ -n "$TEXT_ENCODER_ID" ]]; then
        "$HF" download "$TEXT_ENCODER_ID" --exclude "*.gguf" --exclude "graphs/*" --quiet >/dev/null \
            || die "Не удалось скачать энкодер $TEXT_ENCODER_ID"
        echo "$TEXT_ENCODER_ID — готово"
    fi
    if [[ "$BASE" == 1 ]]; then
        step "Klein Base: трансформер (~11 GB; энкодер и VAE общие с Distilled)"
        "$HF" download "$BASE_ID" "${SKIP[@]}" --exclude "text_encoder/*.safetensors" --exclude "vae/*" --quiet >/dev/null \
            || die "Не удалось скачать $BASE_ID — лицензия принята? https://huggingface.co/$BASE_ID"
        echo "$BASE_ID — готово"
    fi
    if [[ "$UPSAMPLER" == local ]]; then
        step "Апсемплер промптов (~19 GB)"
        "$HF" download "$PUP_ID" --quiet >/dev/null
        echo "$PUP_ID — готово"
    fi
fi

# --------------------------------------------------------------------------- #
step "Автозапуск"
# --------------------------------------------------------------------------- #
CLOUDFLARED=""
if [[ "$PUBLIC" == 1 ]]; then
    CLOUDFLARED="$(command -v cloudflared || true)"
    [[ -z "$CLOUDFLARED" && -x /opt/instance-tools/bin/cloudflared ]] && CLOUDFLARED=/opt/instance-tools/bin/cloudflared
    if [[ -z "$CLOUDFLARED" ]]; then
        case "$(uname -m)" in x86_64) arch=amd64 ;; aarch64) arch=arm64 ;; *) die "cloudflared: неизвестная архитектура" ;; esac
        CLOUDFLARED="$APP_DIR/.bin/cloudflared"
        mkdir -p "$APP_DIR/.bin"
        curl -fsSL -o "$CLOUDFLARED" \
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}"
        chmod +x "$CLOUDFLARED"
    fi
fi
TUNNEL_CMD="$CLOUDFLARED tunnel --no-autoupdate --url http://127.0.0.1:$PORT"
APP_ENV="KLEIN_UPSAMPLER=$UPSAMPLER"

if [[ $VAST == 1 ]]; then
    APP_LOG=/var/log/portal/klein-studio.log
    TUNNEL_LOG=/var/log/portal/klein-studio-tunnel.log

    # Скрипты не используют exit_portal.sh: иначе сервис молча не стартует без записи в /etc/portal.yaml,
    # а для неё нужен свободный внешний порт, которого на инстансе может не быть.
    write_service() {  # имя, команда, autorestart
        cat > "/opt/supervisor-scripts/$1.sh" <<EOF
#!/bin/bash
# Сгенерировано $APP_DIR/setup.sh — правьте там и перезапустите setup.sh.
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
cd "$APP_DIR"
pty $2 2>&1
EOF
        chmod +x "/opt/supervisor-scripts/$1.sh"
        cat > "/etc/supervisor/conf.d/$1.conf" <<EOF
[program:$1]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/$1.sh
autostart=true
autorestart=$3
exitcodes=0
startsecs=5
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=20
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOF
    }
    write_service klein-studio "env $APP_ENV $PY app.py --host 127.0.0.1 --port $PORT" unexpected
    if [[ "$PUBLIC" == 1 ]]; then
        write_service klein-studio-tunnel "$TUNNEL_CMD" true
    elif [[ -f /etc/supervisor/conf.d/klein-studio-tunnel.conf ]]; then
        supervisorctl stop klein-studio-tunnel >/dev/null 2>&1 || true
        rm -f /etc/supervisor/conf.d/klein-studio-tunnel.conf /opt/supervisor-scripts/klein-studio-tunnel.sh
    fi
    tunnel_was_running=$(supervisorctl status klein-studio-tunnel 2>/dev/null | grep -c RUNNING || true)
    supervisorctl reread >/dev/null
    supervisorctl update   # запускает новые сервисы; уже работающий туннель не трогает — ссылка сохраняется
    supervisorctl restart klein-studio >/dev/null   # подхватить новый код
    [[ "$PUBLIC" == 1 && "$tunnel_was_running" == 0 ]] && supervisorctl start klein-studio-tunnel >/dev/null || true

    # Кнопка в портале Vast (с токеном Vast) — только если есть свободный внешний порт.
    if [[ -s /etc/portal.yaml ]] && ! grep -q "Klein Studio" /etc/portal.yaml && command -v vast-capabilities >/dev/null; then
        ext=$(vast-capabilities 2>/dev/null | jq -r '[.instance.open_ports[]? | select(.in_use == false
              and (.self_mapped | not) and .container_port <= 65535)][0].container_port // empty' || true)
        if [[ -n "$ext" && -n "$(printenv "VAST_TCP_PORT_$ext" || true)" ]]; then
            "$PY" - "$ext" "$PORT" <<'EOF'
import sys, yaml
ext, port = int(sys.argv[1]), int(sys.argv[2])
d = yaml.safe_load(open("/etc/portal.yaml")) or {}
d.setdefault("applications", {})["Klein Studio"] = {
    "hostname": "localhost", "external_port": ext, "internal_port": port, "open_path": "/", "name": "Klein Studio"}
yaml.safe_dump(d, open("/etc/portal.yaml", "w"), sort_keys=False)
EOF
            supervisorctl restart caddy >/dev/null
            echo "Портал Vast: «Klein Studio» на порту $ext"
        fi
    fi
else
    # Обычная машина: фоновые процессы. Перезапуск setup.sh заменяет приложение, туннель оставляет.
    mkdir -p "$APP_DIR/logs"
    APP_LOG="$APP_DIR/logs/app.log"
    TUNNEL_LOG="$APP_DIR/logs/tunnel.log"
    alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }
    alive "$APP_DIR/logs/app.pid" && kill "$(cat "$APP_DIR/logs/app.pid")" && sleep 3
    (cd "$APP_DIR" && nohup env $APP_ENV "$PY" app.py --host 127.0.0.1 --port "$PORT" > "$APP_LOG" 2>&1 & echo $! > "$APP_DIR/logs/app.pid")
    if [[ "$PUBLIC" == 1 ]] && ! alive "$APP_DIR/logs/tunnel.pid"; then
        nohup $TUNNEL_CMD > "$TUNNEL_LOG" 2>&1 & echo $! > "$APP_DIR/logs/tunnel.pid"
    fi
    warn "Без supervisor процессы не переживут перезагрузку машины — после неё снова запустите setup.sh"
fi

# --------------------------------------------------------------------------- #
step "Запуск (загрузка моделей ~1–3 мин)"
# --------------------------------------------------------------------------- #
for _ in $(seq 150); do
    curl -fs -o /dev/null "http://127.0.0.1:$PORT/" && break
    if grep -q "Traceback" "$APP_LOG" 2>/dev/null; then
        tail -30 "$APP_LOG"; die "Приложение упало, лог: $APP_LOG"
    fi
    sleep 4
done
curl -fs -o /dev/null "http://127.0.0.1:$PORT/" || die "Приложение не ответило за 10 минут, лог: $APP_LOG"

LINK=""
if [[ "$PUBLIC" == 1 ]]; then
    for _ in $(seq 30); do
        LINK=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | tail -1 || true)
        [[ -n "$LINK" ]] && break
        sleep 2
    done
fi

echo
printf '\033[1;32mГотово.\033[0m\n'
[[ -n "$LINK" ]] && echo "Публичная ссылка (без пароля): $LINK"
[[ "$PUBLIC" == 1 && -z "$LINK" ]] && warn "Туннель ещё не выдал ссылку — посмотрите позже: grep trycloudflare $TUNNEL_LOG"
echo "Локально: http://127.0.0.1:$PORT  (с вашей машины: ssh -L $PORT:127.0.0.1:$PORT <сервер>)"
echo "Логи: $APP_LOG"
[[ $VAST == 1 ]] && echo "Управление: supervisorctl status | restart klein-studio"
exit 0
