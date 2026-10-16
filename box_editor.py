"""Редактор боксов в браузере — для раскладки и для точечной правки; без запроса к серверу на каждый клик.

Общее для обоих режимов: протянуть по холсту — новый бокс, в том числе внутри другого; перенести — за подпись
бокса; маркеры выделенного — размер; клик — выделить меньший бокс под курсором, повторный клик — следующий
под ним; ✕ на боксе, 🗑 в списке или Delete — удалить; Ctrl+Z — отменить; стрелки — сдвинуть (Shift — крупнее);
Ctrl+D — копия. Описание и id правятся в списке под холстом; пока id не задан руками, он берётся из описания.

Режим «layout» (`LayoutEditor`) — раскладка на пустом холсте. Значение:
    {"aspect": "16:9", "boxes": [{"id", "desc", "bbox": [y0, x0, y1, x1]}]}
Режим «edit» (`EditBoxEditor`) — правка поверх референса. Значение:
    {"refs": [{"src": data-URI превью, "w", "h"}], "shown": 0,
     "boxes": [{"id", "desc", "kind": new|move|keep|remove, "ref": N, "src": bbox|None, "tgt": bbox|None}]}
У «↔ Переместить» две рамки: пунктир — где элемент сейчас (src), сплошная — куда его поставить (tgt), между ними
стрелка; у остальных действий одна. Действие меняется у готового бокса кнопками в его строке.
Координаты — доли кадра (0–1), как `boxes` пайплайна; `to_state` переводит значение в состояние `layout.py`.

Устройство (gr.HTML в Gradio 6): шаблон рисует весь редактор из `props` и при каждом изменении `props`
точечно патчит DOM. Во время перетаскивания JS двигает элемент напрямую и меняет `props.value` один раз —
на отпускании, поэтому шаблон не перерисовывается посреди жеста. Шаблон сначала проходит через Handlebars,
поэтому в нём нельзя писать две открывающие фигурные скобки подряд.
"""

from __future__ import annotations

import base64
import io

import gradio as gr
from PIL import Image

import layout

KIND_ICON = {"new": "➕", "move": "↔", "keep": "📌", "remove": "✖"}
PREVIEW_SIDE = 896  # превью референса для холста — в значении компонента, data-URI JPEG

TEMPLATE = r"""${(() => {
  const v = value || ({});
  const edit = mode === 'edit';
  const boxes = Array.isArray(v.boxes) ? v.boxes : [];
  const refs = edit && Array.isArray(v.refs) ? v.refs : [];
  const shown = Math.min(Math.max(0, v.shown || 0), Math.max(0, refs.length - 1));
  const ref = refs[shown];
  let aw = 1, ah = 1;
  if (edit) { if (ref) { aw = ref.w; ah = ref.h; } }
  else { const a = String(v.aspect || '1:1').split(/[:x×\/]/).map(Number); aw = a[0] > 0 ? a[0] : 1; ah = a[1] > 0 ? a[1] : 1; }
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
    c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[c]);
  const pct = n => (n * 100).toFixed(3) + '%';
  const grid = n => Math.round(n * 1000);
  const color = i => colors[i % colors.length];
  const icon = ({new: '➕', move: '↔', keep: '📌', remove: '✖'});
  // Части бокса: [имя, bbox]. У «перенести» две — где сейчас (src) и куда (tgt).
  const parts = b => !edit ? [['tgt', b.bbox]]
    : b.kind === 'move' ? [['src', b.src], ['tgt', b.tgt]]
    : b.kind === 'new' ? [['tgt', b.tgt]] : [['src', b.src]];
  const visible = b => !edit || b.ref === shown;
  const area = bb => (bb[2] - bb[0]) * (bb[3] - bb[1]);
  const all = [];
  boxes.forEach((b, i) => { if (visible(b)) parts(b).forEach(([p, bb]) => { if (bb) all.push({ i, p, bb }); }); });
  const handles = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']
    .map(h => `<i class="kle-h kle-h-${h}" data-h="${h}"></i>`).join('');
  const boxHtml = all.map(({ i, p, bb }, k) => {
    const b = boxes[i];
    const [y0, x0, y1, x1] = bb;
    const sel = selected === i;
    // меньший бокс лежит выше большого; маркеры выделенного — в отдельном слое поверх всех (kle-sel)
    const z = 1 + all.filter((o, m) => area(o.bb) > area(bb) || (area(o.bb) === area(bb) && m < k)).length;
    const kind = edit ? b.kind : 'layout';
    const label = !edit ? `<b>${esc(b.id)}</b>${b.desc ? ' · ' + esc(b.desc) : ''}`
      : kind === 'move' && p === 'tgt' ? `→ <b>${esc(b.id)}</b> сюда`
      : `${icon[kind]} <b>${esc(b.id)}</b>${b.desc ? ' · ' + esc(b.desc) : ''}`;
    return `<div class="kle-box k-${kind} p-${p}${sel ? ' sel' : ''}" data-i="${i}" data-p="${p}" style="left:${pct(x0)};top:${pct(y0)};width:${pct(x1 - x0)};height:${pct(y1 - y0)};z-index:${z};--c:${color(i)}">
      <div class="kle-tag" title="Тяните, чтобы переместить"><span class="kle-move">✥</span>${label}</div>
    </div>`;
  }).join('');
  const arrows = edit ? boxes.map((b, i) => {
    if (!visible(b) || b.kind !== 'move' || !b.src || !b.tgt) return '';
    const cx = bb => (bb[1] + bb[3]) * 50, cy = bb => (bb[0] + bb[2]) * 50;
    return `<line class="kle-arrow" data-i="${i}" x1="${cx(b.src)}" y1="${cy(b.src)}" x2="${cx(b.tgt)}" y2="${cy(b.tgt)}" style="stroke:${color(i)}"/>`;
  }).join('') : '';
  const sb = boxes[selected];
  const sp = sb && visible(sb) ? parts(sb).find(([p]) => p === part) || parts(sb)[0] : null;
  const selHtml = sp && sp[1] ? `<div class="kle-sel" data-p="${sp[0]}" style="left:${pct(sp[1][1])};top:${pct(sp[1][0])};width:${pct(sp[1][3] - sp[1][1])};height:${pct(sp[1][2] - sp[1][0])};--c:${color(selected)}">
      ${handles}<button class="kle-x" data-del="${selected}" title="Удалить (Delete)">✕</button></div>` : '';
  const placeholder = ({new: 'что появится: a small potted cactus', move: 'что переносим: the knight',
                        keep: 'что не трогать: the table', remove: 'что убрать: the lamp'});
  const rows = boxes.map((b, i) => {
    const kinds = edit ? `<span class="kle-kinds">${['new', 'move', 'keep', 'remove'].map(k =>
      `<button class="kle-kind${b.kind === k ? ' on' : ''}" data-kind="${k}" data-i="${i}" title="${({new: 'Новый / заменить', move: 'Переместить', keep: 'Оставить как есть', remove: 'Удалить с картинки'})[k]}">${icon[k]}</button>`).join('')}</span>` : '';
    const where = edit ? `<button class="kle-refno${b.ref === shown ? '' : ' other'}" data-show="${b.ref}" title="Показать референс">№${b.ref + 1}</button>`
      : `<input class="kle-xy" data-f="bbox" data-i="${i}" value="${grid(b.bbox[0])}, ${grid(b.bbox[1])}, ${grid(b.bbox[2])}, ${grid(b.bbox[3])}" spellcheck="false" title="y0, x0, y1, x1 на холсте 0–1000">`;
    return `<div class="kle-row${selected === i ? ' sel' : ''}${visible(b) ? '' : ' dim'}" data-i="${i}">
      <span class="kle-dot" style="background:${color(i)}"></span>${kinds}
      <input class="kle-id" data-f="id" data-i="${i}" value="${esc(b.id)}" spellcheck="false" title="id: латиница, цифры и _">
      <input class="kle-desc" data-f="desc" data-i="${i}" value="${esc(b.desc)}" placeholder="${edit ? placeholder[b.kind] : 'что в боксе: a red balloon'}">
      ${where}
      <button class="kle-del" data-del="${i}" title="Удалить">🗑</button>
    </div>`;
  }).join('');
  const refStrip = refs.length > 1 ? `<div class="kle-refs">${refs.map((r, k) =>
    `<button class="kle-ref${k === shown ? ' on' : ''}" data-ref="${k}" title="Показать №${k + 1}"><img src="${r.src}" alt=""><span>№${k + 1}</span></button>`).join('')}</div>` : '';
  const drawKinds = edit ? `<span class="kle-draw">Новый бокс:${['new', 'move', 'keep', 'remove'].map(k =>
    `<button class="kle-kind${(draw_kind || 'new') === k ? ' on' : ''}" data-dk="${k}" title="${k === 'new' ? 'Новый / заменить' : k === 'move' ? 'Переместить' : k === 'keep' ? 'Оставить как есть' : 'Удалить'}">${icon[k]}</button>`).join('')}</span>` : '';
  const hint = edit
    ? (boxes.length ? 'Действие бокса — кнопки ➕ ↔ 📌 ✖ в его строке · перенос — за подпись ✥' : 'Протяните по картинке — бокс; ↔ — потом перетащите рамку «→ сюда»')
    : (boxes.length ? `Боксов: ${boxes.length} · перенос — за подпись ✥ · Delete — удалить · Ctrl+Z — отменить` : 'Протяните по холсту — новый бокс');
  const empty = edit ? (ref ? '' : '<div class="kle-empty">Загрузите изображение выше</div>')
    : (boxes.length ? '' : '<div class="kle-empty">Зажмите кнопку мыши и протяните — получится бокс</div>');
  return `<div class="kle${edit ? ' kle-edit' : ''}" tabindex="0">
    ${refStrip}
    <div class="kle-bar">
      <span class="kle-hint">${hint}</span>
      ${drawKinds}
      ${edit ? '' : `<label class="kle-snap"><input type="checkbox" data-snap ${snap ? 'checked' : ''}> сетка 5 %</label>`}
      <button data-act="undo" title="Ctrl+Z">↶</button>
      <button data-act="clear" ${boxes.length ? '' : 'disabled'}>Очистить</button>
    </div>
    <div class="kle-stage">
      <div class="kle-canvas${edit && !ref ? ' off' : ''}" style="aspect-ratio:${aw} / ${ah};width:min(100%, calc(var(--kle-max-h) * ${aw / ah}))">
        ${ref ? `<img class="kle-bg" src="${ref.src}" alt="" draggable="false">` : ''}
        ${empty}
        <svg class="kle-arrows" viewBox="0 0 100 100" preserveAspectRatio="none">${arrows}</svg>
        ${boxHtml}
        ${selHtml}
        <div class="kle-draft" hidden></div>
      </div>
    </div>
    <div class="kle-list">${rows || '<div class="kle-none">Боксов пока нет</div>'}</div>
  </div>`;
})()}"""

CSS = r"""
--kle-max-h: 520px;
.kle { outline: none; display: flex; flex-direction: column; gap: 8px; color: var(--body-text-color);
       font-size: 14px; }
.kle-bar { display: flex; align-items: center; gap: 10px; min-height: 30px; }
.kle-hint { flex: 1 1 0; min-width: 0; opacity: .72; font-size: 13px; white-space: nowrap; overflow: hidden;
            text-overflow: ellipsis; }
.kle-snap { display: inline-flex; align-items: center; gap: 4px; font-size: 13px; cursor: pointer; }
.kle button { border: 1px solid var(--border-color-primary); background: var(--button-secondary-background-fill);
              color: var(--button-secondary-text-color); border-radius: var(--radius-md); padding: 3px 10px;
              font-size: 13px; cursor: pointer; }
.kle button:disabled { opacity: .45; cursor: default; }
.kle-draw { display: inline-flex; align-items: center; gap: 3px; font-size: 13px; white-space: nowrap; }
.kle-draw .kle-kind { margin-left: 2px; }
.kle .kle-kind { padding: 2px 6px; font-size: 13px; line-height: 18px; }
.kle .kle-kind.on { background: var(--color-accent); border-color: var(--color-accent); color: #fff; }
.kle-kinds { display: inline-flex; gap: 2px; flex: 0 0 auto; }
.kle-refs { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 2px; }
.kle .kle-ref { position: relative; padding: 0; border-width: 2px; border-radius: 8px; overflow: hidden; flex: 0 0 auto;
                height: 58px; background: #26282e; }
.kle .kle-ref.on { border-color: var(--color-accent); }
.kle-ref img { height: 100%; display: block; }
.kle-ref span { position: absolute; left: 3px; bottom: 2px; font-size: 11px; color: #fff; background: rgba(0,0,0,.6);
                padding: 0 4px; border-radius: 4px; }
.kle-stage { display: flex; justify-content: center; }
.kle-canvas { position: relative; max-width: 100%; border-radius: 8px; overflow: hidden; cursor: crosshair;
              touch-action: none; user-select: none; background-color: #26282e;
              background-image: linear-gradient(to right, rgba(255,255,255,.07) 1px, transparent 1px),
                                linear-gradient(to bottom, rgba(255,255,255,.07) 1px, transparent 1px);
              background-size: 10% 10%; box-shadow: inset 0 0 0 1px rgba(255,255,255,.08); }
.kle-edit .kle-canvas { background-image: none; }
.kle-canvas.off { cursor: default; }
.kle-bg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; user-select: none; }
.kle-arrows { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 900; }
.kle-arrow { stroke-width: 3px; stroke-dasharray: 7 5; vector-effect: non-scaling-stroke;
             filter: drop-shadow(0 0 1px rgba(0,0,0,.8)); }
.kle-empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
             color: rgba(255,255,255,.45); font-size: 14px; pointer-events: none; text-align: center; padding: 12px; }
.kle-box { position: absolute; box-sizing: border-box; border: 2px solid var(--c); pointer-events: none;
           background: color-mix(in srgb, var(--c) 14%, transparent); }
.kle-box.sel { border-width: 3px; background: color-mix(in srgb, var(--c) 24%, transparent);
               box-shadow: 0 0 0 1px rgba(0,0,0,.5); }
.kle-box.k-move.p-src { border-style: dashed; background: color-mix(in srgb, var(--c) 8%, transparent); }
.kle-box.k-remove { border-style: dashed;
  background: linear-gradient(to top right, transparent calc(50% - 1.5px), var(--c) 50%, transparent calc(50% + 1.5px)),
              linear-gradient(to top left, transparent calc(50% - 1.5px), var(--c) 50%, transparent calc(50% + 1.5px)),
              color-mix(in srgb, var(--c) 10%, transparent); }
.kle-box.k-keep { border-style: double; border-width: 4px; background: color-mix(in srgb, var(--c) 6%, transparent); }
.kle-sel { position: absolute; z-index: 1000; pointer-events: none; }
.kle-move { opacity: .8; margin-right: 4px; }
.kle-tag { position: absolute; left: 0; top: 0; max-width: 100%; padding: 1px 6px; font-size: 12px;
           line-height: 18px; color: #fff; background: color-mix(in srgb, var(--c) 82%, black);
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: auto;
           cursor: move; border-bottom-right-radius: 6px; }
.kle-h { position: absolute; pointer-events: auto; width: 12px; height: 12px; background: #fff; border: 2px solid var(--c);
         border-radius: 3px; box-sizing: border-box; }
@media (pointer: coarse) { .kle-h { width: 22px; height: 22px; } }
.kle-h-nw, .kle-h-n, .kle-h-ne { top: 0; transform: translateY(-50%); }
.kle-h-sw, .kle-h-s, .kle-h-se { bottom: 0; transform: translateY(50%); }
.kle-h-w, .kle-h-e { top: 50%; transform: translateY(-50%); }
.kle-h-nw, .kle-h-w, .kle-h-sw { left: 0; margin-left: -6px; }
.kle-h-ne, .kle-h-e, .kle-h-se { right: 0; margin-right: -6px; }
.kle-h-n, .kle-h-s { left: 50%; margin-left: -6px; }
.kle-h-nw, .kle-h-se { cursor: nwse-resize; } .kle-h-ne, .kle-h-sw { cursor: nesw-resize; }
.kle-h-n, .kle-h-s { cursor: ns-resize; } .kle-h-e, .kle-h-w { cursor: ew-resize; }
.kle .kle-x { position: absolute; pointer-events: auto; right: 4px; top: 4px; width: 22px; height: 22px; padding: 0;
              border-radius: 50%; background: rgba(0,0,0,.65); color: #fff; border: 1px solid rgba(255,255,255,.4);
              font-size: 12px; line-height: 20px; }
.kle .kle-x:hover { background: #e03131; }
.kle-draft { position: absolute; border: 2px dashed #fff; background: rgba(255,255,255,.12); pointer-events: none;
             z-index: 1100; }
.kle-list { display: flex; flex-direction: column; gap: 4px; }
.kle-none { opacity: .6; font-size: 13px; }
.kle-row { display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 8px;
           border: 1px solid transparent; }
.kle-row.sel { border-color: var(--border-color-accent, var(--color-accent)); background: var(--background-fill-secondary); }
.kle-row.dim { opacity: .55; }
.kle-dot { flex: 0 0 12px; height: 12px; border-radius: 50%; }
.kle-row input { min-width: 0; border: 1px solid var(--border-color-primary); border-radius: var(--radius-sm);
                 background: var(--input-background-fill); color: var(--body-text-color); padding: 4px 7px;
                 font-size: 13px; }
.kle-id { flex: 0 1 110px; font-family: var(--font-mono); }
.kle-desc { flex: 1 1 180px; }
.kle-xy { flex: 0 0 158px; font-family: var(--font-mono); font-size: 12px !important; opacity: .85; }
.kle .kle-refno { padding: 3px 6px; font-size: 12px; }
.kle .kle-refno.other { border-color: var(--color-accent); }
.kle .kle-del { padding: 3px 7px; }
@media (max-width: 640px) { .kle-xy { display: none; } .kle-draw { display: none; } }
"""

JS = r"""
const MIN = 0.01;          // меньше 1 % кадра — не бокс, а клик
const undo = [];
let drag = null;

const edit = () => props.mode === 'edit';
const clone = o => JSON.parse(JSON.stringify(o));
const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const round = b => b.map(v => Math.round(v * 10000) / 10000);
const snapv = v => props.snap ? Math.round(v * 20) / 20 : v;
const state = () => {
  const v = props.value || {};
  return { ...v, boxes: Array.isArray(v.boxes) ? v.boxes : [] };
};
const shown = () => Math.max(0, state().shown || 0);
const canvas = () => element.querySelector('.kle-canvas');
const STOP = new Set(['a', 'an', 'the', 'one', 'two', 'three', 'some', 'small', 'big', 'large']);
const PREP = new Set(['on', 'in', 'with', 'at', 'near', 'under', 'over', 'of', 'by', 'from', 'behind', 'above',
                      'below', 'beside', 'next', 'holding', 'wearing', 'and', 'that', 'which', 'for', 'to']);

// Части бокса (как в шаблоне): у «перенести» — где сейчас (src) и куда (tgt), у остальных одна.
function parts(b) {
  if (!edit()) return [['tgt', b.bbox]];
  if (b.kind === 'move') return [['src', b.src], ['tgt', b.tgt]];
  return b.kind === 'new' ? [['tgt', b.tgt]] : [['src', b.src]];
}
const visible = b => !edit() || b.ref === shown();
const partBox = (b, p) => (parts(b).find(([q]) => q === p) || parts(b)[0])[1];
function setPart(b, p, bb) {
  if (!edit()) b.bbox = bb;
  else if (b.kind === 'keep') { b.src = bb; b.tgt = bb.slice(); }
  else b[p] = bb;
}
function shifted(bb) {
  // копия-цель для «перенести»: правее (или левее, если не влезает)
  const [y0, x0, y1, x1] = bb, w = x1 - x0;
  let nx = x0 + Math.max(0.12, w * 0.6);
  if (nx + w > 1) nx = x0 - Math.max(0.12, w * 0.6);
  nx = clamp(nx, 0, 1 - w);
  return round([y0, nx, y1, nx + w]);
}
function setKind(b, k) {
  const drawn = b.kind === 'new' ? b.tgt : b.src;   // область, которую обвели
  if (k === 'new') { b.tgt = b.kind === 'move' ? b.tgt : drawn; b.src = null; }
  else if (k === 'move') { b.src = drawn; b.tgt = b.kind === 'move' ? b.tgt : shifted(drawn); }
  else if (k === 'keep') { b.src = drawn; b.tgt = drawn.slice(); }
  else { b.src = drawn; b.tgt = null; }
  b.kind = k;
}
function makeBox(boxes, bb) {
  const id = nextId(boxes, 'obj');
  if (!edit()) return { id, desc: '', bbox: bb, auto: true };
  const b = { id, desc: '', kind: 'new', ref: shown(), src: null, tgt: bb, auto: true };
  setKind(b, props.draw_kind || 'new');
  return b;
}

function slug(s) {
  let id = String(s || '').trim().toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '');
  if (id && !/^[a-z]/.test(id)) id = 'obj_' + id;
  return id;
}
function nextId(boxes, stem) {
  stem = slug(stem) || 'obj';
  const taken = new Set(boxes.map(b => b.id));
  let n = 1;
  while (taken.has(`${stem}_${n}`)) n++;
  return `${stem}_${n}`;
}
function stemFromDesc(desc) {
  // «a red hot-air balloon» → balloon, «two cats on a sofa» → cats: последнее слово до первого предлога.
  const words = (String(desc).toLowerCase().match(/[a-z][a-z0-9]*/g) || []).filter(w => !STOP.has(w));
  const cut = words.findIndex(w => PREP.has(w));
  const head = cut > 0 ? words.slice(0, cut) : words;
  return head.length ? head[head.length - 1] : null;
}
function select(i, p) {
  if (props.selected !== i) props.selected = i;
  if (props.part !== p) props.part = p;
}
function commit(boxes, selected, part) {
  undo.push(clone(state().boxes));
  if (undo.length > 100) undo.shift();
  props.selected = selected;
  props.part = part || null;
  props.value = { ...state(), boxes };
}
function remove(i) {
  const boxes = clone(state().boxes);
  if (!boxes[i]) return;
  boxes.splice(i, 1);
  commit(boxes, null);
}
function showRef(k) {
  if (shown() !== k) props.value = { ...state(), shown: k };
}
function focusField(i, field) {
  // Без прокрутки «по центру»: страница сдвигается ровно настолько, чтобы поле стало видно, холст не уезжает.
  setTimeout(() => {
    const input = element.querySelector(`input[data-f="${field}"][data-i="${i}"]`);
    if (!input) return;
    input.focus({ preventScroll: true });
    input.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, 30);
}
function rel(e) {
  const r = canvas().getBoundingClientRect();
  return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
}
function place(el, b) {
  el.style.left = b[1] * 100 + '%'; el.style.top = b[0] * 100 + '%';
  el.style.width = (b[3] - b[1]) * 100 + '%'; el.style.height = (b[2] - b[0]) * 100 + '%';
}

// Части боксов под точкой, меньшие первыми: меньший лежит выше, до вложенного всегда можно дотянуться.
function hitsAt(x, y) {
  const area = b => (b[2] - b[0]) * (b[3] - b[1]);
  const hits = [];
  state().boxes.forEach((b, i) => {
    if (!visible(b)) return;
    for (const [p, bb] of parts(b))
      if (bb && y >= bb[0] && y <= bb[2] && x >= bb[1] && x <= bb[3]) hits.push({ i, p, bb });
  });
  return hits.sort((a, c) => area(a.bb) - area(c.bb) || c.i - a.i);
}
const partEl = (i, p) => element.querySelector(`.kle-box[data-i="${i}"][data-p="${p}"]`);

// Протянуть где угодно (в том числе внутри другого бокса) — новый бокс. Перенос — за подпись бокса,
// размер — за маркеры выделенного. Клик — выделить меньший бокс под курсором, повторный клик
// в том же месте — следующий под ним (так достаётся перекрытый).
element.addEventListener('pointerdown', e => {
  const cv = canvas();
  if (!cv || cv.classList.contains('off') || !cv.contains(e.target) || e.button !== 0 || e.target.closest('.kle-x')) return;
  e.preventDefault();
  element.querySelector('.kle')?.focus({ preventScroll: true });
  const [x, y] = rel(e);
  const handle = e.target.closest('.kle-h');
  const tag = e.target.closest('.kle-tag');
  const hits = hitsAt(x, y);
  if (handle || tag) {
    const i = handle ? props.selected : +tag.closest('.kle-box').dataset.i;
    const p = handle ? e.target.closest('.kle-sel').dataset.p : tag.closest('.kle-box').dataset.p;
    drag = { mode: handle ? 'resize' : 'move', h: handle?.dataset.h || '', i, p, x, y, hits, moved: false,
             orig: partBox(state().boxes[i], p).slice() };
    select(i, p);
  } else {
    drag = { mode: 'draw', x, y, hits, moved: false, el: cv.querySelector('.kle-draft') };
  }
  cv.setPointerCapture(e.pointerId);
});

element.addEventListener('pointermove', e => {
  if (!drag) return;
  const [x, y] = rel(e);
  const r = canvas().getBoundingClientRect();
  if (!drag.moved && Math.hypot((x - drag.x) * r.width, (y - drag.y) * r.height) < 4) return;
  drag.moved = true;
  let b;
  if (drag.mode === 'draw') {
    const x0 = snapv(Math.min(x, drag.x)), x1 = snapv(Math.max(x, drag.x));
    const y0 = snapv(Math.min(y, drag.y)), y1 = snapv(Math.max(y, drag.y));
    b = [y0, x0, y1, x1];
    drag.el.hidden = false;
  } else if (drag.mode === 'move') {
    const [oy0, ox0, oy1, ox1] = drag.orig, h = oy1 - oy0, w = ox1 - ox0;
    const ny0 = clamp(snapv(oy0 + y - drag.y), 0, 1 - h), nx0 = clamp(snapv(ox0 + x - drag.x), 0, 1 - w);
    b = [ny0, nx0, ny0 + h, nx0 + w];
  } else {
    let [y0, x0, y1, x1] = drag.orig;
    const hd = drag.h;
    if (hd.includes('n')) y0 = Math.min(snapv(y), y1 - MIN);
    if (hd.includes('s')) y1 = Math.max(snapv(y), y0 + MIN);
    if (hd.includes('w')) x0 = Math.min(snapv(x), x1 - MIN);
    if (hd.includes('e')) x1 = Math.max(snapv(x), x0 + MIN);
    b = [clamp(y0), clamp(x0), clamp(y1), clamp(x1)];
  }
  drag.cur = b;
  if (drag.mode === 'draw') { place(drag.el, b); return; }
  [partEl(drag.i, drag.p), element.querySelector('.kle-sel')].forEach(el => el && place(el, b));
  const arrow = element.querySelector(`.kle-arrow[data-i="${drag.i}"]`);
  if (arrow) {
    const end = drag.p === 'src' ? ['x1', 'y1'] : ['x2', 'y2'];
    arrow.setAttribute(end[0], (b[1] + b[3]) * 50); arrow.setAttribute(end[1], (b[0] + b[2]) * 50);
  }
});

function endDrag() {
  if (!drag) return;
  const d = drag;
  drag = null;
  if (d.mode === 'draw') {
    d.el.hidden = true;
    const b = d.cur;
    if (!d.moved || !b || b[2] - b[0] < MIN || b[3] - b[1] < MIN) {
      // клик: меньшая часть под курсором; если выделенная среди них — следующая за ней; мимо боксов — снять
      const sb = state().boxes[props.selected];
      const sp = props.part || (sb ? parts(sb)[0][0] : null);   // без части — первая часть бокса
      const k = d.hits.findIndex(h => h.i === props.selected && h.p === sp);
      const pick = d.hits.length ? d.hits[(k + 1) % d.hits.length] : null;
      select(pick ? pick.i : null, pick ? pick.p : null);
      return;
    }
    const boxes = clone(state().boxes);
    const nb = makeBox(boxes, round(b));
    boxes.push(nb);
    commit(boxes, boxes.length - 1, nb.kind === 'move' ? 'tgt' : null);
    focusField(boxes.length - 1, 'desc');
  } else if (d.moved && d.cur) {
    const boxes = clone(state().boxes);
    setPart(boxes[d.i], d.p, round(d.cur));
    commit(boxes, d.i, d.p);
  }
}
element.addEventListener('pointerup', endDrag);
element.addEventListener('pointercancel', endDrag);

element.addEventListener('click', e => {
  const t = e.target;
  const del = t.closest('[data-del]');
  if (del) { remove(+del.dataset.del); return; }
  const kind = t.closest('[data-kind]');
  if (kind) {
    const i = +kind.dataset.i, boxes = clone(state().boxes);
    if (boxes[i].kind === kind.dataset.kind) return;
    setKind(boxes[i], kind.dataset.kind);
    showRef(boxes[i].ref);
    commit(boxes, i, boxes[i].kind === 'move' ? 'tgt' : null);
    return;
  }
  const dk = t.closest('[data-dk]');
  if (dk) { props.draw_kind = dk.dataset.dk; return; }
  const ref = t.closest('[data-ref]');
  if (ref) { props.selected = null; showRef(+ref.dataset.ref); return; }
  const show = t.closest('[data-show]');
  if (show) { showRef(+show.dataset.show); select(+show.closest('.kle-row').dataset.i, null); return; }
  const act = t.closest('[data-act]');
  if (act) {
    if (act.dataset.act === 'undo' && undo.length) {
      props.selected = null;
      props.value = { ...state(), boxes: undo.pop() };
    } else if (act.dataset.act === 'clear' && state().boxes.length) {
      commit([], null);
    }
    return;
  }
  const row = t.closest('.kle-row');
  if (row) {
    const i = +row.dataset.i;
    showRef(state().boxes[i].ref ?? shown());
    if (props.selected !== i) select(i, null);
  }
});

element.addEventListener('focusin', e => {
  const row = e.target.closest('.kle-row');
  if (row && props.selected !== +row.dataset.i) select(+row.dataset.i, null);
});

element.addEventListener('change', e => {
  const t = e.target;
  if (t.matches('[data-snap]')) { props.snap = t.checked; return; }
  const f = t.dataset.f;
  if (!f) return;
  const i = +t.dataset.i;
  const boxes = clone(state().boxes);
  const b = boxes[i];
  if (!b) return;
  const others = boxes.filter((_, j) => j !== i);
  if (f === 'desc') {
    b.desc = t.value.trim();
    const stem = b.auto && stemFromDesc(b.desc);
    if (stem) b.id = nextId(others, stem);
  } else if (f === 'id') {
    const id = slug(t.value);
    if (!id) { t.value = b.id; return; }
    b.id = others.some(o => o.id === id) ? nextId(others, id) : id;
    b.auto = false;
  } else if (f === 'bbox') {
    let vals = t.value.split(/[\s,;]+/).filter(Boolean).map(Number);
    if (vals.length === 4 && vals.every(Number.isFinite)) {
      if (Math.max(...vals) > 1) vals = vals.map(v => v / 1000);
      const [y0, x0, y1, x1] = vals.map(v => clamp(v));
      if (y1 - y0 >= MIN && x1 - x0 >= MIN) b.bbox = round([y0, x0, y1, x1]);
    }
    if (JSON.stringify(b.bbox) === JSON.stringify(state().boxes[i].bbox)) {
      const g = v => Math.round(v * 1000);
      t.value = b.bbox.map(g).join(', ');   // не разобрали — вернуть как было
      return;
    }
  }
  commit(boxes, i, props.part);
});

element.addEventListener('keydown', e => {
  if (e.target.matches('input')) {
    if (e.key === 'Enter') e.target.blur();
    return;
  }
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === 'z') {
    e.preventDefault();
    element.querySelector('[data-act="undo"]')?.click();
    return;
  }
  const i = props.selected;
  const boxes = clone(state().boxes);
  if (i == null || !boxes[i]) return;
  const b = boxes[i];
  const p = parts(b).find(([q]) => q === props.part) ? props.part : parts(b)[0][0];
  const move = (bb, dy, dx) => {
    const [y0, x0, y1, x1] = bb, h = y1 - y0, w = x1 - x0;
    const ny0 = clamp(y0 + dy, 0, 1 - h), nx0 = clamp(x0 + dx, 0, 1 - w);
    return round([ny0, nx0, ny0 + h, nx0 + w]);
  };
  if (e.key === 'Delete' || e.key === 'Backspace') {
    e.preventDefault();
    remove(i);
  } else if (e.key === 'Escape') {
    select(null, null);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    focusField(i, 'desc');
  } else if (mod && e.key.toLowerCase() === 'd') {
    e.preventDefault();
    const copy = clone(b);
    copy.id = nextId(boxes, b.id.replace(/_\d+$/, ''));
    for (const [q, bb] of parts(copy)) if (bb) setPart(copy, q, move(bb, 0.03, 0.03));
    boxes.push(copy);
    commit(boxes, boxes.length - 1);
  } else if (e.key.startsWith('Arrow')) {
    e.preventDefault();
    const step = e.shiftKey ? 0.05 : 0.005;
    const dy = e.key === 'ArrowUp' ? -step : e.key === 'ArrowDown' ? step : 0;
    const dx = e.key === 'ArrowLeft' ? -step : e.key === 'ArrowRight' ? step : 0;
    setPart(b, p, move(partBox(b, p), dy, dx));
    commit(boxes, i, p);
  }
});
"""


class _BoxEditor(gr.HTML):
    def __init__(self, mode: str, value: dict, **kwargs) -> None:
        super().__init__(value=value, html_template=TEMPLATE, css_template=CSS, js_on_load=JS,
                         apply_default_css=False, mode=mode, selected=None, part=None, snap=False,
                         draw_kind="new", colors=layout.COLORS, **kwargs)


class LayoutEditor(_BoxEditor):
    def __init__(self, aspect: str = "1:1", **kwargs) -> None:
        super().__init__("layout", empty(aspect), **kwargs)

    def api_info(self) -> dict:
        return {"type": "object", "description": '{"aspect": "1:1", "boxes": [{"id", "desc", "bbox": [y0, x0, y1, x1]}]}'}


class EditBoxEditor(_BoxEditor):
    def __init__(self, **kwargs) -> None:
        super().__init__("edit", empty_edit(), **kwargs)

    def api_info(self) -> dict:
        return {"type": "object", "description": '{"refs": [...], "shown": 0, "boxes": [{"id", "desc", "kind", '
                                                 '"ref", "src": [y0, x0, y1, x1] | null, "tgt": [...] | null}]}'}


# --------------------------------------------------------------------------- #
# Раскладка
# --------------------------------------------------------------------------- #
def empty(aspect: str = "1:1") -> dict:
    return {"aspect": aspect, "boxes": []}


def with_aspect(value: dict | None, aspect: str) -> dict:
    return {**(value or empty()), "aspect": aspect}


def to_state(value: dict | None) -> list[dict]:
    """Значение редактора раскладки → состояние боксов `layout.py` (kind=layout, tgt=bbox)."""
    state = []
    for b in (value or {}).get("boxes") or []:
        bbox = _bbox(b.get("bbox"))
        if bbox is None or not layout.BOX_ID.fullmatch(str(b.get("id", ""))):
            continue
        state.append(vars(layout.Box(id=b["id"], kind=layout.KIND_LAYOUT, tgt=bbox, desc=str(b.get("desc") or ""))))
    return state


# --------------------------------------------------------------------------- #
# Точечная правка
# --------------------------------------------------------------------------- #
def empty_edit() -> dict:
    return {"refs": [], "shown": 0, "boxes": []}


def with_refs(value: dict | None, images: list[Image.Image]) -> dict:
    """Новые референсы на холст. Боксы референсов, которых больше нет, убираются."""
    value = value or empty_edit()
    refs = [_preview(im) for im in images]
    boxes = [b for b in value.get("boxes") or [] if 0 <= int(b.get("ref", 0)) < len(refs)]
    shown = min(int(value.get("shown") or 0), max(0, len(refs) - 1))
    return {**value, "refs": refs, "shown": shown, "boxes": boxes}


def to_edit_state(value: dict | None) -> list[dict]:
    """Значение редактора правки → состояние боксов `layout.py` (kind, ref, src, tgt)."""
    state = []
    for b in (value or {}).get("boxes") or []:
        kind = b.get("kind")
        if kind not in KIND_ICON or not layout.BOX_ID.fullmatch(str(b.get("id", ""))):
            continue
        src, tgt = _bbox(b.get("src")), _bbox(b.get("tgt"))
        if (kind in ("move", "keep", "remove") and src is None) or (kind in ("new", "move") and tgt is None):
            continue
        state.append(vars(layout.Box(id=b["id"], kind=kind, ref=int(b.get("ref") or 0), src=src, tgt=tgt,
                                     desc=str(b.get("desc") or ""))))
    return state


def _bbox(v) -> list[float] | None:
    if not v or len(v) != 4:
        return None
    return [float(x) for x in v]


def _preview(image: Image.Image) -> dict:
    im = image.convert("RGB")
    w, h = im.size
    if max(w, h) > PREVIEW_SIDE:
        im = im.copy()
        im.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return {"src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "w": w, "h": h}
