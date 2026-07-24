import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Krea2CharacterRouter";
const JSON_WIDGET = "regions_json";
const CANVAS_LORA_WIDGET = "canvas_lora_json";
const COLORS = ["#f04452", "#338cff", "#35d873", "#f1b42f", "#995de9", "#2cced0", "#ef61b4", "#b2cb3f"];
let loraNames = ["None"];

function installStyles() {
  if (document.getElementById("k2cr-styles")) return;
  const style = document.createElement("style");
  style.id = "k2cr-styles";
  style.textContent = `
    .k2cr { box-sizing:border-box; width:100%; height:100%; min-height:0; padding:8px; overflow:auto;
      color:var(--fg-color,#ddd); background:rgba(18,18,22,.96); font:12px system-ui,sans-serif; }
    .k2cr * { box-sizing:border-box; }
    .k2cr-toolbar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:7px; }
    .k2cr button { border:1px solid #555; border-radius:5px; background:#2a2a31; color:#eee; padding:5px 8px; cursor:pointer; }
    .k2cr button:hover { background:#393943; }
    .k2cr-hint { margin-left:auto; color:#999; font-size:11px; }
    .k2cr-count-label { flex-direction:row !important; align-items:center; gap:5px !important; font-size:11px !important; }
    .k2cr-count-label input { width:48px; padding:5px 4px; }
    .k2cr-scene { margin-bottom:7px; border:1px solid #46464f; border-radius:6px; padding:6px; background:#202027; }
    .k2cr-scene summary { cursor:pointer; color:#ccc; font-weight:650; user-select:none; }
    .k2cr-scene textarea { height:68px; min-height:42px; max-height:220px; margin-top:6px; resize:vertical; }
    .k2cr-canvas-lora { margin-bottom:7px; border:1px solid #5a4b70; border-radius:6px; padding:6px; background:#24202b; }
    .k2cr-canvas-lora summary { cursor:pointer; color:#d7c7ed; font-weight:650; user-select:none; }
    .k2cr-canvas-body { margin-top:7px; }
    .k2cr-canvas-help { margin:0 0 7px; color:#aaa; font-size:11px; line-height:1.35; }
    .k2cr-enable-label { flex-direction:row !important; align-items:center; gap:6px !important; margin-bottom:7px; font-size:11px !important; }
    .k2cr-enable-label input { width:auto; }
    .k2cr-canvas-grid { display:grid; grid-template-columns:1fr 145px 90px 70px; gap:5px; }
    .k2cr-canvas-grid .wide { grid-column:1 / -1; }
    .k2cr-canvas-schedule { display:grid; grid-template-columns:1fr 1fr; gap:5px; margin-top:5px; }
    .k2cr-stage { width:100%; border:1px solid #555; border-radius:6px; display:block; background:#111318;
      touch-action:none; cursor:crosshair; }
    .k2cr-rows { display:flex; flex-direction:column; gap:8px; margin-top:8px; }
    .k2cr-row { border:1px solid #46464f; border-left:4px solid var(--k2-color); border-radius:6px; padding:7px; background:#202027; }
    .k2cr-row-head { display:flex; align-items:center; gap:6px; margin-bottom:6px; }
    .k2cr-row-head input[type=text] { font-weight:650; }
    .k2cr-grid { display:grid; grid-template-columns:1fr 90px 70px 70px; gap:5px; }
    .k2cr-grid .wide { grid-column:1 / -1; }
    .k2cr label { color:#aaa; font-size:10px; display:flex; flex-direction:column; gap:2px; }
    .k2cr input,.k2cr select,.k2cr textarea { width:100%; border:1px solid #4a4a54; border-radius:4px; background:#141419; color:#eee; padding:4px; }
    .k2cr textarea { resize:vertical; min-height:42px; }
    .k2cr textarea.k2cr-description { height:96px; min-height:72px; max-height:320px; line-height:1.4; padding:7px; }
    .k2cr-box-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:4px; margin-top:5px; }
    .k2cr-box-grid input { font-variant-numeric:tabular-nums; }
    .k2cr-combo { position:relative; }
    .k2cr-combo input { padding-right:22px; }
    .k2cr-combo input,.k2cr-combo-option,.k2cr-combo-empty { font-size:14px; }
    .k2cr-combo::after { content:"⌕"; position:absolute; right:6px; top:4px; color:#888; pointer-events:none; }
    .k2cr-combo-list { position:absolute; z-index:10000; left:0; right:0; top:100%; max-height:210px;
      overflow:auto; border:1px solid #666; border-radius:4px; background:#16161c; box-shadow:0 8px 22px #000b; }
    .k2cr-combo-list[hidden] { display:none; }
    .k2cr-combo-option { padding:5px 7px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; cursor:pointer; }
    .k2cr-combo-option:hover,.k2cr-combo-option.active { background:#3a547d; color:#fff; }
    .k2cr-combo-empty { padding:7px; color:#999; font-style:italic; }
  `;
  document.head.appendChild(style);
}

async function loadLoras() {
  if (loraNames.length > 1) return;
  try {
    const response = await api.fetchApi("/object_info/LoraLoader");
    const data = await response.json();
    const names = data?.LoraLoader?.input?.required?.lora_name?.[0];
    if (Array.isArray(names)) loraNames = ["None", ...names.filter((name) => name !== "None")];
  } catch (error) {
    console.warn("[Krea2 Multi-LoRA Composer] unable to load LoRA names", error);
  }
}

function widget(node, name) {
  return node.widgets?.find((item) => item.name === name);
}

function normalizeSupersampleScale(node) {
  const item = widget(node, "supersample_scale");
  if (!item) return;
  const value = Number(item.value);
  if (Number.isFinite(value) && value >= 1 && value <= 2) return;
  item.value = 1.0;
  if (item.inputEl) item.inputEl.value = "1";
  item.callback?.(1.0);
}

function hideWidget(node, name) {
  const item = widget(node, name);
  if (!item) return;
  item.hidden = true;
  item.options = { ...(item.options || {}), hidden: true };
  item.computeSize = () => [0, -4];
  item.draw = () => {};
  if (item.inputEl) item.inputEl.style.display = "none";
}

function readRegions(node) {
  try {
    const value = JSON.parse(widget(node, JSON_WIDGET)?.value || "[]");
    return Array.isArray(value) ? value : [];
  } catch (_) {
    return [];
  }
}

function writeRegions(node, regions) {
  const item = widget(node, JSON_WIDGET);
  if (!item) return;
  item.value = JSON.stringify(regions, null, 2);
  if (item.inputEl) item.inputEl.value = item.value;
  item.callback?.(item.value);
  node.setDirtyCanvas?.(true, true);
}

function defaultCanvasLora() {
  return {
    enabled: false,
    lora: "None",
    trigger: "",
    prompt: "",
    strength: 1.0,
    coverage: "unboxed",
    start: 0.0,
    end: 1.0,
  };
}

function normalizeCanvasLora(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const coverage = ["unboxed", "global"].includes(source.coverage) ? source.coverage : "unboxed";
  return {
    enabled: source.enabled === true,
    lora: String(source.lora || "None"),
    trigger: String(source.trigger || ""),
    prompt: String(source.prompt ?? source.description ?? ""),
    strength: number(source.strength, 1.0),
    coverage,
    start: Math.max(0, Math.min(1, number(source.start ?? source.schedule?.start, 0.0))),
    end: Math.max(0, Math.min(1, number(source.end ?? source.schedule?.end, 1.0))),
  };
}

function readCanvasLora(node) {
  try {
    return normalizeCanvasLora(JSON.parse(widget(node, CANVAS_LORA_WIDGET)?.value || "{}"));
  } catch (_) {
    return defaultCanvasLora();
  }
}

function writeCanvasLora(node, canvasLora) {
  const item = widget(node, CANVAS_LORA_WIDGET);
  if (!item) return;
  const normalized = normalizeCanvasLora(canvasLora);
  item.value = JSON.stringify(normalized, null, 2);
  if (item.inputEl) item.inputEl.value = item.value;
  item.callback?.(item.value);
  node.setDirtyCanvas?.(true, true);
}

function defaultRegion(index) {
  const offset = (index % 4) * 0.12;
  return {
    name: `Character ${index + 1}`, enabled: true, lora: "None", trigger: `CHAR_${index + 1}`,
    prompt: "", strength: 1.0, x: Math.min(0.65, 0.05 + offset), y: 0.08,
    w: 0.36, h: 0.84, start: 0.0, end: 1.0,
  };
}

function layoutRegions(regions) {
  const count = Math.max(1, regions.length);
  const slot = 1 / count;
  const gutter = Math.min(0.04, slot * 0.12);
  regions.forEach((region, index) => {
    region.x = index * slot + gutter / 2;
    region.y = 0.08;
    region.w = slot - gutter;
    region.h = 0.84;
  });
  return regions;
}

function number(value, fallback) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function unwrapJsonText(value) {
  let text = String(value ?? "").trim();
  if (text.startsWith("```")) {
    text = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "");
  }
  return text;
}

function importedDimension(value, fallback) {
  const parsed = number(value, fallback);
  return Math.max(256, Math.min(4096, Math.round(parsed / 8) * 8));
}

function normalizeImportedScene(rawText) {
  let payload;
  try {
    payload = JSON.parse(unwrapJsonText(rawText));
  } catch (error) {
    throw new Error(`The selected text is not valid JSON: ${error.message}`);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("The scene JSON must contain one top-level object.");
  }
  if (!payload.canvas || typeof payload.canvas !== "object") {
    throw new Error("The scene JSON is missing its canvas width and height.");
  }
  if (!Array.isArray(payload.characters) || payload.characters.length < 1) {
    throw new Error("The scene JSON must contain at least one character.");
  }
  if (payload.characters.length > 5) {
    throw new Error("This router supports a maximum of five characters.");
  }

  const width = importedDimension(payload.canvas.width, 1024);
  const height = importedDimension(payload.canvas.height, 1024);
  const router = payload.router && typeof payload.router === "object" ? payload.router : {};
  const supersampling = payload.supersampling && typeof payload.supersampling === "object"
    ? payload.supersampling
    : {};
  const canvasLora = normalizeCanvasLora(
    payload.canvas_lora && typeof payload.canvas_lora === "object"
      ? payload.canvas_lora
      : payload.background_lora,
  );
  const overlapPolicy = ["nearest", "normalize", "allow"].includes(router.overlap_policy)
    ? router.overlap_policy
    : "nearest";
  const regions = payload.characters.map((character, index) => {
    if (!character || typeof character !== "object" || Array.isArray(character)) {
      throw new Error(`Character ${index + 1} is not a valid object.`);
    }
    const placement = character.placement && typeof character.placement === "object"
      ? character.placement
      : {};
    const normalized = placement.normalized && typeof placement.normalized === "object"
      ? placement.normalized
      : placement;
    const schedule = character.schedule && typeof character.schedule === "object"
      ? character.schedule
      : {};
    return clampRegion({
      name: String(character.name || `Character ${index + 1}`),
      enabled: character.enabled !== false,
      lora: String(character.lora || "None"),
      trigger: String(character.trigger || `CHAR_${index + 1}`),
      prompt: String(character.description ?? character.prompt ?? ""),
      strength: number(character.strength, 1),
      x: number(normalized.x, 0.05 + index * 0.45),
      y: number(normalized.y, 0.08),
      w: number(normalized.width ?? normalized.w, 0.4),
      h: number(normalized.height ?? normalized.h, 0.84),
      start: number(schedule.start ?? character.start, 0),
      end: number(schedule.end ?? character.end, 1),
    });
  });

  return {
    width,
    height,
    scenePrompt: String(payload.scene_prompt ?? payload.scenePrompt ?? ""),
    feather: Math.max(0, Math.min(0.5, number(router.feather, 0.08))),
    overlapPolicy,
    scheduleSoftness: Math.max(0, Math.min(0.25, number(router.schedule_softness, 0.04))),
    strict: router.strict !== false,
    supersampleScale: Math.max(1, Math.min(2, number(supersampling.scale, 1))),
    canvasLora,
    regions,
  };
}

function clampRegion(region) {
  region.x = Math.max(0, Math.min(0.999, number(region.x, 0)));
  region.y = Math.max(0, Math.min(0.999, number(region.y, 0)));
  region.w = Math.max(0.001, Math.min(1 - region.x, number(region.w ?? region.width, 0.4)));
  region.h = Math.max(0.001, Math.min(1 - region.y, number(region.h ?? region.height, 0.8)));
  region.start = Math.max(0, Math.min(1, number(region.start, 0)));
  region.end = Math.max(0, Math.min(1, number(region.end, 1.0)));
  return region;
}

function makeInput(type, value, onChange) {
  const input = document.createElement("input");
  input.type = type;
  input.value = value ?? "";
  input.addEventListener(type === "checkbox" ? "change" : "input", () => onChange(input));
  return input;
}

function makeSearchableLoraPicker(value, onSelect) {
  const control = document.createElement("div");
  control.className = "k2cr-combo";
  const input = document.createElement("input");
  input.type = "text";
  input.value = value || "None";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.placeholder = "Type to search LoRAs…";
  const popup = document.createElement("div");
  popup.className = "k2cr-combo-list";
  popup.hidden = true;
  let committed = value || "None";
  let matches = [];
  let active = -1;

  const render = () => {
    const terms = input.value.toLocaleLowerCase().split(/\s+/).filter(Boolean);
    matches = loraNames.filter((name) => {
      const candidate = name.toLocaleLowerCase();
      return terms.every((term) => candidate.includes(term));
    }).slice(0, 120);
    active = matches.length ? 0 : -1;
    popup.replaceChildren();
    if (!matches.length) {
      const empty = document.createElement("div");
      empty.className = "k2cr-combo-empty";
      empty.textContent = "No matching LoRA";
      popup.append(empty);
    } else {
      matches.forEach((name, index) => {
        const option = document.createElement("div");
        option.className = `k2cr-combo-option${index === active ? " active" : ""}`;
        option.textContent = name;
        option.title = name;
        option.addEventListener("pointerdown", (event) => event.preventDefault());
        option.addEventListener("click", () => choose(index));
        popup.append(option);
      });
    }
    popup.hidden = false;
  };

  const updateActive = (next) => {
    if (!matches.length) return;
    active = (next + matches.length) % matches.length;
    [...popup.children].forEach((option, index) => option.classList.toggle("active", index === active));
    popup.children[active]?.scrollIntoView({ block: "nearest" });
  };

  const choose = (index) => {
    const selected = matches[index];
    if (!selected) return;
    input.value = selected;
    committed = selected;
    popup.hidden = true;
    onSelect(selected);
  };

  input.addEventListener("focus", () => {
    input.select();
    render();
  });
  input.addEventListener("input", render);
  input.addEventListener("blur", () => {
    window.setTimeout(() => {
      popup.hidden = true;
      input.value = committed;
    }, 100);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (popup.hidden) render(); else updateActive(active + 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      if (popup.hidden) render(); else updateActive(active - 1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      choose(active);
    } else if (event.key === "Escape") {
      popup.hidden = true;
      input.value = committed;
    }
  });
  control.append(input, popup);
  return control;
}

function label(text, control, className = "") {
  const element = document.createElement("label");
  element.className = className;
  element.append(text, control);
  return element;
}

function drawStage(node, canvas, selected = -1) {
  const regions = readRegions(node).map(clampRegion);
  const width = Math.max(1, number(widget(node, "width")?.value, 1024));
  const height = Math.max(1, number(widget(node, "height")?.value, 1024));
  // Preserve the requested image aspect ratio. The old 150-330px height clamp
  // made portrait and panoramic grids look nearly identical even though the
  // resolution label changed. Use layout pixels rather than getBoundingClientRect:
  // ComfyUI transforms DOM widgets with graph zoom, and screen-pixel measurement
  // would make every focus-triggered redraw progressively resize the stage.
  const cssWidth = Math.max(
    1,
    canvas.offsetWidth || canvas.parentElement?.clientWidth || 420,
  );
  const cssHeight = Math.max(1, cssWidth * height / width);
  const scale = window.devicePixelRatio || 1;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * scale);
  canvas.height = Math.round(cssHeight * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = "#111318";
  ctx.fillRect(0, 0, cssWidth, cssHeight);
  ctx.strokeStyle = "#282b33";
  ctx.lineWidth = 1;
  for (let i = 1; i < 3; i++) {
    ctx.beginPath(); ctx.moveTo(cssWidth * i / 3, 0); ctx.lineTo(cssWidth * i / 3, cssHeight); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cssHeight * i / 3); ctx.lineTo(cssWidth, cssHeight * i / 3); ctx.stroke();
  }
  regions.forEach((region, index) => {
    const x = region.x * cssWidth, y = region.y * cssHeight;
    const w = region.w * cssWidth, h = region.h * cssHeight;
    ctx.fillStyle = `${COLORS[index % COLORS.length]}35`;
    ctx.strokeStyle = COLORS[index % COLORS.length];
    ctx.lineWidth = index === selected ? 3 : 2;
    ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = COLORS[index % COLORS.length];
    ctx.fillRect(x, Math.max(0, y - 18), Math.min(w, 140), 18);
    ctx.fillStyle = "#fff"; ctx.font = "11px system-ui";
    ctx.fillText(region.name || `Character ${index + 1}`, x + 4, Math.max(12, y - 5));
    ctx.fillStyle = COLORS[index % COLORS.length];
    ctx.fillRect(x + w - 7, y + h - 7, 10, 10);
  });
  const resolution = `${Math.round(width)} × ${Math.round(height)}`;
  ctx.font = "11px system-ui";
  const labelWidth = ctx.measureText(resolution).width + 12;
  const labelX = cssWidth - labelWidth - 6;
  ctx.fillStyle = "rgba(5,6,9,.78)";
  ctx.fillRect(labelX, 6, labelWidth, 20);
  ctx.fillStyle = "#d8dbe2";
  ctx.fillText(resolution, labelX + 6, 20);
}

function hookDimensionWidget(node, name, redraw) {
  const item = widget(node, name);
  if (!item || item.__k2crDimensionHooked) return;
  const original = item.callback;
  item.callback = function () {
    const result = original?.apply(this, arguments);
    window.requestAnimationFrame(redraw);
    return result;
  };
  item.__k2crDimensionHooked = true;
}

function attachPanel(node) {
  installStyles();
  hideWidget(node, JSON_WIDGET);
  hideWidget(node, "scene_prompt");
  hideWidget(node, CANVAS_LORA_WIDGET);
  const root = document.createElement("div");
  root.className = "k2cr";
  const scene = document.createElement("details");
  scene.className = "k2cr-scene";
  scene.open = true;
  const sceneSummary = document.createElement("summary");
  sceneSummary.textContent = "Scene prompt";
  const scenePrompt = document.createElement("textarea");
  scenePrompt.placeholder = "Describe the shared setting, action, composition, lighting, and style…";
  scene.append(sceneSummary, scenePrompt);
  const canvasLoraPanel = document.createElement("details");
  canvasLoraPanel.className = "k2cr-canvas-lora";
  const canvasLoraSummary = document.createElement("summary");
  canvasLoraSummary.textContent = "Canvas LoRA (optional)";
  const canvasLoraBody = document.createElement("div");
  canvasLoraBody.className = "k2cr-canvas-body";
  canvasLoraPanel.append(canvasLoraSummary, canvasLoraBody);
  const toolbar = document.createElement("div");
  toolbar.className = "k2cr-toolbar";
  const add = document.createElement("button"); add.textContent = "+ Character";
  const countInput = document.createElement("input");
  countInput.type = "number"; countInput.min = "1"; countInput.max = "5"; countInput.step = "1";
  const countLabel = label("Characters", countInput, "k2cr-count-label");
  const applyCount = document.createElement("button"); applyCount.textContent = "Set + reset";
  const reset = document.createElement("button"); reset.textContent = "Reset layout";
  const refresh = document.createElement("button"); refresh.textContent = "Refresh grid";
  const importJson = document.createElement("button"); importJson.textContent = "Import JSON";
  const pasteJson = document.createElement("button"); pasteJson.textContent = "Paste JSON";
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".json,application/json";
  fileInput.hidden = true;
  const fitContent = document.createElement("button"); fitContent.textContent = "Fit content";
  const compact = document.createElement("button"); compact.textContent = "Compact";
  const hint = document.createElement("span"); hint.className = "k2cr-hint"; hint.textContent = "drag boxes · corner resizes";
  toolbar.append(countLabel, applyCount, add, reset, refresh, importJson, pasteJson, fitContent, compact, hint);
  const canvas = document.createElement("canvas"); canvas.className = "k2cr-stage";
  const rows = document.createElement("div"); rows.className = "k2cr-rows";
  root.append(scene, canvasLoraPanel, toolbar, fileInput, canvas, rows);
  let selected = -1;

  const resizeNodeToContent = (useCompactHeight) => {
    window.requestAnimationFrame(() => {
      const width = Math.max(480, node.size?.[0] || 480);
      const minimum = node.computeSize?.()?.[1] || 560;
      let targetHeight = minimum;
      if (!useCompactHeight) {
        const editorHeight = root.clientHeight || 1;
        const nonEditorHeight = Math.max(0, (node.size?.[1] || minimum) - editorHeight);
        targetHeight = Math.max(minimum, Math.ceil(nonEditorHeight + root.scrollHeight + 6));
      }
      node.setSize?.([width, targetHeight]);
      window.requestAnimationFrame(() => drawStage(node, canvas, selected));
    });
  };

  scenePrompt.addEventListener("input", () => {
    const item = widget(node, "scene_prompt");
    if (!item) return;
    item.value = scenePrompt.value;
    if (item.inputEl) item.inputEl.value = item.value;
    item.callback?.(item.value);
    node.setDirtyCanvas?.(true, true);
  });

  const persist = (regions, rebuild = false) => {
    writeRegions(node, regions.map(clampRegion));
    drawStage(node, canvas, selected);
    if (rebuild) renderRows();
  };

  const setWidgetValue = (name, value) => {
    const item = widget(node, name);
    if (!item) throw new Error(`The router is missing its ${name} control.`);
    item.value = value;
    if (item.inputEl) item.inputEl.value = value;
    item.callback?.(value);
  };

  const renderCanvasLora = () => {
    canvasLoraBody.replaceChildren();
    const state = readCanvasLora(node);
    canvasLoraSummary.textContent = state.enabled
      ? `Canvas LoRA — ${state.coverage === "global" ? "entire canvas" : "unboxed only"}`
      : "Canvas LoRA (optional)";

    const help = document.createElement("p");
    help.className = "k2cr-canvas-help";
    help.textContent = "Unboxed only applies this LoRA to the inverse of every enabled character box. "
      + "Entire canvas behaves like a base style LoRA underneath the routed characters.";

    const enabled = makeInput("checkbox", "", (input) => {
      state.enabled = input.checked;
      writeCanvasLora(node, state);
      canvasLoraSummary.textContent = state.enabled
        ? `Canvas LoRA — ${state.coverage === "global" ? "entire canvas" : "unboxed only"}`
        : "Canvas LoRA (optional)";
    });
    enabled.checked = state.enabled;

    const loraPicker = makeSearchableLoraPicker(state.lora, (selectedLora) => {
      state.lora = selectedLora;
      writeCanvasLora(node, state);
    });
    const coverage = document.createElement("select");
    [
      ["unboxed", "Unboxed area only"],
      ["global", "Entire canvas"],
    ].forEach(([value, title]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = title;
      coverage.append(option);
    });
    coverage.value = state.coverage;
    coverage.addEventListener("change", () => {
      state.coverage = coverage.value;
      writeCanvasLora(node, state);
      canvasLoraSummary.textContent = state.enabled
        ? `Canvas LoRA — ${state.coverage === "global" ? "entire canvas" : "unboxed only"}`
        : "Canvas LoRA (optional)";
    });
    const trigger = makeInput("text", state.trigger, (input) => {
      state.trigger = input.value;
      writeCanvasLora(node, state);
    });
    const strength = makeInput("number", state.strength, (input) => {
      state.strength = number(input.value, 1);
      writeCanvasLora(node, state);
    });
    strength.step = "0.05";
    strength.min = "-4";
    strength.max = "4";
    const prompt = document.createElement("textarea");
    prompt.className = "k2cr-description";
    prompt.value = state.prompt;
    prompt.placeholder = "Describe the environment or style this LoRA should contribute…";
    prompt.addEventListener("input", () => {
      state.prompt = prompt.value;
      writeCanvasLora(node, state);
    });
    const start = makeInput("number", state.start.toFixed(3), (input) => {
      state.start = Math.max(0, Math.min(1, number(input.value, state.start)));
      writeCanvasLora(node, state);
    });
    const end = makeInput("number", state.end.toFixed(3), (input) => {
      state.end = Math.max(0, Math.min(1, number(input.value, state.end)));
      writeCanvasLora(node, state);
    });
    for (const control of [start, end]) {
      control.step = "0.01";
      control.min = "0";
      control.max = "1";
    }

    const grid = document.createElement("div");
    grid.className = "k2cr-canvas-grid";
    grid.append(
      label("Canvas LoRA", loraPicker),
      label("Coverage", coverage),
      label("Trigger", trigger),
      label("Strength", strength),
      label("Description / style instruction", prompt, "wide"),
    );
    const schedule = document.createElement("div");
    schedule.className = "k2cr-canvas-schedule";
    schedule.append(label("Start", start), label("End", end));
    const enabledLabel = label("Enable Canvas LoRA", enabled, "k2cr-enable-label");
    canvasLoraBody.append(help, enabledLabel, grid, schedule);
  };

  const applyImportedScene = (rawText) => {
    // Normalize and validate everything before mutating the current node.
    const imported = normalizeImportedScene(rawText);
    setWidgetValue("width", imported.width);
    setWidgetValue("height", imported.height);
    setWidgetValue("supersample_scale", imported.supersampleScale);
    setWidgetValue("scene_prompt", imported.scenePrompt);
    setWidgetValue("feather", imported.feather);
    setWidgetValue("overlap_policy", imported.overlapPolicy);
    setWidgetValue("schedule_softness", imported.scheduleSoftness);
    setWidgetValue("strict", imported.strict);
    writeCanvasLora(node, imported.canvasLora);
    scenePrompt.value = imported.scenePrompt;
    selected = -1;
    writeRegions(node, imported.regions);
    renderCanvasLora();
    renderRows();

    const missing = [...new Set(
      [
        ...imported.regions.map((region) => region.lora),
        imported.canvasLora.lora,
      ]
        .filter((name) => name && name !== "None" && !loraNames.includes(name)),
    )];
    if (missing.length) {
      window.alert(
        `Scene imported, but ${missing.length} LoRA${missing.length === 1 ? " was" : "s were"} not found locally:\n\n`
        + missing.join("\n"),
      );
    }
  };

  function renderRows() {
    rows.replaceChildren();
    const currentScenePrompt = String(widget(node, "scene_prompt")?.value ?? "");
    if (document.activeElement !== scenePrompt && scenePrompt.value !== currentScenePrompt) {
      scenePrompt.value = currentScenePrompt;
    }
    const regions = readRegions(node).map(clampRegion);
    countInput.value = String(Math.max(1, regions.length));
    add.disabled = regions.length >= 5;
    regions.forEach((region, index) => {
      const row = document.createElement("div"); row.className = "k2cr-row";
      row.style.setProperty("--k2-color", COLORS[index % COLORS.length]);
      const head = document.createElement("div"); head.className = "k2cr-row-head";
      const enabled = makeInput("checkbox", "", (input) => { region.enabled = input.checked; persist(regions); });
      enabled.checked = region.enabled !== false;
      const name = makeInput("text", region.name || `Character ${index + 1}`, (input) => { region.name = input.value; persist(regions); });
      const remove = document.createElement("button"); remove.textContent = "Remove";
      remove.disabled = regions.length <= 1;
      remove.addEventListener("click", () => { regions.splice(index, 1); selected = -1; persist(regions, true); });
      head.append(enabled, name, remove);

      const grid = document.createElement("div"); grid.className = "k2cr-grid";
      const loraPicker = makeSearchableLoraPicker(region.lora, (selected) => {
        region.lora = selected;
        persist(regions);
      });
      const trigger = makeInput("text", region.trigger || "", (input) => { region.trigger = input.value; persist(regions); });
      const strength = makeInput("number", region.strength ?? 1, (input) => { region.strength = number(input.value, 1); persist(regions); });
      strength.step = "0.05"; strength.min = "-4"; strength.max = "4";
      const prompt = document.createElement("textarea");
      prompt.className = "k2cr-description";
      prompt.value = region.prompt || "";
      prompt.addEventListener("input", () => { region.prompt = prompt.value; persist(regions); });
      grid.append(label("Character LoRA", loraPicker), label("Trigger", trigger), label("Strength", strength), document.createElement("span"), label("Description", prompt, "wide"));

      const boxGrid = document.createElement("div"); boxGrid.className = "k2cr-box-grid";
      [["X","x"],["Y","y"],["W","w"],["H","h"],["Start","start"],["End","end"]].forEach(([title,key]) => {
        const input = makeInput("number", Number(region[key] ?? 0).toFixed(3), (field) => { region[key] = number(field.value, region[key]); persist(regions); });
        input.step = "0.01"; input.min = "0"; input.max = "1";
        boxGrid.append(label(title, input));
      });
      row.addEventListener("pointerdown", () => { selected = index; drawStage(node, canvas, selected); });
      row.append(head, grid, boxGrid); rows.append(row);
    });
    drawStage(node, canvas, selected);
  }

  applyCount.addEventListener("click", () => {
    const target = Math.max(1, Math.min(5, Math.round(number(countInput.value, 1))));
    const regions = readRegions(node).map(clampRegion);
    while (regions.length < target) regions.push(defaultRegion(regions.length));
    if (regions.length > target) regions.splice(target);
    layoutRegions(regions);
    countInput.value = String(target);
    selected = -1;
    persist(regions, true);
  });
  countInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") applyCount.click();
  });
  add.addEventListener("click", () => {
    const regions = readRegions(node);
    if (regions.length >= 5) return;
    regions.push(defaultRegion(regions.length));
    selected = regions.length - 1;
    persist(regions, true);
  });
  reset.addEventListener("click", () => {
    const regions = readRegions(node);
    if (!regions.length) regions.push(defaultRegion(0));
    layoutRegions(regions);
    persist(regions, true);
  });
  refresh.addEventListener("click", () => drawStage(node, canvas, selected));
  importJson.addEventListener("click", () => {
    fileInput.value = "";
    fileInput.click();
  });
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    try {
      if (file.size > 2 * 1024 * 1024) {
        throw new Error("The selected JSON file is larger than 2 MB.");
      }
      applyImportedScene(await file.text());
    } catch (error) {
      window.alert(`Could not import scene:\n\n${error.message || error}`);
    } finally {
      fileInput.value = "";
    }
  });
  pasteJson.addEventListener("click", async () => {
    try {
      let text = "";
      try {
        text = await navigator.clipboard.readText();
      } catch (_) {
        text = window.prompt("Paste the complete Krea2 Multi-LoRA Composer scene JSON:") || "";
      }
      if (!text.trim()) throw new Error("The clipboard does not contain any text.");
      applyImportedScene(text);
    } catch (error) {
      window.alert(
        `Could not paste scene JSON:\n\n${error.message || error}\n\n`
        + "Copy the complete JSON text and try again.",
      );
    }
  });
  fitContent.addEventListener("click", () => resizeNodeToContent(false));
  compact.addEventListener("click", () => resizeNodeToContent(true));

  let drag = null;
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 && event.button !== 2) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const nx = (event.clientX - rect.left) / rect.width, ny = (event.clientY - rect.top) / rect.height;
    const regions = readRegions(node).map(clampRegion);
    for (let index = regions.length - 1; index >= 0; index--) {
      const region = regions[index];
      if (nx >= region.x && nx <= region.x + region.w && ny >= region.y && ny <= region.y + region.h) {
        selected = index;
        const resize = Math.abs(nx - (region.x + region.w)) < 0.035 && Math.abs(ny - (region.y + region.h)) < 0.035;
        drag = { index, nx, ny, original: { ...region }, resize };
        canvas.setPointerCapture(event.pointerId); drawStage(node, canvas, selected); break;
      }
    }
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const rect = canvas.getBoundingClientRect();
    const nx = (event.clientX - rect.left) / rect.width, ny = (event.clientY - rect.top) / rect.height;
    const regions = readRegions(node).map(clampRegion), region = regions[drag.index];
    const dx = nx - drag.nx, dy = ny - drag.ny;
    if (drag.resize) {
      region.w = Math.max(0.03, Math.min(1 - region.x, drag.original.w + dx));
      region.h = Math.max(0.03, Math.min(1 - region.y, drag.original.h + dy));
    } else {
      region.x = Math.max(0, Math.min(1 - region.w, drag.original.x + dx));
      region.y = Math.max(0, Math.min(1 - region.h, drag.original.y + dy));
    }
    persist(regions);
  });
  const stopDrag = () => { if (drag) renderRows(); drag = null; };
  canvas.addEventListener("pointerup", stopDrag); canvas.addEventListener("pointercancel", stopDrag);

  const dom = node.addDOMWidget("character_router_ui", "character_router_ui", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 360,
    getValue: () => "",
    setValue: () => {},
  });
  node.resizable = true;
  node.__k2crRoot = root;
  node.__k2crRender = renderRows;
  node.__k2crRenderCanvas = renderCanvasLora;
  hookDimensionWidget(node, "width", () => drawStage(node, canvas, selected));
  hookDimensionWidget(node, "height", () => drawStage(node, canvas, selected));
  renderCanvasLora();
  renderRows();
  window.requestAnimationFrame(() => {
    const minimum = node.computeSize?.()?.[1] || 560;
    if ((node.size?.[1] || 0) < minimum) {
      node.setSize?.([Math.max(480, node.size?.[0] || 480), minimum]);
    }
    drawStage(node, canvas, selected);
  });
}

app.registerExtension({
  name: "Krea2.MultiLoRAComposer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE && nodeType.comfyClass !== NODE_TYPE) return;
    await loadLoras();
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      normalizeSupersampleScale(this);
      if (!this.__k2crRoot) attachPanel(this);
      return result;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configured?.apply(this, arguments);
      queueMicrotask(() => {
        normalizeSupersampleScale(this);
        hideWidget(this, JSON_WIDGET);
        hideWidget(this, "scene_prompt");
        hideWidget(this, CANVAS_LORA_WIDGET);
        this.__k2crRenderCanvas?.();
        this.__k2crRender?.();
      });
      return result;
    };
  },
});
