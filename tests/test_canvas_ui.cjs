const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.listeners = {};
    this.style = { setProperty() {} };
    this.className = "";
    this.value = "";
    this.offsetWidth = 480;
    this.clientWidth = 480;
  }
  append(...children) {
    this.children.push(...children);
    for (const child of children) if (child instanceof Element) child.parentElement = this;
  }
  appendChild(child) { this.append(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, callback) {
    (this.listeners[type] ??= []).push(callback);
  }
  dispatch(type) {
    return Promise.all((this.listeners[type] ?? []).map(
      (callback) => callback({ preventDefault() {}, stopPropagation() {} }),
    ));
  }
  click() { if (!this.disabled) this.dispatch("click"); }
  getContext() {
    this.context ??= {
      labels: [], draws: 0,
      scale() {}, clearRect() { this.draws++; }, fillRect() {}, strokeRect() {},
      beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
      fillText(text) { this.labels.push(text); },
      measureText(text) { return { width: text.length * 6 }; },
    };
    return this.context;
  }
}

function elements(root, predicate) {
  return [
    ...(predicate(root) ? [root] : []),
    ...root.children.filter((child) => child instanceof Element).flatMap((child) => elements(child, predicate)),
  ];
}

function field(root, title) {
  return elements(root, (element) => element.tagName === "label" && element.children[0] === title)[0].children[1];
}

function change(input, value) {
  input.value = value;
  input.dispatch(input.tagName === "select" ? "change" : "input");
}

function editor(initial = {}, { realStage = false } = {}) {
  const head = new Element("head");
  const observers = [];
  const context = vm.createContext({
    console, app: { registerExtension() {} }, api: {},
    document: {
      head, createElement: (tag) => new Element(tag),
      getElementById: (id) => head.children.find((element) => element.id === id),
    },
    window: { requestAnimationFrame: (callback) => callback() },
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; observers.push(this); }
      observe(target) { this.target = target; }
      disconnect() { this.disconnected = true; }
    },
  });
  const source = readFileSync(join(process.cwd(), "web", "character_router.js"), "utf8");
  vm.runInContext(source.replace(/^import .*;\r?\n/gm, ""), context);
  if (!realStage) context.drawStage = () => {};
  const values = {
    regions_json: JSON.stringify([{ name: "Person", enabled: true, lora: "None", trigger: "PERSON" }]),
    canvas_lora_json: JSON.stringify(initial), scene_prompt: "A city", width: 1024, height: 1024,
    supersample_scale: 1, feather: 0.08, overlap_policy: "nearest", schedule_softness: 0.04, strict: true,
  };
  const node = {
    widgets: Object.entries(values).map(([name, value]) => ({ name, value })),
    size: [600, 800], addDOMWidget() {}, setDirtyCanvas() {},
  };
  context.attachPanel(node);
  const panels = () => elements(node.__k2crRoot, (element) => element.className === "k2cr-canvas-lora");
  const button = (text, root = node.__k2crRoot) => elements(
    root, (element) => element.tagName === "button" && element.textContent === text,
  )[0];
  const saved = () => JSON.parse(node.widgets.find((item) => item.name === "canvas_lora_json").value);
  const stage = elements(node.__k2crRoot, (element) => element.tagName === "canvas")[0];
  return { context, node, panels, button, saved, stage, observers };
}

test("legacy workflow retains its coverage and + Canvas has no fixed count limit", () => {
  const { panels, button, saved, node } = editor({
    enabled: true, lora: "paint", coverage: "global", strength: 0.6,
  });
  const originalCharacters = node.widgets[0].value;
  for (let index = 0; index < 7; index++) button("+ Canvas").click();
  assert.equal(panels().length, 8);
  assert.equal(saved().length, 8);
  assert.equal(saved()[0].coverage, "global");
  assert.equal(saved()[0].strength, 0.6);
  assert.equal(saved()[7].enabled, true);
  assert.equal(button("+ Canvas").disabled, undefined);
  assert.equal(node.widgets[0].value, originalCharacters);
});

test("each Canvas has an independent picker, coverage, strength, prompt and schedule", () => {
  const { context, panels, button, saved } = editor({ enabled: true, lora: "paint", coverage: "global" });
  vm.runInContext('loraNames = ["None", "paint", "grain"];', context);
  button("+ Canvas").click();
  const second = panels()[1];
  const picker = field(second, "Canvas LoRA");
  change(picker.children[0], "grain");
  picker.children[1].children[0].click();
  change(field(second, "Coverage"), "global");
  change(field(second, "Strength"), "0.35");
  change(field(second, "Trigger"), "GRAIN");
  change(field(second, "Description / style instruction"), "paper texture");
  change(field(second, "Start"), "0.2");
  change(field(second, "End"), "0.8");
  assert.deepEqual(saved().map((canvas) => canvas.lora), ["paint", "grain"]);
  assert.equal(saved()[0].strength, 1);
  assert.equal(saved()[1].coverage, "global");
  assert.equal(saved()[1].strength, 0.35);
  assert.equal(saved()[1].trigger, "GRAIN");
  assert.equal(saved()[1].prompt, "paper texture");
  assert.equal(saved()[1].start, 0.2);
  assert.equal(saved()[1].end, 0.8);
  assert.match(second.children[0].textContent, /entire canvas/);
  const enabled = field(second, "Enable Canvas LoRA");
  enabled.checked = false;
  enabled.dispatch("change");
  assert.equal(saved()[1].enabled, false);
  assert.equal(saved()[0].enabled, true);
});

test("removing any Canvas preserves the others, including an empty saved collection", () => {
  const { panels, button, saved, node } = editor([
    { enabled: true, lora: "first" }, { enabled: true, lora: "middle" }, { enabled: true, lora: "last" },
  ]);
  button("Remove", panels()[1]).click();
  assert.deepEqual(saved().map((canvas) => canvas.lora), ["first", "last"]);
  change(field(panels()[1], "Strength"), "0.4");
  assert.equal(saved()[1].strength, 0.4);
  button("Remove", panels()[0]).click();
  button("Remove", panels()[0]).click();
  assert.deepEqual(saved(), []);
  node.__k2crRenderCanvas();
  assert.equal(panels().length, 0);
  button("+ Canvas").click();
  assert.equal(saved().length, 1);
});

test("scene import prefers canvas_loras, preserves schedules and supports legacy aliases", () => {
  const { context } = editor();
  const scene = { canvas: { width: 1024, height: 768 }, characters: [{}] };
  const imported = (extra) => context.normalizeImportedScene(JSON.stringify({ ...scene, ...extra }));
  const legacy = { enabled: true, lora: "legacy", coverage: "global" };
  assert.equal(imported({ canvas_lora: legacy }).canvasLoras[0].lora, "legacy");
  assert.equal(imported({ background_lora: legacy }).canvasLoras[0].coverage, "global");
  const multiple = imported({
    canvas_lora: legacy,
    canvas_loras: [
      { enabled: true, lora: "paint", coverage: "global", schedule: { start: 0.2, end: 0.9 } },
      { enabled: true, lora: "grain", coverage: "unboxed" },
    ],
  });
  assert.equal(multiple.canvasLoras.length, 2);
  assert.equal(multiple.canvasLoras[0].lora, "paint");
  assert.equal(multiple.canvasLoras[0].start, 0.2);
  assert.equal(multiple.canvasLoras[0].end, 0.9);
  assert.equal(imported({ canvas_lora: legacy, canvas_loras: [] }).canvasLoras.length, 0);
  assert.equal(imported({}).canvasLoras[0].enabled, false);
});

test("invalid Canvas entries raise an explicit import error", () => {
  const { context } = editor();
  const scene = { canvas: {}, characters: [{}] };
  for (const canvas_loras of [42, "style", [null], [[]]]) {
    assert.throws(
      () => context.normalizeImportedScene(JSON.stringify({ ...scene, canvas_loras })),
      /Canvas LoRA 1 must be a JSON object/,
    );
  }
});

test("file import restores every Canvas and reports every missing LoRA", async () => {
  const { context, node, saved, panels } = editor();
  const alerts = [];
  context.window.alert = (message) => alerts.push(message);
  const fileInput = elements(node.__k2crRoot, (element) => element.type === "file")[0];
  fileInput.files = [{
    size: 1024,
    text: async () => JSON.stringify({
      canvas: { width: 768, height: 1024 }, characters: [{ lora: "None" }],
      canvas_loras: [
        { enabled: true, lora: "paint", coverage: "global", strength: 0.7 },
        { enabled: true, lora: "grain", coverage: "unboxed", schedule: { start: 0.2, end: 0.9 } },
      ],
    }),
  }];
  await fileInput.dispatch("change");
  assert.deepEqual(saved().map((canvas) => canvas.lora), ["paint", "grain"]);
  assert.equal(saved()[1].start, 0.2);
  assert.equal(panels().length, 2);
  assert.equal(alerts.length, 1);
  assert.match(alerts[0], /paint\ngrain/);
});

test("invalid imported Canvas does not mutate the existing workflow", async () => {
  const { context, node } = editor({ enabled: true, lora: "keep", coverage: "global" });
  const original = JSON.stringify(node.widgets.map(({ name, value }) => ({ name, value })));
  const alerts = [];
  context.window.alert = (message) => alerts.push(message);
  const fileInput = elements(node.__k2crRoot, (element) => element.type === "file")[0];
  fileInput.files = [{
    size: 100,
    text: async () => JSON.stringify({ canvas: {}, characters: [{}], canvas_loras: [null] }),
  }];
  await fileInput.dispatch("change");
  assert.equal(JSON.stringify(node.widgets.map(({ name, value }) => ({ name, value }))), original);
  assert.equal(alerts.length, 1);
  assert.match(alerts[0], /Could not import scene/);
});

function connectResolutionSelector(node, ratio = "16:9 (Widescreen)", megapixels = 1, multiple = 8) {
  const selector = {
    type: "ResolutionSelector",
    widgets: [
      { name: "aspect_ratio", value: ratio },
      { name: "megapixels", value: megapixels },
      { name: "multiple", value: multiple },
    ],
    outputs: [{ name: "width" }, { name: "height" }],
  };
  const links = [{ origin_id: 1, origin_slot: 0 }, { origin_id: 1, origin_slot: 1 }];
  const sources = new Map([[1, selector]]);
  node.inputs = [{ name: "width" }, { name: "height" }];
  node.graph = { getNodeById: (id) => sources.get(id) };
  node.getInputLink = (slot) => links[slot];
  return { selector, links, sources };
}

test("grid resolves connected ResolutionSelector outputs instead of stale 1024 widgets", () => {
  const { node, stage } = editor({}, { realStage: true });
  const originalRegions = node.widgets[0].value;
  connectResolutionSelector(node);
  node.__k2crRender();
  assert.equal(stage.context.labels.at(-1), "1368 × 768");
  assert.equal(stage.style.height, `${480 * 768 / 1368}px`);
  assert.equal(node.widgets.find((item) => item.name === "width").value, 1024);
  assert.equal(node.widgets[0].value, originalRegions);
});

test("grid automatically tracks aspect ratio, megapixels, rounding multiple and disconnection", () => {
  const { node, stage } = editor({}, { realStage: true });
  const { selector, links } = connectResolutionSelector(node);
  node.onDrawForeground();
  assert.equal(stage.context.labels.at(-1), "1368 × 768");
  selector.widgets[0].value = "9:16 (Portrait Widescreen)";
  node.onDrawForeground();
  assert.equal(stage.context.labels.at(-1), "768 × 1368");
  selector.widgets[1].value = 2;
  selector.widgets[2].value = 64;
  node.onDrawForeground();
  assert.equal(stage.context.labels.at(-1), "1088 × 1920");
  const draws = stage.context.draws;
  node.onDrawForeground();
  assert.equal(stage.context.draws, draws);
  links.splice(0, links.length);
  node.onDrawForeground();
  assert.equal(stage.context.labels.at(-1), "1024 × 1024");
});

test("local dimension assignments and editor resizing update without widget callbacks", () => {
  const { node, stage, observers } = editor({}, { realStage: true });
  node.widgets.find((item) => item.name === "width").value = 1536;
  node.widgets.find((item) => item.name === "height").value = 768;
  node.onDrawForeground();
  assert.equal(stage.style.height, "240px");
  stage.offsetWidth = 600;
  observers[0].callback();
  assert.equal(stage.style.height, "300px");
  node.onRemoved();
  assert.equal(observers[0].disconnected, true);
});

test("resolution links follow reroutes and honor output slots even when swapped", () => {
  const { node, stage } = editor({}, { realStage: true });
  const { links, sources } = connectResolutionSelector(node);
  const reroute = {
    type: "Reroute", graph: node.graph,
    getInputLink: () => ({ origin_id: 1, origin_slot: 1 }),
  };
  sources.set(2, reroute);
  links[0] = { origin_id: 2, origin_slot: 0 };
  links[1] = { origin_id: 1, origin_slot: 0 };
  node.onDrawForeground();
  assert.equal(stage.context.labels.at(-1), "768 × 1368");
  reroute.getInputLink = () => ({ origin_id: 2, origin_slot: 0 });
  assert.throws(() => node.onDrawForeground(), /reroute cycle/);
});

test("numeric primitive links and reconnecting resolution sources update the grid", () => {
  const { node, stage } = editor({}, { realStage: true });
  const { links, sources } = connectResolutionSelector(node);
  sources.set(2, { type: "PrimitiveNode", widgets: [{ value: 1536 }] });
  sources.set(3, { type: "PrimitiveInt", widgets: [{ name: "value", value: 768 }] });
  links[0] = { origin_id: 2, origin_slot: 0 };
  links[1] = { origin_id: 3, origin_slot: 0 };
  node.onDrawForeground();
  assert.equal(stage.context.labels.at(-1), "1536 × 768");
  links[0] = { origin_id: 1, origin_slot: 0, data: 1024 };
  links[1] = { origin_id: 1, origin_slot: 1, data: 1024 };
  node.onDrawForeground();
  assert.equal(stage.context.labels.at(-1), "1368 × 768");
});

test("selector rounding matches Python at exact half-multiples", () => {
  const { context, node } = editor();
  const { selector } = connectResolutionSelector(node, "1:1 (Square)", 1028 ** 2 / 1024 ** 2);
  assert.equal(context.stageDimensions(node).width, 1024);
  selector.widgets[1].value = 1036 ** 2 / 1024 ** 2;
  assert.equal(context.stageDimensions(node).width, 1040);
});

test("invalid selector settings fail explicitly rather than display a false resolution", () => {
  const { context, node } = editor();
  const { selector } = connectResolutionSelector(node);
  selector.widgets[0].value = "not an aspect ratio";
  assert.throws(() => context.stageDimensions(node), /invalid resolution settings/);
});

test("size watcher preserves existing lifecycle hooks and avoids zoom-dependent growth", () => {
  const { context, stage } = editor({}, { realStage: true });
  const calls = [];
  const node = {
    widgets: [{ name: "width", value: 1536 }, { name: "height", value: 768 }],
    onDrawForeground(value) { calls.push([this, value]); return "drawn"; },
    onRemoved(value) { calls.push([this, value]); return "removed"; },
  };
  let draws = 0;
  context.watchStageSize(node, stage, () => { draws++; context.drawStage(node, stage); });
  for (let index = 0; index < 3; index++) assert.equal(node.onDrawForeground("frame"), "drawn");
  assert.equal(draws, 1);
  context.window.devicePixelRatio = 2;
  node.onDrawForeground("retina");
  assert.equal(stage.width, 960);
  assert.equal(stage.height, 480);
  assert.equal(stage.style.height, "240px");
  assert.equal(node.onRemoved("cleanup"), "removed");
  assert.equal(calls.at(-1)[0], node);
  assert.equal(calls.at(-1)[1], "cleanup");
});
