const $ = (id) => document.getElementById(id);
const copy = (value) => structuredClone(value);
const numberText = (value) => new Intl.NumberFormat("ja-JP").format(value ?? 0);
const paths = (value) => encodeURIComponent(value);

const targetModes = [
  ["document", "本文全体"], ["sentence", "文ごと"], ["paragraph", "段落ごと"],
  ["literal", "語句の一致"], ["regex", "正規表現の一致"],
];
const contextModes = [
  ["target", "対象箇所のみ"], ["sentence", "対象を含む文"], ["window", "対象の前後"],
  ["paragraph", "対象を含む段落"], ["document", "本文全体"],
];
const kinds = [
  ["choice", "単一選択"], ["multi_label", "複数選択"],
  ["boolean", "真偽判定"], ["score", "スコア"],
];
const annotationLabels = { accepted: "採用", review: "要確認", excluded: "除外" };
const jobLabels = {
  queued: "待機中", running: "処理中", paused: "一時停止中", cancelled: "中止",
  completed: "完了", completed_with_errors: "エラーを伴う完了",
};
const originLabels = { demo: "デモ", model: "Jev", verifier: "検証モデル" };
const stageLabels = { primary: "一次判定", secondary: "二次判定", verifier: "検証" };
const qualityPresets = {
  standard: { min_confidence: 0.6, boolean_low: 0.2, boolean_high: 0.8, audit_fraction: 0.05, score_tolerance: 0.5 },
  quick: { min_confidence: 0.5, boolean_low: 0.25, boolean_high: 0.75, audit_fraction: 0, score_tolerance: 1 },
  cautious: { min_confidence: 0.8, boolean_low: 0.1, boolean_high: 0.9, audit_fraction: 0.15, score_tolerance: 0.25 },
};
const qualityInputs = {
  min_confidence: "min-confidence", boolean_low: "boolean-low", boolean_high: "boolean-high",
  audit_fraction: "audit-fraction", score_tolerance: "score-tolerance",
};
const reviewPageSize = 50;
const defaultWindowChars = 180;
let session;
let epoch = 0;
let page = "recipe";
let datasets = [];
let templates = [];
let jobs = [];
let dataset;
let features = [];
let recipeOutput;
let jobOutput;
let preview;
let previewRevision = 0;
let previewRun = 0;
let liveModel;
let activeJob;
let reviewOffset = 0;
let reviewStatus = "all";
let pendingReviewQuery;
let reviewResult;
let currentAnnotation;
let reviewRequest = 0;
let exportGeneration = 0;
let datasetRequest = 0;
let jobsRequest = 0;
let polling = false;

function node(tag, className = "", content = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== "") element.textContent = content;
  return element;
}

function controlButton(label, className, onClick) {
  const button = node("button", "button " + className, label);
  button.type = "button";
  button.addEventListener("click", () => act(button, onClick));
  return button;
}

function field(label, control) {
  const wrapper = node("label", "field", label);
  wrapper.append(control);
  return wrapper;
}

function check(label, checked = false) {
  const input = node("input");
  input.type = "checkbox";
  input.checked = checked;
  const wrapper = node("label", "check");
  wrapper.append(input, document.createTextNode(label));
  return { input, wrapper };
}

function inputField(label, value, options = {}) {
  const input = node("input");
  Object.assign(input, options);
  input.value = value ?? "";
  return { input, wrapper: field(label, input) };
}

function textAreaField(label, value, options = {}) {
  const input = node("textarea");
  Object.assign(input, options);
  input.value = value ?? "";
  return { input, wrapper: field(label, input) };
}

function setOptions(select, entries, selected, missingLabel = "") {
  const options = entries.map(([value, label, disabled]) => {
    const option = node("option", "", label);
    option.value = value;
    option.disabled = Boolean(disabled);
    return option;
  });
  if (missingLabel && selected && !entries.some(([value]) => value === selected)) {
    const option = node("option", "", missingLabel);
    option.value = selected;
    options.push(option);
  }
  select.replaceChildren(...options);
  select.value = selected ?? "";
  if (select.selectedIndex < 0 && options.length) select.selectedIndex = 0;
}

function selectField(label, entries, selected) {
  const input = node("select");
  setOptions(input, entries, selected);
  return { input, wrapper: field(label, input) };
}

function setNotice(message) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}

function showError(message, focus = true) {
  $("alert").textContent = message;
  $("alert").hidden = !message;
  if (message && focus) $("alert").focus();
}

function errorText(detail) {
  if (Array.isArray(detail)) {
    return detail.map((error) => (error.loc ?? []).filter((part) => part !== "body").join(" → ") + "　" + error.msg).join("\n");
  }
  return typeof detail === "string" ? detail : "";
}

async function api(path, { method = "GET", body } = {}) {
  const currentEpoch = epoch;
  const headers = { Accept: "application/json" };
  if (method === "POST" || method === "PATCH") headers["X-Studio-Request"] = "1";
  if (body !== undefined && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }
  const response = await fetch("/api" + path, { method, body, headers, credentials: "same-origin" });
  const contentType = response.headers.get("Content-Type") ?? "";
  const data = contentType.includes("application/json") ? await response.json() : null;
  if (currentEpoch !== epoch) throw new DOMException("古いリクエスト", "AbortError");
  if (response.status === 401 && path !== "/login") {
    lockWorkspace();
    throw new Error("セッションが終了しました。ログインしてください。");
  }
  if (!response.ok) {
    const error = new Error(errorText(data?.detail) || "リクエストに失敗しました（HTTP " + response.status + "）。");
    error.status = response.status;
    throw error;
  }
  return data;
}

async function act(control, operation) {
  const currentEpoch = epoch;
  if (control) control.disabled = true;
  showError("");
  if (control) setNotice("");
  try {
    await operation();
  } catch (error) {
    if (error.name !== "AbortError") showError(error.message || "操作を完了できませんでした。");
  } finally {
    if (control && currentEpoch === epoch) control.disabled = false;
  }
}

function lockWorkspace() {
  epoch += 1;
  session = { auth_required: true, authenticated: false };
  datasets = [];
  templates = [];
  jobs = [];
  dataset = null;
  preview = null;
  features = [];
  jobViews.clear();
  recipeOutput = null;
  jobOutput = null;
  activeJob = null;
  pendingReviewQuery = null;
  reviewResult = null;
  currentAnnotation = null;
  for (const id of ["preview-content", "sample-table", "jobs-list", "annotation-editor", "feature-cards",
    "recipe-output", "job-output", "dataset-select", "template-select", "export-template-select", "review-job",
    "annotation-select", "text-field", "id-field", "group-field"]) $(id).replaceChildren();
  for (const id of ["dataset-meta", "annotation-summary", "review-job-meta"]) $(id).textContent = "";
  for (const id of ["recipe-name", "target-terms", "target-pattern", "model"]) $(id).value = "";
  setNotice("");
  invalidateExport();
  liveModel = "";
  $("workspace").hidden = true;
  $("navigation").hidden = true;
  $("session-badge").hidden = true;
  $("logout").hidden = true;
  $("login-panel").hidden = false;
  $("loading").hidden = true;
  $("access-token").value = "";
  $("login-form").querySelector("button").disabled = false;
  $("access-token").focus();
}

async function initialize() {
  session = await api("/session");
  if (!session.authenticated) {
    lockWorkspace();
    return;
  }
  const [loadedDatasets, loadedTemplates, loadedJobs] = await Promise.all([
    api("/datasets"), api("/templates"), api("/jobs"),
  ]);
  datasets = loadedDatasets;
  templates = loadedTemplates;
  jobs = loadedJobs;
  fillDatasets();
  fillTemplates();
  const initial = templates.find((template) => template.id === "builtin-equipment") ?? templates[0];
  if (!initial) throw new Error("初期テンプレートを読み込めませんでした。");
  applyTemplate(initial);
  if (datasets.length) await chooseDataset(datasets[0].id);
  else fillSourceOptions();
  const limit = session.limits?.upload_bytes;
  const maxRows = session.limits?.rows;
  $("upload-hint").textContent = "CSV、JSON、JSONL、NDJSON、TXT、Markdown、Parquet" +
    (limit ? " · 最大 " + numberText(limit / (1024 * 1024)) + " MiB" : "") +
    (maxRows ? " · 最大 " + numberText(maxRows) + " 行" : "");
  $("session-badge").textContent = session.provider_ready.jev ? "Jev 接続設定済み" : "デモ環境";
  $("session-badge").className = "status-pill " + (session.provider_ready.jev ? "tone-good" : "tone-demo");
  $("session-badge").hidden = false;
  $("logout").disabled = false;
  $("logout").hidden = !session.auth_required;
  $("login-panel").hidden = true;
  $("navigation").hidden = false;
  $("workspace").hidden = false;
  $("loading").hidden = true;
  updateWorkspaceActions();
  renderJobs();
  renderReviewJobSelect();
  await navigate(false);
}

function fillDatasets() {
  setOptions($("dataset-select"), [
    ["", "データセットを選択"], ...datasets.map((item) => [item.id, item.name + "（" + numberText(item.rows) + " 行）"]),
  ], dataset?.id);
}

function fillTemplates(preferred) {
  for (const id of ["template-select", "export-template-select"]) {
    const select = $(id);
    setOptions(select, templates.map((template) =>
      [template.id, template.name + (template.builtin ? "（組み込み）" : "")]), preferred ?? select.value);
  }
  $("load-template").disabled = !templates.length;
  $("load-export-template").disabled = !templates.length;
}

function fillSourceOptions(preferred = {
  text_field: $("text-field").value, id_field: $("id-field").value, group_field: $("group-field").value,
}, preserveMissing = false) {
  const columnNames = dataset?.columns ?? [];
  const columns = columnNames.map((column) => [column, column]);
  const firstString = columnNames.find((column) => dataset.sample?.some((row) => typeof row[column] === "string"));
  const inputOptions = [
    ["text-field", preferred.text_field, columnNames.includes("text") ? "text" : firstString ?? columnNames[0], "本文列を選択"],
    ["id-field", preferred.id_field, columnNames.includes("id") ? "id" : "", "自動で割り当て"],
    ["group-field", preferred.group_field, "", "使用しない"],
  ];
  for (const [id, wanted, fallback, emptyLabel] of inputOptions) {
    const value = dataset && wanted && !columnNames.includes(wanted) && !preserveMissing ? fallback : wanted ?? "";
    const options = [["", emptyLabel], ...columns];
    if (!dataset && value) options.push([value, value]);
    setOptions($(id), options, value, "列がありません · " + value);
  }
  $("dataset-detail").hidden = !dataset;
  recipeOutput?.updateSources();
}

function renderDataset() {
  const tableHost = $("sample-table");
  tableHost.replaceChildren();
  if (!dataset) return;
  $("dataset-meta").textContent = numberText(dataset.rows) + " 行　/　" + numberText(dataset.columns.length) + " 列";
  const table = node("table", "data-table");
  const head = table.createTHead().insertRow();
  for (const column of dataset.columns) {
    const cell = node("th", "", column);
    cell.scope = "col";
    head.append(cell);
  }
  const tbody = table.createTBody();
  for (const record of dataset.sample ?? []) {
    const row = tbody.insertRow();
    for (const column of dataset.columns) row.insertCell().textContent = displayValue(Object.hasOwn(record, column) ? record[column] : null);
  }
  tableHost.append(table);
}

function selectDataset(result) {
  dataset = result;
  fillDatasets();
  fillSourceOptions();
  renderDataset();
  invalidatePreview(true);
  updateWorkspaceActions();
}

async function chooseDataset(id) {
  const request = ++datasetRequest;
  const result = id ? await api("/datasets/" + paths(id)) : null;
  if (request === datasetRequest) selectDataset(result);
}

function updateWorkspaceActions() {
  $("run-preview").disabled = !dataset;
  $("create-job").disabled = !dataset;
  $("jump-job-create").hidden = !dataset;
}

function applyTemplate(template) {
  const recipe = template.recipe;
  $("recipe-name").value = recipe.name;
  $("target-mode").value = recipe.target.mode;
  $("target-terms").value = recipe.target.terms.join("\n");
  $("target-pattern").value = recipe.target.pattern ?? "";
  $("ignore-case").checked = recipe.target.ignore_case;
  $("max-candidates").value = recipe.target.max_candidates;
  $("context-mode").value = recipe.context.mode;
  $("window-chars").value = recipe.context.window_chars;
  $("max-context").value = recipe.context.max_chars;
  liveModel = recipe.model;
  updateProviders(recipe.provider);
  for (const [key, id] of Object.entries(qualityInputs)) $(id).value = recipe.quality[key];
  const secondary = recipe.quality.secondary_context;
  $("secondary-enabled").checked = Boolean(secondary);
  $("secondary-mode").value = secondary?.mode ?? "document";
  $("secondary-window").value = secondary?.window_chars ?? defaultWindowChars;
  $("secondary-max").value = secondary?.max_chars ?? 12000;
  $("verifier").checked = recipe.quality.verifier;
  updateProviderDetails();
  syncPreset();
  syncModes();
  fillSourceOptions(recipe.source, true);
  features = recipe.features.map((feature) => ({ ...copy(feature), choices: Object.entries(feature.choices) }));
  renderFeatures();
  recipeOutput = outputEditor($("recipe-output"), template.export_spec, recipeExportSources, invalidatePreview);
  invalidatePreview(true);
}

function updateProviders(provider) {
  setOptions($("provider"), [
    ["demo", "デモ（合成の判定値）"], ["jev", session.provider_ready.jev ? "Jev" : "Jev（未設定）", !session.provider_ready.jev],
  ], provider);
}

function updateProviderDetails() {
  const demo = $("provider").value === "demo";
  const verifierReady = session.provider_ready.verifier;
  $("model").readOnly = demo;
  $("model").value = demo ? "demo-v1" : liveModel;
  $("verifier").disabled = demo || !verifierReady;
  $("provider-note").textContent = demo
    ? "デモの判定値は合成データです。結果はすべて人の確認対象になります。"
    : session.provider_ready.jev
      ? "対象箇所と選択した文脈を Jev に送信します。"
      : "Jev を実行するにはサーバーに API キーを設定してください。";
  $("verifier-note").textContent = verifierReady
    ? demo ? "検証モデルは Jev を選ぶと使用できます。" : "設定済みの検証モデルにも選択した文脈を送り、判定を照合します。"
    : "サーバーに検証モデルの接続設定がありません。";
}

function syncPreset() {
  $("quality-preset").value = Object.keys(qualityPresets).find((preset) =>
    Object.entries(qualityPresets[preset]).every(([key, value]) =>
      $(qualityInputs[key]).value !== "" && Number($(qualityInputs[key]).value) === value)) ?? "custom";
}

function syncModes() {
  const mode = $("target-mode").value;
  $("term-field").hidden = mode !== "literal";
  $("pattern-field").hidden = mode !== "regex";
  $("case-field").hidden = mode !== "literal" && mode !== "regex";
  $("window-field").hidden = $("context-mode").value !== "window";
  $("secondary-fields").hidden = !$("secondary-enabled").checked;
  $("secondary-window-field").hidden = $("secondary-mode").value !== "window";
}

function numeric(input, label, { optional = false, openMin, openMax } = {}) {
  if (input.value === "" && !input.validity.badInput && optional) return null;
  const value = input.valueAsNumber;
  if (!Number.isFinite(value) || !input.validity.valid ||
      (openMin !== undefined && value <= openMin) || (openMax !== undefined && value >= openMax)) {
    throw new Error(label + "に有効な値を入力してください。");
  }
  return value;
}

function meaningful(value, label, max) {
  if (!value.trim() || Array.from(value).length > max) {
    throw new Error(label + "は 1〜" + max + " 文字で入力してください。");
  }
  return value;
}

function renderFeatures() {
  const host = $("feature-cards");
  host.replaceChildren();
  for (const [index, feature] of features.entries()) {
    const card = node("fieldset", "feature-card");
    card.append(node("legend", "", feature.name || "特徴 " + (index + 1)));
    const top = node("div", "field-grid");
    const name = inputField("表示名", feature.name, { maxLength: 256 });
    const id = inputField("特徴 ID", feature.id, { maxLength: 64, spellcheck: false });
    const kind = selectField("判定形式", kinds, feature.kind);
    const instruction = textAreaField("判定指示", feature.instruction, { rows: 3, maxLength: 4000, placeholder: "何を根拠に、どう判定するかを記述" });
    name.input.addEventListener("input", () => {
      feature.name = name.input.value;
      card.firstElementChild.textContent = feature.name || "特徴 " + (index + 1);
      recipeOutput?.updateSources();
    });
    id.input.addEventListener("input", () => {
      feature.id = id.input.value;
      recipeOutput?.updateSources();
    });
    kind.input.addEventListener("change", () => {
      feature.kind = kind.input.value;
      if (["choice", "multi_label"].includes(feature.kind) && !feature.choices.length) {
        feature.choices = [["", ""]];
        if (feature.kind === "choice") feature.choices.push(["", ""]);
      }
      if (feature.kind === "score" && !feature.levels.length) feature.levels = ["", ""];
      renderFeatures();
      host.children[index].querySelector("select").focus();
      invalidatePreview();
    });
    instruction.input.addEventListener("input", () => { feature.instruction = instruction.input.value; });
    top.append(name.wrapper, id.wrapper, kind.wrapper);
    card.append(top, instruction.wrapper);
    if (["choice", "multi_label"].includes(feature.kind)) renderChoiceFields(card, feature);
    if (feature.kind === "score") renderLevelFields(card, feature);
    if (feature.kind === "boolean") {
      const filter = check("確信をもって「いいえ」と判定した候補を除外する", feature.filter);
      filter.input.addEventListener("change", () => { feature.filter = filter.input.checked; });
      card.append(filter.wrapper);
    }
    const actions = node("div", "action-row feature-actions");
    const remove = controlButton("特徴を削除", "quiet compact", () => {
      features.splice(index, 1);
      redrawFeatures(index);
    });
    remove.disabled = features.length < 2;
    actions.append(remove);
    card.append(actions);
    host.append(card);
  }
  $("add-feature").disabled = features.length >= 32;
}

function redrawFeatures(index) {
  renderFeatures();
  recipeOutput?.updateSources();
  invalidatePreview();
  const next = $("feature-cards").children[Math.min(index, features.length - 1)];
  (next?.querySelector("input") ?? $("add-feature")).focus();
}

function renderChoiceFields(card, feature) {
  const list = node("div", "choice-list");
  function render() {
    list.replaceChildren();
    feature.choices.forEach(([key, description], index) => {
      const row = node("div", "option-row");
      const label = inputField("ラベル " + (index + 1), key, { maxLength: 256 });
      const detail = inputField("説明", description, { maxLength: 4000 });
      label.input.addEventListener("input", () => { feature.choices[index][0] = label.input.value; });
      detail.input.addEventListener("input", () => { feature.choices[index][1] = detail.input.value; });
      const remove = controlButton("削除", "quiet compact", () => {
        feature.choices.splice(index, 1);
        render();
        list.querySelector("input").focus();
        invalidatePreview();
      });
      remove.setAttribute("aria-label", "ラベル " + (index + 1) + " を削除");
      remove.disabled = feature.choices.length <= (feature.kind === "choice" ? 2 : 1);
      row.append(label.wrapper, detail.wrapper, remove);
      list.append(row);
    });
    const add = controlButton("＋ 選択肢", "secondary compact", () => {
      feature.choices.push(["", ""]);
      render();
      list.children[feature.choices.length - 1].querySelector("input").focus();
      invalidatePreview();
    });
    add.disabled = feature.choices.length >= 32;
    list.append(add);
  }
  render();
  card.append(list);
}

function renderLevelFields(card, feature) {
  const list = node("div", "choice-list");
  function render() {
    list.replaceChildren();
    feature.levels.forEach((description, index) => {
      const row = node("div", "option-row score-row");
      const detail = inputField("レベル " + index + " の説明", description, { maxLength: 4000 });
      detail.input.addEventListener("input", () => { feature.levels[index] = detail.input.value; });
      const remove = controlButton("削除", "quiet compact", () => {
        feature.levels.splice(index, 1);
        if (feature.minimum !== null && Number(feature.minimum) > feature.levels.length - 1) feature.minimum = null;
        render();
        list.querySelector("input").focus();
        invalidatePreview();
      });
      remove.setAttribute("aria-label", "レベル " + index + " を削除");
      remove.disabled = feature.levels.length <= 2;
      row.append(detail.wrapper, remove);
      list.append(row);
    });
    const add = controlButton("＋ レベル", "secondary compact", () => {
      feature.levels.push("");
      render();
      list.children[feature.levels.length - 1].querySelector("input").focus();
      invalidatePreview();
    });
    add.disabled = feature.levels.length >= 10;
    const minimum = inputField("採用する最小スコア（任意）", feature.minimum, {
      type: "number", min: 0, max: feature.levels.length - 1, step: "any", placeholder: "しきい値なし",
    });
    minimum.input.addEventListener("input", () => {
      feature.minimum = minimum.input.validity.badInput ? NaN : minimum.input.value === "" ? null : minimum.input.valueAsNumber;
    });
    list.append(add, minimum.wrapper);
  }
  render();
  card.append(list);
}

function buildRecipe({ requireDataset = true, requireReady = true } = {}) {
  const requiredColumn = requireDataset ? dataset?.columns ?? [] : null;
  const textField = $("text-field").value;
  const idField = $("id-field").value || null;
  const groupField = $("group-field").value || null;
  if (requiredColumn && !requiredColumn.includes(textField)) throw new Error("データセットの本文列を選択してください。");
  if (requiredColumn && [idField, groupField].some((value) => value && !requiredColumn.includes(value))) {
    throw new Error("データセットに存在する ID 列とグループ列を選択してください。");
  }
  const mode = $("target-mode").value;
  const terms = mode === "literal" ? $("target-terms").value.split(/\r?\n/).filter((term) => term.trim()) : [];
  if (mode === "literal" && (!terms.length || terms.length > 64 || terms.some((term) => Array.from(term).length > 256))) {
    throw new Error("検索語は 1〜64 語、各 256 文字以内で入力してください。");
  }
  const pattern = mode === "regex" ? meaningful($("target-pattern").value, "正規表現", 2000) : null;
  const provider = $("provider").value;
  if (requireReady && provider === "jev" && !session.provider_ready.jev) throw new Error("Jev の接続設定がありません。");
  const verifier = provider === "jev" && $("verifier").checked;
  if (requireReady && verifier && !session.provider_ready.verifier) throw new Error("検証モデルの接続設定がありません。");
  const used = new Set();
  const serializedFeatures = features.map((feature, index) => {
    const label = "特徴 " + (index + 1);
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(feature.id)) throw new Error(label + "の ID は半角英数字・ハイフン・アンダースコアで入力してください。");
    if (used.has(feature.id)) throw new Error("特徴 ID が重複しています　" + feature.id);
    used.add(feature.id);
    const name = meaningful(feature.name, label + "の表示名", 256).trim();
    const instruction = meaningful(feature.instruction, label + "の判定指示", 4000).trim();
    if ([name.toLowerCase(), feature.id.toLowerCase()].includes(instruction.toLowerCase())) {
      throw new Error(name + "の判定指示に、名前とは別に判定の根拠を記述してください。");
    }
    const choices = Object.create(null);
    const levels = [];
    let minimum = null;
    if (["choice", "multi_label"].includes(feature.kind)) {
      for (const [key, description] of feature.choices) {
        const trimmed = meaningful(key, name + "のラベル", 256).trim();
        if (Object.hasOwn(choices, trimmed)) throw new Error(name + "のラベルが重複しています　" + trimmed);
        choices[trimmed] = meaningful(description, name + "のラベル説明", 4000).trim();
      }
      if (feature.kind === "choice" && Object.keys(choices).length < 2) throw new Error(name + "に選択肢を 2 つ以上設定してください。");
    }
    if (feature.kind === "score") {
      levels.push(...feature.levels.map((level) => meaningful(level, name + "のレベル説明", 4000).trim()));
      if (new Set(levels).size !== levels.length) throw new Error(name + "のレベルには異なる説明を入力してください。");
      minimum = feature.minimum;
      if (minimum !== null && (!Number.isFinite(minimum) || minimum < 0 || minimum > levels.length - 1)) {
        throw new Error(name + "の最小スコアをレベルの範囲内で入力してください。");
      }
    }
    return { id: feature.id, name, kind: feature.kind, instruction, choices, levels, filter: feature.kind === "boolean" && feature.filter, minimum };
  });
  const quality = Object.fromEntries(Object.entries(qualityInputs).map(([key, id]) => [
    key, numeric($(id), $(id).parentElement.firstChild.textContent, {
      openMax: key === "boolean_low" ? 0.5 : undefined,
      openMin: key === "boolean_high" ? 0.5 : undefined,
    }),
  ]));
  const contextMode = $("context-mode").value;
  const secondaryMode = $("secondary-mode").value;
  quality.secondary_context = $("secondary-enabled").checked ? {
    mode: secondaryMode,
    window_chars: secondaryMode === "window"
      ? numeric($("secondary-window"), "二次判定の前後の文字数") : defaultWindowChars,
    max_chars: numeric($("secondary-max"), "二次判定の最大文字数"),
  } : null;
  quality.verifier = verifier;
  return {
    version: 1, name: meaningful($("recipe-name").value.trim(), "レシピ名", 200),
    provider, model: meaningful(provider === "demo" ? liveModel : $("model").value, "モデル名", 256),
    source: { text_field: textField || "text", id_field: idField, group_field: groupField },
    target: {
      mode, terms, pattern, ignore_case: ["literal", "regex"].includes(mode) && $("ignore-case").checked,
      max_candidates: numeric($("max-candidates"), "1 行あたりの候補数"),
    },
    context: {
      mode: contextMode,
      window_chars: contextMode === "window" ? numeric($("window-chars"), "前後の文字数") : defaultWindowChars,
      max_chars: numeric($("max-context"), "文脈の最大文字数"),
    },
    features: serializedFeatures, quality,
  };
}

function exportSources(recipe, columns, grain) {
  const entries = [
    ["source_id", "元の ID"], ["text", "元の本文"], ["target.text", "対象のテキスト"],
    ["target.start", "対象の開始位置"], ["target.end", "対象の終了位置"],
    ["status", "注釈の状態"], ["origin", "判定の出所"], ["reviewed", "人が確認済み"], ["split", "分割名"],
  ];
  if (grain === "annotation") entries.push(["feature_id", "特徴 ID（その行）"], ["value", "値（その行）"]);
  for (const feature of recipe.features) {
    if (feature.id && !entries.some(([id]) => id === "features." + feature.id)) {
      entries.push(["features." + feature.id, "特徴 · " + (feature.name || feature.id)]);
    }
  }
  for (const column of columns) entries.push(["source." + column, "元の列 · " + column]);
  return entries;
}

function recipeExportSources(grain) {
  const sourceColumns = dataset?.columns ?? ["text", "id", "equipment"];
  return exportSources({ features }, sourceColumns, grain);
}

function outputEditor(host, spec, getSources, onChange = () => {}) {
  const initial = copy(spec);
  const output = {
    format: selectField("ファイル形式", [["jsonl", "JSONL"], ["csv", "CSV"], ["parquet", "Parquet"]], initial.format),
    grain: selectField("1 行の単位", [
      ["target", "対象箇所"], ["document", "元の本文"], ["annotation", "対象箇所 × 特徴"],
    ], initial.grain),
    selection: selectField("出力する注釈", [
      ["reviewed", "人が承認した注釈"], ["accepted", "採用済み（自動採用を含む）"], ["all", "すべての注釈"],
    ], initial.selection),
    empty: check("対象がない本文も含める", initial.include_empty),
    csv: check("CSV の数式として解釈されるセルをエスケープ", initial.csv_escape_formulas ?? true),
    split: check("学習・検証・テストに分割する", initial.split.enabled),
    train: inputField("学習の比率", initial.split.train, { type: "number", min: 0, max: 1, step: "any" }),
    validation: inputField("検証の比率", initial.split.validation, { type: "number", min: 0, max: 1, step: "any" }),
    test: inputField("テストの比率", initial.split.test, { type: "number", min: 0, max: 1, step: "any" }),
    seed: inputField("分割のシード", initial.split.seed, { maxLength: 100 }),
  };
  let columns = initial.columns;
  let sourceControls = [];
  const formatRow = node("div", "field-grid three");
  formatRow.append(output.grain.wrapper, output.format.wrapper, output.selection.wrapper);
  const grainNote = node("p", "hint");
  const list = node("div", "output-list");
  const details = node("details", "subdetails");
  const splitFields = node("div", "split-fields");
  const splitGrid = node("div", "field-grid three");
  splitGrid.append(output.train.wrapper, output.validation.wrapper, output.test.wrapper);
  splitFields.append(splitGrid, output.seed.wrapper, node("p", "hint", "同じグループや同一の本文は同じ分割に配置されます。"));
  details.append(node("summary", "", "学習用データの分割"), output.split.wrapper, splitFields);
  details.open = output.split.input.checked;
  host.replaceChildren(formatRow, grainNote, output.empty.wrapper, output.csv.wrapper, list, details);

  function updateSources() {
    const entries = getSources(output.grain.input.value);
    sourceControls.forEach(({ input, column }) => {
      const previous = column.source;
      setOptions(input, [["", "参照元を選択"], ...entries], previous, "使用できません · " + previous);
    });
    return entries;
  }
  function visibility() {
    const grain = output.grain.input.value;
    output.empty.wrapper.hidden = grain !== "document";
    output.csv.wrapper.hidden = output.format.input.value !== "csv";
    splitFields.hidden = !output.split.input.checked;
    grainNote.textContent = {
      target: "対象箇所ごとに 1 行。開始位置には 0 から数えた文字位置を使います。",
      document: "本文ごとに 1 行。対象箇所や特徴の値は、0 件や 1 件の場合も配列になります。",
      annotation: "対象箇所の特徴ごとに 1 行。「特徴 ID（その行）」と「値（その行）」が使えます。",
    }[grain];
  }
  function renderRows() {
    sourceControls = [];
    list.replaceChildren();
    columns.forEach((column, index) => {
      const row = node("div", "output-row");
      const name = inputField("列名", column.name, { maxLength: 100 });
      name.input.addEventListener("input", () => { column.name = name.input.value; });
      const source = selectField("参照元", [], column.source);
      source.input.addEventListener("change", () => { column.source = source.input.value; });
      sourceControls.push({ input: source.input, column });
      const role = selectField("役割", [
        ["input", "入力"], ["target", "正解・教師信号"], ["metadata", "メタデータ"], ["group", "グループ"],
      ], column.role ?? "metadata");
      role.input.addEventListener("change", () => { column.role = role.input.value; });
      const remove = controlButton("削除", "quiet compact", () => {
        columns.splice(index, 1);
        renderRows();
        list.children[Math.min(index, columns.length - 1)].querySelector("input").focus();
        onChange();
      });
      remove.setAttribute("aria-label", "出力列 " + (index + 1) + " を削除");
      remove.disabled = columns.length <= 1;
      row.append(name.wrapper, source.wrapper, role.wrapper, remove);
      list.append(row);
    });
    const add = controlButton("＋ 出力列", "secondary compact", () => {
      columns.push({ name: "", source: "", role: "metadata" });
      renderRows();
      list.children[columns.length - 1].querySelector("input").focus();
      onChange();
    });
    add.disabled = columns.length >= 64;
    list.append(add);
    updateSources();
  }
  output.grain.input.addEventListener("change", () => { visibility(); updateSources(); });
  output.format.input.addEventListener("change", visibility);
  output.split.input.addEventListener("change", visibility);
  host.oninput = onChange;
  host.onchange = onChange;
  renderRows();
  visibility();

  function read({ allowMissingSources = false } = {}) {
    const grain = output.grain.input.value;
    const available = new Set(getSources(grain).map(([source]) => source));
    const names = new Set();
    const used = { input: new Set(), target: new Set() };
    const cleanColumns = columns.map((column) => {
      const name = meaningful(column.name.trim(), "出力列名", 100);
      if (names.has(name)) throw new Error("出力列名が重複しています　" + name);
      if (!column.source || (!allowMissingSources && !available.has(column.source))) {
        throw new Error("出力列「" + name + "」の参照元を選択してください。");
      }
      names.add(name);
      used[column.role]?.add(column.source);
      return { ...column, name };
    });
    if ([...used.input].some((source) => used.target.has(source))) {
      throw new Error("同じ参照元を入力列と正解列の両方には指定できません。");
    }
    const split = {
      enabled: output.split.input.checked, train: 0.8, validation: 0.1, test: 0.1, seed: output.seed.input.value,
    };
    if (split.enabled) {
      for (const key of ["train", "validation", "test"]) {
        split[key] = numeric(output[key].input, output[key].wrapper.firstChild.textContent);
      }
      if (Math.abs(split.train + split.validation + split.test - 1) > 1e-9) throw new Error("学習・検証・テストの比率の合計を 1 にしてください。");
    } else if (["train", "validation", "test"].every((key) => output[key].input.value !== "" && output[key].input.validity.valid)) {
      const [train, validation, test] = ["train", "validation", "test"].map((key) => output[key].input.valueAsNumber);
      if (Math.abs(train + validation + test - 1) <= 1e-9) Object.assign(split, { train, validation, test });
    }
    return {
      grain, format: output.format.input.value, selection: output.selection.input.value, columns: cleanColumns, split,
      include_empty: grain === "document" && output.empty.input.checked,
      csv_escape_formulas: output.csv.input.checked,
    };
  }
  return { read, updateSources };
}

function displayValue(value, feature) {
  if (value == null) return "未確定";
  if (feature?.kind === "boolean") return value ? "はい" : "いいえ";
  if (feature?.kind === "choice") {
    const description = Object.hasOwn(feature.choices, value) ? feature.choices[value] : "";
    return description ? value + " · " + description : value;
  }
  if (feature?.kind === "multi_label" && Array.isArray(value)) return value.length ? value.join("、") : "該当ラベルなし";
  if (feature?.kind === "score" && Number.isInteger(value) && feature.levels[value]) return value + " · " + feature.levels[value];
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function reasonText(reason) {
  const [key, ...detail] = reason.split(":");
  const label = {
    demo: "デモの合成値", audit: "監査に抽出", uncertain: "判定が未確定",
    low_confidence: "信頼度が低い", below_minimum: "スコアが基準未満",
    disagreement: "判定が不一致", filter_negative: "否定による除外条件に一致",
  }[key];
  return label ? label + (detail.length ? "（" + detail.map((piece) => stageLabels[piece] ?? piece).join(" · ") + "）" : "") : reason;
}

function statusPill(label, tone = "") {
  return node("span", "status-pill" + (tone ? " tone-" + tone : ""), label);
}

function annotationBadges(annotation) {
  const container = node("div", "badges");
  const tone = { accepted: "good", review: "warn", excluded: "neutral" }[annotation.status];
  const status = annotation.status === "accepted" && !annotation.reviewed ? "自動採用" : annotationLabels[annotation.status] ?? annotation.status;
  container.append(statusPill(status, tone), statusPill(originLabels[annotation.origin] ?? annotation.origin, annotation.origin === "demo" ? "demo" : ""));
  if (annotation.reviewed) container.append(statusPill("人が確認済み", "good"));
  return container;
}

function highlight(text, start, end, targetText) {
  const block = node("div", "source-text");
  block.tabIndex = 0;
  block.setAttribute("role", "region");
  block.setAttribute("aria-label", "対象箇所を強調したテキスト");
  const points = Array.from(text);
  if (Number.isInteger(start) && Number.isInteger(end) && 0 <= start && start < end && end <= points.length &&
      points.slice(start, end).join("") === targetText) {
    block.append(document.createTextNode(points.slice(0, start).join("")), node("mark", "", targetText),
      document.createTextNode(points.slice(end).join("")));
  } else {
    block.textContent = text;
  }
  return block;
}

function sourcePanel(annotation) {
  const panel = node("div", "annotation-pane");
  panel.append(node("h3", "", "元のテキスト"));
  const target = annotation.target;
  panel.append(node("p", "hint coordinates", String(annotation.source_id) + " · 位置 " + target.start + "–" + target.end),
    highlight(annotation.text, target.start, target.end, target.text));
  return panel;
}

function annotationReasons(annotation) {
  if (!annotation.reasons?.length) return null;
  const reasons = node("ul", "reasons");
  for (const reason of annotation.reasons) reasons.append(node("li", "", reasonText(reason)));
  return reasons;
}

function predictionDetails(annotation, recipe) {
  if (!annotation.predictions?.length) return null;
  const details = node("details", "subdetails prediction-details");
  details.append(node("summary", "", "判定に使った文脈とモデル出力"));
  for (const prediction of annotation.predictions) {
    const section = node("section", "prediction");
    section.append(node("h4", "", (stageLabels[prediction.stage] ?? prediction.stage) + " · " + prediction.model),
      node("p", "hint", originLabels[prediction.origin] ?? prediction.origin));
    if (prediction.context?.text !== undefined) {
      const { text, target_start: start, target_end: end } = prediction.context;
      section.append(highlight(text, start, end, annotation.target.text));
    }
    const answers = node("dl", "feature-values");
    for (const feature of recipe.features) {
      const answer = Object.hasOwn(prediction.answers, feature.id) ? prediction.answers[feature.id] : null;
      if (!answer) continue;
      const probabilities = Object.entries(answer.probabilities ?? {}).map(([name, value]) =>
        name + " " + (100 * value).toFixed(1) + "%");
      answers.append(node("dt", "", feature.name), node("dd", "", displayValue(answer.value, feature)));
      if (answer.confidence != null) answers.append(node("dd", "hint", "モデル信頼度 " + (100 * answer.confidence).toFixed(1) + "%"));
      if (probabilities.length) answers.append(node("dd", "hint", probabilities.join(" / ")));
    }
    section.append(answers);
    details.append(section);
  }
  return details;
}

function listErrors(errors) {
  const details = node("details", "subdetails result-errors");
  details.open = true;
  details.append(node("summary", "", "エラー " + numberText(errors.length) + " 件"));
  const list = node("ul", "reasons");
  for (const error of errors) {
    list.append(node("li", "", (error.record_id ? error.record_id + " · " : "") + error.message));
  }
  details.append(list);
  return details;
}

function invalidatePreview(reset = false) {
  previewRevision += 1;
  if (reset === true) {
    preview = null;
    renderPreview();
  } else if (preview) {
    preview.stale = true;
    const stale = $("preview-content").querySelector(".stale");
    if (stale) stale.hidden = false;
  }
}

function selectedAnnotationLabel(annotation, index) {
  const points = Array.from(annotation.target.text);
  const sample = points.length > 46 ? points.slice(0, 46).join("") + "…" : annotation.target.text;
  const status = annotationLabels[annotation.status] ?? annotation.status;
  return (index + 1) + ". " + String(annotation.source_id) + " · " + status + " · " + sample;
}

function renderPreview() {
  const host = $("preview-content");
  host.replaceChildren();
  if (!preview) {
    const placeholder = node("div", "preview-placeholder");
    const illustration = node("div", "preview-illustration");
    illustration.setAttribute("aria-hidden", "true");
    for (let index = 0; index < 2; index += 1) {
      const lines = node("div");
      lines.append(node("i"), node("i"), node("i"), node("i"));
      illustration.append(lines);
    }
    placeholder.append(illustration, node("h3", "", "結果はここに表示されます"),
      node("p", "", "データを選び、プレビューを実行してください。"));
    host.append(placeholder);
    return;
  }
  const stale = node("p", "inline-banner stale", "設定が変わりました。現在表示している結果は前回の実行時のものです。");
  stale.hidden = !preview.stale;
  const { result, recipe } = preview;
  const summary = node("div", "result-summary");
  summary.append(node("strong", "", numberText(result.stats.records) + " 行 · " + numberText(result.items.length) + " 件の候補"),
    node("span", "", numberText(result.request_count) + " リクエスト"));
  host.append(stale, summary);
  if (result.errors?.length) host.append(listErrors(result.errors));
  if (result.stats.no_target_records) host.append(node("p", "hint", "候補がなかった行 " + numberText(result.stats.no_target_records) + " 件"));
  if (!result.items.length) {
    host.append(node("p", "empty-compact", "表示できる注釈はありません。"));
    return;
  }
  const chooser = selectField("確認する箇所", result.items.map((item, index) =>
    [String(index), selectedAnnotationLabel(item, index)]), String(preview.index));
  const detail = node("div", "preview-result");
  function renderAnnotation() {
    const annotation = result.items[preview.index];
    const panes = node("div", "annotation-panes");
    const data = node("div", "annotation-pane");
    const values = node("dl", "feature-values");
    for (const feature of recipe.features) {
      const value = Object.hasOwn(annotation.features, feature.id) ? annotation.features[feature.id] : null;
      values.append(node("dt", "", feature.name), node("dd", value == null ? "unknown" : "", displayValue(value, feature)));
    }
    data.append(node("h3", "", annotation.status === "review" ? "確認が必要な提案" : "注釈結果"), values);
    panes.append(sourcePanel(annotation), data);
    detail.replaceChildren(annotationBadges(annotation), panes);
    const reasons = annotationReasons(annotation);
    const predictions = predictionDetails(annotation, recipe);
    if (reasons) detail.append(reasons);
    if (predictions) detail.append(predictions);
  }
  chooser.input.addEventListener("change", () => {
    preview.index = Number(chooser.input.value);
    renderAnnotation();
  });
  host.append(chooser.wrapper, detail);
  renderAnnotation();
}

async function runPreview() {
  const recipe = buildRecipe();
  const revision = previewRevision;
  const run = ++previewRun;
  const datasetId = dataset.id;
  const result = await api("/preview", { method: "POST", body: {
    dataset_id: datasetId, recipe, limit: numeric($("preview-limit"), "プレビューの対象行数"),
    max_requests: numeric($("preview-max"), "プレビューの最大リクエスト数"),
  } });
  if (dataset?.id !== datasetId || run !== previewRun) return;
  preview = { result, recipe, index: 0, stale: revision !== previewRevision };
  renderPreview();
  setNotice("プレビューを更新しました。候補 " + numberText(result.items.length) + " 件。");
  $("preview-content").scrollIntoView({ block: "nearest" });
}

const activeJobStatuses = ["queued", "running"];
const jobViews = new Map();
const jobCards = new WeakMap();

function updateJob(job) {
  const index = jobs.findIndex(({ id }) => id === job.id);
  if (index < 0) jobs.unshift(job);
  else jobs[index] = job;
  if (activeJob?.id === job.id) activeJob = job;
  renderJobs();
  renderReviewJobSelect();
}

function defaultJobView(job) {
  const names = new Set(["source_id", "text", "origin", "reviewed"]);
  const labels = job.recipe.features.map(({ id }) => {
    let name = id;
    while (names.has(name)) name = "feature_" + name;
    names.add(name);
    return { name, source: "features." + id, role: "target" };
  });
  return {
    format: "jsonl", grain: "target", selection: "reviewed", include_empty: false,
    columns: [
      { name: "source_id", source: "source_id", role: "metadata" },
      { name: "text", source: "target.text", role: "input" },
      ...labels,
      { name: "origin", source: "origin", role: "metadata" },
      { name: "reviewed", source: "reviewed", role: "metadata" },
    ],
    split: { enabled: false, train: 0.8, validation: 0.1, test: 0.1, seed: "42" },
  };
}

async function createJob() {
  const recipe = buildRecipe();
  const datasetId = dataset.id;
  const stagedView = recipeOutput.read();
  const job = await api("/jobs", { method: "POST", body: {
    dataset_id: datasetId, recipe, max_requests: numeric($("job-max"), "最大リクエスト数"),
    budget_usd: numeric($("job-budget"), "予算上限", { optional: true, openMin: 0 }),
  } });
  jobsRequest += 1;
  jobViews.set(job.id, stagedView);
  updateJob(job);
  setNotice("ジョブを作成しました。");
  location.hash = "jobs";
}

function datasetName(id) {
  return datasets.find((item) => item.id === id)?.name ?? "データセット " + id;
}

function jobMessage(message) {
  if (!message) return "";
  return message === "Request or estimated-cost limit reached. Increase the limits to resume."
    ? "リクエスト数または推定費用の上限に達しました。上限を引き上げて再開してください。"
    : message;
}

function jobDate(value) {
  const date = new Date(typeof value === "number" ? value * 1000 : value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString("ja-JP") : "";
}

function renderJobs() {
  const host = $("jobs-list");
  if (!jobs.length) {
    const empty = node("section", "surface empty-state");
    const link = node("a", "button primary", "レシピへ");
    link.href = "#recipe";
    empty.append(node("h2", "", "ジョブはまだありません"), node("p", "muted", "レシピ画面から処理を開始できます。"), link);
    host.replaceChildren(empty);
    return;
  }
  const cards = new Map(Array.from(host.children).map((card) => [card.dataset.jobId, card]));
  jobs.forEach((job, index) => {
    const previous = cards.get(job.id);
    const state = JSON.stringify(job);
    let card = previous;
    if (!previous || jobCards.get(previous) !== state) {
      card = renderJob(job);
      card.dataset.jobId = job.id;
      jobCards.set(card, state);
      if (previous) {
        const focused = previous.contains(document.activeElement) ? document.activeElement : null;
        const controls = "[data-job-control]";
        const fields = new Map(Array.from(previous.querySelectorAll(controls)).map((input) =>
          [input.dataset.jobControl, input]));
        previous.replaceWith(card);
        for (const element of card.querySelectorAll(controls)) {
          const old = fields.get(element.dataset.jobControl);
          if (old && element.matches("input")) element.value = old.value;
          if (old && focused === old) element.focus();
        }
        if (focused && !card.contains(document.activeElement)) $("refresh-jobs").focus();
      }
    }
    const position = host.children[index] ?? null;
    if (position !== card) host.insertBefore(card, position);
    cards.delete(job.id);
  });
  for (const card of cards.values()) card.remove();
}

function renderJob(job) {
  const card = node("article", "surface job-card");
  const header = node("div", "job-header");
  const info = node("div");
  info.append(node("h2", "", job.recipe.name),
    node("p", "hint", datasetName(job.dataset_id) + " · " + jobDate(job.created_at)));
  const badges = node("div", "badges");
  const tone = activeJobStatuses.includes(job.status) ? "neutral" : job.status === "completed" ? "good" : "warn";
  badges.append(statusPill(jobLabels[job.status] ?? job.status, tone));
  if (job.recipe.provider === "demo") badges.append(statusPill("デモ", "demo"));
  header.append(info, badges);
  const progress = node("progress", "job-progress");
  progress.max = Math.max(1, job.total);
  progress.value = job.completed + job.failed;
  progress.setAttribute("aria-label", "処理済みの行数");
  const counters = node("div", "job-metrics");
  const requestLimit = job.max_requests == null ? "" : " / " + numberText(job.max_requests);
  const cost = job.estimated_cost_usd == null ? "未計測" : "USD " + Number(job.estimated_cost_usd).toFixed(6);
  counters.append(node("span", "", numberText(job.completed) + " / " + numberText(job.total) + " 行完了"),
    node("span", "", "失敗 " + numberText(job.failed) + " 行"),
    node("span", "", "リクエスト " + numberText(job.request_count) + requestLimit),
    node("span", "", "推定費用 " + cost), node("span", "", "入力トークン " + numberText(job.actual_input_tokens)));
  if (job.no_target_records) counters.append(node("span", "", "候補なし " + numberText(job.no_target_records) + " 行"));
  card.append(header, progress, counters);
  if (job.message) card.append(node("p", "inline-banner", jobMessage(job.message)));
  if (job.errors?.length) card.append(listErrors(job.errors));
  const actions = node("div", "action-row");
  const toReview = node("a", "button primary", "レビュー・出力へ");
  toReview.href = "#review";
  toReview.dataset.jobControl = "review";
  toReview.addEventListener("click", (event) => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    $("review-job").value = job.id;
    if (location.hash === "#review") void act(null, () => openReviewJob(job.id));
    else location.hash = "review";
  });
  actions.append(toReview);
  if (activeJobStatuses.includes(job.status)) {
    actions.append(controlButton("一時停止", "secondary", () => controlJob(job.id, "pause")));
  }
  if (activeJobStatuses.includes(job.status) || job.status === "paused") {
    actions.append(controlButton("中止", "secondary", () => controlJob(job.id, "cancel")));
  }
  if (job.failed > 0 && ["completed_with_errors", "paused", "completed"].includes(job.status)) {
    actions.append(controlButton("失敗した行を再試行", "secondary", () => controlJob(job.id, "retry-failed")));
  }
  for (const button of actions.querySelectorAll("button")) button.dataset.jobControl = button.textContent;
  card.append(actions);
  if (job.status === "paused") {
    const details = node("details", "subdetails job-resume");
    details.open = true;
    const fields = node("div", "field-grid three");
    const requests = inputField("再開後の最大リクエスト数（任意）", "", {
      type: "number", min: 1, max: 1000000, step: 1, placeholder: job.max_requests ? String(job.max_requests) : "",
    });
    const budget = inputField("再開後の予算上限（USD、任意）", "", {
      type: "number", min: 0, step: "any", placeholder: job.budget_usd == null ? "現在の設定を維持" : String(job.budget_usd),
    });
    const resume = controlButton("再開", "primary", () => {
      const maxRequests = numeric(requests.input, "再開後の最大リクエスト数", { optional: true });
      const budgetUsd = numeric(budget.input, "再開後の予算上限", { optional: true, openMin: 0 });
      const changes = {};
      if (maxRequests != null) changes.max_requests = maxRequests;
      if (budgetUsd != null) changes.budget_usd = budgetUsd;
      return controlJob(job.id, "resume", Object.keys(changes).length ? changes : undefined);
    });
    requests.input.dataset.jobControl = "max-requests";
    budget.input.dataset.jobControl = "budget";
    resume.dataset.jobControl = "resume";
    fields.append(requests.wrapper, budget.wrapper, resume);
    details.append(node("summary", "", "上限を調整して再開"), fields);
    card.append(details);
  }
  return card;
}

async function controlJob(id, action, body) {
  const status = await api("/jobs/" + paths(id) + "/" + action, { method: "POST", body });
  jobsRequest += 1;
  updateJob(status);
  setNotice("ジョブの状態を更新しました　" + (jobLabels[status.status] ?? status.status));
  $("refresh-jobs").focus();
}

async function refreshJobs() {
  const request = ++jobsRequest;
  const data = await api("/jobs");
  if (request !== jobsRequest) return;
  const changed = JSON.stringify(data) !== JSON.stringify(jobs);
  jobs = data;
  if (activeJob) activeJob = jobs.find((job) => job.id === activeJob.id) ?? activeJob;
  if (changed) {
    renderJobs();
    renderReviewJobSelect();
  }
}

function renderReviewJobSelect() {
  const select = $("review-job");
  setOptions(select, [
    ["", "ジョブを選択"], ...jobs.map((job) => [
      job.id, job.recipe.name + " · " + datasetName(job.dataset_id) + "（" + (jobLabels[job.status] ?? job.status) + "）",
    ]),
  ], select.value);
  $("refresh-review").disabled = !select.value;
  updateReviewMeta();
}

function updateReviewMeta() {
  $("review-job-meta").hidden = !activeJob;
  if (!activeJob) return;
  const label = jobLabels[activeJob.status] ?? activeJob.status;
  $("review-job-meta").textContent = label + " · " + numberText(activeJob.completed) + " / " + numberText(activeJob.total) +
    " 行完了" + (activeJob.recipe.provider === "demo" ? " · デモ" : "");
}

function invalidateExport() {
  exportGeneration += 1;
  $("export-result").hidden = true;
  $("export-result").replaceChildren();
  return exportGeneration;
}

async function openReviewJob(id) {
  const request = ++reviewRequest;
  reviewOffset = 0;
  reviewStatus = "all";
  pendingReviewQuery = null;
  reviewResult = null;
  currentAnnotation = null;
  activeJob = null;
  $("review-workbench").hidden = true;
  $("refresh-review").disabled = !id;
  $("create-export").disabled = true;
  updateReviewMeta();
  invalidateExport();
  if (!id) return;
  const job = await api("/jobs/" + paths(id));
  if (request !== reviewRequest) return;
  activeJob = job;
  $("annotation-filter").value = "all";
  $("annotation-editor").replaceChildren();
  $("annotation-summary").textContent = "";
  const columns = datasets.find((item) => item.id === job.dataset_id)?.columns ?? [];
  jobOutput = outputEditor($("job-output"), jobViews.get(id) ?? defaultJobView(job),
    (grain) => exportSources(job.recipe, columns, grain), invalidateExport);
  $("export-provider-note").textContent = "デモ由来の注釈には、レビュー後もデモの出所が記録されます。";
  $("export-provider-note").hidden = job.recipe.provider !== "demo";
  updateReviewMeta();
  $("review-workbench").hidden = false;
  await loadAnnotations();
}

async function loadAnnotations({
  offset = pendingReviewQuery?.offset ?? reviewOffset,
  status = pendingReviewQuery?.status ?? reviewStatus,
  preferred = currentAnnotation?.id,
} = {}) {
  if (!activeJob) return false;
  const request = ++reviewRequest;
  const jobId = activeJob.id;
  pendingReviewQuery = { offset, status };
  const current = () => request === reviewRequest && activeJob?.id === jobId;
  let result;
  try {
    while (true) {
      result = await api("/jobs/" + paths(jobId) + "/annotations?status=" + status +
        "&offset=" + offset + "&limit=" + reviewPageSize);
      if (!current()) return false;
      if (!offset || result.items.length) break;
      const lastPage = Math.floor(Math.max(0, result.total - 1) / reviewPageSize) * reviewPageSize;
      offset = Math.min(offset - reviewPageSize, lastPage);
    }
  } catch (error) {
    if (!current()) return false;
    pendingReviewQuery = null;
    $("annotation-filter").value = reviewStatus;
    throw error;
  }
  pendingReviewQuery = null;
  reviewOffset = offset;
  reviewStatus = status;
  $("annotation-filter").value = status;
  reviewResult = result;
  currentAnnotation = result.items.find((item) => item.id === preferred) ?? result.items[0];
  renderAnnotations();
  return true;
}

function renderAnnotations() {
  const { items, total, counts } = reviewResult;
  $("annotation-summary").textContent = "要確認 " + numberText(counts.review) + " · 採用 " +
    numberText(counts.accepted) + " · 除外 " + numberText(counts.excluded);
  const select = $("annotation-select");
  setOptions(select, items.length ? items.map((item, index) =>
    [item.id, selectedAnnotationLabel(item, reviewOffset + index)]) : [["", "確認する箇所がありません"]],
  currentAnnotation?.id);
  select.disabled = !items.length;
  const pagination = $("annotation-pagination");
  const previous = controlButton("前のページ", "secondary compact", async () => {
    if (await loadAnnotations({ offset: Math.max(0, reviewOffset - reviewPageSize) })) select.focus();
  });
  const next = controlButton("次のページ", "secondary compact", async () => {
    if (await loadAnnotations({ offset: reviewOffset + reviewPageSize })) select.focus();
  });
  previous.disabled = reviewOffset === 0;
  next.disabled = reviewOffset + reviewPageSize >= total;
  const range = total ? numberText(reviewOffset + 1) + "–" + numberText(reviewOffset + items.length) : "0";
  pagination.replaceChildren(previous, node("span", "hint", range + " / " + numberText(total) + " 件"), next);
  pagination.hidden = total <= reviewPageSize;
  $("create-export").disabled = false;
  renderAnnotationEditor();
}

function editableFeature(feature, value) {
  if (feature.kind === "choice" || feature.kind === "boolean") {
    const values = feature.kind === "choice" ? Object.entries(feature.choices).map(([key, description]) =>
      [key, key + " · " + description]) : [["true", "はい"], ["false", "いいえ"]];
    const selection = selectField(feature.name, [["", "未確定"], ...values], value == null ? "" : String(value));
    return { wrapper: selection.wrapper, read: () => {
      const selected = selection.input.value;
      if (!selected) return null;
      return feature.kind === "boolean" ? selected === "true" : selected;
    } };
  }

  const group = node("fieldset", "review-feature");
  group.append(node("legend", "", feature.name));
  if (feature.kind === "multi_label") {
    const known = check("ラベルを人が確定する", value !== null);
    const choices = node("div", "review-choices");
    const choicesByLabel = Object.entries(feature.choices).map(([key, description]) => {
      const choice = check(key + " · " + description, value?.includes(key));
      choices.append(choice.wrapper);
      return { key, input: choice.input };
    });
    function toggleChoices() {
      for (const { input } of choicesByLabel) input.disabled = !known.input.checked;
    }
    known.input.addEventListener("change", toggleChoices);
    toggleChoices();
    group.append(known.wrapper, choices, node("p", "hint", "確定後にもラベルを選ばない場合は「該当ラベルなし」になります。"));
    return { wrapper: group, read: () => known.input.checked ?
      choicesByLabel.filter(({ input }) => input.checked).map(({ key }) => key) : null };
  }

  const score = inputField("スコア（未確定なら空欄）", value, {
    type: "number", min: 0, max: feature.levels.length - 1, step: "any",
  });
  const scale = node("ul", "reasons score-scale");
  feature.levels.forEach((level, index) => scale.append(node("li", "", index + " · " + level)));
  group.append(score.wrapper, scale);
  return { wrapper: group, read: () => numeric(score.input, feature.name, { optional: true }) };
}

function renderAnnotationEditor() {
  const host = $("annotation-editor");
  host.replaceChildren();
  if (!currentAnnotation) {
    host.append(node("p", "empty-compact", "この条件の注釈がありません。実行中のジョブは、処理後に再読み込みできます。"));
    return;
  }
  const annotation = currentAnnotation;
  const job = activeJob;
  const panes = node("div", "annotation-panes review-panes");
  const values = node("div", "annotation-pane review-values");
  values.append(node("h3", "", "判定の確認"));
  const fields = job.recipe.features.map((feature) => {
    const value = Object.hasOwn(annotation.features, feature.id) ? annotation.features[feature.id] : null;
    const control = editableFeature(feature, value);
    const wrapper = node("div", "review-field");
    wrapper.append(control.wrapper, node("p", "hint", feature.instruction));
    values.append(wrapper);
    return [feature.id, control];
  });
  panes.append(sourcePanel(annotation), values);
  const note = textAreaField("レビューのメモ（任意）", "", { rows: 3, maxLength: 4000 });
  const actions = node("div", "action-row");
  const approve = controlButton("承認する", "primary", () => save("approved"));
  const reject = controlButton("除外する", "secondary", () => save("rejected"));
  actions.append(approve, reject);
  const reasons = annotationReasons(annotation);
  const predictions = predictionDetails(annotation, job.recipe);
  host.append(annotationBadges(annotation), panes);
  if (reasons) host.append(reasons);
  if (predictions) host.append(predictions);
  host.append(note.wrapper, actions);

  async function save(decision) {
    const edited = Object.fromEntries(fields.map(([id, fieldControl]) => [id, fieldControl.read()]));
    if (decision === "approved" && Object.values(edited).includes(null)) {
      throw new Error("未確定の特徴を入力してから承認してください。未確定のままなら除外を選べます。");
    }
    approve.disabled = true;
    reject.disabled = true;
    try {
      await api("/annotations/" + paths(annotation.id), { method: "PATCH", body: {
        revision: annotation.revision, decision, features: edited, note: note.input.value,
      } });
      if (activeJob?.id !== job.id) return;
      invalidateExport();
      if (currentAnnotation?.id !== annotation.id) return;
      if (!await loadAnnotations({ preferred: annotation.id })) return;
      setNotice(decision === "approved" ? "注釈を承認しました。" : "注釈を除外しました。");
      if (!$("annotation-select").disabled) $("annotation-select").focus();
    } finally {
      approve.disabled = false;
      reject.disabled = false;
    }
  }
}

async function createExport() {
  const jobId = activeJob.id;
  const view = jobOutput.read();
  jobViews.set(jobId, view);
  const generation = invalidateExport();
  const current = () => generation === exportGeneration && activeJob?.id === jobId;
  let result;
  try {
    result = await api("/jobs/" + paths(jobId) + "/exports", { method: "POST", body: view });
  } catch (error) {
    if (!current()) return;
    throw error;
  }
  if (!current()) return;
  const url = new URL(result.download_url, location.origin);
  if (url.origin !== location.origin || !url.pathname.startsWith("/api/exports/")) {
    throw new Error("ダウンロード先を確認できませんでした。");
  }
  const host = $("export-result");
  const download = node("a", "button primary", "ZIP をダウンロード");
  download.href = url.href;
  download.download = "";
  host.replaceChildren(node("p", "", "出力 " + numberText(result.rows) + " 行"), download);
  const rowCounts = result.manifest?.row_counts;
  if (rowCounts && Object.keys(rowCounts).length > 1) {
    const names = { train: "学習", validation: "検証", test: "テスト" };
    host.append(node("p", "hint", Object.entries(rowCounts).map(([split, count]) =>
      (names[split] ?? split) + " " + numberText(count) + " 行").join(" · ")));
  }
  host.hidden = false;
  download.focus();
}

async function navigate(focus = true) {
  if (!session?.authenticated) return;
  const requested = location.hash.slice(1);
  if (requested === "main") return;
  const next = ["recipe", "jobs", "review"].includes(requested) ? requested : "recipe";
  page = next;
  for (const section of ["recipe", "jobs", "review"]) $(section + "-page").hidden = section !== page;
  for (const link of $("navigation").querySelectorAll("[data-page]")) {
    if (link.dataset.page === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (focus) {
    $(page + "-page").querySelector("h1").focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }
  if (next === "jobs" || next === "review") await refreshJobs();
  if (next === page && page === "review") {
    const select = $("review-job");
    const id = select.value || activeJob?.id || jobs[0]?.id;
    if (id && activeJob?.id !== id) {
      select.value = id;
      await openReviewJob(id);
    }
  }
}

function bindButton(id, callback) {
  $(id).addEventListener("click", () => act($(id), callback));
}

function bindChange(id, callback) {
  $(id).addEventListener("change", () => act(null, callback));
}

async function uploadDataset(path, body) {
  const result = await api(path, { method: "POST", body });
  datasetRequest += 1;
  datasets = [result, ...datasets.filter((item) => item.id !== result.id)];
  selectDataset(result);
  setNotice("データセットを読み込みました　" + result.name + " · " + numberText(result.rows) + " 行");
}

function loadExportTemplate() {
  const template = templates.find((item) => item.id === $("export-template-select").value);
  const job = activeJob;
  if (!template || !job) return;
  const columns = datasets.find((item) => item.id === job.dataset_id)?.columns ?? [];
  jobOutput = outputEditor($("job-output"), template.export_spec,
    (grain) => exportSources(job.recipe, columns, grain), invalidateExport);
  jobViews.set(job.id, copy(template.export_spec));
  invalidateExport();
  setNotice("テンプレートの出力ビューを適用しました。");
}

setOptions($("target-mode"), targetModes, "sentence");
setOptions($("context-mode"), contextModes, "window");
setOptions($("secondary-mode"), contextModes, "document");

const login = $("login-form").querySelector("button");
$("login-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void act(login, async () => {
    await api("/login", { method: "POST", body: { token: $("access-token").value } });
    $("access-token").value = "";
    await initialize();
    if (session.authenticated) $(page + "-page").querySelector("h1").focus();
  });
});

bindButton("logout", async () => {
  await api("/logout", { method: "POST" });
  lockWorkspace();
});
bindButton("load-sample", () => uploadDataset("/datasets/sample"));

$("upload-file").addEventListener("change", () => {
  const input = $("upload-file");
  if (!input.files.length) return;
  const data = new FormData();
  data.append("file", input.files[0]);
  input.value = "";
  void act(input, () => uploadDataset("/datasets", data));
});
bindChange("dataset-select", () => chooseDataset($("dataset-select").value));
for (const id of ["target-mode", "context-mode", "secondary-enabled", "secondary-mode"]) $(id).addEventListener("change", syncModes);
$("model").addEventListener("input", () => { liveModel = $("model").value; });
$("provider").addEventListener("change", () => {
  if ($("provider").value === "demo") $("verifier").checked = false;
  updateProviderDetails();
});
$("quality-preset").addEventListener("change", () => {
  const choice = qualityPresets[$("quality-preset").value];
  if (!choice) return;
  for (const [key, id] of Object.entries(qualityInputs)) $(id).value = choice[key];
  if ($("quality-preset").value === "quick") {
    $("secondary-enabled").checked = false;
    $("verifier").checked = false;
    syncModes();
  }
});
for (const id of Object.values(qualityInputs)) $(id).addEventListener("input", syncPreset);
$("recipe-page").addEventListener("input", (event) => {
  if (event.target.closest(".preview-rail, #recipe-output, #template-select, #job-max, #job-budget")) return;
  invalidatePreview();
});
$("add-feature").addEventListener("click", () => {
  if (features.length >= 32) return;
  const used = new Set(features.map((feature) => feature.id));
  let next = 1;
  while (used.has("feature_" + next)) next += 1;
  features.push({ id: "feature_" + next, name: "", kind: "boolean", instruction: "", choices: [], levels: [], filter: false, minimum: null });
  redrawFeatures(features.length - 1);
});

bindButton("load-template", () => {
  const template = templates.find((item) => item.id === $("template-select").value);
  if (!template) return;
  applyTemplate(template);
  setNotice("テンプレートを読み込みました　" + template.name);
});
bindButton("save-template", async () => {
  const recipe = buildRecipe({ requireDataset: false, requireReady: false });
  const exportSpec = recipeOutput.read({ allowMissingSources: !dataset });
  const saved = await api("/templates", { method: "POST", body: { name: recipe.name, recipe, export_spec: exportSpec } });
  templates.push(saved);
  fillTemplates(saved.id);
  setNotice("テンプレートを保存しました　" + saved.name);
});
bindButton("run-preview", runPreview);
bindButton("create-job", createJob);
bindButton("refresh-jobs", async () => {
  await refreshJobs();
  setNotice("ジョブ一覧を更新しました。");
});
bindChange("review-job", () => openReviewJob($("review-job").value));
bindButton("refresh-review", async () => {
  if (!activeJob) await openReviewJob($("review-job").value);
  else await loadAnnotations();
  setNotice("注釈を再読み込みしました。");
});
bindChange("annotation-filter", () => loadAnnotations({ offset: 0, status: $("annotation-filter").value }));
$("annotation-select").addEventListener("change", () => {
  currentAnnotation = reviewResult.items.find((item) => item.id === $("annotation-select").value);
  renderAnnotationEditor();
});
bindButton("load-export-template", loadExportTemplate);
bindButton("create-export", createExport);

window.addEventListener("hashchange", () => act(null, navigate));
for (const [link, target] of [["jump-preview", "preview-heading"], ["jump-job-create", "job-create-heading"]]) {
  $(link).addEventListener("click", (event) => {
    event.preventDefault();
    $(target).focus({ preventScroll: true });
    $(target).scrollIntoView({ block: "center" });
  });
}
document.querySelector(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  $("main").focus();
});
window.setInterval(async () => {
  if (polling || !session?.authenticated || document.hidden || !["jobs", "review"].includes(page) ||
      !jobs.some((job) => activeJobStatuses.includes(job.status))) return;
  polling = true;
  try {
    await refreshJobs();
  } catch (error) {
    if (error.name !== "AbortError") showError(error.message, false);
  } finally {
    polling = false;
  }
}, 3000);

async function start() {
  try {
    await initialize();
  } catch (error) {
    if (error.name === "AbortError") return;
    $("loading").replaceChildren(node("p", "", "ワークスペースを読み込めませんでした。"),
      controlButton("再読み込み", "secondary", start));
    showError(error.message, false);
  }
}
void start();
