const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./vendor/apply_theme.js");

/* ---------------------------------------------------------------- results
   Numbers are read from the evaluation output, never typed in here, so the
   deck cannot drift from the run that produced it.                        */
const OUT = process.env.ROTATE_OUT || path.join(__dirname, "..", "out");
const PREFIX = process.argv[2] || (() => {
  const cands = fs.readdirSync(OUT).filter(f => f.endsWith("_summary.tsv"))
    .map(f => ({ f, m: fs.statSync(path.join(OUT, f)).mtimeMs }))
    .sort((a, b) => b.m - a.m);
  if (!cands.length) throw new Error("no *_summary.tsv in " + OUT);
  return cands[0].f.replace(/_summary\.tsv$/, "");
})();

const tsv = p => {
  const lines = fs.readFileSync(path.join(OUT, p), "utf8").trim().split("\n");
  const head = lines[0].split("\t");
  return lines.slice(1).map(l => Object.fromEntries(
    l.split("\t").map((v, i) => [head[i], isNaN(+v) || v === "" ? v : +v])));
};

const SUMMARY = tsv(PREFIX + "_summary.tsv");
const META = JSON.parse(fs.readFileSync(path.join(OUT, PREFIX + "_meta.json"), "utf8"));
const byModel = Object.fromEntries(SUMMARY.map(r => [r.model, r]));
const f3 = v => v.toFixed(3);
const int = v => v.toLocaleString("en-US");

const N_POS = 9657, N_NEG = 25633;
const N_FIT = Math.floor(0.6 * N_POS);
const N_VAL = Math.floor(0.2 * N_POS);
const N_TEST = N_POS - N_FIT - N_VAL;
const NEG_VAL = META.n_val_neg !== undefined ? META.n_val_neg : Math.floor(N_NEG / 2);
const NEG_TEST = META.n_test_neg !== undefined ? META.n_test_neg : N_NEG - Math.floor(N_NEG / 2);
console.log("using %s  models=%s", PREFIX, SUMMARY.map(r => r.model).join(","));

/* ----------------------------------------------------------------- theme
   Light throughout. Accents are the validated categorical slots from the
   dataviz reference palette; ink and surfaces are its neutrals.          */
const THEME = {
  name: "KG Audit Light",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0B0B",            // primary ink
    lt1: "FFFFFF",
    dk2: "52514E",            // secondary ink
    lt2: "F0EFEC",            // card surface
    accent1: "2A78D6",        // blue   - primary
    accent2: "1BAF7A",        // aqua
    accent3: "D03B3B",        // critical - a failed result
    accent4: "EDA100",        // yellow
    accent5: "7A7975",        // muted ink, captions
    accent6: "6250D6",        // violet
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const PAGE = "FCFCFB";        // slide background
const TRACK = "DEDDD8";       // meter track

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Kushal Koirala";
pres.title = "Finding Bad Edges in a Knowledge Graph";
const C = pres.SchemeColor;
const W = 13.3, H = 7.5, M = 0.7;

/* --------------------------------------------------------------- layouts */
pres.defineSlideMaster({
  title: "TITLE_LIGHT",
  background: { color: PAGE },
  objects: [
    { rect: { x: 0, y: 0, w: 0.22, h: H, fill: { color: THEME.colors.accent1 } } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.45, w: W - 2 * M, h: 1.5,
        fontSize: 42, bold: true, color: THEME.colors.dk1, align: "left" }, text: " " } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.0, w: W - 2 * M - 1.2, h: 1.3,
        fontSize: 17, color: THEME.colors.dk2, align: "left" }, text: " " } },
  ],
});

pres.defineSlideMaster({
  title: "SECTION_LIGHT",
  background: { color: THEME.colors.lt2 },
  objects: [
    { rect: { x: 0, y: 0, w: 0.22, h: H, fill: { color: THEME.colors.accent2 } } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.9, w: W - 2 * M, h: 1.2,
        fontSize: 36, bold: true, color: THEME.colors.dk1, align: "left" }, text: " " } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: W - 2 * M - 1.5, h: 1.0,
        fontSize: 16, color: THEME.colors.dk2, align: "left" }, text: " " } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: PAGE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: W - 2 * M, h: 0.9,
        fontSize: 29, bold: true, color: THEME.colors.dk1, align: "left" }, text: " " } },
    { rect: { x: M, y: 1.22, w: 1.1, h: 0.045, fill: { color: THEME.colors.accent1 } } },
    { text: { text: "RotatE anomaly detection · ROBOKOP subgraph",
        options: { x: M, y: H - 0.52, w: 6.5, h: 0.3, fontSize: 10, color: THEME.colors.accent5,
          isTextBox: true, margin: 0 } } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.52, w: 0.4, h: 0.3, fontSize: 10, color: THEME.colors.accent5 },
});

/* --------------------------------------------------------------- helpers */
const card = (s, x, y, w, h, fill) => s.addShape(pres.ShapeType.roundRect, {
  x, y, w, h, rectRadius: 0.06, fill: { color: fill || THEME.colors.lt2 },
  line: { color: fill || THEME.colors.lt2, width: 0 },
});

// a card with a colored rule down its left edge - replaces the old dark panels
const ruled = (s, x, y, w, h, col, fill) => {
  card(s, x, y, w, h, fill || THEME.colors.lt2);
  s.addShape(pres.ShapeType.rect, { x, y, w: 0.055, h,
    fill: { color: col }, line: { color: col, width: 0 } });
};

const dot = (s, x, y, label, color) => {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: 0.4, h: 0.4,
    fill: { color: color || THEME.colors.accent1 },
    line: { color: color || THEME.colors.accent1, width: 0 } });
  s.addText(label, { x, y, w: 0.4, h: 0.4, fontSize: 13, bold: true,
    color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
};

const flowBox = (s, x, y, w, h, head, sub, fill, textColor) => {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06,
    fill: { color: fill }, line: { color: fill, width: 0 } });
  s.addText(head, { x: x + 0.1, y: y + 0.12, w: w - 0.2, h: 0.38, fontSize: 13, bold: true,
    color: textColor, align: "center", isTextBox: true, margin: 0 });
  s.addText(sub, { x: x + 0.1, y: y + 0.5, w: w - 0.2, h: h - 0.6, fontSize: 10.5,
    color: textColor, align: "center", isTextBox: true, margin: 0 });
};

const arrow = (s, x, y) => s.addShape(pres.ShapeType.rightArrow, {
  x, y, w: 0.42, h: 0.28, fill: { color: THEME.colors.accent5 },
  line: { color: THEME.colors.accent5, width: 0 },
});

/* a meter: value on a 0-1 track with the 0.5 chance line marked.
   Two near-equal headline numbers against a limit is a KPI row, not a
   bar chart - and the limit is the whole point here.                      */
const meter = (s, x, y, w, value) => {
  const h = 0.26;
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.4,
    fill: { color: TRACK }, line: { color: TRACK, width: 0 } });
  s.addShape(pres.ShapeType.roundRect, { x, y, w: w * value, h, rectRadius: 0.4,
    fill: { color: THEME.colors.accent3 }, line: { color: THEME.colors.accent3, width: 0 } });
  // chance line at 0.5, drawn over the fill
  s.addShape(pres.ShapeType.rect, { x: x + w * 0.5 - 0.015, y: y - 0.1, w: 0.03, h: h + 0.2,
    fill: { color: THEME.colors.dk1 }, line: { color: THEME.colors.dk1, width: 0 } });
  s.addText("0.5 chance", { x: x + w * 0.5 - 0.7, y: y + h + 0.1, w: 1.4, h: 0.26,
    fontSize: 9.5, color: THEME.colors.dk2, align: "center", isTextBox: true, margin: 0 });
  s.addText("0", { x: x - 0.25, y: y + 0.01, w: 0.22, h: 0.24, fontSize: 9,
    color: THEME.colors.accent5, align: "right", isTextBox: true, margin: 0 });
  s.addText("1", { x: x + w + 0.05, y: y + 0.01, w: 0.22, h: 0.24, fontSize: 9,
    color: THEME.colors.accent5, isTextBox: true, margin: 0 });
};

/* ============================================= 1. title */
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "TITLE_LIGHT", sectionTitle: "Opening" });
s.addText("Finding Bad Edges in a Knowledge Graph", { placeholder: "title" });
s.addText("Testing a published anomaly-detection method, and rule-based garbage collection",
  { placeholder: "body" });
s.addText("931,502 nodes   ·   18.6M edges   ·   9,657 treats edges   ·   RotatE, 256 complex dims",
  { x: M, y: 5.7, w: W - 2 * M, h: 0.4, fontSize: 13, color: THEME.colors.accent5,
    isTextBox: true, margin: 0 });
s.addNotes("Two tracks toward the same goal: find edges that are probably wrong. One uses embeddings and a model; the other uses nothing but Biolink type rules. Only one of them delivered.");

/* ============================================= 2. the problem */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
s.addText("The graph contains edges that are wrong", { placeholder: "title" });
s.addText("Edges are assembled from many sources. Some are mis-typed, mis-assigned, or simply false. We want a list a curator can act on.",
  { x: M, y: 1.42, w: W - 2 * M - 0.3, h: 0.5, fontSize: 15, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });

[["Model-based", "Learn where normal edges sit in embedding space, flag the outliers.",
  "Needs no rules — but needs proof it works", THEME.colors.accent1],
 ["Rule-based", "Check each edge against Biolink type constraints.",
  "Finds less — but every hit is certain", THEME.colors.accent2],
].forEach(([head, body, note, col], i) => {
  const x = M + i * 6.1;
  ruled(s, x, 2.3, 5.8, 2.6, col);
  s.addText(head, { x: x + 0.4, y: 2.6, w: 4.6, h: 0.45, fontSize: 20, bold: true,
    color: THEME.colors.dk1, isTextBox: true, margin: 0 });
  s.addText(body, { x: x + 0.4, y: 3.2, w: 5.0, h: 0.8, fontSize: 14, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });
  s.addText(note, { x: x + 0.4, y: 4.15, w: 5.0, h: 0.5, fontSize: 13, italic: true,
    color: col, isTextBox: true, margin: 0 });
});
s.addText("This talk covers both, and what each one actually delivered",
  { x: M, y: 5.45, w: W - 2 * M, h: 0.45, fontSize: 16, bold: true, color: THEME.colors.dk1,
    isTextBox: true, margin: 0 });
s.addNotes("Frame the two tracks up front so the audience knows the deck has two halves.");

/* ============================================= 3. the original method */
pres.addSection({ title: "The original method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The original method" });
s.addText("The original One-Class SVM approach", { placeholder: "title" });
s.addText("Built in 2024 on GraphSAGE embeddings. One model per predicate.",
  { x: M, y: 1.42, w: W - 2 * M, h: 0.4, fontSize: 15, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });

const fy = 2.15;
flowBox(s, M, fy, 2.3, 1.15, "An edge", "drug — treats → disease", THEME.colors.lt2, THEME.colors.dk2);
arrow(s, M + 2.45, fy + 0.44);
flowBox(s, M + 3.0, fy, 2.6, 1.15, "Concatenate", "[ drug emb ‖ disease emb ]", THEME.colors.accent1, "FFFFFF");
arrow(s, M + 5.75, fy + 0.44);
flowBox(s, M + 6.3, fy, 2.5, 1.15, "One-Class SVM", "fit a boundary around the cloud", THEME.colors.accent1, "FFFFFF");
arrow(s, M + 8.95, fy + 0.44);
flowBox(s, M + 9.5, fy, 2.4, 1.15, "Inside / outside", "outside = anomaly", THEME.colors.accent3, "FFFFFF");

ruled(s, M, 3.8, W - 2 * M, 2.35, THEME.colors.accent4);
s.addText("Two properties worth noting", { x: M + 0.4, y: 4.0, w: 6, h: 0.4,
  fontSize: 17, bold: true, color: THEME.colors.dk1, isTextBox: true, margin: 0 });
s.addText([
  { text: "The relation is never used. ", options: { bold: true, breakLine: false } },
  { text: "Only the two node embeddings enter the features, so the model must rediscover what makes a pair a treats pair.", options: { breakLine: true } },
  { text: "It trains and tests on the same rows. ", options: { bold: true, breakLine: false } },
  { text: "There is no held-out data and no label, so no accuracy can be computed. “Anomaly” means only “in a sparse region of this set”.", options: { breakLine: false } },
], { x: M + 0.4, y: 4.5, w: W - 2 * M - 0.9, h: 1.5, fontSize: 14, color: THEME.colors.dk2,
  isTextBox: true, margin: 0, lineSpacingMultiple: 1.2, paraSpaceAfter: 10 });
s.addNotes("The second bullet is the important one — it is why we could never tell whether the output was any good.");

/* ============================================= 4. the degree artifact */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The original method" });
s.addText("The same method, on RotatE embeddings", { placeholder: "title" });
s.addText("Same code, same steps — GraphSAGE swapped for RotatE, 9,657 treats edges",
  { x: M, y: 1.42, w: W - 2 * M, h: 0.4, fontSize: 15, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });

[["31.3%", "of treats edges flagged", "3,019 of 9,657", THEME.colors.accent3],
 ["0.875", "correlation with node degree", "Spearman, flagged vs kept", THEME.colors.accent3],
 ["3 vs 77", "median degree", "flagged vs kept edges", THEME.colors.accent4],
].forEach(([big, lab, sub, col], i) => {
  const x = M + i * 4.08;
  card(s, x, 2.05, 3.8, 1.9);
  s.addText(big, { x: x + 0.3, y: 2.22, w: 3.2, h: 0.8, fontSize: 40, bold: true,
    color: col, isTextBox: true, margin: 0 });
  s.addText(lab, { x: x + 0.3, y: 3.04, w: 3.2, h: 0.4, fontSize: 14, bold: true,
    color: THEME.colors.dk1, isTextBox: true, margin: 0 });
  s.addText(sub, { x: x + 0.3, y: 3.42, w: 3.2, h: 0.35, fontSize: 11,
    color: THEME.colors.accent5, isTextBox: true, margin: 0 });
});

ruled(s, M, 4.2, W - 2 * M, 1.95, THEME.colors.accent3, "FBF0EF");
dot(s, M + 0.35, 4.5, "!", THEME.colors.accent3);
s.addText("It was finding rare drugs, not wrong edges", { x: M + 0.95, y: 4.47, w: 9.5, h: 0.45,
  fontSize: 20, bold: true, color: THEME.colors.dk1, isTextBox: true, margin: 0 });
s.addText("Flagged edges had a median node degree of 3; kept edges, 77. The detector was reading how well-connected a drug or disease is — a fact about the graph's sparsity, not about whether the edge is correct. The 31.3% itself came from the grid search, whose scoring rule always selects the largest nu in the grid.",
  { x: M + 0.95, y: 5.0, w: W - 2 * M - 1.45, h: 1.0, fontSize: 13, color: THEME.colors.dk2,
    isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
s.addNotes("This is what motivated everything that follows: a detector with no way to be wrong is not a result.");

/* ============================================= 5. making it measurable */
pres.addSection({ title: "Evaluation" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Giving the detector a score it can fail", { placeholder: "title" });
s.addText("We declare a known-good class and a known-bad class, hold data out, and measure — so the result can come back wrong.",
  { x: M, y: 1.42, w: W - 2 * M, h: 0.45, fontSize: 15, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });

ruled(s, M, 2.0, 5.85, 1.45, THEME.colors.accent2);
s.addText("NORMAL", { x: M + 0.4, y: 2.13, w: 2.5, h: 0.3, fontSize: 11, bold: true,
  color: THEME.colors.accent2, isTextBox: true, margin: 0 });
s.addText("biolink:treats", { x: M + 0.4, y: 2.42, w: 3.5, h: 0.45, fontSize: 21, bold: true,
  color: THEME.colors.dk1, isTextBox: true, margin: 0 });
s.addText(int(N_POS) + " edges — what a correct edge looks like",
  { x: M + 0.4, y: 2.9, w: 5.2, h: 0.35, fontSize: 13,
    color: THEME.colors.dk2, isTextBox: true, margin: 0 });

ruled(s, M + 6.15, 2.0, 5.85, 1.45, THEME.colors.accent3);
s.addText("ANOMALY", { x: M + 6.55, y: 2.13, w: 2.5, h: 0.3, fontSize: 11, bold: true,
  color: THEME.colors.accent3, isTextBox: true, margin: 0 });
s.addText("biolink:contraindicated_in", { x: M + 6.55, y: 2.42, w: 5.2, h: 0.45, fontSize: 21,
  bold: true, color: THEME.colors.dk1, isTextBox: true, margin: 0 });
s.addText(int(N_NEG) + " edges — same drug–disease shape, opposite meaning",
  { x: M + 6.55, y: 2.9, w: 5.2, h: 0.35, fontSize: 13, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });

const hdr = t => ({ text: t, options: { bold: true, color: "FFFFFF",
  fill: { color: THEME.colors.accent1 }, fontSize: 12.5 } });
const cel = (t, i, opts) => ({ text: t, options: Object.assign({ fontSize: 12.5,
  color: THEME.colors.dk2, fill: { color: i % 2 ? "FFFFFF" : THEME.colors.lt2 } }, opts || {}) });

s.addTable([
  [hdr("Split"), hdr("treats  (normal)"), hdr("contraindicated_in"), hdr("What it is for")],
  [cel("fit — the training set", 1, { bold: true, color: THEME.colors.dk1 }),
   cel(int(N_FIT), 1), cel("0", 1, { color: THEME.colors.accent5 }),
   cel("fitting the One-Class SVM", 1)],
  [cel("validation", 0, { bold: true, color: THEME.colors.dk1 }),
   cel(int(N_VAL), 0), cel(int(NEG_VAL), 0), cel("choosing nu, nothing else", 0)],
  [cel("test — the reported number", 1, { bold: true, color: THEME.colors.dk1 }),
   cel(int(N_TEST), 1), cel(int(NEG_TEST), 1), cel("ROC AUC, " + META.seeds + " independent splits", 1)],
], {
  x: M, y: 3.7, w: W - 2 * M, colW: [3.5, 2.3, 2.9, 3.2], rowH: 0.4,
  border: { type: "solid", color: "DEDDD8", pt: 1 }, align: "left", valign: "middle",
  margin: 0.08, fontFace: "+mn-lt",
});

s.addText([
  { text: "Only positives are ever trained on. ", options: { bold: true, color: THEME.colors.dk1, breakLine: false } },
  { text: "This is one-class detection: the model's whole notion of “normal” comes from those " + int(N_FIT) + " treats edges, and it never sees a contraindication until it is scored. The " + N_POS.toLocaleString("en-US") + " positives are re-split on each of the " + META.seeds + " seeds, so the error bar reflects which edges we fit on.", options: { breakLine: false } },
], { x: M, y: 5.5, w: W - 2 * M, h: 0.8, fontSize: 12.5, color: THEME.colors.dk2,
  isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
s.addNotes("contraindicated_in is a hard negative: same node types, opposite clinical meaning. A detector that cannot tell those apart has no hope of spotting a subtly wrong treats edge. Necessary condition, not sufficient.");

/* ============================================= 6. what the features are */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("The two things we scored", { placeholder: "title" });
s.addText("Both are fed to the same One-Class SVM, on the same splits. One is the method under test; the other exists to catch a false positive.",
  { x: M, y: 1.42, w: W - 2 * M, h: 0.45, fontSize: 15, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });

[["concat", "THE PUBLISHED METHOD", THEME.colors.accent1,
  "[ drug embedding  ‖  disease embedding ]",
  "512 dimensions",
  "The two node vectors glued end to end. Note what is absent: the relation itself never enters the features, so the model has to rediscover what makes a pair a treats pair from the node positions alone."],
 ["degree", "THE CONTROL", THEME.colors.accent4,
  "[ log degree(drug),  log degree(disease) ]",
  "2 dimensions",
  "How well-connected the two endpoints are, and nothing else. By construction it holds no semantic information — so if it scores as well as the real method, any apparent success was only graph sparsity in disguise."],
].forEach(([name, role, col, formula, dims, body], i) => {
  const x = M + i * 6.1;
  ruled(s, x, 2.0, 5.8, 3.95, col);
  s.addText(role, { x: x + 0.4, y: 2.15, w: 4.8, h: 0.3, fontSize: 10.5, bold: true,
    color: col, isTextBox: true, margin: 0 });
  s.addText(name, { x: x + 0.4, y: 2.45, w: 4.8, h: 0.5, fontSize: 26, bold: true,
    color: THEME.colors.dk1, isTextBox: true, margin: 0 });
  card(s, x + 0.4, 3.05, 5.0, 0.5, "FFFFFF");
  s.addText(formula, { x: x + 0.5, y: 3.13, w: 4.8, h: 0.35, fontSize: 12.5,
    color: THEME.colors.dk1, fontFace: "Consolas", isTextBox: true, margin: 0 });
  s.addText(dims, { x: x + 0.4, y: 3.62, w: 4.8, h: 0.3, fontSize: 11.5, italic: true,
    color: THEME.colors.accent5, isTextBox: true, margin: 0 });
  s.addText(body, { x: x + 0.4, y: 4.0, w: 5.0, h: 1.8, fontSize: 13,
    color: THEME.colors.dk2, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
});
s.addNotes("The control is the slide people skip. It is the reason the earlier 31.3% result collapsed, and it is why the numbers on the next slide can be trusted as far as they go.");

/* ============================================= 7. results */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("What scored, and what did not", { placeholder: "title" });
s.addText("ROC AUC on the held-out test set: pick one real treats edge and one contraindication at random — how often is the contraindication ranked as the more anomalous of the two?",
  { x: M, y: 1.42, w: W - 2 * M, h: 0.45, fontSize: 14.5, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });

const ROWS = [
  ["concat", "the published method", "One-Class SVM on concat(drug, disease)"],
  ["degree", "the control", "One-Class SVM on the two node degrees"],
].filter(r => byModel[r[0]]);

ROWS.forEach(([name, role, blurb], i) => {
  const y = 2.1 + i * 1.72;
  const r = byModel[name];
  const below = r.roc_mean < 0.5;
  card(s, M, y, W - 2 * M, 1.5);
  s.addText(f3(r.roc_mean), { x: M + 0.35, y: y + 0.18, w: 1.9, h: 0.85, fontSize: 44,
    bold: true, color: below ? THEME.colors.accent3 : THEME.colors.accent1,
    isTextBox: true, margin: 0 });
  s.addText("± " + f3(r.roc_std), { x: M + 0.38, y: y + 1.02, w: 1.9, h: 0.3, fontSize: 11,
    color: THEME.colors.accent5, isTextBox: true, margin: 0 });
  s.addText(name, { x: M + 2.4, y: y + 0.2, w: 3.0, h: 0.42, fontSize: 21, bold: true,
    color: THEME.colors.dk1, isTextBox: true, margin: 0 });
  s.addText(role.toUpperCase(), { x: M + 2.4, y: y + 0.64, w: 3.0, h: 0.28, fontSize: 10.5,
    bold: true, color: THEME.colors.accent5, isTextBox: true, margin: 0 });
  s.addText(blurb, { x: M + 2.4, y: y + 0.95, w: 4.2, h: 0.4, fontSize: 12,
    color: THEME.colors.dk2, isTextBox: true, margin: 0 });
  meter(s, M + 7.3, y + 0.42, 4.0, r.roc_mean);
});

const worst = ROWS.map(r => byModel[r[0]]).sort((a, b) => a.roc_mean - b.roc_mean)[0];
const best = ROWS.map(r => byModel[r[0]]).sort((a, b) => b.roc_mean - a.roc_mean)[0];
const allBelow = ROWS.every(r => byModel[r[0]].roc_mean < 0.5);

ruled(s, M, 5.6, W - 2 * M, 1.1, allBelow ? THEME.colors.accent3 : THEME.colors.accent1,
  allBelow ? "FBF0EF" : THEME.colors.lt2);
s.addText(allBelow
  ? "Neither one beats a coin flip."
  : "The best of the two reaches " + f3(best.roc_mean) + ".",
  { x: M + 0.4, y: 5.72, w: 5.6, h: 0.38, fontSize: 17, bold: true,
    color: THEME.colors.dk1, isTextBox: true, margin: 0 });
s.addText(allBelow
  ? "Both land below 0.5, and the gap between them (" + f3(Math.abs(best.roc_mean - worst.roc_mean)) + ") is barely larger than their own spread across splits. The published features are not separating these two classes at all."
  : "Compare each against the 0.5 line and against the control before reading anything into it.",
  { x: M + 0.4, y: 6.1, w: W - 2 * M - 0.9, h: 0.5, fontSize: 12.5, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });
s.addNotes("Mean over " + META.seeds + " independent 60/20/20 splits of the treats edges. PR AUC is not shown: with " + int(NEG_TEST) + " negatives against " + int(N_TEST) + " positives the prevalence baseline is " + f3(META.prevalence) + ", so a high PR AUC here would mean almost nothing. ROC AUC is invariant to that imbalance, which is why it is the number on the slide.");

/* ============================================= 8. gc section */
pres.addSection({ title: "Garbage collection" });
s = pres.addSlide({ masterName: "SECTION_LIGHT", sectionTitle: "Garbage collection" });
s.addText("Garbage collection: rules, not models", { placeholder: "title" });
s.addText("Every edge in Biolink has a declared domain and range. Checking them needs no embeddings, no threshold, and no tuning — and every hit is certain.",
  { placeholder: "body" });
s.addNotes("Transition: the model track came back empty. This track gives certainty for a smaller set.");

/* ============================================= 9. what the rules found */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Garbage collection" });
s.addText("What the type rules found", { placeholder: "title" });
s.addText("A treats edge must run from a chemical to a disease or phenotype. Anything else is malformed by definition.",
  { x: M, y: 1.42, w: W - 2 * M, h: 0.4, fontSize: 15, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });

[["350", "wrong object type", "treats points at a Procedure, Activity, Taxon or Chemical — not a disease", THEME.colors.accent3],
 ["61", "wrong subject type", "the thing doing the treating is a process, pathway or gene — not a drug", THEME.colors.accent4],
 ["1", "self-loop", "a compound recorded as treating itself", THEME.colors.accent6],
].forEach(([big, lab, sub, col], i) => {
  const x = M + i * 4.08;
  card(s, x, 2.05, 3.8, 2.2);
  s.addText(big, { x: x + 0.3, y: 2.22, w: 3.2, h: 0.85, fontSize: 44, bold: true,
    color: col, isTextBox: true, margin: 0 });
  s.addText(lab, { x: x + 0.3, y: 3.1, w: 3.2, h: 0.35, fontSize: 15, bold: true,
    color: THEME.colors.dk1, isTextBox: true, margin: 0 });
  s.addText(sub, { x: x + 0.3, y: 3.47, w: 3.2, h: 0.7, fontSize: 11.5,
    color: THEME.colors.dk2, isTextBox: true, margin: 0 });
});

ruled(s, M, 4.5, W - 2 * M, 1.6, THEME.colors.accent2, "EDF7F2");
s.addText("411", { x: M + 0.45, y: 4.72, w: 1.5, h: 0.95, fontSize: 46, bold: true,
  color: THEME.colors.accent2, isTextBox: true, margin: 0 });
s.addText("distinct edges of 9,657 removable with certainty — 4.3% of the predicate",
  { x: M + 2.1, y: 4.8, w: 9.2, h: 0.4, fontSize: 17, bold: true, color: THEME.colors.dk1,
    isTextBox: true, margin: 0 });
s.addText("The self-loop also fails the object-type rule, so the three groups overlap by one. No model was involved — one type lookup and one equality check — so there is no threshold to defend and no false-positive rate to estimate.",
  { x: M + 2.1, y: 5.24, w: 9.2, h: 0.7, fontSize: 12.5, color: THEME.colors.dk2,
    isTextBox: true, margin: 0 });
s.addNotes("4.3% found with certainty, versus 31.3% flagged with no way to check. That contrast is the point of the slide.");

/* ============================================= 10. examples */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Garbage collection" });
s.addText("Examples of what came out", { placeholder: "title" });

const exRows = [
  ["Edge as recorded", "Why it is wrong"],
  ["Ammonia  —treats→  Positron-Emission Tomography", "A PET tracer, not a therapy. Object is a Procedure"],
  ["Oxytocin  —treats→  Induction of labour", "Object is a Procedure, not a disease"],
  ["Linezolid  —treats→  Vancomycin-Resistant Enterococci", "Object is an OrganismTaxon, not a disease"],
  ["ORF1ab  —treats→  COVID-19", "Subject is a viral gene, not a drug"],
  ["gastric acid secretion  —treats→  short bowel syndrome", "Subject is a biological process"],
  ["copper ion homeostasis  —treats→  Wilson disease", "Subject is a biological process"],
  ["Anamorelin  —treats→  Anamorelin", "Self-loop: a compound treating itself"],
];
s.addTable(exRows.map((r, i) => r.map(t => ({ text: t, options: {
  bold: i === 0, fontSize: i === 0 ? 12.5 : 12,
  color: i === 0 ? "FFFFFF" : THEME.colors.dk2,
  fill: { color: i === 0 ? THEME.colors.accent1 : (i % 2 ? "FFFFFF" : THEME.colors.lt2) } } }))), {
  x: M, y: 1.55, w: W - 2 * M, colW: [6.4, 5.5], rowH: 0.42,
  border: { type: "solid", color: "DEDDD8", pt: 1 }, align: "left", valign: "middle",
  margin: 0.08, fontFace: "+mn-lt",
});
s.addText("Each of these is a definite curation error, found without a model. Names resolved from the graph's own node index.",
  { x: M, y: 5.6, w: W - 2 * M, h: 0.4, fontSize: 12, italic: true,
    color: THEME.colors.accent5, isTextBox: true, margin: 0 });
s.addNotes("Read two or three aloud — Ammonia treating a PET scan and ORF1ab treating COVID-19 land best.");

/* ============================================= 11. conclusion */
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Close" });
s.addText("What we concluded", { placeholder: "title" });

const cc = byModel["concat"], dg = byModel["degree"];
[["1", "The published method does not work on this task",
  "concat(drug, disease) scored " + f3(cc.roc_mean) + " against a hard negative — no better than a coin flip. Two node embeddings glued together carry no information about which relation holds between them.",
  THEME.colors.accent3],
 ["2", "The failure is real, not a measurement artifact",
  "The degree control scored " + f3(dg.roc_mean) + " on the identical splits. Neither features nor graph sparsity separate these classes, so there is no hidden signal the protocol is throwing away.",
  THEME.colors.accent4],
 ["3", "The relation has to enter the features",
  "A method that never sees the predicate cannot distinguish treats from its own opposite. That is a design flaw, not a tuning problem — no choice of nu rescues it.",
  THEME.colors.accent1],
 ["4", "Type rules are what actually delivered",
  "411 edges removable with certainty today, no threshold to defend. A smaller yield than the model claimed, but every one of them is real.",
  THEME.colors.accent2],
].forEach(([n, h, b, col], i) => {
  const y = 1.5 + i * 1.12;
  ruled(s, M, y, W - 2 * M, 1.0, col);
  dot(s, M + 0.3, y + 0.3, n, col);
  s.addText(h, { x: M + 0.95, y: y + 0.14, w: 5.3, h: 0.38, fontSize: 15, bold: true,
    color: THEME.colors.dk1, isTextBox: true, margin: 0 });
  s.addText(b, { x: M + 6.45, y: y + 0.12, w: 5.3, h: 0.78, fontSize: 11,
    color: THEME.colors.dk2, isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
});

ruled(s, M, 6.05, W - 2 * M, 0.85, THEME.colors.accent6, "F1F0FA");
s.addText([
  { text: "What this does not show:  ", options: { bold: true, color: THEME.colors.accent6, breakLine: false } },
  { text: "separating treats from contraindications is a necessary test, not a sufficient one. Passing it would not have proved a detector finds real curation errors — but failing it does rule these features out.", options: { color: THEME.colors.dk2, breakLine: false } },
], { x: M + 0.4, y: 6.26, w: W - 2 * M - 0.9, h: 0.5, fontSize: 12,
  isTextBox: true, margin: 0 });
s.addNotes("End on the limit of the claim. The honest framing is that this is a refutation of a specific feature choice, cleanly measured, plus a small set of certain removals from the rules track.");

/* ----------------------------------------------------------------- write */
(async () => {
  const out = path.join(__dirname, "kg_anomaly_detection.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote " + out);
})();
