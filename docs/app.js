/* Glaucoma screening - in-browser pipeline (ONNX Runtime Web).
   Same steps as the Python tool: disc localisation -> disc/cup segmentation -> tight cup refinement -> CDRs and rim (ISNT)
   -> PPA segmentation -> peripapillary ring CNN -> disc-crop CNN + class-activation map -> calibrated decision -> quality check.
   Everything runs locally in the browser; the photo never leaves the device. */
"use strict";
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const MODELS = [["localiser", 49], ["discup", 25], ["tight", 26], ["ppa", 25], ["ring", 22], ["cnn", 47]];
let S = {}, CFG = null, loading = null;
ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const $ = id => document.getElementById(id);
function setStatus(t) { $("status").textContent = t; }

async function fetchWithProgress(url, onBytes) {
  const r = await fetch(url); if (!r.ok) throw new Error("download failed: " + url);
  const reader = r.body.getReader(); const chunks = []; let n = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); n += value.length; onBytes(value.length); }
  const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out;
}
async function loadModels() {
  if (loading) return loading;
  loading = (async () => {
    CFG = await (await fetch("models/config.json")).json();
    const total = MODELS.reduce((a, m) => a + m[1], 0) * 1e6; let got = 0;
    const bar = $("loadbar"), lab = $("loadlabel"); $("loader").hidden = false;
    for (const [name] of MODELS) {
      const buf = await fetchWithProgress(`models/${name}.onnx`, b => { got += b; bar.style.width = Math.min(100, 100 * got / total) + "%"; lab.textContent = `Downloading models ${Math.round(got / 1e6)} / ${Math.round(total / 1e6)} MB (first visit only)`; });
      S[name] = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    }
    lab.textContent = "Models ready"; setTimeout(() => $("loader").hidden = true, 800);
  })();
  loading.catch(() => { loading = null; });
  return loading;
}

/* ---------- image helpers ---------- */
function canvas(w, h) { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; }
function ctx2d(c) { const x = c.getContext("2d", { willReadFrequently: true }); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = "high"; return x; }
function tensorFrom(c) {                       // canvas (RGB) -> normalised NCHW float32
  const w = c.width, h = c.height, d = ctx2d(c).getImageData(0, 0, w, h).data, n = w * h, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) f[k * n + i] = (d[4 * i + k] / 255 - MEAN[k]) / STD[k];
  return new ort.Tensor("float32", f, [1, 3, h, w]);
}
function squareCrop(src, cx, cy, side, size) {   // square of `side` px centred at (cx, cy) from the original photo, zero padded
  side = Math.floor(side); const x1 = Math.floor(cx) - Math.floor(side / 2), y1 = Math.floor(cy) - Math.floor(side / 2);
  const c = canvas(size, size), g = ctx2d(c); g.fillStyle = "#000"; g.fillRect(0, 0, size, size);
  const k = size / side, sx = Math.max(0, x1), sy = Math.max(0, y1), ex = Math.min(src.width, x1 + side), ey = Math.min(src.height, y1 + side);
  if (ex > sx && ey > sy) g.drawImage(src, sx, sy, ex - sx, ey - sy, (sx - x1) * k, (sy - y1) * k, (ex - sx) * k, (ey - sy) * k);
  return c;
}
async function run(name, t) { const o = await S[name].run({ input: t }); return o; }
const plane = (t, ch) => { const [, , h, w] = t.dims; return t.data.subarray(ch * h * w, (ch + 1) * h * w); };

/* ---------- mask helpers (8-connectivity, like OpenCV) ---------- */
function largestCC(m, w, h) {
  const lab = new Int32Array(w * h), out = new Uint8Array(w * h); let best = 0, bestN = 0, cur = 0; const st = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (!m[i] || lab[i]) continue; cur++; let top = 0, cnt = 0; st[top++] = i; lab[i] = cur;
    while (top) { const p = st[--top]; cnt++; const x = p % w, y = (p / w) | 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue; const q = yy * w + xx; if (m[q] && !lab[q]) { lab[q] = cur; st[top++] = q; } } }
    if (cnt > bestN) { bestN = cnt; best = cur; }
  }
  if (cur <= 1) { for (let i = 0; i < w * h; i++) out[i] = m[i] ? 1 : 0; return { mask: out, n: cur }; }
  for (let i = 0; i < w * h; i++) out[i] = lab[i] === best ? 1 : 0; return { mask: out, n: cur };
}
function stats(m, w, h) {
  let n = 0, sx = 0, sy = 0, x0 = w, x1 = -1, y0 = h, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) { n++; sx += x; sy += y; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  return { n, cx: sx / Math.max(n, 1), cy: sy / Math.max(n, 1), w: x1 - x0 + 1, h: y1 - y0 + 1 };
}
function cdr(disc, cup, w, h) {
  const d = stats(disc, w, h), c = stats(cup, w, h); if (!d.n) return [NaN, NaN, NaN];
  return [c.n ? c.h / d.h : 0, c.n ? c.w / d.w : 0, c.n / d.n];
}
function rimProfile(disc, cup, w, h, temporalRight) {   // rim width along 360 rays, relative to disc diameter
  const d = stats(disc, w, h); if (d.n < 50) return null; const R = Math.max(w, h), rd = new Float64Array(360), rc = new Float64Array(360);
  for (let a = 0; a < 360; a++) { const t = a * Math.PI / 180, s = Math.sin(t), co = Math.cos(t); let ld = 0, lc = 0;
    for (let k = 0; k < 2 * R; k++) { const r = k * 0.5; let py = Math.trunc(d.cy + r * s), px = Math.trunc(d.cx + r * co); py = Math.min(Math.max(py, 0), h - 1); px = Math.min(Math.max(px, 0), w - 1);
      if (disc[py * w + px]) ld = r; if (cup[py * w + px]) lc = r; }
    rd[a] = ld; rc[a] = lc; }
  let mean = 0; for (let a = 0; a < 360; a++) mean += rd[a]; const diam = Math.max(2 * mean / 360, 1);
  const rim = Array.from(rd, (v, a) => Math.max(v - rc[a], 0) / diam);
  const avg = f => { let s = 0, n = 0; for (let a = 0; a < 360; a++) if (f(a)) { s += rim[a]; n++; } return s / n; };
  const I = avg(a => a >= 45 && a < 135), Sx = avg(a => a >= 225 && a < 315), right = avg(a => a < 45 || a >= 315), left = avg(a => a >= 135 && a < 225);
  const T = temporalRight ? right : left, N = temporalRight ? left : right;
  return { rim_I: I, rim_S: Sx, rim_N: N, rim_T: T, rim_min: Math.min(...rim), isnt: I >= Sx && Sx >= N && N >= T };
}
function dilate(m, w, h, r) {                          // square structuring element (2r+1)
  const t = new Uint8Array(w * h), o = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = 0; for (let k = -r; k <= r && !v; k++) { const xx = x + k; if (xx >= 0 && xx < w && m[y * w + xx]) v = 1; } t[y * w + x] = v; }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = 0; for (let k = -r; k <= r && !v; k++) { const yy = y + k; if (yy >= 0 && yy < h && t[yy * w + x]) v = 1; } o[y * w + x] = v; }
  return o;
}
function drawContour(g, m, w, h, color) {
  const id = g.getImageData(0, 0, w, h), d = id.data, [r, gg, b] = color;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = y * w + x; if (!m[i]) continue;
    const edge = x === 0 || y === 0 || x === w - 1 || y === h - 1 || !m[i - 1] || !m[i + 1] || !m[i - w] || !m[i + w];
    if (edge) for (let dy = -1; dy <= 0; dy++) for (let dx = -1; dx <= 0; dx++) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0) continue; const j = 4 * (yy * w + xx); d[j] = r; d[j + 1] = gg; d[j + 2] = b; } }
  g.putImageData(id, 0, 0);
}
function jet(v) { const r = Math.min(Math.max(1.5 - Math.abs(4 * v - 3), 0), 1), g = Math.min(Math.max(1.5 - Math.abs(4 * v - 2), 0), 1), b = Math.min(Math.max(1.5 - Math.abs(4 * v - 1), 0), 1); return [r * 255, g * 255, b * 255]; }

/* ---------- the pipeline ---------- */
async function analyse(img, step) {
  const W = img.width, H = img.height, src = canvas(W, H); ctx2d(src).drawImage(img, 0, 0); const r = { image_size: `${W}x${H}` }; const t0 = performance.now();
  step(1, "Finding the optic disc");
  const s = 512 / Math.max(W, H), rw = Math.round(W * s), rh = Math.round(H * s), x0 = Math.floor((512 - rw) / 2), y0 = Math.floor((512 - rh) / 2);
  const lb = canvas(512, 512), g = ctx2d(lb); g.fillStyle = "#000"; g.fillRect(0, 0, 512, 512); g.drawImage(src, x0, y0, rw, rh);
  const pl = plane((await run("localiser", tensorFrom(lb))).prob, 0); let mx = 0; const raw = new Uint8Array(512 * 512);
  for (let i = 0; i < pl.length; i++) { if (pl[i] > mx) mx = pl[i]; raw[i] = pl[i] > 0.5; }
  r.localiser_confidence = mx; const loc = largestCC(raw, 512, 512).mask, ls = stats(loc, 512, 512);
  if (ls.n < 10) { r.warning = "Optic disc not found - this may not be a fundus photograph, or it is of very poor quality."; return { r }; }
  const cx = (ls.cx - x0) / s, cy = (ls.cy - y0) / s, dd = Math.max(ls.w, ls.h) / s, side = 3 * dd;
  step(2, "Outlining disc and cup");
  const c3 = squareCrop(src, cx, cy, side, 512), p = (await run("discup", tensorFrom(c3))).prob, p0 = plane(p, 0), p1 = plane(p, 1), N = 512 * 512;
  const dRaw = new Uint8Array(N); for (let i = 0; i < N; i++) dRaw[i] = p0[i] > 0.5; const dcc = largestCC(dRaw, 512, 512), disc = dcc.mask;
  let cupRaw = new Uint8Array(N), anyCup = 0; for (let i = 0; i < N; i++) { cupRaw[i] = p1[i] > 0.5 && disc[i]; anyCup |= cupRaw[i]; }
  const cup = anyCup ? largestCC(cupRaw, 512, 512).mask : cupRaw; const [v, hc, a] = cdr(disc, cup, 512, 512);
  const qd = stats(disc, 512, 512), k = side / 512, cx2 = cx - side / 2 + qd.cx * k, cy2 = cy - side / 2 + qd.cy * k, dd2 = Math.max(qd.w, qd.h) * k;
  const t = (await run("tight", tensorFrom(squareCrop(src, cx2, cy2, 1.5 * dd2, 512)))).prob, t0p = plane(t, 0), t1p = plane(t, 1);
  const tdr = new Uint8Array(N); for (let i = 0; i < N; i++) tdr[i] = t0p[i] > 0.5; const td = largestCC(tdr, 512, 512).mask;
  let tcr = new Uint8Array(N), anyT = 0; for (let i = 0; i < N; i++) { tcr[i] = t1p[i] > 0.5 && td[i]; anyT |= tcr[i]; }
  const v2 = cdr(td, anyT ? largestCC(tcr, 512, 512).mask : tcr, 512, 512);
  r.vCDR_base = v; r.vCDR = (v + v2[0]) / 2; r.hCDR = hc; r.area_CDR = (a + v2[2]) / 2;
  let ps = 0, pd = 0; for (let i = 0; i < N; i++) { ps += p1[i] * p0[i]; pd += p0[i]; } r.soft_area_CDR = ps / Math.max(pd, 1e-6);
  step(3, "Measuring rim and atrophy");
  Object.assign(r, rimProfile(disc, cup, 512, 512, cx < W / 2) || {});
  const q = (await run("ppa", tensorFrom(c3))).prob, q0 = plane(q, 0), q1 = plane(q, 1); const ppaRaw = new Uint8Array(N); let nP = 0, nD = 0;
  for (let i = 0; i < N; i++) { ppaRaw[i] = q1[i] > 0.5 && !(q0[i] > 0.5); nP += ppaRaw[i]; nD += q0[i] > 0.5; }
  r.PPA_to_disc_area = nP / Math.max(1, nD); const ppa = nP > 50 ? largestCC(ppaRaw, 512, 512).mask : new Uint8Array(N);
  r.PPA_outline = nP > 50 ? "outlined in red" : "no atrophy region found by the pixel model";
  const i3 = canvas(384, 384); ctx2d(i3).drawImage(c3, 0, 0, 384, 384);
  const ringC = canvas(384, 384), rg = ctx2d(ringC); rg.drawImage(i3, 0, 0); const rd = rg.getImageData(0, 0, 384, 384);
  for (let y = 0; y < 384; y++) for (let x = 0; x < 384; x++) { const rr = Math.hypot(y - 192, x - 192) / 64; if (rr < 0.9 || rr > 2.2) { const j = 4 * (y * 384 + x); rd.data[j] = rd.data[j + 1] = rd.data[j + 2] = 0; } }
  rg.putImageData(rd, 0, 0); r.ring_score = (await run("ring", tensorFrom(ringC))).prob.data[0];
  step(4, "Deep classifier and heat map");
  const out = await run("cnn", tensorFrom(i3)); r.cnn_logit = out.logit.data[0];
  const F = out.features, [, C, fh, fw] = F.dims, cam = new Float32Array(fh * fw), wv = CFG.cnn_fc.w;
  for (let c = 0; c < C; c++) { const wc = wv[c], o = c * fh * fw; for (let i = 0; i < fh * fw; i++) cam[i] += wc * F.data[o + i]; }
  let cm = 0; for (let i = 0; i < cam.length; i++) { cam[i] = Math.max(cam[i], 0); if (cam[i] > cm) cm = cam[i]; }
  step(5, "Decision and quality check");
  const D = CFG.deploy, feats = { cnn_logit: r.cnn_logit, vcdr_e2e: r.vCDR_base, rim_I: r.rim_I, rim_S: r.rim_S, rim_N: r.rim_N, rim_T: r.rim_T, rim_min: r.rim_min };
  let z = D.intercept; D.features.forEach((f, i) => { const x = (feats[f] - D.mean[i]) / D.scale[i]; z += (isFinite(x) ? x : 0) * D.coef[i]; });
  r.probability = 1 / (1 + Math.exp(-z)); r.threshold = D.threshold; r.refer = r.probability >= D.threshold; r.paper_rule = r.area_CDR > 0.25;
  const band = dilate(dRaw, 512, 512, 7), eps = 1e-6; let e0 = 0, e1 = 0, nb = 0, nraw = 0;
  for (let i = 0; i < N; i++) { nraw += dRaw[i]; if (!band[i]) continue; nb++; const a0 = p0[i], a1 = p1[i]; e0 += -(a0 * Math.log(a0 + eps) + (1 - a0) * Math.log(1 - a0 + eps)); e1 += -(a1 * Math.log(a1 + eps) + (1 - a1) * Math.log(1 - a1 + eps)); }
  const Q = CFG.quality, warn = []; const de = nb ? e0 / nb : 1, ce = nb ? e1 / nb : 1, frac = nraw / N;
  if (de > Q.disc_entropy || ce > Q.cup_entropy) warn.push("the model is uncertain about the disc/cup boundary (blur, glare, poor focus or an unusual image)");
  if (frac < Q.disc_area_lo || frac > Q.disc_area_hi || dcc.n !== 1) warn.push("disc size or shape is outside the range seen in training");
  if (r.localiser_confidence < 0.8) warn.push("disc localisation confidence is low");
  r.quality_warning = warn.join("; ");
  // pictures
  const ov = canvas(512, 512), og = ctx2d(ov); og.drawImage(c3, 0, 0); drawContour(og, disc, 512, 512, [0, 255, 255]); drawContour(og, cup, 512, 512, [255, 0, 255]); drawContour(og, ppa, 512, 512, [255, 40, 40]);
  const camC = canvas(fw, fh), cg = ctx2d(camC), cd = cg.createImageData(fw, fh);
  for (let i = 0; i < fh * fw; i++) { const [R, G, B] = jet(cm > 0 ? cam[i] / cm : 0); cd.data[4 * i] = R; cd.data[4 * i + 1] = G; cd.data[4 * i + 2] = B; cd.data[4 * i + 3] = 255; }
  cg.putImageData(cd, 0, 0); const hm = canvas(384, 384), hg = ctx2d(hm); hg.drawImage(i3, 0, 0); hg.globalAlpha = 0.45; hg.drawImage(camC, 0, 0, 384, 384);
  r.seconds = (performance.now() - t0) / 1000;
  return { r, overlay: ov.toDataURL("image/jpeg", 0.92), heat: hm.toDataURL("image/jpeg", 0.92) };
}
window.Glaucoma = { loadModels, analyse };
