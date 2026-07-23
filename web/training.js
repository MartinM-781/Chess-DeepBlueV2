/* Page « Entraînement en direct » : courbes canvas maison auto-actualisées.
   Forme reprise de la page équivalente du projet poker (training.js). */
"use strict";

const $ = (id) => document.getElementById(id);
const REFRESH_MS = 5000;
/* Paliers d'adversaires jouables — marqueurs verticaux sur les courbes. */
const MILESTONES_H = [1, 3, 10, 30, 100]; /* doit refléter checkpoints.rs::MILESTONES_H */

let lastOk = 0;
let lastData = null;

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fr = (n, d = 0) => n.toLocaleString("fr-FR", { maximumFractionDigits: d });

/* Apparie deux colonnes de metrics.csv en points [x, y] finis. */
function pairs(xs, ys) {
  const out = [];
  if (!xs || !ys) return out;
  const n = Math.min(xs.length, ys.length);
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) out.push([xs[i], ys[i]]);
  }
  return out;
}

function niceTicks(min, max, count) {
  const span = max - min;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= count + 1) || mag * 10;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) ticks.push(t);
  return ticks;
}

/* Heures sur l'axe X : entier si ≥ 10, sinon une décimale utile. */
const fmtH = (v) => (v >= 10 ? String(Math.round(v)) : String(+v.toFixed(1)));

/* ---------------------------------------------------- graphique générique */
/* series : [{color, pts: [[heures, valeur], ...], dots?}]
   opts   : {yMin?, yMax?, baseline?, yFmt?, zeroFloor?}                    */

function drawChart(canvasId, emptyId, series, opts) {
  const canvas = $(canvasId);
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const has = series.some((s) => s.pts.length);
  $(emptyId).hidden = has;
  if (!has) return;

  const P = { l: 52, r: 14, t: 18, b: 32 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const xMax = Math.max(...series.flatMap((s) => s.pts.map((p) => p[0])), 0.1) * 1.03;

  let yMin, yMax;
  if (opts.yMin !== undefined) {
    yMin = opts.yMin; yMax = opts.yMax;
  } else {
    const ys = series.flatMap((s) => s.pts.map((p) => p[1]));
    yMin = Math.min(...ys); yMax = Math.max(...ys);
    const pad = Math.max((yMax - yMin) * 0.12, (Math.abs(yMax) || 1) * 0.05);
    yMin -= pad; yMax += pad;
    if (opts.zeroFloor && yMin < 0) yMin = 0;   // une loss ne descend pas sous 0
  }

  const X = (v) => P.l + (v / xMax) * iw;
  const Y = (v) => P.t + (1 - (v - yMin) / (yMax - yMin)) * ih;

  // Grille + graduations Y
  ctx.strokeStyle = css("--grid");
  ctx.lineWidth = 1;
  ctx.fillStyle = css("--muted");
  ctx.font = "10.5px system-ui";
  ctx.textAlign = "right";
  for (const t of niceTicks(yMin, yMax, 4)) {
    ctx.beginPath(); ctx.moveTo(P.l, Y(t)); ctx.lineTo(W - P.r, Y(t)); ctx.stroke();
    ctx.fillText(opts.yFmt ? opts.yFmt(t) : String(Math.round(t)), P.l - 7, Y(t) + 3.5);
  }

  // Ligne de référence (50 % = égalité)
  if (opts.baseline !== undefined && opts.baseline > yMin && opts.baseline < yMax) {
    ctx.strokeStyle = css("--baseline");
    ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(P.l, Y(opts.baseline)); ctx.lineTo(W - P.r, Y(opts.baseline)); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Graduations X (heures)
  ctx.textAlign = "center";
  for (const t of niceTicks(0, xMax, 5)) {
    if (t < 0 || t > xMax) continue;
    ctx.fillStyle = css("--muted");
    ctx.fillText(fmtH(t), X(t), H - P.b + 16);
  }
  ctx.fillText("heures d'entraînement →", P.l + iw / 2, H - 5);

  // Marqueurs verticaux des paliers 1 h / 3 h / 10 h / 30 h
  for (const m of MILESTONES_H) {
    if (m > xMax) continue;
    ctx.strokeStyle = css("--series-3");
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(X(m), P.t); ctx.lineTo(X(m), H - P.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = css("--series-3");
    ctx.textAlign = "center";
    ctx.fillText(`${m} h`, X(m), P.t - 6);
  }

  // Séries
  for (const s of series) {
    if (!s.pts.length) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    s.pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.stroke();
    if (s.dots || s.pts.length === 1) {
      ctx.fillStyle = s.color;
      s.pts.forEach(([x, y]) => {
        ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, Math.PI * 2); ctx.fill();
      });
    }
  }
}

/* -------------------------------------------------------------- rendu */

function render(data) {
  const m = data.metrics || {};
  const st = data.state || {};

  // Cartouches chiffres-clés
  $("t-hours").textContent = `${fr((st.trained_secs || 0) / 3600, 1)} h`;
  $("t-games").textContent = fr(st.games || 0);
  $("t-positions").textContent = fr(st.positions || 0);
  $("t-cycles").textContent = fr(st.cycles || 0);

  const ptsRandom = pairs(m.elapsed_hours, m.pct_vs_random);
  const ptsMaterial = pairs(m.elapsed_hours, m.pct_vs_material);
  const ptsLoss = pairs(m.elapsed_hours, m.loss);

  const lastR = ptsRandom.length ? ptsRandom[ptsRandom.length - 1][1] : null;
  const lastM = ptsMaterial.length ? ptsMaterial[ptsMaterial.length - 1][1] : null;
  $("pct-last").textContent = lastR === null && lastM === null ? "" :
    `Dernière mesure : ${lastR === null ? "—" : fr(lastR, 1) + " %"} vs Aléatoire · ` +
    `${lastM === null ? "—" : fr(lastM, 1) + " %"} vs Matériel`;

  drawChart("chart-pct", "empty-pct", [
    { color: css("--series-1"), pts: ptsRandom, dots: true },
    { color: css("--series-2"), pts: ptsMaterial, dots: true },
  ], { yMin: 0, yMax: 100, baseline: 50, yFmt: (v) => `${Math.round(v)} %` });

  // Courbe Elo : échantillonnée moins souvent que les métriques (1er cycle après
  // un lancement puis 1 cycle sur 15) ; sa source est data.elo (models/elo.csv).
  const e = data.elo || {};
  const ptsElo = pairs(e.elapsed_hours, e.elo);
  const lastE = ptsElo.length ? ptsElo[ptsElo.length - 1][1] : null;
  $("elo-last").textContent = lastE === null ? "" :
    `Dernière estimation : ~${fr(Math.round(lastE))} Elo`;
  drawChart("chart-elo", "empty-elo", [
    { color: css("--series-4"), pts: ptsElo, dots: true },
  ], { yFmt: (v) => fr(Math.round(v)) });

  drawChart("chart-loss", "empty-loss", [
    { color: css("--series-3"), pts: ptsLoss },
  ], {
    zeroFloor: true,
    yFmt: (v) => (Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3)),
  });
}

/* ------------------------------------------------------------ refresh */

async function refresh() {
  try {
    const r = await fetch("/api/progress");
    if (!r.ok) throw new Error(r.status);
    lastData = await r.json();
    lastOk = Date.now();
    $("tr-dot").classList.remove("stale");
    $("tr-updated").textContent = `en direct — maj à ${new Date().toLocaleTimeString("fr-FR")}`;
    render(lastData);
  } catch (_) {
    $("tr-dot").classList.add("stale");
    $("tr-updated").textContent = lastOk
      ? `connexion perdue (dernière maj ${new Date(lastOk).toLocaleTimeString("fr-FR")})`
      : "serveur injoignable — lance « cargo run --release --bin serve »";
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
window.addEventListener("resize", () => { if (lastData) render(lastData); });
