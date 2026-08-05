/* Échecs IA — page « Match » : retransmission du match Champion contre le
   Fantôme de Deep Blue (Stockfish UCI_Elo bridé). Lecture seule : polling de
   GET /api/match (models/match_live.json écrit par match.exe à chaque coup)
   toutes les 400 ms. Rendu du plateau repris de live.js (parseFen +
   renderBoard). Convention du contrat (src/bin/match.rs) : v_champion /
   v_fantome sont DÉJÀ du point de vue des blancs — affichage tel quel. */
"use strict";

const $ = (id) => document.getElementById(id);
const FILES = "abcdefgh";
const GLYPHS = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};
const FEN_INITIALE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
/* Cadence de polling du direct (ms). */
const PERIODE_MS = 400;

let enVol = false; // une requête à la fois, même si le réseau traîne

/* ------------------------------------------------------ FEN → pièces */
function parseFen(fen) {
  const map = {};
  const rows = fen.split(" ")[0].split("/");
  for (let r = 0; r < 8; r++) {
    const rank = 8 - r;
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= "1" && ch <= "8") file += +ch;
      else map[FILES[file++] + rank] = ch;
    }
  }
  return map;
}

/* --------------------------------------------------------------- plateau */
function renderBoard(boardMap, lastMove) {
  const lastFrom = lastMove ? lastMove.slice(0, 2) : null;
  const lastTo = lastMove ? lastMove.slice(2, 4) : null;
  const board = $("board");
  board.replaceChildren();

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const rank = 8 - r;
      const sq = FILES[c] + rank;
      const el = document.createElement("div");
      el.className = "sq " + ((c + rank) % 2 === 0 ? "light" : "dark");
      if (sq === lastFrom || sq === lastTo) el.classList.add("last");
      const piece = boardMap[sq];
      if (piece) {
        const span = document.createElement("span");
        span.className = "piece " + (piece === piece.toUpperCase() ? "w" : "b");
        span.textContent = GLYPHS[piece];
        el.append(span);
      }
      if (c === 0) {
        const co = document.createElement("span");
        co.className = "coord rank";
        co.textContent = rank;
        el.append(co);
      }
      if (r === 7) {
        const co = document.createElement("span");
        co.className = "coord file";
        co.textContent = FILES[c];
        el.append(co);
      }
      board.append(el);
    }
  }
}

/* ---------------------------------------------------------------- jauges */
/* Jauge bicolore (repris de live.js) : part blanche = (v + 1) / 2. */
function majJauge(nom, v) {
  const jauge = $("g-" + nom);
  const fill = $("fill-" + nom);
  const val = $("val-" + nom);
  if (v === null || v === undefined || !isFinite(v)) {
    jauge.classList.add("off");
    fill.style.width = "50%";
    val.textContent = "—";
    return;
  }
  const borne = Math.max(-1, Math.min(1, v));
  jauge.classList.remove("off");
  fill.style.width = (((borne + 1) / 2) * 100).toFixed(1) + "%";
  val.textContent = (borne >= 0 ? "+" : "") + borne.toFixed(2);
}

/* ------------------------------------------------------------ historique */
function renderHistory(h) {
  const box = $("moves");
  if (!h || !h.length) {
    box.innerHTML = '<div class="moves-empty">La partie vient de commencer…</div>';
    return;
  }
  let html = "<table>";
  for (let i = 0; i < h.length; i += 2) {
    const wCur = i === h.length - 1 ? " cur" : "";
    const bCur = i + 1 === h.length - 1 ? " cur" : "";
    html += `<tr><td class="num">${i / 2 + 1}.</td>` +
            `<td class="mv${wCur}">${h[i]}</td>` +
            `<td class="mv${bCur}">${h[i + 1] || ""}</td></tr>`;
  }
  box.innerHTML = html + "</table>";
  box.scrollTop = box.scrollHeight;
}

/* -------------------------------------------------------------- panneau */

/* mm:ss d'un cumul de millisecondes. */
function horloge(ms) {
  if (ms === null || ms === undefined || !isFinite(ms)) return "—";
  const s = Math.floor(ms / 1000);
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

/* ½ pour les demi-points (2.5 → « 2½ »). */
function scoreTexte(x) {
  if (x === null || x === undefined) return "—";
  const entier = Math.floor(x);
  return x - entier >= 0.5 ? (entier ? entier + "½" : "½") : String(entier);
}

function texteResultat(r) {
  if (r === "1-0") return "1-0 — victoire des blancs";
  if (r === "0-1") return "0-1 — victoire des noirs";
  if (r === "1/2-1/2") return "½ – ½ — nulle";
  return "—";
}

function majPanneau(st) {
  $("tile-partie").textContent =
    st.partie != null ? `${st.partie} / ${st.games}` : "—";
  $("tile-ply").textContent = st.ply != null ? st.ply : "—";
  $("tile-horloge-champion").textContent = horloge(st.temps_champion_ms);
  $("tile-horloge-fantome").textContent = horloge(st.temps_fantome_ms);
  $("score-champion").textContent = scoreTexte(st.score_champion);
  $("score-fantome").textContent = scoreTexte(st.score_fantome);
  $("score-sub").textContent =
    `Champion · Fantôme (UCI_Elo ${st.elo_fantome ?? "?"})`;
  // Noms des jauges : rappellent qui a quelle couleur dans la partie en cours.
  const cb = st.champion_blanc;
  $("nom-champion").textContent = "Champion — " + (cb ? "blancs" : "noirs");
  $("nom-fantome").textContent = "Fantôme de Deep Blue — " + (cb ? "noirs" : "blancs");
}

/* ------------------------------------------------------------- affichage */

function setStatus(text) { $("status").textContent = text; }

/* Personne au micro (404 ou serveur injoignable). */
function afficherHorsLigne() {
  $("live-dot").classList.add("stale");
  $("banner").hidden = true;
  setStatus("Pas de match en cours — match.exe tourne ?");
}

function afficherEtat(st) {
  $("live-dot").classList.remove("stale");
  renderBoard(parseFen(st.fen), st.last_move);
  majJauge("champion", st.v_champion);
  majJauge("fantome", st.v_fantome);
  renderHistory(st.history_san);
  majPanneau(st);

  const banner = $("banner");
  if (st.termine) {
    $("banner-title").textContent = "Match terminé";
    $("banner-sub").textContent =
      `Champion ${scoreTexte(st.score_champion)} — Fantôme ${scoreTexte(st.score_fantome)}` +
      ` · dernière partie : ${st.result} (${st.result_reason})`;
    banner.hidden = false;
    setStatus(`Match terminé : Champion ${scoreTexte(st.score_champion)} — ` +
      `Fantôme ${scoreTexte(st.score_fantome)}.`);
  } else if (st.result) {
    $("banner-title").textContent = st.result === "1/2-1/2" ? "Nulle" :
      st.result === "1-0" ? "Victoire des blancs" : "Victoire des noirs";
    $("banner-sub").textContent =
      `${st.result} (${st.result_reason}) · ${st.history_san.length} plis — la suite arrive…`;
    banner.hidden = false;
    setStatus(`Partie ${st.partie} terminée : ${texteResultat(st.result)}.`);
  } else {
    banner.hidden = true;
    const trait = st.fen.split(" ")[1] === "b" ? "noirs" : "blancs";
    const auTrait = (trait === "blancs") === st.champion_blanc ? "Champion" : "Fantôme";
    setStatus(`En direct — partie ${st.partie}/${st.games}, pli ${st.ply}, ` +
      `trait aux ${trait} (${auTrait} réfléchit…).`);
  }
}

/* ------------------------------------------------------------- polling */

async function tick() {
  if (enVol) return;
  enVol = true;
  try {
    const res = await fetch("/api/match");
    if (!res.ok) { afficherHorsLigne(); return; }   // 404 : rien de publié
    const st = await res.json();
    if (!st || st.actif !== true) { afficherHorsLigne(); return; }
    afficherEtat(st);
  } catch (_) {
    afficherHorsLigne();                             // serveur injoignable
  } finally {
    enVol = false;
  }
}

/* ------------------------------------------------------------------ init */

(function init() {
  renderBoard(parseFen(FEN_INITIALE), null);
  tick();
  setInterval(tick, PERIODE_MS);
})();
