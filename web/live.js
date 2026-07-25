/* Échecs IA — page « Direct » : retransmission d'une partie de self-play de
   l'entraînement. Lecture seule : polling de GET /api/live (models/live.json
   écrit par l'entraîneur) toutes les 400 ms. Rendu du plateau repris de
   app.js (parseFen + renderBoard), débarrassé de toute interactivité.
   Convention du contrat : v_eleve / v_prof sont DÉJÀ du point de vue des
   blancs dans live.json (la conversion depuis la perspective du trait est
   faite côté entraîneur) — ici on affiche tel quel. */
"use strict";

const $ = (id) => document.getElementById(id);
const FILES = "abcdefgh";
const GLYPHS = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};
const FEN_INITIALE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
/* Écart |v_eleve − v_prof| à partir duquel on signale un fort désaccord. */
const SEUIL_DESACCORD = 0.4;
/* Cadence de polling du direct (ms). */
const PERIODE_MS = 400;

let enVol = false; // une requête à la fois, même si le réseau traîne

/* ------------------------------------------------------ FEN → pièces */
/* Copié de app.js : case ("e4") → lettre FEN de la pièce ("P", "k", ...). */
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
/* Rendu repris de app.js, version spectateur : toujours côté blancs, aucune
   case cliquable — seule reste la surbrillance du dernier coup. */
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
      // a1 sombre : (file + rangée) pair → case claire
      el.className = "sq " + ((c + rank) % 2 === 0 ? "light" : "dark");
      if (sq === lastFrom || sq === lastTo) el.classList.add("last");
      const piece = boardMap[sq];
      if (piece) {
        const span = document.createElement("span");
        span.className = "piece " + (piece === piece.toUpperCase() ? "w" : "b");
        span.textContent = GLYPHS[piece];
        el.append(span);
      }
      // Coordonnées : colonne de gauche (rangées) et rangée du bas (colonnes)
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
/* Jauge bicolore : largeur de la part blanche = (v + 1) / 2, v côté blancs.
   v null (ex. prof hors mode mentor) → jauge éteinte, valeur « — ». */
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

function majJauges(st) {
  majJauge("eleve", st.v_eleve);
  majJauge("prof", st.v_prof);
  const note = $("diverge-note");
  const cadre = $("gauges");
  const lesDeux = st.v_eleve != null && st.v_prof != null;
  const ecart = lesDeux ? Math.abs(st.v_eleve - st.v_prof) : 0;
  if (lesDeux && ecart >= SEUIL_DESACCORD) {
    cadre.classList.add("diverge");
    note.textContent = `Fort désaccord élève / prof : Δ = ${ecart.toFixed(2)}`;
  } else {
    cadre.classList.remove("diverge");
    note.textContent = "";
  }
}

/* ------------------------------------------------------------ historique */
/* Historique SAN à deux colonnes numérotées (repris de app.js), qui suit
   le direct : défilement collé au dernier coup joué. */
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

function texteResultat(r) {
  if (r === "1-0") return "1-0 — victoire des blancs";
  if (r === "0-1") return "0-1 — victoire des noirs";
  if (r === "1/2-1/2") return "½ – ½ — nulle";
  return "—";
}

function majPanneau(st) {
  $("tile-cycle").textContent = st.cycle != null ? st.cycle : "—";
  $("tile-ply").textContent = st.ply != null ? st.ply : "—";
  const badge = $("badge-phase");
  const ouverture = st.phase === "ouverture";
  badge.textContent = ouverture ? "ouverture" : "normale";
  badge.className = "badge-phase " + (ouverture ? "ouverture" : "normale");
  $("tile-prev").textContent = texteResultat(st.resultat_precedent);
}

/* ------------------------------------------------------------- affichage */

function setStatus(text) { $("status").textContent = text; }

/* Personne au micro (actif = false, 404 ou entraîneur éteint). */
function afficherHorsLigne() {
  $("live-dot").classList.add("stale");
  $("banner").hidden = true;
  setStatus("Pas de partie au micro — l'entraîneur tourne ?");
}

function afficherEtat(st) {
  $("live-dot").classList.remove("stale");
  renderBoard(parseFen(st.fen), st.last_move);
  majJauges(st);
  renderHistory(st.history_san);
  majPanneau(st);

  const banner = $("banner");
  if (st.result) {
    // Partie au micro terminée : bannière de résultat ; la partie suivante
    // prendra le relais naturellement au prochain tick.
    $("banner-title").textContent = st.result === "1/2-1/2" ? "Nulle" :
      st.result === "1-0" ? "Victoire des blancs" : "Victoire des noirs";
    $("banner-sub").textContent = `${st.result} · ${st.history_san.length} coups — la partie suivante arrive…`;
    banner.hidden = false;
    setStatus(`Partie terminée : ${texteResultat(st.result)}.`);
  } else {
    banner.hidden = true;
    const trait = st.fen.split(" ")[1] === "b" ? "noirs" : "blancs";
    setStatus(`En direct — cycle ${st.cycle}, pli ${st.ply}, trait aux ${trait}.`);
  }
}

/* ------------------------------------------------------------- polling */

async function tick() {
  if (enVol) return;
  enVol = true;
  try {
    const res = await fetch("/api/live");
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
  // Plateau décoratif (position initiale) en attendant le premier état
  renderBoard(parseFen(FEN_INITIALE), null);
  tick();
  setInterval(tick, PERIODE_MS);
})();
