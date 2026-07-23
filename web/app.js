/* Échecs IA — plateau jouable. Parle au serveur Rust (serve.rs) via /api/*.
   Forme et logique reprises du dashboard poker (C:\dev\poker\web\app.js). */
"use strict";

const $ = (id) => document.getElementById(id);
const FILES = "abcdefgh";
const GLYPHS = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};
const FEN_INITIALE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

let currentState = null;   // dernier état renvoyé par l'API
let boardMap = {};         // case ("e4") → lettre FEN de la pièce ("P", "k", ...)
let selected = null;       // case sélectionnée ("e2") ou null
let targets = new Map();   // case cible → uci complet à envoyer (promotion incluse)
let opponents = [];        // liste renvoyée par /api/checkpoints
let pollTimer = null;      // polling 1 s, actif SEULEMENT quand l'IA doit jouer
let pollInFlight = false;
let busy = false;          // vrai pendant un aller-retour réseau qui modifie la partie

/* ------------------------------------------------------------------ API */

async function api(path, body) {
  const opts = body !== undefined
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined;
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = JSON.parse(text); } catch (_) { /* erreur renvoyée en texte brut */ }
  if (!res.ok) throw new Error((data && data.error) || text || `HTTP ${res.status}`);
  if (data === null) throw new Error("Réponse illisible du serveur");
  return data;
}

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

/* La pièce appartient-elle à l'humain ? (majuscules = blancs) */
function isMine(piece) {
  const st = currentState;
  if (!piece || !st) return false;
  return st.your_color === "white"
    ? piece === piece.toUpperCase()
    : piece === piece.toLowerCase();
}

/* --------------------------------------------------------------- plateau */

function renderBoard() {
  const st = currentState;
  const flipped = st && st.your_color === "black";  // plateau retourné côté noirs
  const lastFrom = st && st.last_move ? st.last_move.slice(0, 2) : null;
  const lastTo = st && st.last_move ? st.last_move.slice(2, 4) : null;
  const humanTurn = st && !st.result && st.turn === st.your_color;
  const board = $("board");
  board.replaceChildren();

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const fileIdx = flipped ? 7 - c : c;
      const rank = flipped ? r + 1 : 8 - r;
      const sq = FILES[fileIdx] + rank;
      const el = document.createElement("div");
      // a1 sombre : (file + rangée) pair → case claire
      el.className = "sq " + ((fileIdx + rank) % 2 === 0 ? "light" : "dark");
      if (sq === lastFrom || sq === lastTo) el.classList.add("last");
      if (sq === selected) el.classList.add("selected");
      if (targets.has(sq)) {
        el.classList.add("target");
        if (boardMap[sq]) el.classList.add("capture");
      }
      const piece = boardMap[sq];
      if (piece) {
        const span = document.createElement("span");
        span.className = "piece " + (piece === piece.toUpperCase() ? "w" : "b");
        span.textContent = GLYPHS[piece];
        el.append(span);
      }
      if (humanTurn && (targets.has(sq) || isMine(piece))) el.classList.add("playable");
      // Coordonnées : colonne affichée de gauche (rangées) et rangée du bas (colonnes)
      if (c === 0) {
        const co = document.createElement("span");
        co.className = "coord rank";
        co.textContent = rank;
        el.append(co);
      }
      if (r === 7) {
        const co = document.createElement("span");
        co.className = "coord file";
        co.textContent = FILES[fileIdx];
        el.append(co);
      }
      el.addEventListener("click", () => onSquareClick(sq));
      board.append(el);
    }
  }
}

/* ------------------------------------------------------- sélection / coup */

function onSquareClick(sq) {
  const st = currentState;
  if (!st || st.result || busy) return;
  if (st.turn !== st.your_color) return;       // pas ton tour : rien à cliquer
  if (selected && targets.has(sq)) { playMove(targets.get(sq)); return; }
  const piece = boardMap[sq];
  if (isMine(piece) && sq !== selected) {
    selected = sq;
    targets = computeTargets(sq, piece, st.legal || []);
  } else {
    selected = null;
    targets = new Map();
  }
  renderBoard();
}

/* Cases cibles depuis la liste des coups légaux (préfixe uci = case de départ).
   Promotion : auto-dame — on ajoute "q" si un pion part de sa 7e rangée
   (relative) vers la dernière ; le serveur validera. */
function computeTargets(from, piece, legal) {
  const map = new Map();
  for (const u of legal) {
    if (u.slice(0, 2) !== from) continue;
    const to = u.slice(2, 4);
    if (map.has(to)) continue;                  // e7e8q/e7e8r/... : une seule cible
    const promo = (piece === "P" && from[1] === "7" && to[1] === "8")
               || (piece === "p" && from[1] === "2" && to[1] === "1");
    map.set(to, promo ? from + to + "q" : from + to);
  }
  return map;
}

async function playMove(uci) {
  busy = true;
  selected = null;
  targets = new Map();
  // Affichage optimiste du coup humain (le serveur renverra l'état exact,
  // roque et prise en passant compris)
  const from = uci.slice(0, 2), to = uci.slice(2, 4);
  let piece = boardMap[from];
  if (uci.length === 5) piece = currentState.your_color === "white" ? "Q" : "q";
  delete boardMap[from];
  boardMap[to] = piece;
  renderBoard();
  setStatus("L'IA réfléchit…");
  try {
    processState(await api("/api/move", { uci }));
  } catch (e) {
    setStatus(`Coup refusé : ${e.message}`);
    if (currentState) boardMap = parseFen(currentState.fen);
    renderBoard();
  } finally {
    busy = false;
  }
}

/* ------------------------------------------------------------- affichage */

function processState(state) {
  currentState = state;
  selected = null;
  targets = new Map();
  boardMap = parseFen(state.fen);
  renderBoard();
  renderHistory(state);
  renderTiles(state);
  renderStatus(state);
  schedulePolling(state);
}

function setStatus(text) { $("status").textContent = text; }

const REASONS = {
  "mat": "échec et mat",
  "pat": "pat",
  "50 coups": "règle des 50 coups",
  "3 répétitions": "3 répétitions",
  "matériel insuffisant": "matériel insuffisant",
  "abandon": "abandon",
};
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

/* Résultat en clair : « Échec et mat — victoire de l'IA », « Nulle — 3 répétitions »… */
function resultText(st) {
  const reason = REASONS[st.result_reason] || st.result_reason || "fin de partie";
  if (st.result === "1/2-1/2") {
    return { title: "Nulle", sub: `${cap(reason)} · ½ – ½`, plain: `Nulle — ${reason}.` };
  }
  const winner = st.result === "1-0" ? "white" : "black";
  const youWin = winner === st.your_color;
  return {
    title: youWin ? "Tu gagnes ! 🎉" : "Victoire de l'IA",
    sub: `${cap(reason)} · ${st.result}`,
    plain: `${cap(reason)} — ${youWin ? "tu gagnes !" : "victoire de l'IA."}`,
  };
}

function renderStatus(st) {
  const banner = $("banner");
  if (st.result) {
    const r = resultText(st);
    $("banner-title").textContent = r.title;
    $("banner-sub").textContent = r.sub;
    banner.hidden = false;
    setStatus(r.plain);
    return;
  }
  banner.hidden = true;
  if (st.turn === st.your_color) {
    const lastSan = st.history_san && st.history_san.length
      ? st.history_san[st.history_san.length - 1] : "";
    const check = lastSan.endsWith("+") ? "Échec ! " : "";
    setStatus(check + `À toi de jouer — tu as les ${st.your_color === "white" ? "blancs" : "noirs"}.`);
  } else {
    setStatus("L'IA réfléchit…");
  }
}

/* Historique SAN à deux colonnes numérotées */
function renderHistory(st) {
  const box = $("moves");
  const h = st.history_san || [];
  if (!h.length) {
    box.innerHTML = '<div class="moves-empty">La partie n\'a pas encore commencé.</div>';
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

function opponentLabel(id) {
  const o = opponents.find((o) => o.id === id);
  return o ? o.label : id;
}

function renderTiles(st) {
  $("tile-opponent").textContent = st.opponent ? opponentLabel(st.opponent) : "—";
  const ms = st.thinking_ms;
  $("tile-think").textContent = ms == null ? "—"
    : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`;
}

/* ------------------------------------------------------------- polling */
/* Polling léger de /api/state (1 s) SEULEMENT quand c'est à l'IA de jouer. */

function schedulePolling(st) {
  const aiToMove = !st.result && st.turn !== st.your_color;
  if (aiToMove && !pollTimer) pollTimer = setInterval(pollState, 1000);
  if (!aiToMove && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollState() {
  if (busy || pollInFlight) return;
  pollInFlight = true;
  try {
    processState(await api("/api/state"));
  } catch (_) {
    /* silencieux : on retentera au tick suivant */
  } finally {
    pollInFlight = false;
  }
}

/* ------------------------------------------------------- nouvelle partie */

async function newGame() {
  if (busy) return;
  busy = true;  // armé AVANT le fetch : bloque tout double-clic pendant la latence
  const opponent = $("sel-opponent").value;
  const color = document.querySelector('input[name="color"]:checked').value;
  setStatus("Nouvelle partie…");
  try {
    processState(await api("/api/new-game", { opponent, color }));
  } catch (e) {
    setStatus(`Impossible de démarrer : ${e.message}`);
  } finally {
    busy = false;
  }
}

/* Sélecteur d'adversaire depuis /api/checkpoints ; paliers non atteints grisés. */
async function loadCheckpoints() {
  const sel = $("sel-opponent");
  try {
    const data = await api("/api/checkpoints");
    opponents = data.opponents || [];
    sel.replaceChildren(...opponents.map((o) => {
      const opt = document.createElement("option");
      opt.value = o.id;
      opt.textContent = o.label + (o.available ? "" : " (pas encore atteint)");
      opt.disabled = !o.available;
      return opt;
    }));
    // Par défaut : l'adversaire disponible le plus avancé de la liste
    for (let i = opponents.length - 1; i >= 0; i--) {
      if (opponents[i].available) { sel.value = opponents[i].id; break; }
    }
  } catch (_) {
    /* le statut d'init signalera déjà le serveur injoignable */
  }
}

/* ------------------------------------------------------------------ init */

$("btn-new").addEventListener("click", newGame);

document.addEventListener("keydown", (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey) return;
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (e.key.toLowerCase() === "n") newGame();
});

(async function init() {
  // Plateau décoratif (position initiale) en attendant le serveur
  boardMap = parseFen(FEN_INITIALE);
  renderBoard();
  await loadCheckpoints();
  try {
    processState(await api("/api/state"));
  } catch (e) {
    setStatus("Serveur injoignable — lance « cargo run --release --bin serve » puis recharge.");
  }
})();
