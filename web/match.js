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

/* --------------------------------------------- horloges vivantes (état) ---
   Le json publié par match.exe donne les temps CUMULÉS au dernier coup joué ;
   il ne porte pas d'heure d'écriture exploitable côté client. On interpole
   donc LOCALEMENT : à chaque fois que le polling voit un nouveau pli
   (partie/ply/fen changent), on note performance.now() comme origine, et
   entre deux polls l'horloge du camp au trait avance de (now − origine).
   Remise à zéro de l'interpolation à chaque nouveau coup ; horloges figées
   sur le cumul connu quand la partie est finie ou le serveur muet. */
let dernierEtat = null;  // dernier état « actif » reçu du serveur
let horsLigne = false;   // serveur muet / 404 : on fige l'interpolation
let cleCoup = null;      // identité du pli affiché ("partie:ply:fen")
let baseLocale = null;   // performance.now() à l'apparition de ce pli

/* --------------------------------------- navigation historique (état) ---
   history_fen (contrat src/bin/match.rs) : tableau des FEN après chaque pli,
   position initiale incluse — history_fen[i] = position après i plis.
   indexNav = null → la page suit le direct ; sinon index dans history_fen
   (décrochage automatique dès qu'on navigue, retour au direct par le bouton
   « ● direct », la touche Fin, ou en re-avançant jusqu'au dernier pli).
   Un JSON d'ancienne génération (sans history_fen) désactive la navigation
   sans rien casser. */
let indexNav = null;     // null = direct ; sinon index de la FEN affichée
let partieNav = null;    // n° de partie du dernier état : changement → direct

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
/* `cur` : index (0-based) du coup à surligner — dernier coup joué en direct,
   coup atteint par la navigation sinon (-1 : aucun). */
function renderHistory(h, cur) {
  const box = $("moves");
  if (!h || !h.length) {
    box.innerHTML = '<div class="moves-empty">La partie vient de commencer…</div>';
    return;
  }
  if (cur === undefined) cur = h.length - 1;
  let html = "<table>";
  for (let i = 0; i < h.length; i += 2) {
    const wCur = i === cur ? " cur" : "";
    const bCur = i + 1 === cur ? " cur" : "";
    html += `<tr><td class="num">${i / 2 + 1}.</td>` +
            `<td class="mv${wCur}">${h[i]}</td>` +
            `<td class="mv${bCur}">${h[i + 1] || ""}</td></tr>`;
  }
  box.innerHTML = html + "</table>";
  // En direct on suit la fin de la liste ; en navigation on laisse l'œil
  // où il est (le surlignage suffit à se repérer).
  if (cur === h.length - 1) box.scrollTop = box.scrollHeight;
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

/* Nombre compact de nœuds (1234567 → « 1.23 M »). */
function texteNoeuds(n) {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + " M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + " k";
  return String(Math.round(n));
}

/* Panneau « pensée en direct » : profondeur / éval / nœuds / nœuds-par-
   seconde du champion, publiés par le hook d'itération de match.exe
   (objet st.pensee, throttle ~1/s côté moteur). Absent (ancien json, tour
   du Fantôme, partie close) → panneau éteint, tirets. */
function majPensee(st) {
  const box = $("pensee");
  const p = st ? st.pensee : null;
  const eteint = !p || !!st.termine || !!st.result;
  box.classList.toggle("off", eteint);
  $("pensee-prof").textContent = !eteint && p.profondeur != null ? p.profondeur : "—";
  const evalTexte = !eteint && isFinite(p.eval)
    ? (p.eval >= 0 ? "+" : "") + Number(p.eval).toFixed(2) : "—";
  $("pensee-eval").textContent = evalTexte;
  $("pensee-noeuds").textContent = !eteint ? texteNoeuds(p.noeuds) : "—";
  const nps = !eteint && p.ecoule_ms > 0 ? (p.noeuds / (p.ecoule_ms / 1000)) : null;
  $("pensee-nps").textContent = nps ? texteNoeuds(nps) + "/s" : "—";
  // Rangée d'analyse [éval en gras] [ligne SAN] : p.pv (variante principale,
  // marche de TT côté moteur). Absente (ancien json, pv null, panneau
  // éteint) → rangée masquée ; un coup joué remet pensee à null côté
  // match.exe, ce qui vide la rangée au poll suivant.
  const pv = !eteint && typeof p.pv === "string" && p.pv ? p.pv : null;
  const ligne = $("pensee-ligne");
  ligne.hidden = !pv;
  $("pensee-ligne-eval").textContent = pv ? evalTexte : "—";
  $("pensee-ligne-pv").textContent = pv || "";
}

/* --------------------------------------------------- navigation (rendu) */

/* history_fen exploitable, ou null (json d'ancienne génération). */
function fensDisponibles(st) {
  return st && Array.isArray(st.history_fen) && st.history_fen.length
    ? st.history_fen : null;
}

/* Plateau + historique + barre de navigation, selon indexNav (direct ou
   position historique). Appelée à chaque état reçu ET à chaque action de
   navigation (sans attendre le poll suivant). */
function rendrePosition(st) {
  const fens = fensDisponibles(st);
  if (indexNav !== null && fens) {
    // Décroché du direct : FEN historique, pas de surlignage last_move
    // (seul le direct connaît le dernier coup en UCI).
    indexNav = Math.max(0, Math.min(indexNav, fens.length - 1));
    renderBoard(parseFen(fens[indexNav]), null);
    renderHistory(st.history_san, indexNav - 1);
  } else {
    indexNav = null; // pas d'historique de FEN : le direct est le seul mode
    renderBoard(parseFen(st.fen), st.last_move);
    renderHistory(st.history_san);
  }
  majBarreNav(st);
}

function majBarreNav(st) {
  const fens = fensDisponibles(st);
  const direct = indexNav === null;
  $("nav-pos").textContent = direct
    ? "direct" : `coup ${indexNav} / ${fens.length - 1}`;
  $("navbar").classList.toggle("detache", !direct);
  // Bornes : pas de recul avant la position initiale, pas d'avance en direct.
  $("nav-prec").disabled = !fens || indexNav === 0;
  $("nav-suiv").disabled = direct;
  $("nav-live").disabled = direct;
}

/* ------------------------------------------------ navigation (actions) */

function navPrecedent() {
  const fens = fensDisponibles(dernierEtat);
  if (!fens) return;
  // Depuis le direct : premier recul → avant-dernière position.
  indexNav = indexNav === null ? Math.max(0, fens.length - 2)
                               : Math.max(0, indexNav - 1);
  rendrePosition(dernierEtat);
}

function navSuivant() {
  const fens = fensDisponibles(dernierEtat);
  if (!fens || indexNav === null) return;
  indexNav += 1;
  if (indexNav >= fens.length - 1) indexNav = null; // recolle au direct
  rendrePosition(dernierEtat);
}

function navDirect() {
  if (indexNav === null || !dernierEtat) return;
  indexNav = null;
  rendrePosition(dernierEtat);
}

function majPanneau(st) {
  $("tile-partie").textContent =
    st.partie != null ? `${st.partie} / ${st.games}` : "—";
  $("tile-ply").textContent = st.ply != null ? st.ply : "—";
  // Horloges : rendues par majHorloges() (interpolation locale continue).
  $("score-champion").textContent = scoreTexte(st.score_champion);
  $("score-fantome").textContent = scoreTexte(st.score_fantome);
  $("score-sub").textContent =
    `Champion · Fantôme (UCI_Elo ${st.elo_fantome ?? "?"})`;
  // Noms des jauges : rappellent qui a quelle couleur dans la partie en cours.
  const cb = st.champion_blanc;
  $("nom-champion").textContent = "Champion — " + (cb ? "blancs" : "noirs");
  $("nom-fantome").textContent = "Fantôme de Deep Blue — " + (cb ? "noirs" : "blancs");
}

/* ----------------------------------------------------- horloges vivantes */

/* Camp en train de réfléchir, ou null si la partie est close : le trait est
   lu dans la FEN, puis rapporté à champion/fantôme via champion_blanc. */
function penseurActuel(st) {
  if (!st || st.actif !== true || st.termine || st.result) return null;
  const traitBlanc = st.fen.split(" ")[1] !== "b";
  return traitBlanc === st.champion_blanc ? "champion" : "fantome";
}

/* Pose/retire la pastille « en réflexion » (jauge + tuile horloge). */
function marquerPense(penseur) {
  for (const camp of ["champion", "fantome"]) {
    const actif = penseur === camp;
    $("g-" + camp).classList.toggle("pense", actif);
    const tuile = $("tile-horloge-" + camp).closest(".tile");
    if (tuile) tuile.classList.toggle("pense", actif);
  }
}

/* Rendu des deux horloges : cumul publié + interpolation locale pour le camp
   qui pense. Appelée à chaque état reçu ET par un minuteur (~5 Hz) pour que
   le chrono du penseur avance en continu entre deux polls. */
function majHorloges() {
  const st = dernierEtat;
  if (!st) return;
  let tc = st.temps_champion_ms;
  let tf = st.temps_fantome_ms;
  // Serveur muet : aucune animation, on fige sur le dernier cumul connu.
  const penseur = horsLigne ? null : penseurActuel(st);
  if (penseur && baseLocale != null) {
    const extra = Math.max(0, performance.now() - baseLocale);
    if (penseur === "champion") tc += extra;
    else tf += extra;
  }
  $("tile-horloge-champion").textContent = horloge(tc);
  $("tile-horloge-fantome").textContent = horloge(tf);
  marquerPense(penseur);
}

/* ------------------------------------------------------------- affichage */

function setStatus(text) { $("status").textContent = text; }

/* Personne au micro (404 ou serveur injoignable). */
function afficherHorsLigne() {
  $("live-dot").classList.add("stale");
  $("banner").hidden = true;
  setStatus("Pas de match en cours — match.exe tourne ?");
  // Fige les horloges sur le dernier cumul connu et éteint les pastilles.
  horsLigne = true;
  majHorloges();
}

function afficherEtat(st) {
  $("live-dot").classList.remove("stale");
  // Horloges vivantes : nouveau pli (ou nouvelle partie) → nouvelle origine
  // locale d'interpolation ; même pli → l'origine court toujours.
  horsLigne = false;
  const cle = `${st.partie}:${st.ply}:${st.fen}`;
  if (cle !== cleCoup) {
    cleCoup = cle;
    baseLocale = performance.now();
  }
  dernierEtat = st;
  // Nouvelle partie : la navigation de l'ancienne n'a plus de sens, on
  // recolle au direct.
  if (st.partie !== partieNav) {
    partieNav = st.partie;
    indexNav = null;
  }
  majHorloges();
  rendrePosition(st);
  majJauge("champion", st.v_champion);
  majJauge("fantome", st.v_fantome);
  majPensee(st);
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
  // Navigation historique : boutons + flèches clavier (◀/▶ pour remonter et
  // redescendre, Fin ou « ● direct » pour recoller au direct).
  $("nav-prec").addEventListener("click", navPrecedent);
  $("nav-suiv").addEventListener("click", navSuivant);
  $("nav-live").addEventListener("click", navDirect);
  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); navPrecedent(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); navSuivant(); }
    else if (e.key === "End") { e.preventDefault(); navDirect(); }
  });
  tick();
  setInterval(tick, PERIODE_MS);
  // Animation continue des horloges entre deux polls (~5 Hz : mm:ss affiché,
  // inutile d'aller plus vite ; indépendant du réseau).
  setInterval(majHorloges, 200);
})();
