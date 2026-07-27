# ♟️ Chess DeepBlue V2 — une IA d'échecs qui apprend, 100 % Rust

Une IA d'échecs entraînée par self-play sur un simple laptop — et surtout : **on peut
affronter ses versions successives** (l'IA après 1 h d'entraînement, 3 h, 10 h, 30 h…)
et sentir soi-même chaque étage de sa progression. Zéro framework de ML, zéro Python :
le réseau de neurones, le moteur de recherche et le serveur web sont écrits à la main
en Rust.

Projet frère de [PokerIA](https://github.com/MartinM-781/PokerIA). Objectif affiché :
le niveau de Deep Blue.

## Le dashboard (port 8778)

| Page | Ce qu'on y fait |
|---|---|
| **/** | Jouer contre n'importe quelle génération : bots de référence, « IA — 1 h », « IA — 3 h », … ou le champion du jour (promu par gating) |
| **/training** | Les courbes en direct : Elo estimé, % de points contre les bots de référence, loss, duels de gating, marqueurs des changements de régime |
| **/live** | 🔴 Regarder l'IA s'entraîner : le plateau d'une partie de self-play en cours, avec les évaluations de l'élève **et** du professeur côte à côte |

## Architecture

- **Règles** : [`shakmaty`](https://crates.io/crates/shakmaty) (bitboards).
- **Recherche** (`search.rs`) : négamax alpha-bêta fail-soft à approfondissement
  itératif — table de transposition (scores de mat ajustés au ply), quiescence,
  tri coup TT / MVV-LVA / killers / historique, élagage null-move.
- **Évaluation** (`nn.rs` + `nnue.rs`) : MLP maison à architecture libre
  (actuellement [773, 1024, 128, 1], ReLU + tanh, Adam écrit à la main) servi en
  recherche par des **accumulateurs incrémentaux type NNUE** — deux perspectives
  (nos features basculent avec le trait), deltas par coup, seule la tête est
  recalculée. Mesuré : ×12 sur la vitesse de recherche, puis ×3,8 de plus avec
  `target-cpu=native` (AVX2/FMA) → **~65 000 nœuds/s** par thread.
- **Apprentissage** (`train.rs` + `selfplay.rs`) : self-play TD-leaf — l'IA joue
  contre elle-même (ouvertures diversifiées à haute température, arbitrage des
  positions décidées), chaque position est étiquetée par une **valeur de
  recherche** plutôt que par le seul résultat final. Trois régimes d'étiquetage :
  auto (sa propre recherche), **mentoré** (la recherche d'un réseau champion
  figé) ou **oracle** (un moteur UCI externe pleine force — Stockfish — note
  chaque position, ~15 ms/position sur un pool de moteurs réutilisés).
  Tampon de rejeu d'1,2 M de positions, apprentissage parallélisé (rayon).
- **Gating** (`train.rs`) : un réseau n'est promu champion que s'il **bat** le
  champion en titre à ≥ 55 % sur un duel à ouvertures aléatoires appariées.
  Aucune régression ne peut atteindre le plateau.
- **Paliers** (`checkpoints.rs`) : instantané du réseau à 1 h / 3 h / 10 h /
  30 h / 100 h de temps d'entraînement cumulé — les adversaires historiques
  jouables du plateau.
- **Mesure** (`elo.rs`, `calibrate.rs`) : Elo estimé en continu contre une
  échelle d'ancres interne (ajustement par maximum de vraisemblance), et
  calibration contre Stockfish à force limitée (UCI_Elo) pour recaler l'échelle.
- **Distillation** (`distill.rs`) : transvaser un réseau dans une architecture
  plus grande (l'élève apprend à imiter le prof, puis l'entraînement reprend).

## Journal de bord (mesuré, pas raconté)

Le chemin réel, avec ses impasses — chaque leçon est un commit :

1. **Self-play 1 pli, étiquettes = résultat final** : apprend la valeur du
   matériel, plafonne (~750 Elo maison), puis **dérive** — le self-play devenu
   déterministe apprend son propre écho. *Leçon : la diversité des données est
   vitale ; les paliers ont servi de sauvegarde.*
2. **Recherche complète + TD-leaf** : +500 Elo en une nuit (pic 1326). Puis
   plateau : le cerveau [512] était plein.
3. **Réseau élargi par distillation** : l'élève [1024] a d'abord **régressé**
   en s'entraînant sur ses propres étiquettes floues (chambre d'écho, gatings
   17 %), a convergé sous mentorat (50 %)… sans pouvoir dépasser son professeur.
   *Leçon : un élève ne dépasse pas le prof dont il copie toutes les notes.*
4. **Oracle Stockfish** : les étiquettes passent de ~1300 à ~2600+ Elo de
   qualité. Première promotion en 2 cycles, puis le cliquet s'installe :
   **30 générations de champions** en ~40 h de calcul (147 duels de gating),
   Elo maison dans la bande 1450-1650 (pic à ~1660) après 75 h d'entraînement
   cumulé et 100 000+ parties de self-play.
5. **Première partie contre un humain** (l'opérateur du projet, aidé de
   l'analyse) : **nulle par triple répétition en 39 coups**. Le champion a
   perdu un cavalier dès l'ouverture (aucun livre : il improvise), puis a
   puni trois imprécisions adverses en 150 ms chacune pour égaliser et
   prendre l'avantage... avant d'accepter la répétition en finale de tours.
   Diagnostic gratuit : tactiquement impitoyable, ouvertures et finales
   encore naïves — les deux prochains chantiers s'écrivent d'eux-mêmes.

## Lancer

```bash
cargo build --release

# Entraînement (régime oracle — recommandé si un Stockfish est disponible)
./target/release/train --out models --threads 18 --search-nodes 8000 \
  --oracle engines/stockfish/stockfish-windows-x86-64-avx2.exe --oracle-movetime 15

# Serveur web (plateau + courbes + direct)
./target/release/serve

# Calibration Elo contre Stockfish à force limitée
./target/release/calibrate --games 24 --movetime 60 --elos 1320,1450,1600

# Élargir le réseau par distillation
./target/release/distill --teacher models/chess_best.bin --sizes 773,1024,128,1
```

`train --help` liste tous les réglages (tampon de rejeu, gating, températures,
λ TD-leaf, poids du mentor…). Tout est sauvegardé à chaque cycle : couper et
relancer reprend exactement où on en était. `.cargo/config.toml` compile pour
le processeur local (`target-cpu=native`) — recompiler sur chaque machine.

Stockfish n'est **pas** embarqué : déposer un binaire UCI dans `engines/` pour
les régimes oracle et la calibration (testé avec Stockfish 18).
