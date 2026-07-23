# ♟️ Chess — IA d'échecs par self-play, 100 % Rust

Une IA d'échecs qui apprend **en jouant contre elle-même**, à partir de zéro : aucune
ouverture, aucune table, aucune connaissance humaine — juste les règles et un réseau
de neurones qui découvre le jeu partie après partie. Et surtout : **on peut affronter
ses versions successives** — l'IA après 1 h d'entraînement, après 3 h, après 10 h… —
et sentir la différence.

Projet frère de [PokerIA](https://github.com/MartinM-781/PokerIA), avec la leçon
retenue : le chemin chaud est compilé dès le premier jour (Rust intégral, zéro Python).

## Comment ça marche

- **Moteur de règles** : [`shakmaty`](https://crates.io/crates/shakmaty) (bitboards,
  écosystème lichess) — génération de coups exacte et très rapide.
- **Réseau de valeur** : MLP maison `773 → 512 → 64 → 1` (ReLU, sortie tanh), écrit à
  la main avec Adam, sans framework. L'encodage est toujours **du point de vue du
  trait** : 12 plans pièce×case (miroir vertical quand les noirs jouent), droits de
  roque, en passant. Sortie ∈ [-1, 1] = espérance de gain du camp au trait.
- **Self-play** : parties parallèles (rayon) avec exploration softmax en température ;
  chaque position est étiquetée par le résultat final de la partie, vu de son trait.
  Le réseau apprend par régression — le matériel, les mats, puis la stratégie
  émergent seuls des statistiques de victoire.
- **Paliers** : le trainer photographie le modèle à **1 h, 3 h, 10 h, 30 h, 100 h**
  de temps d'entraînement cumulé (`models/chess_t1h.bin`, …). Le temps survit aux
  redémarrages (`models/state.json`).
- **Adversaires de référence** pour la courbe de progression : bot aléatoire et bot
  matériel (alpha-bêta profondeur 2).

## Lancer

```bash
cargo build --release

# Entraînement (Ctrl-C quand vous voulez : tout est sauvegardé à chaque cycle)
./target/release/train --out models --threads 10

# Serveur web : plateau + courbe d'entraînement
./target/release/serve
```

Puis ouvrir **http://localhost:8778** :

- **/** — le plateau : choisissez votre adversaire (« IA — 1 h d'entraînement »,
  « IA — 3 h », … les paliers non atteints sont grisés), votre couleur, et jouez.
- **/training** — la courbe d'apprentissage en direct : loss, % de points contre le
  bot aléatoire et contre le bot matériel, en fonction des heures d'entraînement.

## Un agent Claude comme chef de chantier

Ce projet est construit et piloté par un agent Claude (architecture, implémentation
par escouade d'agents parallèles, audits adversariaux croisés — signes/perspective,
règles de nulle, cohérence API — puis entraînement et évaluation en continu).
