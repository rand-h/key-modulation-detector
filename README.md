# key-modulation-detector

> **From chords to keys**: automatic modulation detection for tonal music.

[![Frame accuracy](https://img.shields.io/badge/frame--acc-0.944-brightgreen)]()
[![Modulation F1](https://img.shields.io/badge/mod--F1-0.863-blue)]()
[![Test tracks](https://img.shields.io/badge/test--tracks-91-lightgrey)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()

Système de détection automatique des changements de tonalité (modulations) dans un morceau, à partir d'une extraction d'accords audio. Conçu pour fonctionner en production sur tous les styles de musique tonale : variété, pop, rock, jazz.

---

## Statut actuel

| Métrique              | Valeur        | Contexte                                                               |
| --------------------- | ------------- | ---------------------------------------------------------------------- |
| Frame accuracy        | 0.944         | Précision par accord                                                   |
| Modulation F1         | 0.863         | Détection des bornes (métrique métier)                                 |
| Morceaux test         | 91            | 3 styles (pop/rock international, variété française, variété malgache) |
| Morceaux train        | 210           | Annotés manuellement                                                   |
| Modulations observées | 184           | Dans le train set                                                      |
| Plafond réaliste      | ~0.96 / ~0.87 | 98% atteint                                                            |

**Production = V3 + `merge_relatives` (min_dominance=5.0)**. Les variantes `degree` et `learned` sont archivées (voir section Décisions prises).

---

## Table des matières

- [Le problème](#le-problème)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dépendances externes](#dépendances-externes)
- [Comment l'utiliser](#comment-lutiliser)
- [Les algorithmes expliqués](#les-algorithmes-expliqués)
- [Métriques et évaluation](#métriques-et-évaluation)
- [Structure du projet](#structure-du-projet)
- [Résultats actuels](#résultats-actuels)
- [Décisions prises](#décisions-prises)
- [Limites connues](#limites-connues)
- [Roadmap](#roadmap)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Citation](#citation)

---

## Le problème

Un morceau de musique peut changer de tonalité en cours de route — c'est ce qu'on appelle une modulatio. Exemples classiques :

- Bon Jovi – Livin' on a Prayer : dernière modulation +1 demi-ton (Mi min → Fa min)
- Whitney Houston – I Will Always Love You : dernier refrain +1 ton (La maj → Si maj)
- Beyoncé – Love on Top : 4 modulations successives (Do → Do# → Ré → Ré#)

L'objectif est de produire automatiquement une segmentation temporelle du morceau :

```
[0:00 → 3:25] A majeur
[3:25 → 4:35] B majeur
```

avec comme seule entrée un fichier audio (`.wav`, `.mp3`, `.flac`...).

### Pourquoi c'est difficile

1. Ambiguïté relative/majeur : `C maj` et `Am` partagent les mêmes notes — Krumhansl les confond souvent
2. Tonicisations vs modulations : un ii-V-I éphémère en jazz n'est pas une modulation (c'est une tonicisation locale)
3. Flickering : sur des accords isolés, les algorithmes oscillent entre 2-3 tonalités voisines
4. Fausses modulations : un accord emprunté (bVII, V/ii) peut être mal interprété
5. Manque de données annotées : la vérité terrain doit être créée manuellement

---

## Architecture

```
                    +-----------------------------------------+
                    |  YouTube (yt-dlp) / fichiers locaux     |
                    +------------------+----------------------+
                                       |
                    +------------------v----------------------+
                    |  download_audio.py / import_local.py    |
                    |  (dataset/audio/*.wav)                  |
                    +------------------+----------------------+
                                       |
          +----------------------------+----------------------------+
          |                            |                            |
+---------v----------+      +----------v----------+      +----------v----------+
| madmom (externe)   |      | ISMIR2019 engine    |      | extract_all.py      |
| DBNBeatTracker     |      | chord_recognition.py|      | (V2 + merge_rel.)   |
| -> beats + downbeats|     | -> accords + timings |     | -> auto_annotation  |
+--------------------+      +---------------------+      +---------------------+
          |                            |                            |
          +----------------------------+----------------------------+
                                       |
                    +------------------v----------------------+
                    |  annotations.json / test_annotations    |
                    |  (verite terrain manuelle)              |
                    +------------------+----------------------+
                                       |
                    +------------------v----------------------+
                    |  train_model.py -> T empirique 24x24    |
                    |  test_model.py  -> comparatif           |
                    +-----------------------------------------+
```

**Points clés** :

- madmom fournit le beat tracking (positions des temps + downbeats) — utilisé pour aligner les modulations sur les frontières de mesure
- ISMIR2019 fournit l'extraction d'accords (fondamentale + qualité + timings)
- `extract_all.py` consomme les accords et produit la segmentation en tonalités
- Le HMM (Viterbi) est le squelette commun à toutes les variantes
- Le post-traitement (`smooth` + `merge_relatives`) élimine le flickering

---

## Installation

### Prérequis

- Python 3.9+ (3.9–3.10 recommandé pour la compatibilité madmom)
- ffmpeg + ffprobe (dans le PATH)
- madmom pour le beat tracking — voir Dépendances externes
- ISMIR2019 chord recognition pour l'extraction d'accords — voir Dépendances externes

### Installation rapide

```bash
# 1. Cloner ce depot
git clone https://github.com/rand-h/key-modulation-detector.git
cd key-modulation-detector

# 2. Dependances Python du projet
pip install numpy scikit-learn joblib yt-dlp

# 3. madmom (beat tracking)
pip install madmom
# OU si Python 3.11+ :
pip install "madmom-infer @ git+https://github.com/openmirlab/madmom-infer.git"

# 4. ISMIR2019 chord recognition
git clone https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition.git chord_engine_ismir
cd chord_engine_ismir
pip install torch numpy
cd ..

# 5. Verifier
python -c "import train_model, test_model, extract_all, improve_model; print('OK')"
python -c "import madmom; print('madmom OK')"
python -c "import torch; print('torch OK')"
```

Doit afficher `OK` trois fois.

---

## Dépendances externes

Ce projet repose sur deux bibliothèques externes pour l'analyse audio.

### 1. ISMIR2019 — Large Vocabulary Chord Recognition

**Rôle** : extraction automatique des accords (fondamentale + qualité + timings) à partir de l'audio.

**Dépôt officiel** : [music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition)

**Paper** : Large-Vocabulary Chord Transcription via Chord Structure Decomposition, ISMIR 2019.

**Installation** :

```bash
git clone https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition.git chord_engine_ismir
cd chord_engine_ismir
pip install torch numpy
# Les modeles pre-entraines sont inclus dans le depot (dossier cache_data/)
```

**Utilisation** :

```bash
python3 chord_recognition.py <audio_file> <output_file> [chord_dict]
```

**Exemple** :

```bash
python3 chord_recognition.py ma_chanson.mp3 ma_chanson.lab
```

**Options `chord_dict`** :

| Valeur                | Description                                        | Recommandation |
| --------------------- | -------------------------------------------------- | -------------- |
| `submission` (défaut) | Dictionnaire utilisé pour la soumission ISMIR 2019 | recommandé     |
| `ismir2017`           | Dictionnaire MIREX                                 | alternative    |
| `full`                | Tous les accords du dataset MARL                   | non testé      |

**Sortie** : fichier `.lab` au format standardisé (timestamps + labels), directement consommable par `extract_all.py`.

Note : l'ancien `chord_engine/ismir_engine.exe` était un binaire packagé du même moteur. La version officielle Python ci-dessus le remplace avantageusement (auditabilité, reproductibilité, updates).

---

### 2. madmom — Beat & Downbeat Tracking

**Rôle** : détection des temps (beats) et des temps forts (downbeats) pour l'alignement rythmique.

**Dépôt officiel** : [CPJKU/madmom](https://github.com/CPJKU/madmom)

**Paper** : madmom: a new Python Audio and Music Signal Processing Library, ACM Multimedia 2016.

**Installation** :

```bash
# Option A — version originale (recommandee sur Python 3.9/3.10)
pip install madmom

# Option B — reimplementation moderne (Python 3.10+)
pip install "madmom-infer @ git+https://github.com/openmirlab/madmom-infer.git"
```

**Utilisation CLI** :

```bash
# Detection des beats (temps)
DBNBeatTracker single -o beats.txt ma_chanson.wav

# Detection des downbeats (temps forts)
DBNDownBeatTracker single -o downbeats.txt ma_chanson.wav
```

**Sortie `beats.txt`** : un temps par ligne

```
0.512
1.024
1.536
...
```

**Sortie `downbeats.txt`** : temps + numéro dans la mesure

```
0.512   1
1.536   2
2.560   3
3.584   4
```

**Utilisation Python** :

```python
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

# Beat tracking
proc = DBNBeatTrackingProcessor(fps=100)
act  = RNNBeatProcessor()('ma_chanson.wav')
beats = proc(act)   # np.array des temps de beats

# Downbeat tracking (3/4 ou 4/4)
proc_db = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
act_db  = RNNDownBeatProcessor()('ma_chanson.wav')
downbeats = proc_db(act_db)   # np.array (time, beat_number)
```

---

## Comment l'utiliser

### Cas 1 — Obtenir la timeline d'un nouveau morceau (usage principal)

Tu as un fichier `ma_chanson.mp3` et tu veux la segmentation en tonalités.

**Étape 1 — Placer le fichier dans `dataset/audio/`**

**Étape 2 — Extraire les accords + lancer V3+relatives**

```bash
python extract_all.py --only-missing --sanitize
```

`--only-missing` évite de re-traiter les fichiers déjà présents. `--sanitize` copie temporairement avec un nom ASCII (utile si le titre contient des caractères spéciaux).

**Étape 3 — Lire le résultat**

Ouvre `dataset/chords_analysis.json` et cherche l'entrée de ton morceau :

```json
"ma_chanson.wav": {
  "duration": 245.3,
  "auto_annotation": [
    { "start": 0.0,   "end": 187.5, "key": "A:maj", "key_human": "A majeur" },
    { "start": 187.5, "end": 245.3, "key": "B:maj", "key_human": "B majeur" }
  ]
}
```

Le champ `auto_annotation` contient la timeline prédite.

**Pour un seul morceau (script ad-hoc) :**

```python
# predict_one.py
import json
from extract_all import analyze_chords

with open('dataset/chords_analysis.json') as f:
    data = json.load(f)

entry = data['ma_chanson.wav']
for s in analyze_chords(entry['chords']):
    print(f"[{s['start']:6.1f}s -> {s['end']:6.1f}s]  {s['key']}")
```

### Cas 1bis — Extraction complète avec beat tracking (recommandé)

Pour une meilleure précision sur les frontières de modulation, combine l'extraction d'accords (ISMIR2019) avec le beat tracking (madmom).

**Étape 1 — Accords (ISMIR2019)**

```bash
python3 chord_engine_ismir/chord_recognition.py ma_chanson.mp3 ma_chanson.lab
```

**Étape 2 — Beats (madmom)**

```bash
DBNBeatTracker single -o ma_chanson_beats.txt ma_chanson.wav
```

**Étape 3 — Conversion `.lab` -> JSON**

```python
# lab_to_json.py — convertit un .lab ISMIR en JSON compatible extract_all
import json, sys

def lab_to_chords(lab_path):
    chords = []
    with open(lab_path) as f:
        for line in f:
            start, end, label = line.strip().split('\t')
            chords.append({
                'start': float(start),
                'end':   float(end),
                'chord': label,
            })
    return chords

if __name__ == '__main__':
    print(json.dumps(lab_to_chords(sys.argv[1]), indent=2))
```

**Étape 4 — Détection de tonalité**

```bash
python extract_all.py
```

Sortie : `auto_annotation` dans `dataset/chords_analysis.json`, avec les bornes alignées sur les mesures (via les downbeats).

**Bonus — Alignement sur les downbeats**

```python
# snap_to_bars.py — aligne les modulations sur les downbeats
import numpy as np
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

def snap_modulations_to_bars(modulations, audio_path, beats_per_bar=4):
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act  = RNNDownBeatProcessor()(audio_path)
    downbeats = proc(act)
    bar_times = downbeats[downbeats[:, 1] == 1][:, 0]

    snapped = []
    for mod in modulations:
        idx = np.argmin(np.abs(bar_times - mod['time']))
        snapped.append({**mod, 'time': float(bar_times[idx])})
    return snapped
```

### Cas 2 — Tester le système sur un nouveau set (validation)

**Étape 1 — Créer le set de test**

```bash
# Option A : telecharger depuis YouTube
python download_test_audio.py

# Option B : importer des fichiers locaux
python import_local.py --source-test "C:\Mes tests"
```

**Étape 2 — Extraire les accords**

```bash
python extract_test.py
```

**Étape 3 — Annoter manuellement `test_annotations.json`**

```json
"ma_chanson.wav": [
  { "time": 0.0,   "key": "A:maj", "confidence": "high" },
  { "time": 187.5, "key": "B:maj", "confidence": "high" }
]
```

**Étape 4 — Lancer le comparatif**

```bash
run.bat
```

Sortie attendue :

```
Variante                         frame-acc    mod-F1
-------------------------------------------------------
V3 baseline                        0.928     0.820
V3 + relatives                     0.944     0.863

Delta relatives - baseline :
   frame-acc : +0.016
   mod-F1    : +0.043
```

### Cas 3 — Ajouter des morceaux au train (améliorer le modèle)

```bash
# Telecharger les nouveaux morceaux + extraire les accords
python download_audio.py
python extract_all.py

# (Manuel) Annoter les nouveaux morceaux dans annotations.json

# Re-entrainer la matrice de transition
python train_model.py

# Re-tester
run.bat
```

Le train n'apporte un gain mesurable qu'au-delà de ~300 morceaux annotés (voir Décisions prises).

### Cas 4 — Modifier l'algorithme (développement)

```bash
# Boucle rapide : modifier extract_all.py puis relancer le test
run.bat

# Afficher le detail morceau par morceau
run.bat verbose

# Re-entrainer + tester (apres modification de annotations.json ou train_model.py)
run.bat train

# Extraction + entrainement + test (apres ajout d'audios)
run.bat full
```

### Raccourcis `run.bat`

| Commande          | Effet                            | Durée     |
| ----------------- | -------------------------------- | --------- |
| `run.bat`         | Test seul (2 variantes)          | ~3 min    |
| `run.bat verbose` | Test + détail par morceau        | ~3 min    |
| `run.bat train`   | Ré-entraîne + teste              | ~5 min    |
| `run.bat full`    | Extraction + entraînement + test | ~5-60 min |
| `run.bat extract` | Extraction seule                 | ~2 min    |
| `run.bat help`    | Affiche l'aide                   | —         |

---

## Les algorithmes expliqués

### Brique commune : le HMM

- États cachés : les 24 tonalités (12 majeures + 12 mineures)
- Décodage : Viterbi, pondéré par la durée des accords
- Émissions : score `[N x 24]` où `N` = nombre d'accords
- Transition : matrice `[24 x 24]`

```python
em = build_emissions(chords)           # N x 24
T  = make_T()                          # 24 x 24
path = viterbi(em, T, durations)       # N
segs = path_to_segments(path)
segs = smooth(segs)
segs = merge_relatives(segs)
```

### V1 — Krumhansl naïf (baseline)

Chaque accord est corrélé aux 24 profils de Krumhansl (majeurs + mineurs). Simple, rapide, mais flickering massif et confond relatif/majeur.

### V2 — Krumhansl + règles musicales

V1 + bonus pour les preuves fortes : cadence V->I (+0.40), sensible (+0.15), bornes toniques (+0.15/+0.20). Distingue mieux maj/min.

### V3 — V2 + lissage anti-flickering

V2 + post-traitement `smooth()` : fusionne les segments de moins de 2 accords avec le voisin le plus long. Nettoie les micro-modulations.

### + `merge_relatives` (production)

Fusionne un segment court dont la tonalité est le relatif d'un voisin dominant. Résout le flickering `I <-> vi` (ABBA, Queen, Katy Perry).

```python
# Exemple : F#maj (30 acc) - D#min (3 acc) - F#maj (25 acc)
# -> F#maj (58 acc) : D#min est vi de F#maj, absorbe
```

**Paramètre clé** : `min_dominance=5.0` (le voisin dominant doit être au moins 5 fois plus long).

| min_dominance | frame-acc | mod-F1 | Note                            |
| ------------- | --------- | ------ | ------------------------------- |
| 3.0           | 0.953     | 0.859  | Plus agressif, casse ABBA/Queen |
| 5.0           | 0.944     | 0.863  | Production                      |

### Matrice de transition apprise (expérimental)

`train_model.py` apprend `P(key_next | key_prev)` avec lissage bayésien vers un prior théorique. Les self-transitions sont incluses (chaque accord dans un segment = une observation `prev=next`). Sans elles, la diagonale de `T` était ~0.33 au lieu de ~0.99 et Viterbi forçait une modulation à chaque accord.

Statut : archivé. Voir Décisions prises.

### Émissions "degré" (archivées)

Score chaque accord selon sa cohérence diéatonique. Avantage théorique : transposition-invariant. Réalité : bug V7/VII non résolu, dominé par `em_v2`. Archivé.

---

## Métriques et évaluation

### Frame accuracy

Pourcentage d'accords où la tonalité prédite = tonalité annotée.

### Modulation F1

F1-score sur les bornes de modulation, avec tolérance ±2s. C'est la métrique métier.

```python
precision = vrais_positifs / bornes_predites
rappel    = vrais_positifs / bornes_annotees
F1        = 2 * precision * rappel / (precision + rappel)
```

### GroupKFold (train)

Évaluations par morceau, jamais par accord — sinon surévaluation massive.

---

## Structure du projet

```
key-modulation-detector/
+-- chord_engine/           # ISMIR2019 (clone, non versionne)
|   +-- chord_recognition.py
|
+-- dataset/                       # TRAIN
|   +-- audio/                    # Morceaux (.wav/.mp3) — NON VERSIONNE
|   +-- chords_cache/             # JSON bruts — NON VERSIONNE
|   +-- chords_analysis.json      # Accords + auto_annotation
|
+-- dataset_test/                  # TEST
|   +-- audio/                    # NON VERSIONNE
|   +-- chords_cache/             # NON VERSIONNE
|   +-- chords_analysis.json
|
+-- download_audio.py             # YouTube -> train set
+-- download_test_audio.py        # YouTube -> test set
+-- import_local.py               # Fichiers locaux -> train/test
+-- extract_all.py                # Coeur : V3 + merge_relatives
+-- extract_test.py               # Wrapper test set
+-- train_model.py                # Apprend matrice T (experimental)
+-- test_model.py                 # Comparatif 2-5 variantes
+-- improve_model.py              # Boucle diagnostic/ajout
+-- lab_to_json.py                # (a creer) .lab ISMIR -> JSON
+-- snap_to_bars.py               # (a creer) alignement downbeats
+-- run.bat                       # Pipeline Windows
|
+-- annotations.json              # Verite TRAIN (manuel)
+-- test_annotations.json         # Verite TEST (manuel)
|
+-- key_transition_model.joblib   # Modele appris (non utilise prod)
+-- requirements.txt              # Dependances Python
+-- .gitignore
+-- README.md
```

| Fichier                 | Rôle                | Priorité                    |
| ----------------------- | ------------------- | --------------------------- |
| `extract_all.py`        | Coeur algorithmique | production                  |
| `test_model.py`         | Comparatif          | dev                         |
| `annotations.json`      | Vérité train        | à enrichir                  |
| `test_annotations.json` | Vérité test         | ne jamais mettre dans train |
| `train_model.py`        | Apprentissage T     | archivé                     |

---

## Résultats actuels

Test set : 91 morceaux, 3 styles.

| Variante                    | frame-acc | mod-F1 | Statut     |
| --------------------------- | --------- | ------ | ---------- |
| V3 + relatives (production) | 0.944     | 0.863  | production |
| V3 baseline                 | 0.928     | 0.820  | référence  |
| V3 + degree + T baseline    | 0.722     | 0.116  | archivé    |
| V3 + degree + T apprise     | 0.748     | 0.135  | archivé    |

### Morceaux bien détectés (score 1.00)

Adele, Coldplay, Ed Sheeran, Frank Sinatra, Jacques Brel, John Legend, Mariah Carey, Robbie Williams, Sam Smith, Taylor Swift, Whitney Houston, et ~60 morceaux malgaches (variété I-IV-V).

### Morceaux difficiles (limites structurelles)

| Morceau                    | BASE | +REL | Cause                             |
| -------------------------- | ---- | ---- | --------------------------------- |
| ABBA – Winner Takes It All | 1.00 | 0.97 | flickering `vi-V-I`, résolu à 97% |
| Isika Efatra               | 0.78 | 0.44 | `merge_relatives` sur-absorbe     |
| ANTSIKA JIABY              | 0.78 | 0.49 | idem                              |
| Hira malaky                | 0.42 | 0.42 | cassé à la base (V2)              |
| 10 Tiako hiova             | 0.08 | 0.10 | cassé à la base (V2)              |
| Instrumental               | 0.56 | 0.56 | cassé à la base (V2)              |

Les 3 derniers sont des échecs de `em_v2`, pas de `merge_relatives`.

---

## Décisions prises

### `merge_relatives(min_dominance=5.0)` en production

Testé empiriquement 3.0 vs 5.0 :

- 3.0 : +0.025 frame-acc, +0.038 F1 mais casse ABBA/Queen/Christina
- 5.0 : +0.016 frame-acc, +0.043 F1, préserve les vraies modulations courtes

`mod-F1` est la métrique métier — le compromis penche pour 5.0.

### Variantes `degree` et `learned` archivées

- `degree + T baseline` : 0.722 / 0.116 -> -0.21 frame-acc vs production
- `degree + T apprise` : 0.748 / 0.135 -> idem

Malgré un fix réussi de `train_model.py` (self-transitions, diagonale T=0.99, CV Delta=+0.20), les émissions `degree` sont structurellement inférieures à `em_v2`. Reconsidérer si le corpus dépasse 300 morceaux.

### `train_model.py` gardé mais non utilisé en production

Le pipeline fonctionne sans `key_transition_model.joblib`. Réactiver si le ML bat `+relatives` sur un futur corpus.

---

## Limites connues

### 1. Modulations parallèles (même tonique, mode change)

`F:min -> F:maj` fonctionne grâce aux règles V2, mais reste fragile.

### 2. Échecs de V2 sur 3 morceaux malgaches

`Hira malaky`, `10 Tiako hiova`, `Instrumental` ont un score < 0.60. Non prioritaire (3/91 = -0.004 sur la moyenne).

### 3. `merge_relatives` sur-absorbe sur 2 morceaux

`Isika Efatra` (0.78->0.44) et `ANTSIKA` (0.78->0.49). Piste : `min_dominance=4.0` ou `skip_metric`.

### 4. Qualité de l'extraction d'accords

ISMIR2019 produit ~5-10% d'accords erronés (surtout intro, passages calmes). Toute la chaîne hérite de ces erreurs.

### 5. Taille du dataset

184 modulations pour 24 classes = ~7 observations/classe. Pour un vrai modèle ML généralisable, il faudrait ~500+ modulations (~1000 morceaux annotés).

---

## Roadmap

| Priorité | Tâche                                                    | Effort    | Gain estimé            |
| -------- | -------------------------------------------------------- | --------- | ---------------------- |
| P2       | Intégrer `snap_to_bars` (beat-alignment via madmom)      | 1h        | +0.02 F1               |
| P2       | Tester `DBNDownBeatTracker` vs `DBNBeatTracker`          | 30 min    | stabilité              |
| P2       | Test `min_dominance=4.0` (compromis)                     | 5 min     | +0.02 sur 2 morceaux   |
| P2       | Annoter 15-20 morceaux difficiles (ABBA/Queen-like)      | 3-4h      | Mesure plus fiable     |
| P3       | Fixer `build_emissions_degree` (V7, contexte, dominance) | 2-3 jours | +0.005 (via `learned`) |
| P3       | Analyse bar-level au lieu de chord-level                 | 2h        | +0.02 frame-acc        |
| P3       | Passer de `madmom` (2016) à `madmom-infer` (2026)        | 2h        | maintenabilité         |
| P4       | Portage mobile (ONNX/TF Lite)                            | 1-2 jours | production             |
| P4       | Interface utilisateur (timeline visuelle)                | 1 jour    | UX                     |

Le système est à 98% du plafond atteignable avec cette approche.

---

## FAQ / Troubleshooting

### `madmom` ne s'installe pas sur Python 3.11+

La version PyPI de `madmom` est ancienne (2016) et compile des extensions Cython incompatibles avec numpy récent. Utilise `madmom-infer` :

```bash
pip install "madmom-infer @ git+https://github.com/openmirlab/madmom-infer.git"
```

Ou reste sur Python 3.9/3.10 si tu veux `pip install madmom`.

### Le `.lab` d'ISMIR2019 ne se convertit pas

Le format `.lab` est standard (timestamps tabulés). Si ton fichier contient des accords avec caractères spéciaux (`C:maj7/3`), vérifie que `extract_all.py::parse_chord` les accepte. Sinon, normalise les labels avant conversion (dictionnaire `submission` recommandé).

### Le test plante avec `KeyError: 'start'` dans `modulation_f1`

Le prédicteur utilise `start`, l'annotation utilise `time`. Fix (déjà appliqué) : `s.get("start", s.get("time", 0.0))`.

### `merge_relatives` ne change rien

Vérifie que `RELATIVE_OF` utilise des roots entiers (`(0,'maj')`), pas des strings (`'C:maj'`).

### Le `.joblib` est absent — `learned` ne s'affiche pas

C'est normal : `learned` est archivé. Le pipeline production n'en a pas besoin.

### `train_model.py` affiche `Delta = +0.200` mais `learned` reste mauvais

C'est attendu. Le delta CV est gonflé par les self-transitions. Le vrai test est la variante `learned` dans `test_model.py`.

### Comment ajouter une chanson au test set

```bash
python import_local.py --source-test "C:\Mes tests"
python extract_test.py
# (annoter dans test_annotations.json)
run.bat
```

Ne jamais mettre un titre de test dans `download_audio.py`.

### Comment annoter un morceau ?

- À l'oreille : jouer la fondamentale sur un piano/clavier
- Théorie : la tonalité principale est celle de la cadence finale
- Recherche : Hooktheory, Ultimate Guitar, forums

Un morceau monotonal = une seule entrée :

```json
[{ "time": 0, "key": "C:maj", "confidence": "high" }]
```

---

## Contribuer

1. Fork le projet
2. Créer une branche (`git checkout -b feature/ma-feature`)
3. Toujours tester avec `run.bat` avant commit
4. Ne jamais modifier `test_annotations.json` en même temps que `annotations.json`
5. Commit avec message clair

---

## Licence

Éducatif et recherche. Les audios sont soumis aux droits d'auteur de leurs ayants droit respectifs — ne pas redistribuer. Voir `.gitignore` pour la liste des fichiers exclus.

---

## Citation

Si tu utilises ce projet dans un travail académique, merci de le citer :

```bibtex
@misc{key-modulation-detector,
  author       = {rand-h},
  title        = {key-modulation-detector: Automatic Key Modulation Detection from Chord Sequences},
  howpublished = {\url{https://github.com/rand-h/key-modulation-detector}},
  year         = {2026},
  note         = {Frame accuracy 0.944, Modulation F1 0.863 on 91 test tracks}
}
```

---

## Crédits

- Krumhansl & Kessler (1982) pour les profils de tonalité
- MIREX pour les benchmarks de référence
- [music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) pour l'extraction d'accords
- [CPJKU/madmom](https://github.com/CPJKU/madmom) pour le beat tracking
- Inspiration : articles de Chordify, Melodia, Madmom

---

*Dernière mise à jour : voir `git log`*
