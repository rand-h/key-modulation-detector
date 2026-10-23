# train_model.py — Apprend la matrice de transition empirique 24x24
import json, os
import numpy as np
from collections import Counter
from sklearn.model_selection import GroupKFold
import joblib

# ============================================================
# CONSTANTES — ordre MUST match build_profiles() dans extract_all.py
# ============================================================
NOTE_TO_PC = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,
              'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
ROOTS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# IMPORTANT : tous les maj d'abord, puis tous les min
# (même ordre que build_profiles() dans extract_all.py)
ALL_KEYS = ([f"{r}:maj" for r in ROOTS] + [f"{r}:min" for r in ROOTS])
KEY_IDX  = {k:i for i,k in enumerate(ALL_KEYS)}

ENH = {'Db':'C#','Eb':'D#','Gb':'F#','Ab':'G#','Bb':'A#',
       'Cb':'B','Fb':'E','E#':'F','B#':'C'}

# Paramètres ajustés
PRIOR_STRENGTH = 5.0    # lissage bayésien (3.0 → 5.0 : plus robuste avec self-transitions)
MIN_REAL_MODS  = 10     # seuil sur les VRAIES modulations (pas les self)


def norm_key(k):
    """Normalise enharmonique (Db → C#, etc.)."""
    if not k or ':' not in k: return None
    r, m = k.split(':', 1)
    return f"{ENH.get(r, r)}:{m}"


def key_to_tuple(k):
    """Renvoie (int_root, mode) pour make_T_interval_based()."""
    r, m = k.split(':')
    return (NOTE_TO_PC[r], m)


# ============================================================
# PRIOR (théorie musicale, depuis extract_all.py)
# ============================================================
def _make_prior(keys_tuples):
    """Import différé pour éviter le cycle d'import au chargement."""
    from extract_all import make_T_interval_based
    return make_T_interval_based(keys_tuples)


# ============================================================
# EXTRACTION DES TRANSITIONS (self + vraies modulations)
# ============================================================
def extract_modulations(chords_data, ann_data):
    """
    Extrait TOUTES les transitions accord→accord, y compris les self.

    - self        : chaque accord dans un segment = 1 observation prev=next=segment_key
    - modulation  : transition entre deux segments de tonalités différentes

    Pourquoi les self : sans elles, la matrice T apprise estime
        T[k,k] = prior_strength * prior_T[k,k] / (row_sum + prior_strength)
    ce qui donne ~0.33 au lieu de ~0.97 → Viterbi force une modulation par accord.
    Les self-transitions dominent 95% du signal réel.
    """
    mods = []
    for fname, entry in chords_data.items():
        if 'error' in entry:
            continue
        if fname not in ann_data:
            continue
        chords = entry.get('chords', [])
        if len(chords) < 5:
            continue
        anns = sorted(ann_data[fname], key=lambda a: a['time'])
        if len(anns) < 1:
            continue

        total_dur = chords[-1]['end'] if chords else 1.0

        # 1) SELF-TRANSITIONS
        for i, ann in enumerate(anns):
            k = norm_key(ann.get('key', ''))
            conf = ann.get('confidence', 'low')
            if not k or k not in KEY_IDX or conf == 'low':
                continue
            start_t = ann['time']
            end_t = anns[i+1]['time'] if i+1 < len(anns) else total_dur
            n_chords = sum(1 for c in chords if start_t <= c['start'] < end_t)
            for _ in range(n_chords):
                mods.append({
                    'prev': k, 'next': k,
                    'prev_dur': 1.0,
                    'position': 0.5,
                    'fname': fname,
                    'conf': 'self',
                })

        # 2) VRAIES MODULATIONS
        for i in range(len(anns) - 1):
            prev = norm_key(anns[i].get('key', ''))
            nxt  = norm_key(anns[i+1].get('key', ''))
            if not prev or not nxt or prev == nxt:
                continue
            if prev not in KEY_IDX or nxt not in KEY_IDX:
                continue
            conf = anns[i+1].get('confidence', 'low')
            if conf == 'low':
                continue
            mods.append({
                'prev': prev, 'next': nxt,
                'prev_dur': anns[i+1]['time'] - anns[i]['time'],
                'position': anns[i+1]['time'] / max(total_dur, 1.0),
                'fname': fname,
                'conf': conf,
            })
    return mods


# ============================================================
# MATRICE DE TRANSITION EMPIRIQUE + LISSAGE BAYÉSIEN
# ============================================================
def build_transition_matrix(mods, prior_T, prior_strength=PRIOR_STRENGTH):
    """
    Construit la matrice de transition 24×24 à partir des transitions observées,
    avec lissage bayésien vers le prior théorique.

    Formule : T[i,j] = (counts[i,j] + prior_strength * prior_T[i,j])
                     / (row_sum[i]   + prior_strength)

    Retourne : (T, counts)
    """
    K = len(ALL_KEYS)
    counts = np.zeros((K, K))

    for m in mods:
        counts[KEY_IDX[m['prev']], KEY_IDX[m['next']]] += 1.0

    row_sum = counts.sum(axis=1, keepdims=True)
    T = (counts + prior_strength * prior_T) / (row_sum + prior_strength)
    return T, counts


# ============================================================
# CROSS-VALIDATION (GroupKFold par morceau)
# ============================================================
def cross_validate(mods, prior_T, prior_strength=PRIOR_STRENGTH):
    """
    Évalue la matrice apprise via GroupKFold par morceau.

    GroupKFold : les transitions d'un même morceau sont corrélées → split par morceau.
    Métrique  : log-lik moyenne par échantillon (plus haut = mieux).
    """
    if len(mods) < 10:
        return None

    groups = np.array([m['fname'] for m in mods])
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return None

    prev_idx = np.array([KEY_IDX[m['prev']] for m in mods])
    nxt_idx  = np.array([KEY_IDX[m['next']] for m in mods])

    lls = []
    for tr, te in GroupKFold(n_splits=n_splits).split(prev_idx, nxt_idx, groups):
        tr_mods = [mods[i] for i in tr]
        T, _ = build_transition_matrix(tr_mods, prior_T, prior_strength)
        ll = sum(np.log(T[prev_idx[i], nxt_idx[i]] + 1e-10) for i in te)
        lls.append(ll / len(te))

    return float(np.mean(lls)), float(np.std(lls))


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists('dataset/chords_analysis.json'):
        raise SystemExit("❌ Lance d'abord extract_all.py")
    if not os.path.exists('annotations.json'):
        raise SystemExit("❌ annotations.json manquant")

    with open('dataset/chords_analysis.json', encoding='utf-8') as f:
        chords_data = json.load(f)
    with open('annotations.json', encoding='utf-8') as f:
        ann_data = json.load(f)

    # 1. Extraction
    mods = extract_modulations(chords_data, ann_data)

    n_self = sum(1 for m in mods if m['conf'] == 'self')
    n_real = len(mods) - n_self
    print(f"Transitions observées   : {len(mods)}  (self: {n_self}, modulations: {n_real})")
    print(f"Morceaux source         : {len(set(m['fname'] for m in mods))}")

    # Distribution des intervalles : uniquement les VRAIES modulations
    real_mods = [m for m in mods if m['conf'] != 'self']
    if real_mods:
        intervals = Counter()
        for m in real_mods:
            r1, _ = m['prev'].split(':')
            r2, _ = m['next'].split(':')
            intervals[(NOTE_TO_PC[r2] - NOTE_TO_PC[r1]) % 12] += 1
        print("Distribution des modulations réelles :")
        for iv, c in sorted(intervals.items(), key=lambda x: -x[1])[:8]:
            print(f"   +{iv:2d} demi-tons : {c:3d}")

    # Seuil sur les VRAIES modulations (pas sur len(mods) qui inclut les self)
    if n_real < MIN_REAL_MODS:
        raise SystemExit(
            f"Pas assez de vraies modulations ({n_real}, min {MIN_REAL_MODS}). "
            f"Annote plus de morceaux."
        )

    # 2. Prior théorique
    keys_tuples = [key_to_tuple(k) for k in ALL_KEYS]
    prior_T = _make_prior(keys_tuples)

    # 3. Cross-validation
    cv = cross_validate(mods, prior_T, prior_strength=PRIOR_STRENGTH)
    if cv is not None:
        mean_ll, std_ll = cv
        ll_prior = sum(
            np.log(prior_T[KEY_IDX[m['prev']], KEY_IDX[m['next']]] + 1e-10)
            for m in mods
        ) / len(mods)

        print(f"\nCV log-lik / sample (appris)  : {mean_ll:+.3f} ± {std_ll:.3f}")
        print(f"Log-lik / sample (prior seul) : {ll_prior:+.3f}")

        delta = mean_ll - ll_prior
        if delta > 0.05:
            verdict = "✅ gain"
        elif delta > -0.05:
            verdict = "⚠️ marginal"
        else:
            verdict = "❌ perte"
        print(f"Δ (appris - prior)            : {delta:+.3f} {verdict}")

    # 4. Modèle final
    T_final, counts = build_transition_matrix(mods, prior_T, prior_strength=PRIOR_STRENGTH)
    print(f"\nTransitions observées : {int(counts.sum())}")
    print(f"Paires distinctes     : {int((counts > 0).sum())} / {len(ALL_KEYS)**2}")

    # Vérification rapide : diagonale (self-transitions dominantes ?)
    diag = np.diag(T_final)
    print(f"Diagonale T (self-transitions) : min={diag.min():.3f}, "
          f"median={np.median(diag):.3f}, max={diag.max():.3f}")

    # 5. Sauvegarde
    joblib.dump({
        'T': T_final,
        'prior': prior_T,
        'keys': ALL_KEYS,
        'n_modulations': n_real,          # ← vrai compte (pas les self)
        'n_self': n_self,
        'n_total': len(mods),
        'counts': counts,
        'prior_strength': PRIOR_STRENGTH,
    }, 'key_transition_model.joblib')
    print("✅ Sauvegardé : key_transition_model.joblib")


if __name__ == '__main__':
    main()