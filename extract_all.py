# extract_all.py — Extraction + pré-annotation avec options CLI
import json, os, sys, subprocess, glob, time, argparse, shutil, tempfile, re
from datetime import datetime
import numpy as np

# ============================================================
# CONFIG PAR DÉFAUT (surchargeable via CLI)
# ============================================================
DEFAULT_AUDIO_DIR   = 'dataset/audio'
DEFAULT_OUTPUT_JSON = 'dataset/chords_analysis.json'
DEFAULT_ENGINE      = 'chord_engine/ismir_engine.exe'
DEFAULT_CACHE_DIR   = 'dataset/chords_cache'
LOG_FILE            = 'extract_all.log'

# ---------- Théorie musicale ----------
NOTE_TO_PC = {
    'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,
    'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11,
}
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHORD_INTERVALS = {
    'maj':[0,4,7], 'min':[0,3,7], 'dim':[0,3,6], 'aug':[0,4,8],
    'maj7':[0,4,7,11], 'min7':[0,3,7,10], '7':[0,4,7,10],
    'dim7':[0,3,6,9], 'm7b5':[0,3,6,10], 'sus4':[0,5,7], 'sus2':[0,2,7],
}
KRUMHANSL_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KRUMHANSL_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# ============================================================
# LOGGING (écran + fichier)
# ============================================================
class Logger:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
    def __call__(self, s=''):
        print(s)
        self.f.write(s + '\n')
        self.f.flush()
    def close(self):
        self.f.close()

log = None  # initialisé dans main()

# ============================================================
# THÉORIE MUSICALE
# ============================================================
def parse_chord(symbol):
    if symbol in ('N/A','NC','',None): return None
    parts = symbol.split('/')
    first = parts[0].strip()
    if ':' in first:
        root_str, quality = first.split(':', 1)
    else:
        m = re.match(r'^([A-Ga-g][#b]?)(.*)$', first)
        if not m: return None
        root_str, quality = m.group(1), m.group(2) or 'maj'
    root_str = root_str.strip().capitalize()
    quality = quality.strip().lower() or 'maj'
    if root_str not in NOTE_TO_PC: return None
    return (NOTE_TO_PC[root_str], quality)

def chord_pcs(symbol):
    p = parse_chord(symbol)
    if p is None: return set()
    root, quality = p
    if quality not in CHORD_INTERVALS:
        for q in ['maj7','min7','dim7','m7b5','sus4','sus2','maj','min','dim','aug','7']:
            if quality.startswith(q): quality = q; break
        else: return {root}
    return {(root + i) % 12 for i in CHORD_INTERVALS[quality]}

def key_name(key):
    root, mode = key
    return f"{NOTE_NAMES[root]} {'majeur' if mode == 'maj' else 'mineur'}"

def key_label(key):
    return f"{NOTE_NAMES[key[0]]}:{key[1]}"

def build_profiles():
    return {**{(r,'maj'): np.roll(KRUMHANSL_MAJOR, r) for r in range(12)},
            **{(r,'min'): np.roll(KRUMHANSL_MINOR, r) for r in range(12)}}

def hist_from_chord(s):
    h = np.zeros(12)
    for pc in chord_pcs(s): h[pc] += 1
    if h.sum() > 0: h /= h.sum()
    return h

def key_corr(h, profiles):
    if h.std() == 0: return {k: 0.0 for k in profiles}
    return {k: np.corrcoef(h, p)[0,1] for k, p in profiles.items()}

def is_dom(s, root_pc):
    p = parse_chord(s)
    return p is not None and p[0] == (root_pc+7)%12 and p[1].startswith(('maj','7'))

def is_ton(s, root_pc, mode):
    p = parse_chord(s)
    return p is not None and p[0] == root_pc and p[1].startswith('maj' if mode=='maj' else 'min')

def has_lead(s, root_pc):
    return (root_pc - 1) % 12 in chord_pcs(s)

# ============================================================
# PIPELINE V2 + VITERBI + V3
# ============================================================
def build_emissions_v2(chords, keys, profiles):
    N, K = len(chords), len(keys); em = np.zeros((N, K))
    for i, s in enumerate(chords):
        if s == 'N/A':
            em[i] = 1.0/K; continue
        h = hist_from_chord(s)
        if h.sum() == 0:
            em[i] = 1.0/K; continue
        arr = np.array([key_corr(h, profiles)[k] for k in keys]).copy()
        for j, (root, mode) in enumerate(keys):
            if i > 0 and is_dom(chords[i-1], root) and is_ton(s, root, mode): arr[j] += 0.40
            if has_lead(s, root): arr[j] += 0.15
            if i == 0 and is_ton(s, root, mode): arr[j] += 0.15
            if i == N-1 and is_ton(s, root, mode): arr[j] += 0.20
        arr = np.exp(arr * 3.0); em[i] = arr / arr.sum()
    return em

def viterbi(em, T, dur):
    N, K = em.shape
    le = np.log(em + 1e-10); lt = np.log(T + 1e-10)
    V = np.zeros((N,K)); ptr = np.zeros((N,K), dtype=int)
    V[0] = dur[0] * le[0]
    for t in range(1, N):
        for k in range(K):
            vals = V[t-1] + lt[:,k]
            best = int(np.argmax(vals))
            V[t,k] = vals[best] + dur[t] * le[t,k]
            ptr[t,k] = best
    path = np.zeros(N, dtype=int); path[-1] = int(np.argmax(V[-1]))
    for t in range(N-2,-1,-1): path[t] = ptr[t+1, path[t+1]]
    return path

def make_T(K, keys, self_p=0.85):
    T = np.full((K,K), (1-self_p)/(K-1))
    for i,(r1,m1) in enumerate(keys):
        for j,(r2,m2) in enumerate(keys):
            if i==j: T[i,j]=self_p; continue
            dist = min((r1-r2)%12, (r2-r1)%12)
            if dist in (5,7): T[i,j] += 0.03
            elif dist in (3,9): T[i,j] += 0.02
            elif r1==r2 and m1!=m2: T[i,j] += 0.02
    return T / T.sum(axis=1, keepdims=True)

# ============================================================
# PATCHES V3+ — degrés, relatives, transitions transposition-invariantes
# ============================================================

# ---------- 1. Fusion des relatifs (fix flickering ABBA/Katy Perry) ----------
# Fix : use int roots to match build_profiles() key format (int, mode)
RELATIVE_PAIRS = [
    ((0,'maj'),  (9,'min')),   # C maj - A min
    ((1,'maj'),  (10,'min')),  # C# maj - A# min
    ((2,'maj'),  (11,'min')),  # D maj - B min
    ((3,'maj'),  (0,'min')),   # D# maj - C min
    ((4,'maj'),  (1,'min')),   # E maj - C# min
    ((5,'maj'),  (2,'min')),   # F maj - D min
    ((6,'maj'),  (3,'min')),   # F# maj - D# min
    ((7,'maj'),  (4,'min')),   # G maj - E min
    ((8,'maj'),  (5,'min')),   # G# maj - F min
    ((9,'maj'),  (6,'min')),   # A maj - F# min
    ((10,'maj'), (7,'min')),   # A# maj - G min
    ((11,'maj'), (8,'min')),   # B maj - G# min
]
RELATIVE_OF = {}
for _maj, _mn in RELATIVE_PAIRS:
    RELATIVE_OF[_mn] = _maj
    RELATIVE_OF[_maj] = _mn

# Fonctions diatoniques : quels degrés appartiennent à une tonalité ?
DIATONIC_TO_MAJOR = {
    (0, 'min'): (0, 'maj'), (2, 'min'): (2, 'maj'), (4, 'min'): (4, 'maj'),
    (5, 'maj'): (0, 'maj'), (7, 'maj'): (0, 'maj'), (7, 'dom'): (0, 'maj'),
    (9, 'min'): (0, 'maj'), (11, 'dim'): (0, 'maj'),
}

def _belongs_to(candidate, target_root, target_mode):
    """True si candidate (root, mode) est diatonique à (target_root, target_mode)."""
    cr, cm = candidate
    deg = (cr - target_root) % 12
    if target_mode == 'maj':
        diatonic = {0: ('maj',), 2: ('min',), 4: ('min',), 5: ('maj',),
                    7: ('maj', 'dom'), 9: ('min',), 11: ('dim',)}
    else:
        diatonic = {0: ('min',), 2: ('dim',), 3: ('maj',), 5: ('min',),
                    7: ('min', 'maj', 'dom'), 8: ('maj',), 10: ('maj',), 11: ('dim', 'maj')}
    return deg in diatonic and cm in diatonic[deg]

def merge_relatives(segs, min_ratio=0.15, context_window=3):
    """Fusionne les segments courts diatoniques à un voisin dominant."""
    if len(segs) < 2: return segs
    total = sum(s['length'] for s in segs)
    if total == 0: return segs
    out = list(segs)

    # --- Passe 1 : segment court entouré de deux voisins = son relatif ---
    i = 1
    while i < len(out) - 1:
        s, left, right = out[i], out[i-1], out[i+1]
        rel = RELATIVE_OF.get(s['key'])
        if (s['length'] / total < min_ratio
                and left['key'] == right['key'] == rel):
            left['end'] = right['end']
            left['length'] += s['length'] + right['length']
            out.pop(i+1); out.pop(i)
            i = max(1, i-1)
        else:
            i += 1

    # --- Passe 2 : segment court adjacent à son relatif direct ---
    i = 0
    while i < len(out):
        s = out[i]
        if s['length'] / total < min_ratio:
            rel = RELATIVE_OF.get(s['key'])
            if i > 0 and out[i-1]['key'] == rel:
                out[i-1]['end'] = s['end']
                out[i-1]['length'] += s['length']
                out.pop(i); continue
            if i < len(out)-1 and out[i+1]['key'] == rel:
                out[i+1]['start'] = s['start']
                out[i+1]['length'] += s['length']
                out.pop(i); continue
        i += 1

    # --- Passe 2bis (NOUVEAU) : absorber si diatonique au dominant du voisinage ---
    i = 0
    while i < len(out):
        s = out[i]
        if s['length'] / total < min_ratio:
            # Trouver le segment le plus long dans un voisinage de ±context_window
            lo = max(0, i - context_window)
            hi = min(len(out), i + context_window + 1)
            neighbors = [(j, out[j]) for j in range(lo, hi) if j != i]
            if neighbors:
                dom_j, dom = max(neighbors, key=lambda x: x[1]['length'])
                # Ne fusionner que si dom est BEAUCOUP plus long (au moins 3x)
                if dom['length'] >= 5 * s['length'] and _belongs_to(s['key'], *dom['key']):
                    # Fusionner s dans dom
                    new_start = min(dom['start'], s['start'])
                    new_end = max(dom['end'], s['end'])
                    dom['start'] = new_start
                    dom['end'] = new_end
                    dom['length'] += s['length']
                    out.pop(i)
                    # Recalculer l'index : le segment fusionné peut être à gauche ou à droite
                    i = max(0, i - 1)
                    continue
        i += 1

    # --- Passe 3 : re-fusionner adjacents identiques ---
    merged = []
    for s in out:
        if merged and merged[-1]['key'] == s['key']:
            merged[-1]['end'] = s['end']
            merged[-1]['length'] += s['length']
        else:
            merged.append(s)
    return merged

# ---------- 2. Matrice de transition transposition-invariante ----------
def make_T_interval_based(keys, base_self_p=0.88):
    """
    T[i,j] dépend uniquement de (next.tonic - prev.tonic) mod 12 et du mode.
    Hiérarchise les modulations selon la réalité statistique de la musique tonale.
    """
    INTERVAL_PRIORS = {
        0: 0.88,   # même tonalité
        5: 0.030,  # IV (sous-dominante)
        7: 0.025,  # V (dominante)
        1: 0.015,  # +1 demi-ton (final chorus pop)
        2: 0.012,  # +2 demi-tons (final chorus +1 ton)
        3: 0.008,  # tierce mineure (relatif mineur)
        10: 0.005, # -2 demi-tons
        8: 0.003,  # -4 demi-tons
        9: 0.002,  # +9 = -3 (relatif majeur)
        4: 0.0005, # -8 (très rare)
        6: 0.0005, # triton (jazz)
        11: 0.0005 # -1 (très rare)
    }
    K = len(keys)
    T = np.zeros((K, K))
    for i, (r1, m1) in enumerate(keys):
        for j, (r2, m2) in enumerate(keys):
            interval = (r2 - r1) % 12
            base = INTERVAL_PRIORS.get(interval, 0.001)
            # Modulations parallèles (même tonique, mode ≠)
            if interval == 0 and m1 != m2:
                base = 0.02
            # Relatif mineur (maj → min à +3)
            if interval == 3 and m1 == 'maj' and m2 == 'min':
                base += 0.015
            # Relatif majeur (min → maj à +9)
            if interval == 9 and m1 == 'min' and m2 == 'maj':
                base += 0.015
            T[i, j] = base
    T = T / T.sum(axis=1, keepdims=True)
    return T

# ---------- 3. Émissions basées sur les degrés diéatoniques ----------
# Chaque degré accepte PLUSIEURS qualités (maj/min/dom/dim/aug)
DIATONIC_MAJOR = {
    0: ('maj',),
    2: ('min',),
    4: ('min',),
    5: ('maj',),
    7: ('maj', 'dom'),   # V ou V7 — FIX Bug A
    9: ('min',),
    11: ('dim', 'min'),  # vii° ou vii (rare en pop)
}
DIATONIC_MINOR = {
    0: ('min',),
    2: ('dim', 'min'),   # ii° ou ii (Dorian)
    3: ('maj',),         # III (relatif majeur)
    5: ('min', 'maj'),   # iv ou IV (Dorian)
    7: ('min', 'maj', 'dom'),  # v, V ou V7 — FIX Bug A
    8: ('maj',),         # VI
    10: ('maj',),        # VII naturel — FIX Bug B
    11: ('dim', 'maj'),  # vii° ou VII (harmonique)
}

def chord_quality_class(q):
    """Réduit une qualité d'accord à sa classe fonctionnelle."""
    q = (q or '').lower()
    if q.startswith('maj') or q == '': return 'maj'
    if q.startswith('min') or q == 'm': return 'min'
    if q.startswith('dim'): return 'dim'
    if q.startswith('aug') or q == '+': return 'aug'
    if q.startswith('7') or q == 'dom7': return 'dom'
    if q.startswith('sus'): return 'sus'
    return 'maj'

def score_chord_in_key(chord_root, chord_quality, key_root, key_mode):
    """Score [-0.8 .. 1.5] d'un accord dans une tonalité candidate."""
    deg = (chord_root - key_root) % 12
    qc = chord_quality_class(chord_quality)
    expected = DIATONIC_MAJOR if key_mode == 'maj' else DIATONIC_MINOR

    # Match diéatonique (avec qualités multiples acceptées)
    if deg in expected and qc in expected[deg]:
        if deg == 0: return 1.5          # tonique
        if deg == 7 and qc in ('maj', 'dom'): return 1.3   # V/V7
        if deg == 7: return 0.6          # v mineur
        if deg == 5: return 1.0          # IV
        return 0.6

    # I7/IV7 (dominante secondaire sur place)
    if qc == 'dom' and deg in expected and 'maj' in expected[deg]:
        return 0.4

    # V/x : dominante secondaire qui résout vers un degré diéatonique
    if qc == 'dom' and ((deg - 7) % 12) in expected:
        return 0.4

    # Emprunt au mode parallèle
    parallel = DIATONIC_MINOR if key_mode == 'maj' else DIATONIC_MAJOR
    if deg in parallel and qc in parallel[deg]:
        return 0.2

    if qc == 'sus':
        return 0.3

    return -0.4

def build_emissions_degree(chords_data, keys, window=8):
    """
    chords_data : liste de dicts {chord, start, end} (PAS de strings)
    Émissions basées sur la cohérence diéatonique dans une fenêtre glissante.
    """
    N = len(chords_data)
    em = np.zeros((N, len(keys)))
    half = max(1, window // 2)

    for i in range(N):
        lo, hi = max(0, i - half), min(N, i + half + 1)
        local = chords_data[lo:hi]

        for k_idx, (kr, km) in enumerate(keys):
            score = 0.0
            n_diatonic = 0
            n_valid = 0
            last = None

            for c in local:
                p = parse_chord(c['chord'])
                if p is None: continue
                root, q = p
                n_valid += 1
                s = score_chord_in_key(root, q, kr, km)
                score += s
                if s > 0.5: n_diatonic += 1

                # Bonus cadence V→I
                if last is not None:
                    lp, lq = last
                    if ((lp - kr) % 12 == 7 and (root - kr) % 12 == 0
                            and chord_quality_class(lq) in ('maj', 'dom')
                            and chord_quality_class(q) in ('maj', 'min')):
                        score += 0.8

                last = (root, q)

            if n_valid > 0:
                em[i, k_idx] = (n_diatonic / n_valid) * 2.0 + score / n_valid
            else:
                em[i, k_idx] = 0.0

    em = np.exp(em * 1.0)
    em = em / em.sum(axis=1, keepdims=True)
    return em

def smooth(segs, min_len=2):
    while True:
        changed = False; new = []
        for i, s in enumerate(segs):
            if s['length'] < min_len and len(segs) > 1:
                left  = segs[i-1] if i>0 else None
                right = segs[i+1] if i<len(segs)-1 else None
                tgt = left if (left and (not right or left['length'] >= right['length'])) else right
                tgt['length'] += s['length']
                if left is tgt: tgt['end'] = s['end']
                else: tgt['start'] = s['start']
                changed = True
            else: new.append(s)
        merged = []
        for s in new:
            if merged and merged[-1]['key'] == s['key']:
                merged[-1]['end'] = s['end']; merged[-1]['length'] += s['length']
            else: merged.append(s)
        if not changed and len(merged) == len(segs):
            segs = merged; break
        segs = merged
    return segs

def analyze_chords(chords_data):
    keys = list(build_profiles().keys())
    profiles = build_profiles()
    chords_str = [c['chord'] for c in chords_data]
    durs = np.array([c['end'] - c['start'] for c in chords_data])
    durs = durs / durs.mean()

    em = build_emissions_v2(chords_str, keys, profiles)
    T  = make_T(len(keys), keys)
    path = viterbi(em, T, durs)

    segs, start = [], 0
    for i in range(1, len(path)):
        if path[i] != path[i-1]:
            segs.append({
                'start': chords_data[start]['start'],
                'end':   chords_data[i-1]['end'],
                'key':   keys[path[i-1]],
                'length': i - start,
            })
            start = i
    segs.append({
        'start': chords_data[start]['start'],
        'end':   chords_data[-1]['end'],
        'key':   keys[path[-1]],
        'length': len(path) - start,
    })

    segs = smooth(segs, 2)
    segs = merge_relatives(segs)   # ← SEUL ajout par rapport au V3 original
    return segs

# ============================================================
# EXTRACTION
# ============================================================
def get_duration(path):
    try:
        r = subprocess.run(
            ['ffprobe','-v','error','-show_entries','format=duration',
             '-of','default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip() or 0)
    except: return 0.0

def convert_to_wav(src, dst):
    subprocess.run(
        ['ffmpeg','-y','-loglevel','error','-i', src,
         '-ar','22050','-ac','1', dst],
        check=True, capture_output=True,
    )

def sanitize_copy(src):
    """Copie le fichier dans /tmp avec un nom ASCII safe.
    Retourne le chemin temporaire (à supprimer après usage)."""
    ext = os.path.splitext(src)[1]
    tmp = os.path.join(tempfile.gettempdir(),
                       f"sanitized_{os.getpid()}_{int(time.time()*1000)}{ext}")
    shutil.copy2(src, tmp)
    return tmp

def extract_one(audio_path, engine_path, cache_dir, force=False, sanitize=False):
    """Retourne (json_path, was_cached)."""
    base = os.path.splitext(os.path.basename(audio_path))[0]
    cache_json = os.path.join(cache_dir, base + '.json')

    # Cache hit
    if not force and os.path.exists(cache_json) and \
       os.path.getmtime(cache_json) >= os.path.getmtime(audio_path):
        return cache_json, True

    # Sanitize si demandé (évite les erreurs ismir_engine sur caractères spéciaux)
    src = audio_path
    sanitized_tmp = None
    if sanitize:
        try:
            sanitized_tmp = sanitize_copy(audio_path)
            src = sanitized_tmp
        except Exception as e:
            if log: log(f"   ⚠️  Sanitize échoué : {e}")

    # Convertir en WAV si nécessaire
    tmp_wav = None
    if not src.lower().endswith('.wav'):
        tmp_wav = os.path.join(tempfile.gettempdir(),
                                f"ismir_{os.getpid()}_{int(time.time()*1000)}.wav")
        convert_to_wav(src, tmp_wav)
        wav_in = tmp_wav
    else:
        wav_in = src

    try:
        r = subprocess.run([engine_path, wav_in, cache_json],
                            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ismir_engine code {r.returncode} : "
                                f"{(r.stderr or r.stdout).strip()[:200]}")
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try: os.remove(tmp_wav)
            except: pass
        if sanitized_tmp and os.path.exists(sanitized_tmp):
            try: os.remove(sanitized_tmp)
            except: pass
    return cache_json, False

# ============================================================
# I/O JSON
# ============================================================
def load_existing(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        if log: log(f"⚠️  Impossible de charger {path} : {e}")
        return {}

def save_json(path, data):
    """Sauvegarde atomique (tmp + rename) pour éviter la corruption."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# ============================================================
# DÉCISION DE TRAITEMENT
# ============================================================
def should_process(fname, existing_entry, cache_json, audio_path,
                   force, retry_errors, only_missing, skip_errors):
    """Retourne (bool, raison_string)."""
    in_existing = existing_entry is not None
    is_error = in_existing and 'error' in existing_entry
    cache_ok = os.path.exists(cache_json) and \
               os.path.getmtime(cache_json) >= os.path.getmtime(audio_path)

    if force:
        return True, "force"
    if skip_errors and is_error:
        return False, "erreur ignorée (--skip-errors)"
    if retry_errors:
        if is_error:
            return True, "retry erreur"
        return False, "pas une erreur"
    if only_missing:
        if not in_existing:
            return True, "nouveau"
        if is_error:
            return True, "retry erreur (auto)"
        return False, "déjà traité"
    # Défaut : traite si pas dans existing OU en erreur OU cache invalide
    if not in_existing:
        return True, "nouveau"
    if is_error:
        return True, "retry erreur (auto)"
    if not cache_ok:
        return True, "cache invalide"
    return False, "déjà OK"

# ============================================================
# MAIN
# ============================================================
def main():
    global log

    parser = argparse.ArgumentParser(
        description="Extrait les accords et pré-annote les tonalités.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python extract_all.py                        # Défaut : skip OK, retry erreurs, ajoute nouveaux
  python extract_all.py --force                # Réextrait TOUT
  python extract_all.py --only-missing         # Traite uniquement les fichiers jamais traités
  python extract_all.py --retry-errors         # Réessaie SEULEMENT les fichiers en erreur
  python extract_all.py --sanitize             # Copie les fichiers avec un nom safe (fix caractères spéciaux)
  python extract_all.py --list                 # Affiche ce qui serait fait, sans rien exécuter
  python extract_all.py --audio-dir=dataset/audio --output=out.json
        """,
    )
    parser.add_argument('--audio-dir',  default=DEFAULT_AUDIO_DIR,
                        help=f"Dossier des audios (défaut: {DEFAULT_AUDIO_DIR})")
    parser.add_argument('--output',     default=DEFAULT_OUTPUT_JSON,
                        help=f"Fichier JSON de sortie (défaut: {DEFAULT_OUTPUT_JSON})")
    parser.add_argument('--engine',     default=DEFAULT_ENGINE,
                        help=f"Chemin vers ismir_engine (défaut: {DEFAULT_ENGINE})")
    parser.add_argument('--cache-dir',  default=DEFAULT_CACHE_DIR,
                        help=f"Dossier cache (défaut: {DEFAULT_CACHE_DIR})")
    parser.add_argument('--log',        default=LOG_FILE,
                        help=f"Fichier de log (défaut: {LOG_FILE})")

    # Modes
    parser.add_argument('-f', '--force',        action='store_true',
                        help="Réextrait TOUT (ignore cache et JSON existant)")
    parser.add_argument('-r', '--retry-errors', action='store_true',
                        help="Réessaie UNIQUEMENT les fichiers en erreur")
    parser.add_argument('-m', '--only-missing', action='store_true',
                        help="Traite UNIQUEMENT les fichiers jamais traités")
    parser.add_argument('-s', '--skip-errors',  action='store_true',
                        help="Ignore les fichiers déjà en erreur")
    parser.add_argument('--sanitize',           action='store_true',
                        help="Copie les fichiers avec un nom ASCII safe avant extraction")
    parser.add_argument('--list',               action='store_true',
                        help="Affiche ce qui serait fait, sans exécuter")
    parser.add_argument('--keep-user-annotations', action='store_true', default=True,
                        help="Préserve les user_annotation existantes (défaut: True)")

    args = parser.parse_args()

    # Validation
    if not os.path.isfile(args.engine):
        print(f"❌ Engine introuvable : {args.engine}")
        return 1
    if not os.path.isdir(args.audio_dir):
        print(f"❌ Dossier audio introuvable : {args.audio_dir}")
        return 1
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    log = Logger(args.log)

    # Charger l'existant pour le merge
    existing = load_existing(args.output)

    # Liste des fichiers audio
    audio_files = sorted(
        glob.glob(os.path.join(args.audio_dir, '*.wav')) +
        glob.glob(os.path.join(args.audio_dir, '*.mp3')) +
        glob.glob(os.path.join(args.audio_dir, '*.flac')) +
        glob.glob(os.path.join(args.audio_dir, '*.ogg')) +
        glob.glob(os.path.join(args.audio_dir, '*.m4a'))
    )
    if not audio_files:
        log(f"❌ Aucun fichier audio dans {args.audio_dir}")
        log.close()
        return 1

    # En-tête
    log(f"{'='*70}")
    log(f"📊 EXTRACTION DES ACCORDS + PRÉ-ANNOTATION")
    log(f"{'='*70}")
    log(f"Dossier audio : {os.path.abspath(args.audio_dir)}")
    log(f"Sortie        : {os.path.abspath(args.output)}")
    log(f"Cache         : {os.path.abspath(args.cache_dir)}")
    log(f"Engine        : {os.path.abspath(args.engine)}")
    log(f"Fichiers      : {len(audio_files)}")
    log(f"Existant      : {len(existing)} entrées déjà dans le JSON")
    log(f"Mode          : "
        f"{'FORCE (tout réextrait)' if args.force else ''}"
        f"{'RETRY-ERRORS' if args.retry_errors else ''}"
        f"{'ONLY-MISSING' if args.only_missing else ''}"
        f"{'SKIP-ERRORS' if args.skip_errors else ''}"
        f"{'DÉFAUT (skip OK + retry erreurs auto)' if not any([args.force,args.retry_errors,args.only_missing,args.skip_errors]) else ''}")
    if args.sanitize: log(f"Sanitize      : activé (copie ASCII temporaire)")
    log(f"{'='*70}\n")

    # Pré-calcul des décisions
    plan = []
    for audio in audio_files:
        fname = os.path.basename(audio)
        cache_json = os.path.join(args.cache_dir,
                                  os.path.splitext(fname)[0] + '.json')
        existing_entry = existing.get(fname)
        do_it, reason = should_process(
            fname, existing_entry, cache_json, audio,
            force=args.force, retry_errors=args.retry_errors,
            only_missing=args.only_missing, skip_errors=args.skip_errors,
        )
        plan.append({
            'audio': audio, 'fname': fname, 'cache_json': cache_json,
            'existing': existing_entry, 'process': do_it, 'reason': reason,
        })

    # Stats du plan
    n_process = sum(1 for p in plan if p['process'])
    n_skip    = len(plan) - n_process

    log(f"📋 Plan : {n_process} à traiter, {n_skip} à skipper\n")

    if args.list:
        log("Fichiers qui SERAIENT traités :")
        for p in plan:
            marker = '✓' if p['process'] else '·'
            log(f"  {marker} [{p['reason']:30s}] {p['fname']}")
        log.close()
        return 0

    # ============================================================
    # EXÉCUTION
    # ============================================================
    results = dict(existing)  # copie pour merge
    n_ok, n_err, n_skip_done, n_retry_ok = 0, 0, 0, 0
    t_global = time.time()

    for idx, item in enumerate(plan, 1):
        fname = item['fname']

        if not item['process']:
            log(f"[{idx}/{len(plan)}] ⏩ {fname}  ({item['reason']})")
            n_skip_done += 1
            continue

        log(f"[{idx}/{len(plan)}] 🔄 {fname}  ({item['reason']})")
        t0 = time.time()

        try:
            cache_json, cached = extract_one(
                item['audio'], args.engine, args.cache_dir,
                force=args.force, sanitize=args.sanitize,
            )
            with open(cache_json, encoding='utf-8') as f:
                chords = json.load(f)
            t_extract = time.time() - t0

            n_chords = len(chords)
            dur = chords[-1]['end'] if chords else get_duration(item['audio'])

            log(f"   📀 {n_chords} accords  "
                f"({'cache' if cached else f'{t_extract:.1f}s'})")

            # Pré-annotation V3
            seg_info = []
            try:
                segs = analyze_chords(chords)
                seg_info = [
                    {'start': s['start'], 'end': s['end'],
                     'key': key_label(s['key']),
                     'key_human': key_name(s['key']),
                     'n_chords': s['length']}
                    for s in segs
                ]
                log(f"   🎼 {len(segs)} segment(s) :")
                for s in seg_info:
                    log(f"      [{s['start']:6.1f}s → {s['end']:6.1f}s] "
                        f"{s['key_human']:15s} ({s['n_chords']} acc.)")
            except Exception as e:
                log(f"   ⚠️  Analyse V3 échouée : {e}")

            # Préserver les user_annotation existantes
            user_ann = None
            if args.keep_user_annotations and item['existing']:
                user_ann = item['existing'].get('user_annotation')

            results[fname] = {
                'duration': dur,
                'n_chords': n_chords,
                'chords': chords,
                'auto_annotation': seg_info,
                'user_annotation': user_ann,
            }
            n_ok += 1
            if item['reason'].startswith('retry'):
                n_retry_ok += 1

            # Sauvegarde incrémentale (résiste aux crashs)
            save_json(args.output, results)
            log(f"   💾 Sauvegardé")

        except Exception as e:
            log(f"   ❌ {e}")
            results[fname] = {'error': str(e)}
            save_json(args.output, results)
            n_err += 1

        log()

    # ============================================================
    # BILAN
    # ============================================================
    dt = time.time() - t_global
    n_success = sum(1 for r in results.values() if 'error' not in r)
    n_errors  = sum(1 for r in results.values() if 'error' in r)

    log(f"{'='*70}")
    log(f"✅ TERMINÉ en {dt:.1f}s")
    log(f"📄 {os.path.abspath(args.output)}")
    log(f"📋 {os.path.abspath(args.log)}")
    log(f"{'='*70}")
    log(f"Stats cette session :")
    log(f"   Traités OK        : {n_ok}  (dont {n_retry_ok} retry réussis)")
    log(f"   Échecs            : {n_err}")
    log(f"   Skippés           : {n_skip_done}")
    log(f"Stats JSON total :")
    log(f"   ✅ Succès         : {n_success}")
    log(f"   ❌ Erreurs        : {n_errors}")
    log(f"   Total             : {len(results)}")
    log(f"{'='*70}")

    if n_errors > 0:
        log(f"\n💡 Fichiers encore en erreur :")
        for fn, entry in results.items():
            if 'error' in entry:
                log(f"   • {fn}  →  {entry['error'][:80]}")
        log(f"\n   Essayez : python extract_all.py --retry-errors --sanitize")

    log.close()
    return 0

if __name__ == '__main__':
    sys.exit(main())