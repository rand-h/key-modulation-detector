# test_model.py — Compare V3-baseline / +relatives / degree / learned
import json, os, argparse
import numpy as np
from collections import defaultdict
import joblib

from extract_all import (
    build_profiles, build_emissions_v2, build_emissions_degree,
    make_T, make_T_interval_based, viterbi, smooth, merge_relatives,
    key_label,
)

# ---------- Alignement ----------
def truth_to_frames(chords, ann_segments):
    if not ann_segments: return None
    anns = sorted(ann_segments, key=lambda a: a["time"])
    labels = []
    for c in chords:
        mid = 0.5 * (c["start"] + c["end"])
        active = None
        for a in anns:
            if a["time"] <= mid: active = a
            else: break
        labels.append(active["key"] if active else None)
    return labels

def segments_to_frames(chords, segs):
    labels = []
    for c in chords:
        mid = 0.5 * (c["start"] + c["end"])
        k = None
        for s in segs:
            if s["start"] <= mid < s["end"]: k = key_label(s["key"]); break
        if k is None and segs: k = key_label(segs[-1]["key"])
        labels.append(k)
    return labels

# ---------- Décodage ----------
def decode(chords_data, keys, em_fn, T, durs, use_merge_relatives=False):
    em = em_fn(chords_data, keys)
    path = viterbi(em, T, durs)
    segs, start = [], 0
    for i in range(1, len(path)):
        if path[i] != path[i-1]:
            segs.append({"start": chords_data[start]["start"], "end": chords_data[i-1]["end"],
                         "key": keys[path[i-1]], "length": i - start})
            start = i
    segs.append({"start": chords_data[start]["start"], "end": chords_data[-1]["end"],
                 "key": keys[path[-1]], "length": len(path) - start})
    segs = smooth(segs, 2)
    if use_merge_relatives:
        segs = merge_relatives(segs)
    return segs

# ---------- Émissions ----------
def em_v2(chords_data, keys):
    """V2 original : attend une liste de strings."""
    profiles = build_profiles()
    return build_emissions_v2([c['chord'] for c in chords_data], keys, profiles)

def em_degree(chords_data, keys):
    """Archivé — voir audit. Ne pas réactiver sans fix de build_emissions_degree."""
    return build_emissions_degree(chords_data, keys, window=8)

# ---------- Métriques ----------
def frame_accuracy(pred_frames, true_frames):
    pairs = [(p, t) for p, t in zip(pred_frames, true_frames) if t]
    if not pairs: return None
    return sum(1 for p, t in pairs if p == t) / len(pairs)

def modulation_f1(pred_segs, true_segs, tol=2.0):
    def boundaries(segs):
        def get_time(s):
            return s.get("start", s.get("time", 0.0))
        return [get_time(s) for s in segs[1:]]
    pb, tb = boundaries(pred_segs), boundaries(true_segs)
    if not tb: return None, len(pb), 0
    tp = sum(1 for t in tb if any(abs(t - p) <= tol for p in pb))
    prec = tp / max(len(pb), 1)
    rec  = tp / max(len(tb), 1)
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0
    return f1, len(pb), len(tb)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chords",  default="dataset_test/chords_analysis.json")
    ap.add_argument("--annot",   default="test_annotations.json")
    ap.add_argument("--model",   default="key_transition_model.joblib")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.chords, encoding="utf-8") as f:
        chords_data = json.load(f)
    truth = {}
    if os.path.exists(args.annot):
        with open(args.annot, encoding="utf-8") as f:
            truth = json.load(f)

    # T apprise (optionnelle)
    T_learned = None
    if os.path.exists(args.model):
        bundle = joblib.load(args.model)
        T_learned = bundle['T']
        print(f"📦 Modèle de transition chargé ({bundle['n_modulations']} modulations)")

    keys = list(build_profiles().keys())
    T_base = make_T(len(keys), keys)
    T_interval = make_T_interval_based(keys)

    # Accumulateurs
    results = defaultdict(lambda: {
        "baseline_acc": [], "relatives_acc": [], "degree_base_acc": [],
        "degree_int_acc": [], "learned_acc": [],
        "baseline_f1": [], "relatives_f1": [], "degree_base_f1": [],
        "degree_int_f1": [], "learned_f1": [],
    })
    n_truth = 0

    for fname, entry in chords_data.items():
        if "error" in entry: continue
        chords = entry.get("chords", [])
        if len(chords) < 5: continue
        durs = np.array([max(c["end"]-c["start"], 0.01) for c in chords])
        durs /= durs.mean()

        # === Variante 1 : baseline (V3 original) ===
        segs_b = decode(chords, keys, em_v2, T_base, durs, use_merge_relatives=False)

        # === Variante 2 : V2 + merge_relatives ===
        segs_r = decode(chords, keys, em_v2, T_base, durs, use_merge_relatives=True)

        # === Variante 3 : degré + T baseline + merge ===
        #segs_d2 = decode(chords, keys, em_degree, T_base, durs, use_merge_relatives=True)

        # === Variante 4 : degré + T interval + merge ===
        #segs_d = decode(chords, keys, em_degree, T_interval, durs, use_merge_relatives=True)

        # === Variante 5 : degré + T apprise + merge ===
        #segs_l = None
        #if T_learned is not None:
            #segs_l = decode(chords, keys, em_degree, T_learned, durs, use_merge_relatives=True)

        # Les variables restent définies à None pour que le code ci-dessous ne plante pas
        segs_d2 = segs_d = segs_l = None

        # === Évaluation ===
        tsegs = truth.get(fname)
        if tsegs and any(s.get("skip_metric") for s in tsegs):
            tsegs = None  # skip des métriques
                   
        if tsegs:
            n_truth += 1
            t_frames = truth_to_frames(chords, tsegs)

            for tag, segs in [("baseline", segs_b), ("relatives", segs_r)]:
                if segs is None: continue
                frames = segments_to_frames(chords, segs)
                acc = frame_accuracy(frames, t_frames)
                if acc is not None: results[fname][f"{tag}_acc"].append(acc)
                f1, _, _ = modulation_f1(segs, tsegs)
                if f1 is not None: results[fname][f"{tag}_f1"].append(f1)

        if args.verbose:
            print(f"\n📀 {fname}")
            for tag, segs in [("BASE", segs_b), ("+REL", segs_r),
                              ("D+B ", segs_d2), ("D+I ", segs_d),
                              ("D+L ", segs_l)]:
                if segs is None: continue
                print(f"   {tag} : {len(segs)} segs → "
                      f"{[(round(s['start'],1), key_label(s['key'])) for s in segs]}")

    # === Bilan ===
    print("\n" + "="*70)
    print("BILAN TEST")
    print("="*70)
    n_tested = sum(1 for e in chords_data.values()
                   if 'error' not in e and len(e.get('chords',[])) >= 5)
    print(f"Morceaux testés      : {n_tested}")
    print(f"Morceaux avec vérité : {n_truth}")
    if n_truth == 0:
        print("\n⚠️  Aucune annotation. Crée test_annotations.json")
        return

    def avg(lst): return np.mean([x for x in lst if x is not None]) if lst else None

    # Dans test_model.py, commente les variantes 3-5
    variants = [
        ("V3 baseline",       "baseline"),
        ("V3 + relatives",    "relatives"),
        # ("V3 + degree + T baseline", "degree_base"),
        # ("V3 + degree + T interval", "degree_int"),
        # ("V3 + degree + T apprise",  "learned"),
    ]

    print(f"\n{'Variante':32s}  {'frame-acc':>10s}  {'mod-F1':>8s}")
    print("-" * 58)
    for label, tag in variants:
        accs = [x for v in results.values() for x in v[f"{tag}_acc"]]
        f1s  = [x for v in results.values() for x in v[f"{tag}_f1"]]
        if not accs: continue
        print(f"{label:32s}  {avg(accs):10.3f}  {avg(f1s):8.3f}")

    # --- Deltas ---
    base_acc = avg([x for v in results.values() for x in v["baseline_acc"]])
    base_f1  = avg([x for v in results.values() for x in v["baseline_f1"]])
    rel_acc  = avg([x for v in results.values() for x in v["relatives_acc"]])
    rel_f1   = avg([x for v in results.values() for x in v["relatives_f1"]])

    print(f"\nΔ relatives - baseline :")
    print(f"   frame-acc : {rel_acc - base_acc:+.3f}")
    print(f"   mod-F1    : {rel_f1  - base_f1 :+.3f}")

    d_base_acc = avg([x for v in results.values() for x in v["degree_base_acc"]])
    d_base_f1  = avg([x for v in results.values() for x in v["degree_base_f1"]])
    if d_base_acc is not None:
        print(f"\nΔ degree+T baseline - baseline :")
        print(f"   frame-acc : {d_base_acc - base_acc:+.3f}")
        print(f"   mod-F1    : {d_base_f1  - base_f1 :+.3f}")

    d_int_acc = avg([x for v in results.values() for x in v["degree_int_acc"]])
    d_int_f1  = avg([x for v in results.values() for x in v["degree_int_f1"]])
    if d_int_acc is not None:
        print(f"\nΔ degree+T interval - baseline :")
        print(f"   frame-acc : {d_int_acc - base_acc:+.3f}")
        print(f"   mod-F1    : {d_int_f1  - base_f1 :+.3f}")

    l_acc = avg([x for v in results.values() for x in v["learned_acc"]])
    l_f1  = avg([x for v in results.values() for x in v["learned_f1"]])
    if l_acc is not None:
        print(f"\nΔ learned - baseline :")
        print(f"   frame-acc : {l_acc - base_acc:+.3f}")
        print(f"   mod-F1    : {l_f1  - base_f1 :+.3f}")

    # --- Détail par morceau ---
    def fmt(x): return f"{x:.2f}" if x is not None else "  --"

    print("\nDétail par morceau :")
    print(f"  {'morceau':45s}  {'BASE':>5s}  {'+REL':>5s}  {'D+B':>5s}  {'D+I':>5s}  {'D+L':>5s}")
    for fname, v in results.items():
        vals = {
            'b':  avg(v["baseline_acc"]),
            'r':  avg(v["relatives_acc"]),
            'db': avg(v["degree_base_acc"]),
            'di': avg(v["degree_int_acc"]),
            'dl': avg(v["learned_acc"]),
        }
        if vals['b'] is None: continue
        present = [x for x in vals.values() if x is not None]
        best = max(present) if present else None
        def mark(x): return "✓" if x is not None and abs(x - best) < 1e-9 else " "
        print(f"  {fname[:45]:45s}  "
              f"{mark(vals['b'])}{fmt(vals['b']):>5s}  "
              f"{mark(vals['r'])}{fmt(vals['r']):>5s}  "
              f"{mark(vals['db'])}{fmt(vals['db']):>5s}  "
              f"{mark(vals['di'])}{fmt(vals['di']):>5s}  "
              f"{mark(vals['dl'])}{fmt(vals['dl']):>5s}")

if __name__ == "__main__":
    main()