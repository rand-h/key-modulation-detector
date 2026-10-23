# improve_model.py — Boucle : diagnostique → propose → ré-entraîne → re-teste
import json, os, subprocess, shutil
from collections import Counter
import joblib

TEST_ANN   = "test_annotations.json"
TRAIN_ANN  = "annotations.json"
TEST_MODEL = "key_transition_model.joblib"

def load(p):
    if not os.path.exists(p): return {}
    with open(p, encoding="utf-8") as f: return json.load(f)

def save(p, d):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)

def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def diagnostic():
    chords_path = "dataset_test/chords_analysis.json"
    if not os.path.exists(chords_path):
        print("❌ Lance d'abord extract_test.py"); return

    chords = load(chords_path)
    truth  = load(TEST_ANN)

    # 1) Morceaux sans annotation
    missing = [f for f in chords if f not in truth and 'error' not in chords[f]]
    if missing:
        print(f"\n⚠️  {len(missing)} morceaux sans annotation de test :")
        for m in missing[:10]: print(f"   - {m}")

    # 2) Erreurs d'extraction
    errored = [f for f, e in chords.items() if 'error' in e]
    if errored:
        print(f"\n❌ {len(errored)} morceaux en erreur d'extraction :")
        for m in errored[:5]: print(f"   - {m}")

    # 3) Couverture train vs test
    train_truth = load(TRAIN_ANN)
    train_keys = Counter()
    for segs in train_truth.values():
        for s in segs:
            train_keys[s.get("key", "?")] += 1
    test_keys = Counter()
    for segs in truth.values():
        for s in segs:
            test_keys[s.get("key", "?")] += 1

    all_keys = sorted(set(train_keys) | set(test_keys))
    print("\nCouverture train vs test (nombre de segments) :")
    print(f"  {'tonalité':10s}  {'train':>6s}  {'test':>6s}")
    for k in all_keys:
        tr, te = train_keys.get(k, 0), test_keys.get(k, 0)
        flag = " ⚠️ absent du train" if tr == 0 and te > 0 else ""
        print(f"  {k:10s}  {tr:>6d}  {te:>6d}{flag}")

def add_tracks_to_train(new_tracks):
    path = "download_audio.py"
    with open(path, encoding="utf-8") as f:
        src = f.read()
    marker = "tracks = ["
    idx = src.index(marker) + len(marker)
    to_add = "\n" + "\n".join(f'    "{t}",' for t in new_tracks)
    src = src[:idx] + to_add + src[idx:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"✅ {len(new_tracks)} titres ajoutés à download_audio.py")

def parse_test_output(stdout):
    import re
    metrics = {}
    for line in stdout.splitlines():
        # accepte espaces multiples et caractères unicode dans le label
        m = re.match(r'^(V3\s+.*?)\s{2,}([\d.]+)\s+([\d.]+)\s*$', line.strip())
        if m:
            metrics[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return metrics

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-tracks", nargs="*", default=[],
                    help="Titres YouTube à ajouter au train")
    ap.add_argument("--annot-file", default=None,
                    help="JSON avec annotations à fusionner dans annotations.json")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="Affiche un delta avant/après ré-entraînement")
    args = ap.parse_args()

    print("="*60); print("ÉTAPE 1 — DIAGNOSTIC"); print("="*60)
    diagnostic()

    if args.new_tracks:
        print("\n" + "="*60); print("ÉTAPE 2 — AJOUT AU TRAIN"); print("="*60)
        add_tracks_to_train(args.new_tracks)
        run(["python", "download_audio.py"])
        run(["python", "extract_all.py"])

    if args.annot_file and os.path.exists(args.annot_file):
        print("\n" + "="*60); print("ÉTAPE 2b — FUSION ANNOTATIONS"); print("="*60)
        new = load(args.annot_file)
        train = load(TRAIN_ANN)
        added = 0
        for fname, segs in new.items():
            if fname not in train:
                train[fname] = segs; added += 1
        save(TRAIN_ANN, train)
        print(f"✅ {added} morceaux ajoutés à annotations.json")

    # Capture baseline AVANT ré-entraînement
    baseline_metrics = None
    if args.compare and not args.skip_test:
        print("\n" + "="*60); print("ÉTAPE 2c — BASELINE (avant ré-entraînement)"); print("="*60)
        r = subprocess.run(["python", "test_model.py"], capture_output=True, text=True)
        print(r.stdout)
        baseline_metrics = parse_test_output(r.stdout)

    if not args.skip_train:
        print("\n" + "="*60); print("ÉTAPE 3 — RÉ-ENTRAÎNEMENT"); print("="*60)
        backup = TEST_MODEL + ".bak"
        if os.path.exists(TEST_MODEL):
            shutil.copy2(TEST_MODEL, backup)
            print(f"📦 Backup : {backup}")
        run(["python", "train_model.py"])

    if not args.skip_test:
        print("\n" + "="*60); print("ÉTAPE 4 — RÉ-ÉVALUATION"); print("="*60)
        r = subprocess.run(["python", "test_model.py"], capture_output=True, text=True)
        print(r.stdout)
        after_metrics = parse_test_output(r.stdout)

        if baseline_metrics and after_metrics:
            print("\n" + "="*60); print("COMPARAISON AVANT/APRÈS"); print("="*60)
            for label in baseline_metrics:
                if label in after_metrics:
                    b_acc, b_f1 = baseline_metrics[label]
                    a_acc, a_f1 = after_metrics[label]
                    print(f"{label}")
                    print(f"  frame-acc : {b_acc:.3f} → {a_acc:.3f}  (Δ {a_acc-b_acc:+.3f})")
                    print(f"  mod-F1    : {b_f1:.3f} → {a_f1:.3f}  (Δ {a_f1-b_f1:+.3f})")

    print("\n💡 Interprétation :")
    print("   Si Δ improved - baseline > +0.02 → garde la nouvelle version")
    print("   Si Δ learned - improved  < -0.02 → la matrice apprise dégrade, à ignorer")
    print("   Si Δ learned - improved  > +0.02 → le modèle appris aide, à garder")

if __name__ == "__main__":
    main()