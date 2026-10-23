# import_local.py — Importe des audios locaux dans le dataset train et/ou test
#
# Usage simple (1 dossier) :
#   python import_local.py --source "D:\Ma zic\playlist"
#   python import_local.py --source "D:\Ma zic\test" --target test
#
# Usage double (2 dossiers en une commande) :
#   python import_local.py --source-train "D:\Train" --source-test "D:\Test"
#   python import_local.py --source-train "D:\Train"                       # train seulement
#   python import_local.py --source-test "D:\Test"                         # test seulement
#
# Options :
#   --dry-run       Simule sans copier
#   --overwrite     Remplace les fichiers existants
#   --keep-bitrate  Ne pas reconvertir les WAV (copie directe)
#
# Après import :
#   train : python extract_all.py
#   test  : python extract_test.py

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
SUPPORTED_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.wma'}

TARGETS = {
    'train': {
        'audio_dir': 'dataset/audio',
        'manifest': 'dataset/download_manifest.json',
        'extract_cmd': 'python extract_all.py',
    },
    'test': {
        'audio_dir': 'dataset_test/audio',
        'manifest': 'dataset_test/download_manifest.json',
        'extract_cmd': 'python extract_test.py',
    },
}

# ============================================================
# HELPERS
# ============================================================
def load_manifest(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_manifest(path, manifest):
    """Sauvegarde atomique (tmp + rename)."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def safe_basename(name, max_len=200):
    """Nettoie un nom de fichier pour éviter les caractères interdits Windows."""
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']:
        name = name.replace(ch, '_')
    if len(name) > max_len:
        name = name[:max_len]
    return name.strip()

def ffprobe_ok(path):
    """Vérifie que le fichier contient au moins une piste audio valide."""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and 'audio' in r.stdout.lower()
    except Exception:
        return False

def wav_is_conform(path):
    """Vrai si le WAV est déjà 22050 Hz mono 16-bit (format pipeline)."""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=sample_rate,channels,codec_name',
             '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return False
        parts = r.stdout.strip().split(',')
        if len(parts) < 3:
            return False
        codec, rate, channels = parts[0], parts[1], parts[2]
        return codec == 'pcm_s16le' and rate == '22050' and channels == '1'
    except Exception:
        return False

def convert_to_wav(src, dst):
    """Convertit n'importe quel format audio en WAV 22050 Hz mono 16-bit."""
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src),
         '-ar', '22050', '-ac', '1', '-acodec', 'pcm_s16le', str(dst)],
        check=True, capture_output=True, timeout=300,
    )

def scan_source(source_dir):
    """Retourne la liste triée de tous les fichiers audio trouvés (récursif)."""
    src = Path(source_dir)
    if not src.is_dir():
        return None  # signale dossier introuvable
    files = []
    for p in sorted(src.rglob('*')):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files

# ============================================================
# IMPORT D'UN DOSSIER VERS UN TARGET
# ============================================================
def import_folder(source_dir, target, dry_run=False, overwrite=False, keep_bitrate=False):
    """
    Importe un dossier source dans le dataset cible.
    Retourne un dict de stats : {imported, skipped, failed, errors}
    """
    cfg = TARGETS[target]
    audio_dir = Path(cfg['audio_dir'])
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(cfg['manifest'])

    print(f"🔍 Scan de : {Path(source_dir).resolve()}")
    files = scan_source(source_dir)
    if files is None:
        print(f"❌ Dossier source introuvable : {source_dir}")
        return {'imported': 0, 'skipped': 0, 'failed': 1,
                'errors': [(source_dir, 'dossier introuvable')]}

    if not files:
        print(f"⚠️  Aucun fichier audio trouvé (extensions : {', '.join(sorted(SUPPORTED_EXTS))})")
        return {'imported': 0, 'skipped': 0, 'failed': 0, 'errors': []}

    print(f"📂 {len(files)} fichier(s) audio trouvé(s)")
    print(f"🎯 Cible    : {target}  →  {audio_dir.resolve()}")
    print(f"📄 Manifeste: {cfg['manifest']}")
    if dry_run:
        print("🧪 Mode DRY-RUN : aucune modification")
    print()

    existing_names = {p.name.lower(): p for p in audio_dir.iterdir() if p.is_file()}

    n_imported = n_skipped = n_failed = 0
    errors = []
    total = len(files)

    for i, src in enumerate(files, 1):
        base = safe_basename(src.stem)
        target_name = base + '.wav'
        target_path = audio_dir / target_name

        # Skip si déjà présent
        if target_name.lower() in existing_names and not overwrite:
            print(f"[{i}/{total}] ⏩ Déjà présent : {target_name}")
            n_skipped += 1
            continue

        # Validation
        if not ffprobe_ok(src):
            print(f"[{i}/{total}] ⚠️  Pas un audio valide : {src.name}")
            n_failed += 1
            errors.append((str(src), 'pas une piste audio valide'))
            continue

        if dry_run:
            size_mb = src.stat().st_size / 1e6
            print(f"[{i}/{total}] 🔍 (dry-run) {src.name} → {target_name} ({size_mb:.1f} Mo)")
            n_imported += 1
            continue

        try:
            if src.suffix.lower() == '.wav' and (keep_bitrate or wav_is_conform(src)):
                shutil.copy2(src, target_path)
                action = 'copié'
            elif src.suffix.lower() == '.wav' and not keep_bitrate:
                convert_to_wav(src, target_path)
                action = 'ré-encodé'
            else:
                convert_to_wav(src, target_path)
                action = 'converti'

            size_mb = round(target_path.stat().st_size / 1e6, 2)
            key = f"local:{src.name}"
            manifest[key] = {
                'path': str(target_path),
                'size_mb': size_mb,
                'source': 'local',
                'original': str(src.resolve()),
            }
            save_manifest(cfg['manifest'], manifest)
            existing_names[target_name.lower()] = target_path
            print(f"[{i}/{total}] ✅ {src.name}  →  {target_name}  ({action}, {size_mb} Mo)")
            n_imported += 1

        except subprocess.TimeoutExpired:
            print(f"[{i}/{total}] ❌ Timeout : {src.name}")
            n_failed += 1
            errors.append((str(src), 'timeout ffmpeg'))
        except Exception as e:
            print(f"[{i}/{total}] ❌ {src.name} : {e}")
            n_failed += 1
            errors.append((str(src), str(e)))

    return {'imported': n_imported, 'skipped': n_skipped,
            'failed': n_failed, 'errors': errors}

# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Importe des musiques locales dans le dataset train et/ou test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Import simple (1 dossier)
  python import_local.py --source "D:\\Musique\\playlist"
  python import_local.py --source "D:\\Musique\\test" --target test

  # Import des DEUX dossiers en une commande
  python import_local.py --source-train "D:\\Train" --source-test "D:\\Test"
  python import_local.py --source-train "D:\\Train"
  python import_local.py --source-test  "D:\\Test"

  # Simulation
  python import_local.py --source-train "D:\\Train" --source-test "D:\\Test" --dry-run
""",
    )
    ap.add_argument('--source', help="Dossier source (mode simple, 1 dossier)")
    ap.add_argument('--target', choices=['train', 'test'], default='train',
                    help="Dataset cible du mode simple (défaut: train)")

    ap.add_argument('--source-train', help="Dossier source pour le TRAIN")
    ap.add_argument('--source-test',  help="Dossier source pour le TEST")

    ap.add_argument('--dry-run', action='store_true',
                    help="Affiche ce qui serait fait sans rien copier")
    ap.add_argument('--overwrite', action='store_true',
                    help="Remplace les fichiers existants")
    ap.add_argument('--keep-bitrate', action='store_true',
                    help="Ne pas reconvertir les .wav (copie directe)")

    args = ap.parse_args()

    # Vérifs ffmpeg
    if shutil.which('ffprobe') is None:
        raise SystemExit("❌ ffprobe introuvable. Installe ffmpeg.")
    if shutil.which('ffmpeg') is None:
        raise SystemExit("❌ ffmpeg introuvable. Installe ffmpeg.")

    # Déterminer les jobs à exécuter
    jobs = []  # [(source, target, label)]

    if args.source_train or args.source_test:
        if args.source:
            raise SystemExit("❌ Utilise SOIT --source SOIT --source-train/--source-test, pas les deux.")
        if args.source_train:
            jobs.append((args.source_train, 'train', 'TRAIN'))
        if args.source_test:
            jobs.append((args.source_test, 'test', 'TEST'))
    elif args.source:
        jobs.append((args.source, args.target, args.target.upper()))
    else:
        raise SystemExit("❌ Aucune source spécifiée. Utilise --source, --source-train ou --source-test.")

    # Exécution
    all_stats = {}
    for source, target, label in jobs:
        print(f"\n{'='*70}")
        print(f"  IMPORT {label} : {source}")
        print(f"{'='*70}\n")
        stats = import_folder(source, target,
                              dry_run=args.dry_run,
                              overwrite=args.overwrite,
                              keep_bitrate=args.keep_bitrate)
        all_stats[target] = stats

    # Bilan global
    print(f"\n{'='*70}")
    print(f"  BILAN GLOBAL")
    print(f"{'='*70}")
    total_imported = total_skipped = total_failed = 0
    for target, s in all_stats.items():
        print(f"\n  [{target.upper()}]")
        print(f"    Importés : {s['imported']}")
        print(f"    Skippés  : {s['skipped']}")
        print(f"    Échecs   : {s['failed']}")
        total_imported += s['imported']
        total_skipped  += s['skipped']
        total_failed   += s['failed']

    print(f"\n  TOTAL")
    print(f"    ✅ Importés : {total_imported}")
    print(f"    ⏩ Skippés  : {total_skipped}")
    print(f"    ❌ Échecs   : {total_failed}")

    if total_failed > 0:
        print(f"\n  Détail des erreurs :")
        for target, s in all_stats.items():
            for name, err in s['errors'][:10]:
                print(f"    [{target}] {Path(name).name} : {err}")

    # Suggestions
    if total_imported > 0 and not args.dry_run:
        print(f"\n💡 Prochaines étapes :")
        if 'train' in all_stats and all_stats['train']['imported'] > 0:
            print(f"   python extract_all.py")
        if 'test' in all_stats and all_stats['test']['imported'] > 0:
            print(f"   python extract_test.py")
        print(f"\n   Puis annote les nouveaux morceaux dans :")
        if 'train' in all_stats and all_stats['train']['imported'] > 0:
            print(f"     • annotations.json       (train)")
        if 'test' in all_stats and all_stats['test']['imported'] > 0:
            print(f"     • test_annotations.json  (test)")
        print(f"\n   Enfin :")
        print(f"     python train_model.py    (si train modifié)")
        print(f"     run.bat                  (comparatif)")


if __name__ == '__main__':
    sys.exit(main())