# extract_test.py — Extrait accords + V3 pour les chansons de TEST
import sys
from extract_all import main as _extract_main

if __name__ == "__main__":
    sys.argv = [
        "extract_test.py",
        "--audio-dir", "dataset_test/audio",
        "--output",    "dataset_test/chords_analysis.json",
        "--cache-dir", "dataset_test/chords_cache",
        "--log",       "extract_test.log",
        "--sanitize",  # recommandé : titres YouTube = caractères spéciaux
    ] + sys.argv[1:]
    sys.exit(_extract_main())