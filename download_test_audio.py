# download_test_audio.py — Set de TEST (jamais vu à l'entraînement)
import yt_dlp, os, json

OUTPUT_DIR    = "dataset_test/audio"
ARCHIVE_FILE  = "dataset_test/downloaded.txt"
MANIFEST_FILE = "dataset_test/download_manifest.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chansons avec modulations connues, styles variés, AUCUNE dans download_audio.py
tracks = [
    # --- Pop avec modulation finale (+1 demi-ton typiquement) ---
    "Kelly Clarkson Because of You",
    "Bon Jovi Always",
    "Mariah Carey Hero",
    "Christina Aguilera Hurt",
    "Celine Dion Because You Loved Me",

    # --- Modulations multiples / structurelles ---
    "ABBA The Winner Takes It All",
    "Queen Somebody to Love",
    "Elton John Can You Feel the Love Tonight",
    "Katy Perry Firework",
    "Whitney Houston I Will Always Love You",

    # --- Sans modulation (test de stabilité — le modèle ne doit PAS inventer) ---
    "Ed Sheeran Perfect",
    "Bruno Mars When I Was Your Man",
    "Adele Someone Like You",
    "Coldplay Fix You",

    # --- Jazz / standards (le terrain le plus dur) ---
    "Frank Sinatra Fly Me to the Moon",
    "Nat King Cole Unforgettable",
    "Ella Fitzgerald Blue Moon",

    # --- Variété / world (robustesse hors distribution) ---
    "Charles Aznavour La Boheme",
    "Jacques Brel Ne Me Quitte Pas",

    # ============================================================
    # --- NOUVEAUX AJOUTS (session 1) ---
    # ============================================================

    # --- Pop avec modulation finale (nouveaux exemples) ---
    "Taylor Swift Love Story",
    "Mariah Carey Vision of Love",
    "Whitney Houston One Moment in Time",
    "Celine Dion It's All Coming Back to Me Now",
    "Celine Dion The Power of Love",

    # --- Pop sans modulation (test de stabilité) ---
    "Adele Rolling in the Deep",
    "Adele Hello",
    "Sam Smith Stay With Me",
    "John Legend All of Me",
    "Alicia Keys If I Ain't Got You",

    # --- Rock / alternatif ---
    "Oasis Wonderwall",
    "Radiohead Creep",
    "Scorpions Wind of Change",

    # --- Variété internationale ---
    "Robbie Williams Angels",
    "Coldplay The Scientist",
]

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(m):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

def download(track):
    captured = {"path": None}
    def _hook(d):
        if d.get("status") == "finished":
            fp = d.get("info_dict", {}).get("filepath") or d.get("filename")
            if fp and fp.endswith(".wav"):
                captured["path"] = fp
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(OUTPUT_DIR, "%(title)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "wav", "preferredquality": "192"}],
        "quiet": True, "no_warnings": True,
        "download_archive": ARCHIVE_FILE,
        "noplaylist": True, "ignoreerrors": True,
        "progress_hooks": [_hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"ytsearch1:{track}"])
    except Exception as e:
        print(f"   ❌ {e}"); return None
    return captured["path"]

def main():
    manifest = load_manifest()
    new_dl, failed = 0, []
    for i, t in enumerate(tracks, 1):
        if t in manifest and os.path.exists(manifest[t]["path"]):
            print(f"[{i}/{len(tracks)}] ⏩ {t}"); continue
        print(f"[{i}/{len(tracks)}] ⬇️  {t}")
        p = download(t)
        if p and os.path.exists(p):
            manifest[t] = {"path": p, "size_mb": round(os.path.getsize(p)/1e6, 2)}
            save_manifest(manifest); new_dl += 1
            print(f"         ✅ {os.path.basename(p)}")
        else:
            failed.append(t); print("         ⚠️  échec")
    print(f"\n✅ {new_dl} nouveaux | ❌ {len(failed)} échecs")
    for t in failed: print(f"   - {t}")

if __name__ == "__main__":
    main()