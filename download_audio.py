# download_audio.py
import yt_dlp
import os
import json

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR       = "dataset/audio"
ARCHIVE_FILE     = "dataset/downloaded.txt"
MANIFEST_FILE    = "dataset/download_manifest.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LISTE DES TITRES
# ============================================================
tracks = [
    # --- Modulations franches et emblématiques (Pop/Rock) ---
    "Michael Jackson Man in the Mirror",
    "Bon Jovi Livin' on a Prayer",
    "Whitney Houston I Wanna Dance With Somebody",
    "Céline Dion My Heart Will Go On",
    "Beyoncé Love On Top",
    "a-ha Take On Me",
    "Bonnie Tyler Total Eclipse of the Heart",
    "Westlife You Raise Me Up",
    "Diana Ross Chain Reaction",
    "The Osmonds The Proud One",
    "Four Seasons Ronnie",
    "Mariah Carey Whitney Houston When You Believe",
    "Adele All I Ask",
    "Lady Gaga Perfect Illusion",
    "Ariana Grande Greedy",
    "Mandondona - THE SINGERS OF JESUS",
    "ZAKAIOSY TGC",
    "Jaojoby - mangala vaiavy",
    "Jaojoby - Alima",
    "Jerry Marcoss & Mima - Samy mafoaka",
    "NY AVO VITA SEDRA",
    "Groupe Ny Avo - Voady",
    "Nanie sy Lalie- Feo roa",
    "Rano sy Vary - Mahaleo (Dadah)",
    "Chicago - You're the Inspiration",
    "Earth, Wind & Fire - After The Love Has Gone",

    # --- Modulations multiples et rapides ---
    "Shania Twain If You Wanna Touch Her, Ask!",
    "Tommy Roe Dizzy",
    "Oasis All Around The World",
    "The Who My Generation",
    "Yes Close to the Edge",
    "The Oak Ridge Boys Elvira",
    "The Communards Don't Leave Me This Way",

    # --- Modulations parallèles et relatives ---
    "The Beatles While My Guitar Gently Weeps",
    "The Beatles Fool on the Hill",
    "The Beatles Here, There and Everywhere",
    "Derek and the Dominos Layla",
    "Jim Croce Time in a Bottle",

    # --- Jazz et modulations par cycles ---
    "John Coltrane Giant Steps",
    "Jerome Kern All the Things You Are",
    "John Coltrane Just Friends",
    "John Coltrane Joyspring",
    "Victor Young Stella by Starlight",
    "Bill Evans My Foolish Heart",
    "Thelonious Monk 'Round About Midnight",
    "Ray Henderson Bye Bye Blackbird",
    "Arthur Schwartz Alone Together",
    "Fats Waller Black Narcissus",

    # --- Autres cas notables ---
    "The Police Every Breath You Take",
    "Eagles Hotel California",
    "The Police So Lonely",
    "The Temptations My Girl",
    "Herman's Hermits No Milk Today",
    "Frank Sinatra Something Stupid",
    "Mr. Big Wild World",
    "Seal Kiss from a Rose",
    "Genesis Invisible Touch",
    "Mr. Big To Be With You",
    "Backstreet Boys I Want It That Way",
    "Michael Bolton How Can We Be Lovers?",
    "Eric Clapton Layla Unplugged",

    # ============================================================
    # --- NOUVEAUX AJOUTS (session 1) ---
    # ============================================================

    # --- Modulations finales classiques (final chorus +1 ton) ---
    "Mariah Carey All I Want For Christmas Is You",
    "Wham Last Christmas",
    "Michael Bublé Home",
    "Michael Bublé Haven't Met You Yet",
    "Christina Aguilera Beautiful",
    "Leona Lewis Bleeding Love",
    "Pink Raise Your Glass",
    "Kelly Clarkson Stronger",

    # --- Rock avec modulations ---
    "Guns N' Roses Sweet Child O' Mine",
    "Bryan Adams Heaven",
    "Journey Don't Stop Believin",
    "Europe The Final Countdown",
    "Bon Jovi Bed of Roses",

    # --- R&B / Soul ---
    "Boyz II Men End of the Road",
    "R. Kelly I Believe I Can Fly",
    "Lionel Richie Hello",
    "George Michael Careless Whisper",
    "Michael Jackson Billie Jean",
    "Michael Jackson Beat It",

    # --- Pop internationale (dont sans modulation — test stabilité) ---
    "Elton John Your Song",
    "John Lennon Imagine",
    "The Beatles Let It Be",
    "The Beatles Yesterday",
    "Jeff Buckley Hallelujah",
    "Eric Clapton Wonderful Tonight",
    "Norah Jones Don't Know Why",

    # --- Variété française ---
    "Édith Piaf La Vie En Rose",
    "Joe Dassin Les Champs-Élysées",
    "Claude François Comme d'habitude",
    "Daniel Balavoine Le Chanteur",
    "Michel Sardou Les Lacs du Connemara",

    # --- Malagasy (renforcement) ---
    "Mahaleo - Zanaka",
    "Jaojoby - Salegy",
    "Rossy - Ravoravo",

    # --- Rock alternatif / pop-rock avec modulation ---
    "Green Day Wake Me Up When September Ends",
    "U2 With Or Without You",
    "Linkin Park Numb",
    "Katy Perry Roar",
    "Lady Gaga Bad Romance",
    "Beyoncé Halo",

        # ============================================================
    # --- NOUVEAUX AJOUTS (session 2) : >100 titres avec modulation ---
    # ============================================================

    # --- Pop / Soul : modulations finales emblématiques ---
    "Cher - If I Could Turn Back Time",
    "Cher - Strong Enough",
    "Dusty Springfield - You Don't Have to Say You Love Me",
    "Dusty Springfield - Son of a Preacher Man",
    "The Supremes - Stop! In the Name of Love",
    "Cheryl Lynn - Got to Be Real",
    "Belinda Carlisle - Heaven Is a Place on Earth",
    "Celine Dion - All by Myself",
    "Celine Dion - That's the Way It Is",
    "Whitney Houston - How Will I Know",
    "Whitney Houston - I Have Nothing",
    "Whitney Houston - I Will Always Love You",
    "Mariah Carey - Hero",
    "Mariah Carey - Vision of Love",
    "Mariah Carey - Emotions",
    "Ariana Grande - Greedy",
    "Ariana Grande - Popular",
    "Lizzo - 2 Be Loved (Am I Ready)",
    "Rihanna - Only Girl (In the World)",
    "Britney Spears - Where Are You Now",
    "Backstreet Boys - Shape of My Heart",
    "All-4-One - I Swear",
    "Michael Bublé - Everything",
    "Lorde - Green Light",

    # --- Pop / Rock : changements de tonalité multiples ou dramatiques ---
    "The Beach Boys - Wouldn't It Be Nice",
    "The Beach Boys - Don't Talk (Put Your Head on My Shoulder)",
    "The Beatles - Penny Lane",
    "The Beatles - Hey Jude",
    "The Beatles - Something",
    "The Beatles - Strawberry Fields Forever",
    "The Beatles - Drive My Car",
    "The Beatles - Another Girl",
    "The Beatles - You're Gonna Lose That Girl",
    "The Beatles - A Day in the Life",
    "The Doors - Love Street",
    "The Who - Eminence Front",
    "The Who - My Generation",
    "The Kinks - Waterloo Sunset",
    "David Bowie - Quicksand",
    "David Bowie - Warszawa",
    "Queen - Keep Yourself Alive",
    "Led Zeppelin - All My Love",
    "The Eagles - Already Gone",
    "Cheap Trick - Surrender",
    "The Pretenders - Back on the Chain Gang",
    "Madness - Our House",
    "The Searchers - When I Get Home",
    "Cold Chisel - Flame Trees",
    "The Cranberries - Dreams",
    "The Cardigans - Lovefool",
    "The Corrs - Breathless",
    "The Smashing Pumpkins - In the Arms of Sleep",
    "Weezer - Across the Sea",
    "Coldplay - 42",
    "Coldplay - Fix You",
    "Radiohead - Creep",
    "Muse - Knights of Cydonia",
    "Fun. - One Foot",
    "Mother Mother - Hay Loft",
    "Wilco - Everlasting Everything",
    "Red Hot Chili Peppers - Under the Bridge",
    "Red Hot Chili Peppers - Wet Sand",
    "The Fray - Never Say Never",
    "Damien Rice - The Blower's Daughter",
    "Bright Eyes - Spun Out",
    "Devendra Banhart - Seahorse",
    "Ray LaMontagne - Pick Up a Gun",
    "Elliott Smith - Miss Misery",
    "Fucked Up - Vivian Girls",
    "MGMT - Electric Feel",
    "Empire of the Sun - Walking on a Dream",

    # --- Rock / Metal : modulations par tierces ou sauts ---
    "Mr. Big - To Be With You",
    "Bon Jovi - Blaze of Glory",
    "Bon Jovi - Livin' on a Prayer",
    "Jon Bon Jovi - Blaze of Glory",
    "Scorpions - Wind of Change",
    "Foreigner - Waiting for a Girl Like You",
    "Heart - Alone",
    "Derek and the Dominos - Layla",
    "Derek and the Dominos - Bell Bottom Blues",
    "Europe - The Final Countdown",
    "Guns N' Roses - Sweet Child o' Mine",
    "Motörhead - Damage Case",
    "The Dead Kennedys - Kill the Poor",
    "The Prodigy - Charly",

    # --- R&B / Soul / Gospel : modulations expressives ---
    "The Temptations - My Girl",
    "Stevie Wonder - Lately",
    "Stevie Wonder - Innervisions (album tracks)",
    "Sam Cooke - A Change Is Gonna Come",
    "Dionne Warwick - Do You Know the Way to San Jose",
    "The Stylistics - You Make Me Feel Brand New",
    "The Delfonics - Didn't I (Blow Your Mind This Time)",
    "Sisqó - Thong Song",
    "Boyz II Men - End of the Road",
    "Mariah Carey & Whitney Houston - When You Believe",
    "Kirk Franklin - Stomp",
    "Donnie McClurkin - Stand",

    # --- Variété française / internationale ---
    "Céline Dion - It's All Coming Back to Me Now",
    "Céline Dion - The Power of Love",
    "Céline Dion - My Heart Will Go On",
    "Edith Piaf - Non, je ne regrette rien",
    "Charles Aznavour - La Bohème",
    "Jacques Brel - Ne me quitte pas",
    "Claude François - Comme d'habitude",
    "Michel Sardou - Les Lacs du Connemara",
    "Daniel Balavoine - Le Chanteur",
    "Joe Dassin - Les Champs-Élysées",
    "Frank Sinatra - Something Stupid",
    "Frank Sinatra - My Way",
    "Paul Simon - Still Crazy After All These Years",
    "Johnny Mathis - Chances Are",

    # --- Jazz : modulations par cycles, tierces, ou chromatisme ---
    "John Coltrane - Giant Steps",
    "John Coltrane - Countdown",
    "John Coltrane - Spiral",
    "John Coltrane - Moment's Notice",
    "Jerome Kern - All the Things You Are",
    "Ray Henderson - Bye Bye Blackbird",
    "Arthur Schwartz - Alone Together",
    "Victor Young - Stella by Starlight",
    "Bill Evans - My Foolish Heart",
    "Thelonious Monk - 'Round Midnight",
    "Fats Waller - Black Narcissus",
    "Wayne Shorter - Footprints",
    "Herbie Hancock - Maiden Voyage",
    "Miles Davis - So What",
    "Antônio Carlos Jobim - Girl from Ipanema",

    # --- Malagasy / Océan Indien : à vérifier manuellement ---
    "Mahaleo - Zanaka",
    "Mahaleo - Ravoravo",
    "Jaojoby - Salegy",
    "Rossy - Ravoravo",
    "Tarika - Malagasy Anie Ianao",
    "Justin Vali Trio - Malagasy Folk Dance Medley",
    "Nanie sy Lalie - Feo roa",
    "Rano sy Vary - Mahaleo (Dadah)",

    # --- Comédies musicales / Dessins animés (modulations dramatiques) ---
    "Alan Menken - Beauty and the Beast",
    "Alan Menken - A Whole New World",
    "The Music Man - Lida Rose / Will I Ever Tell You",
    "Frozen - For the First Time in Forever",
    "The Secret of NIMH - Flying Dreams",
    "All Dogs Go to Heaven - You Can't Keep a Good Dog Down",

    # ============================================================
    # --- NOUVEAUX AJOUTS (session 3 - Modulations diverses) ---
    # ============================================================
    
    # --- Pop & Variété Internationale ---
    "Stevie Wonder - I Just Called To Say I Love You",
    "Michael Jackson - Earth Song",
    "Taylor Swift - Getaway Car",
    "Backstreet Boys - Quit Playing Games (With My Heart)",
    "Toni Braxton - Un-Break My Heart",
    "Barry Manilow - Copacabana",
    "Roxette - It Must Have Been Love",
    "Gloria Gaynor - I Will Survive",
    "Rick Astley - Never Gonna Give You Up",
    "The Beatles - Penny Lane",

    # --- Rock & Pop-Rock ---
    "Queen - Bohemian Rhapsody",
    "The Beatles - Lucy in the Sky with Diamonds",
    "Nirvana - Smells Like Teen Spirit",
    "The Beach Boys - Good Vibrations",
    "Elton John - Goodbye Yellow Brick Road",
    "David Bowie - Life on Mars?",
    "The Who - Baba O'Riley",
    "Fleetwood Mac - Go Your Own Way",
    "Bon Jovi - You Give Love a Bad Name",
    "Oasis - Don't Look Back in Anger",

    # --- Variété Française & Chanson ---
    "Céline Dion - S'il suffisait d'aimer",
    "Lara Fabian - Je t'aime",
    "Johnny Hallyday - Allumer le feu",
    "Jean-Jacques Goldman - Là-bas",
    "Michel Berger - La groupie du pianiste",
    "Francis Cabrel - Je l'aime à mourir",
    "Edith Piaf - La Foule",
    "Charles Trenet - La Mer",
    "Mylène Farmer - Désenchantée",
    "Grégory Lemarchal - Écris l'histoire",

    # --- Soul, Funk & R&B ---
    "Aretha Franklin - Respect",
    "Stevie Wonder - Sir Duke",
    "Earth, Wind & Fire - September",
    "Marvin Gaye - What's Going On",
    "Whitney Houston - Saving All My Love for You",
    "Boyz II Men - I'll Make Love to You",
    "Alicia Keys - Fallin'",
    "Bruno Mars - Uptown Funk",
    "Prince - Purple Rain",
    "Chaka Khan - Ain't Nobody",

    # --- Standards de Jazz ---
    "Duke Ellington - Take the 'A' Train",
    "Miles Davis - All Blues",
    "Frank Sinatra - The Way You Look Tonight",
    "Bill Evans - Waltz for Debby",
    "Antonio Carlos Jobim - Desafinado",
    "John Coltrane - Blue Train",
    "Thelonious Monk - Blue Monk",
    "Charlie Parker - Confirmation",

    # ============================================================
    # --- NOUVEAUX AJOUTS (session 4 - Modulations très riches / complexes) ---
    # ============================================================

    # --- Modulations multiples ("en escalier" ou répétées) ---
    "Barry Manilow - Can't Smile Without You",          # Module d'un demi-ton à chaque couplet
    "Stevie Wonder - Golden Lady",                      # Fin en boucle avec modulations ascendantes
    "R.E.M. - Stand",                                   # Modulations répétées à la fin
    "The Beatles - Good Day Sunshine",                  # Nombreux changements de tonalité très rapprochés
    "Burt Bacharach - Anyone Who Had A Heart",          # Labyrinthe tonal (changements de métrique et de clé)

    # --- Pop/Rock à l'harmonie très riche (Verse et Chorus dans des clés éloignées) ---
    "The Beach Boys - God Only Knows",                  # Ambiguïté tonale constante et modulations fluides
    "Toto - Africa",                                    # Couplet en Si majeur, Refrain en Fa# mineur
    "The Carpenters - We've Only Just Begun",           # Ponts avec des modulations très sophistiquées
    "Radiohead - Paranoid Android",                     # Structure en plusieurs actes avec tonalités distinctes
    "Billy Joel - Scenes From An Italian Restaurant",   # Medley avec de vraies ruptures tonales
    "Elton John - Tiny Dancer",                         # Transition harmonique complexe vers le refrain
    
    # --- R&B, Funk & Soul (Modulations jazz-pop complexes) ---
    "Stevie Wonder - Don't You Worry 'Bout a Thing",    # Progressions latines et modulations riches
    "Earth, Wind & Fire - Fantasy",                     # Harmonie extrêmement riche et changeante
    "Chaka Khan - Through the Fire",                    # Modulations typiques de l'âge d'or du R&B
    "Jamiroquai - Virtual Insanity",                    # Changements de centres tonaux très rapides
    "Grover Washington Jr. - Just the Two of Us",       # Séquences harmoniques mouvantes
    "Michael Jackson - Rock With You",                  # Pont très riche menant à une modulation finale

    # --- Prog Rock & Rhapsodies (Ruptures totales de tonalité) ---
    "Queen - The March of the Black Queen",             # Encore plus complexe que Bohemian Rhapsody
    "Genesis - Firth of Fifth",                         # Sauts de tonalités instrumentaux
    "Pink Floyd - The Great Gig in the Sky",            # Progressions d'accords très denses
    "Muse - United States of Eurasia",                  # Influences classiques (Chopin) avec modulations intenses
    "Supertramp - The Logical Song",                    # Changements de tonalité très marqués entre les sections
    
    # --- J-Pop / City Pop (Connues pour leurs modulations frénétiques) ---
    "Mariya Takeuchi - Plastic Love",                   # Harmonie City Pop très dense
    "Utada Hikaru - First Love",                        # Progressions J-Pop avec modulations typiques
]

# ============================================================
# MANIFESTE : requête → fichier WAV
# ============================================================
def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

def already_downloaded(track, manifest):
    """Vérifie si le morceau a déjà été téléchargé et que le fichier existe."""
    entry = manifest.get(track)
    if not entry:
        return None
    wav_path = entry.get("path")
    if wav_path and os.path.exists(wav_path):
        return wav_path
    return None

# ============================================================
# TÉLÉCHARGEMENT
# ============================================================
def download_track(track, manifest):
    """Télécharge un morceau, retourne le chemin du WAV ou None."""

    # Callback pour capturer le vrai nom de fichier après post-processing
    captured = {"path": None}

    def _hook(d):
        if d.get("status") == "finished":
            # 'filepath' pointe vers le WAV après FFmpegExtractAudio
            filepath = d.get("info_dict", {}).get("filepath") or d.get("filename")
            if filepath:
                # yt-dlp peut renvoyer le chemin du fichier temporaire
                # mais 'filepath' après post-processing est le WAV final
                if filepath.endswith(".wav"):
                    captured["path"] = filepath
                else:
                    base = os.path.splitext(filepath)[0]
                    if os.path.exists(base + ".wav"):
                        captured["path"] = base + ".wav"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(OUTPUT_DIR, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        # --- Anti re-téléchargement ---
        "download_archive": ARCHIVE_FILE,
        "noplaylist": True,
        "progress_hooks": [_hook],
        "ignoreerrors": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{track}"])
    except Exception as e:
        print(f"   ❌ Erreur yt-dlp : {e}")
        return None

    # Fallback : si le hook n'a rien capturé, on cherche par correspondance
    if not captured["path"]:
        safe = track.replace("/", "_")
        candidates = [f for f in os.listdir(OUTPUT_DIR)
                      if f.endswith(".wav") and safe.split()[0] in f]
        if candidates:
            # On prend le plus récent
            candidates.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
                            reverse=True)
            captured["path"] = os.path.join(OUTPUT_DIR, candidates[0])

    return captured["path"]

# ============================================================
# MAIN
# ============================================================
def main():
    manifest = load_manifest()
    total    = len(tracks)
    skipped  = 0
    new_dl   = 0
    failed   = []

    print(f"📂 Dossier cible : {os.path.abspath(OUTPUT_DIR)}")
    print(f"📋 {total} titres dans la liste\n")

    for i, track in enumerate(tracks, 1):
        # 1) Vérifie le manifeste (le plus fiable : on connaît le fichier exact)
        cached = already_downloaded(track, manifest)
        if cached:
            print(f"[{i}/{total}] ⏩ Déjà téléchargé : {track}")
            print(f"         → {os.path.basename(cached)}")
            skipped += 1
            continue

        # 2) Sinon, télécharge
        print(f"[{i}/{total}] ⬇️  Téléchargement : {track}")
        wav_path = download_track(track, manifest)

        if wav_path and os.path.exists(wav_path):
            manifest[track] = {"path": wav_path, "size_mb": round(os.path.getsize(wav_path) / 1e6, 2)}
            save_manifest(manifest)
            print(f"         ✅ {os.path.basename(wav_path)} "
                  f"({manifest[track]['size_mb']} Mo)")
            new_dl += 1
        else:
            print(f"         ⚠️  Échec (fichier non trouvé)")
            failed.append(track)

    # ============================================================
    # BILAN
    # ============================================================
    print(f"\n{'='*60}")
    print(f"BILAN")
    print(f"{'='*60}")
    print(f"  Total demandés   : {total}")
    print(f"  Déjà présents    : {skipped}")
    print(f"  Nouvellement dl  : {new_dl}")
    print(f"  Échecs           : {len(failed)}")
    if failed:
        print(f"\n  Titres en échec :")
        for t in failed:
            print(f"    - {t}")
    print(f"\n📄 Manifeste : {MANIFEST_FILE}")
    print(f"📄 Archive   : {ARCHIVE_FILE}")

if __name__ == "__main__":
    main()