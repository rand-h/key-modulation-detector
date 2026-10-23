# annotate.py — Annotation interactive avec curseur déplaçable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import soundfile as sf
import json
import os
import sys
import threading

try:
    import sounddevice as sd
except (ImportError, OSError) as e:
    print(f"❌ sounddevice : {e}")
    print("   pip install sounddevice")
    sys.exit(1)

ANN_FILE       = 'annotations.json'
AUDIO_DIR      = 'dataset/audio'
DISPLAY_POINTS = 8000
VALID_NOTES    = ['C','C#','Db','D','D#','Eb','E','F','F#','Gb',
                  'G','G#','Ab','A','A#','Bb','B']


# ============================================================
# UTILITAIRES
# ============================================================
def fmt(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:05.2f}"


def get_key(prompt):
    """Demande une tonalité. Accepte C, Cm, C:maj, F#:min, Bb..."""
    while True:
        k = input(prompt).strip()
        if k.lower() in ('', 'q'):
            return None
        if ':' in k:
            note, mode = k.split(':', 1)
            note, mode = note.strip(), mode.strip().lower()
        elif k.endswith('m') and len(k) > 1:
            note, mode = k[:-1], 'min'
        else:
            note, mode = k, 'maj'
        if note not in VALID_NOTES:
            print(f"   ⚠️  Note invalide. Exemples : C, Cm, C:maj, F#:min, Bb")
            continue
        if mode in ('m', 'min', 'minor', 'mineur'):
            mode = 'min'
        elif mode in ('maj', 'major', 'majeur', ''):
            mode = 'maj'
        else:
            print(f"   ⚠️  Mode doit être 'maj' ou 'min'")
            continue
        return f"{note}:{mode}"


# ============================================================
# LECTEUR AUDIO (seek-able)
# ============================================================
class AudioPlayer:
    def __init__(self, path):
        self.data, self.sr = sf.read(path, dtype='float32', always_2d=False)
        if self.data.ndim > 1:
            self.data = self.data.mean(axis=1)
        peak = float(np.max(np.abs(self.data)))
        if peak > 0:
            self.data = self.data / peak * 0.9
        self.pos = 0
        self.playing = False
        self.stream = None
        self._lock = threading.Lock()

    def start(self):
        if self.playing:
            return

        def callback(outdata, frames, time_info, status):
            with self._lock:
                pos = self.pos
            remaining = len(self.data) - pos
            if remaining <= 0:
                outdata[:] = 0
                raise sd.CallbackStop()
            n = min(frames, remaining)
            outdata[:n, 0] = self.data[pos:pos+n]
            if n < frames:
                outdata[n:, 0] = 0
            with self._lock:
                self.pos = pos + n

        self.stream = sd.OutputStream(
            samplerate=self.sr, channels=1, callback=callback,
            blocksize=1024, dtype='float32',
        )
        self.stream.start()
        self.playing = True

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.playing = False

    def seek(self, seconds):
        seconds = max(0.0, min(seconds, self.duration - 0.01))
        with self._lock:
            self.pos = int(seconds * self.sr)

    @property
    def current_time(self):
        with self._lock:
            return self.pos / self.sr

    @property
    def duration(self):
        return len(self.data) / self.sr


# ============================================================
# UI MATPLOTLIB
# ============================================================
class AnnotatorUI:
    def __init__(self, audio_path, filename):
        self.filename = filename
        self.player = AudioPlayer(audio_path)
        self.markers = []
        self.selected_marker = None
        self.dragging = False

        # Downsample waveform pour l'affichage
        wf = self.player.data
        if len(wf) > DISPLAY_POINTS:
            step = len(wf) // DISPLAY_POINTS
            wf = wf[:step * DISPLAY_POINTS].reshape(-1, step).max(axis=1)
        self.display_waveform = wf

        # Création figure
        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.fig.subplots_adjust(bottom=0.12, top=0.88)

        times = np.linspace(0, self.player.duration, len(self.display_waveform))
        self.ax.plot(times, self.display_waveform, color='steelblue', linewidth=0.5)
        self.ax.set_xlim(0, self.player.duration)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlabel('Temps (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(0, color='gray', linewidth=0.5)

        title = (f'{filename}\n'
                 f'ESPACE: Play/Pause  ·  Clic gauche: Seek  ·  '
                 f'Clic droit: Marqueur  ·  Glisser: Ajuster  ·  '
                 f'Entrée: Valider  ·  Échap: Annuler  ·  Molette: Zoom')
        self.ax.set_title(title, fontsize=9, loc='left')

        # Playhead (ligne verte qui suit la lecture)
        self.playhead = self.ax.axvline(0, color='green', linewidth=1.8, alpha=0.75, zorder=10)

        # Animation
        self.anim = animation.FuncAnimation(
            self.fig, self.update_playhead, interval=50,
            blit=False, cache_frame_data=False,
        )

        # Events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        self.cancelled = False
        self.done = False

    # ---------- Animation ----------
    def update_playhead(self, frame):
        if self.player.playing:
            if not (self.player.stream and self.player.stream.active):
                self.player.playing = False
                return
            pos = self.player.current_time
            self.playhead.set_xdata([pos, pos])
            self.fig.canvas.draw_idle()

    # ---------- Événements souris ----------
    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        if event.button == 1:  # clic gauche = seek ou drag
            for m in self.markers:
                if abs(m['time'] - event.xdata) < max(0.3, (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) / 100):
                    self.selected_marker = m
                    self.dragging = True
                    return
            self.player.seek(event.xdata)

        elif event.button == 3:  # clic droit = ajout marqueur
            self.add_marker(event.xdata)

    def on_motion(self, event):
        if not self.dragging or not self.selected_marker:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        t = max(0.1, min(event.xdata, self.player.duration - 0.1))
        self.selected_marker['time'] = t
        self.selected_marker['line'].set_xdata([t, t])
        self.selected_marker['label'].set_position((t, 1.05))
        self.selected_marker['label'].set_text(fmt(t))
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.selected_marker = None
            self.markers.sort(key=lambda m: m['time'])

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        cur = self.ax.get_xlim()
        xdata = event.xdata
        scale = 0.8 if event.button == 'up' else 1.25
        new_width = (cur[1] - cur[0]) * scale
        rel = (xdata - cur[0]) / (cur[1] - cur[0])
        left = xdata - rel * new_width
        self.ax.set_xlim(left, left + new_width)
        self.fig.canvas.draw_idle()

    def on_close(self, event):
        self.cancelled = True
        self.player.stop()

    # ---------- Événements clavier ----------
    def on_key(self, event):
        if event.key == ' ':
            if self.player.playing:
                self.player.stop()
            else:
                if self.player.current_time >= self.player.duration - 0.1:
                    self.player.seek(0)
                self.player.start()

        elif event.key == 'enter':
            self.done = True
            self.player.stop()
            plt.close(self.fig)

        elif event.key == 'escape':
            self.cancelled = True
            self.player.stop()
            plt.close(self.fig)

        elif event.key in ('backspace', 'delete'):
            if self.markers:
                m = self.markers.pop()
                m['line'].remove()
                m['label'].remove()
                self.fig.canvas.draw_idle()

        elif event.key == 'm':
            self.add_marker(self.player.current_time)

    # ---------- Gestion des marqueurs ----------
    def add_marker(self, time):
        if time <= 0 or time >= self.player.duration:
            return
        for m in self.markers:
            if abs(m['time'] - time) < 0.2:
                return
        line = self.ax.axvline(time, color='red', linewidth=1.5,
                                linestyle='--', alpha=0.85, zorder=5)
        label = self.ax.text(time, 1.05, fmt(time), ha='center', va='bottom',
                              color='red', fontsize=7, fontweight='bold', zorder=6)
        self.markers.append({'time': time, 'line': line, 'label': label})
        self.markers.sort(key=lambda m: m['time'])
        self.fig.canvas.draw_idle()

    # ---------- Run ----------
    def run(self):
        print(f"\n   💡 Raccourcis :")
        print(f"      ESPACE       → Play / Pause")
        print(f"      Clic gauche  → Déplacer le curseur (seek)")
        print(f"      Clic droit   → Ajouter un marqueur de modulation")
        print(f"      Glisser      → Ajuster un marqueur existant")
        print(f"      M            → Ajouter marqueur à la position actuelle")
        print(f"      Backspace    → Annuler le dernier marqueur")
        print(f"      Molette      → Zoomer / Dézoomer")
        print(f"      ENTRÉE       → Valider et passer à l'attribution")
        print(f"      ÉCHAP        → Annuler ce morceau\n")

        plt.show()

        if self.cancelled or not self.done:
            return None
        return sorted([m['time'] for m in self.markers])


# ============================================================
# ANNOTATION D'UN MORCEAU
# ============================================================
def annotate_track(audio_path, filename):
    ui = AnnotatorUI(audio_path, filename)
    markers = ui.run()
    if markers is None:
        return None

    sections = [0.0] + markers + [ui.player.duration]
    n_sections = len(sections) - 1

    print(f"\n   🎼 Attribution des tonalités ({n_sections} section(s)) :")
    print(f"   Format : C, Cm, C:maj, F#:min, Bb...\n")

    annotations = []
    for i in range(n_sections):
        start, end = sections[i], sections[i+1]
        print(f"   Section {i+1}/{n_sections} : {fmt(start)} → {fmt(end)}")
        key = get_key("   → ")
        if key is None:
            print(f"   ⏭  Annulé")
            return None
        annotations.append({"time": round(start, 2), "key": key})
        print(f"   ✓ {key}")

    return annotations


# ============================================================
# FICHIERS D'ANNOTATION
# ============================================================
def load_annotations():
    if os.path.exists(ANN_FILE):
        with open(ANN_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_annotations(data):
    with open(ANN_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.isdir(AUDIO_DIR):
        print(f"❌ Dossier introuvable : {AUDIO_DIR}")
        return

    wavs = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')])
    if not wavs:
        print(f"❌ Aucun .wav dans {AUDIO_DIR}")
        return

    annotations = load_annotations()
    redo = '--redo' in sys.argv
    start_from = None
    for a in sys.argv:
        if a.startswith('--start='):
            start_from = a.split('=', 1)[1]

    to_do = [w for w in wavs if redo or w not in annotations]

    if start_from:
        try:
            to_do = to_do[to_do.index(start_from):]
        except ValueError:
            print(f"⚠️  '{start_from}' n'est pas dans la liste à annoter")

    print(f"{'='*60}")
    print(f"🎵 ANNOTATION DE TONALITÉ")
    print(f"{'='*60}")
    print(f"Dossier    : {os.path.abspath(AUDIO_DIR)}")
    print(f"Total      : {len(wavs)}")
    print(f"Annotés    : {len(annotations)}")
    print(f"À annoter  : {len(to_do)}")
    if redo:
        print(f"Mode REDO  : tout sera réannoté")
    print(f"{'='*60}")

    if not to_do:
        print(f"\n✅ Rien à faire. --redo pour tout refaire.")
        return

    for i, wav in enumerate(to_do, 1):
        path = os.path.join(AUDIO_DIR, wav)
        print(f"\n\n{'█'*60}")
        print(f"  [{i}/{len(to_do)}] {wav}")
        print(f"{'█'*60}")

        try:
            result = annotate_track(path, wav)
            if result is not None:
                annotations[wav] = result
                save_annotations(annotations)
                print(f"\n   💾 Sauvegardé ({len(result)} section(s))")
            else:
                print(f"\n   ⏭  Non sauvegardé")
        except KeyboardInterrupt:
            print(f"\n   ⏸  Interruption")
            break
        except Exception as e:
            import traceback
            print(f"\n   ❌ Erreur : {e}")
            traceback.print_exc()

        if i < len(to_do):
            cont = input("\n   Continuer ? [O/n] ").strip().lower()
            if cont == 'n':
                break

    # Bilan
    print(f"\n\n{'='*60}")
    print(f"✅ {len(annotations)} morceau(x) annoté(s)")
    print(f"📄 {os.path.abspath(ANN_FILE)}")

    total_mod = sum(len(a) - 1 for a in annotations.values())
    no_mod = sum(1 for a in annotations.values() if len(a) == 1)
    print(f"\n📊 Statistiques :")
    print(f"   Sans modulation : {no_mod}")
    print(f"   Avec modulation : {len(annotations) - no_mod}")
    print(f"   Modulations     : {total_mod}")


if __name__ == '__main__':
    main()