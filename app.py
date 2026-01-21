import re
import json
import shutil
import hashlib
import tempfile
import subprocess
import textwrap
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

# --- FIX Pillow>=10 (MoviePy usa Image.ANTIALIAS en algunas versiones) ---
if not hasattr(Image, "ANTIALIAS"):
    try:
        Image.ANTIALIAS = Image.Resampling.LANCZOS  # Pillow>=10
    except Exception:
        Image.ANTIALIAS = Image.LANCZOS

from moviepy.editor import (
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
)
from moviepy.audio.fx.all import audio_loop, audio_fadein, audio_fadeout, volumex
import imageio_ffmpeg


# =========================
# CONFIG
# =========================
HEADERS_FAKE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": "https://www.eltiempo.com/",
}

RES_W, RES_H = 1920, 1080
DEFAULT_FPS = 15

DEFAULT_SLIDE_DURATION = 7.0
MIN_SLIDE_DURATION_WITH_VOICE = 3.5

ELEVEN_BASE = "https://api.elevenlabs.io"

SLIDE_TEXT_MAX_CHARS = 60
SLIDE_TEXT_MAX_SENTENCES = 1

RENDER_WRAP_WIDTH = 30
RENDER_LINES_PER_IMAGE = 2

FONT_SIZE = 50

# ✅ mínimo de imágenes
MIN_IMAGES_REQUIRED = 5

# Voces
VOICE_OPTIONS = {
    "Luisa": "7nCYbNPCi8RLAKVnYEoO",
    "JC News": "4XUsiqPDK4UACIM2BILe",
    "Juan": "WOSzFvlJRm2hkYb3KA5w",
    "Isabella": "p18tR9wFA5Ng9WhfWI0o",
    "El Faraón": "W1hAcdh0RNsPYUA7fkJh",
    "Fernando": "dlGxemPxFMTY7iXagmOj",
}


# =========================
# PIN gate
# =========================
def require_pin_if_configured():
    app_pin = st.secrets.get("APP_PIN", "")
    if not app_pin:
        return

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.warning("🔒 Esta app está protegida con PIN.")
    pin = st.text_input("Ingresa el PIN", type="password")
    if st.button("Entrar", type="primary"):
        if pin == app_pin:
            st.session_state.authenticated = True
            st.success("Acceso concedido ✅")
            st.rerun()
        else:
            st.error("PIN incorrecto ❌")

    st.stop()


# =========================
# TEXT NORMALIZATION + SEGMENTATION
# =========================
QUOTES_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "´": "'",
        "…": "...",
        "\u00A0": " ",
    }
)

ABREVIATURAS = [
    "Sr.",
    "Sra.",
    "Dr.",
    "Dra.",
    "Ing.",
    "Lic.",
    "No.",
    "Nro.",
    "Art.",
    "Pág.",
    "p. ej.",
    "p.ej.",
    "etc.",
    "EE. UU.",
    "U.S.",
    "vs.",
]


def normalizar_texto(texto: str) -> str:
    if not texto:
        return ""
    t = texto.translate(QUOTES_MAP)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t


def dividir_en_frases(texto: str) -> list[str]:
    t = normalizar_texto(texto)
    if not t:
        return []

    protect = {}
    for i, ab in enumerate(ABREVIATURAS):
        key = f"__ABR{i}__"
        protect[key] = ab
        t = t.replace(ab, key)

    partes = re.split(r"(?<=[\.\!\?])\s+", t)

    frases = []
    for p in partes:
        for k, ab in protect.items():
            p = p.replace(k, ab)
        p = p.strip()
        if p:
            frases.append(p)
    return frases


def segmentar_para_slides(texto: str, max_chars: int, max_sentences: int) -> list[str]:
    frases = dividir_en_frases(texto)
    if not frases:
        t = normalizar_texto(texto)
        return [t[:max_chars]] if t else []

    slides = []
    buf = []

    def flush():
        nonlocal buf
        if buf:
            slides.append(" ".join(buf).strip())
            buf = []

    for fr in frases:
        fr = fr.strip()
        if not fr:
            continue

        cand = (" ".join(buf + [fr])).strip()

        if len(cand) <= max_chars and len(buf) < max_sentences:
            buf.append(fr)
            if len(buf) >= max_sentences:
                flush()
            continue

        flush()

        if len(fr) > max_chars:
            words = fr.split()
            tmp = []
            for w in words:
                cand2 = (" ".join(tmp + [w])).strip()
                if len(cand2) <= max_chars:
                    tmp.append(w)
                else:
                    if tmp:
                        slides.append(" ".join(tmp))
                    tmp = [w]
            if tmp:
                slides.append(" ".join(tmp))
        else:
            buf = [fr]

    flush()
    return [s for s in slides if s.strip()]


def split_paragraphs_from_manual(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    return [p for p in parts if len(normalizar_texto(p)) >= 10]


# =========================
# FONT
# =========================
def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


# =========================
# SCRAPER ELTIEMPO (cached)
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def extraer_contenido_articulo(url: str) -> tuple[str, list[str], list[str]]:
    r = requests.get(url, headers=HEADERS_FAKE, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")

    titulo_el = soup.find("h1", class_="c-articulo__titulo") or soup.find("h1")
    if not titulo_el:
        raise ValueError("No se pudo encontrar el título del artículo.")
    titulo = titulo_el.get_text(" ", strip=True)

    cuerpo = soup.find("div", class_="c-cuerpo") or soup.find("article")
    if not cuerpo:
        raise ValueError("No se pudo encontrar el cuerpo del artículo.")

    parrafos = []
    divs = cuerpo.find_all("div", class_="paragraph")
    for d in divs:
        t = d.get_text(" ", strip=True)
        if t:
            parrafos.append(t)

    if not parrafos:
        for p in cuerpo.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) > 50:
                parrafos.append(t)

    imagenes_urls = set()

    def limpiar_url(u):
        if not u:
            return None
        u = u.strip()
        if u.startswith("//"):
            return "https:" + u
        if u.startswith("/"):
            return "https://www.eltiempo.com" + u
        return u

    def agregar_img(tag):
        if not tag:
            return
        u = limpiar_url(tag.get("data-full-src") or tag.get("src"))
        if u and (not u.lower().endswith(".svg")) and ("icon" not in u.lower()):
            imagenes_urls.add(u)

    apertura = soup.find("div", class_="c-articulo-apertura__media__thumb")
    if apertura:
        agregar_img(apertura.find("img"))

    for media in cuerpo.find_all(
        ["figure", "div"],
        class_=["c-cuerpo__media__thumb", "c-cuerpo__media", "c-detail__media"],
    ):
        for img in media.find_all("img"):
            agregar_img(img)
        zoom = media.find("div", class_="c-cuerpo__media__thumb__zoom")
        if zoom:
            agregar_img(zoom.find("img"))

    galeria = soup.find("div", class_="c-galeria")
    if galeria:
        for img in galeria.find_all("img"):
            agregar_img(img)

    return titulo, parrafos, list(imagenes_urls)


# =========================
# IMAGES: download + render
# =========================
def ajustar_imagen(imagen: Image.Image) -> Image.Image:
    relacion = imagen.width / max(1, imagen.height)
    nueva_w = int(RES_H * relacion)
    return imagen.resize((nueva_w, RES_H), Image.LANCZOS)


def descargar_imagenes(urls: list[str], out_dir: Path, max_workers: int = 10) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch(u: str) -> Path | None:
        try:
            resp = requests.get(u, headers=HEADERS_FAKE, timeout=20)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            img = ajustar_imagen(img).convert("RGB")
            name = hashlib.md5(u.encode("utf-8")).hexdigest()
            p = out_dir / f"img_{name}.jpg"
            img.save(p, quality=92)
            return p
        except Exception:
            return None

    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(fetch, u) for u in urls]
        for f in as_completed(futs):
            p = f.result()
            if p:
                paths.append(p)
    return paths


def guardar_imagenes_subidas(uploaded_files, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uf in (uploaded_files or []):
        try:
            img = Image.open(BytesIO(uf.read()))
            img = ajustar_imagen(img).convert("RGB")
            safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", uf.name)
            pth = out_dir / f"upload_{safe_name}"
            img.save(pth, quality=92)
            paths.append(pth)
        except Exception:
            continue
    return paths


def render_slide(imagen_path: Path, texto: str, idx: int, out_dir: Path, font_size: int = FONT_SIZE) -> list[Path]:
    imagen = Image.open(imagen_path).convert("RGB")
    imagen = ajustar_imagen(imagen)

    fondo = Image.new("RGB", (RES_W, RES_H), color="black")
    pos = ((RES_W - imagen.width) // 2, (RES_H - imagen.height) // 2)
    fondo.paste(imagen, pos)

    texto = (texto or "").strip()
    if not texto:
        out = out_dir / f"slide_{idx:04d}.jpg"
        fondo.save(out, quality=92)
        return [out]

    fuente = load_font(font_size)

    lineas = textwrap.wrap(
        re.sub(r"\s+", " ", texto).strip(),
        width=RENDER_WRAP_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    )

    bloques = [lineas[i : i + RENDER_LINES_PER_IMAGE] for i in range(0, len(lineas), RENDER_LINES_PER_IMAGE)]
    outs = []

    for j, bloque in enumerate(bloques):
        base = fondo.copy()
        d = ImageDraw.Draw(base)

        heights = []
        for linea in bloque:
            bbox = d.textbbox((0, 0), linea, font=fuente)
            heights.append(bbox[3] - bbox[1])

        total_h = sum(heights) + 10 * (len(bloque) - 1)
        y0 = RES_H - total_h - 110
        y = y0

        for linea, h in zip(bloque, heights):
            bbox = d.textbbox((0, 0), linea, font=fuente)
            w = bbox[2] - bbox[0]
            x = (RES_W - w) // 2

            d.text(
                (x, y),
                linea,
                font=fuente,
                fill="#ffffff",
                stroke_width=3,
                stroke_fill="#000000",
            )
            y += h + 10

        out = out_dir / f"slide_{idx:04d}_{j:02d}.jpg"
        base.save(out, quality=92)
        outs.append(out)

    return outs


# =========================
# ElevenLabs: long TTS
# =========================
def model_char_limit(model_id: str) -> int:
    if model_id == "eleven_multilingual_v2":
        return 9500
    if model_id in ("eleven_flash_v2_5", "eleven_turbo_v2_5"):
        return 39000
    if model_id == "eleven_v3":
        return 4800
    return 4500


def split_text_chunks(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        cand = (cur + " " + s).strip() if cur else s
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            if len(s) > max_chars:
                words = s.split()
                tmp = ""
                for w in words:
                    cand2 = (tmp + " " + w).strip() if tmp else w
                    if len(cand2) <= max_chars:
                        tmp = cand2
                    else:
                        if tmp:
                            chunks.append(tmp)
                        tmp = w
                if tmp:
                    chunks.append(tmp)
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def eleven_tts_long_to_mp3(
    text: str,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    voice_settings: dict,
    out_mp3: Path,
    work_dir: Path,
) -> Path:
    max_chars = model_char_limit(model_id)
    chunks = split_text_chunks(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("Texto vacío para narración.")

    parts_dir = work_dir / "tts_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    part_files = []
    for i, chunk in enumerate(chunks, start=1):
        url = f"{ELEVEN_BASE}/v1/text-to-speech/{voice_id}"
        params = {"output_format": output_format}
        payload = {"text": chunk, "model_id": model_id, "voice_settings": voice_settings}

        r = requests.post(
            url,
            params=params,
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120,
        )

        if r.status_code >= 400:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:400]
            raise RuntimeError(f"ElevenLabs error {r.status_code}: {detail}")

        p = parts_dir / f"part_{i:04d}.mp3"
        p.write_bytes(r.content)
        part_files.append(p)

    list_file = (parts_dir / "concat_list.txt").resolve()
    with list_file.open("w", encoding="utf-8") as f:
        for p in part_files:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    out_mp3 = out_mp3.resolve()

    cmd = [ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_mp3)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        cmd2 = [
            ffmpeg_exe(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(out_mp3),
        ]
        proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc2.returncode != 0:
            raise RuntimeError("ffmpeg falló concatenando audio:\n" + (proc2.stderr[-1500:] or proc.stderr[-1500:]))

    return out_mp3


# =========================
# AUDIO: build final audio clip (voice/music)
# =========================
def build_final_audio_clip(
    voice_path: Path | None,
    music_path: Path | None,
    target_duration: float,
    voice_volume: float,
    music_volume: float,
    fade_sec: float,
):
    if not voice_path and not music_path:
        return None

    fade = min(max(0.0, fade_sec), max(0.0, target_duration / 4.0))

    voice_clip = None
    if voice_path and voice_path.exists():
        voice_clip = AudioFileClip(str(voice_path))
        voice_clip = volumex(voice_clip, voice_volume)

    music_clip = None
    if music_path and music_path.exists():
        music_clip = AudioFileClip(str(music_path))
        music_clip = audio_loop(music_clip, duration=target_duration)
        music_clip = volumex(music_clip, music_volume)
        if fade > 0:
            music_clip = audio_fadein(music_clip, fade)
            music_clip = audio_fadeout(music_clip, fade)

    if voice_clip and music_clip:
        return CompositeAudioClip([music_clip, voice_clip]).set_duration(target_duration)

    if voice_clip:
        return voice_clip

    if music_clip:
        return music_clip

    return None


# =========================
# END CLIP (CIERRE) + FIT TO 1080P
# =========================
def fit_clip_to_1080p(clip: VideoFileClip) -> VideoFileClip:
    c = clip.resize(height=RES_H)
    if c.w > RES_W:
        x1 = (c.w - RES_W) / 2
        x2 = x1 + RES_W
        c = c.crop(x1=x1, x2=x2)
    elif c.w < RES_W:
        c = c.on_color(size=(RES_W, RES_H), color=(0, 0, 0), pos=("center", "center"))
    if c.h != RES_H:
        c = c.resize(height=RES_H)
    return c


def save_uploaded_video(uploaded_file, out_dir: Path, name: str) -> Path | None:
    if not uploaded_file:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    p = out_dir / f"{name}{suffix}"
    p.write_bytes(uploaded_file.getvalue())
    return p


def save_uploaded_audio(uploaded_file, out_dir: Path) -> Path | None:
    if not uploaded_file:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", uploaded_file.name)
    p = out_dir / f"bgm_{safe_name}"
    p.write_bytes(uploaded_file.getvalue())
    return p


def save_uploaded_png(uploaded_file, out_dir: Path, name: str) -> Path | None:
    if not uploaded_file:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.png"
    p.write_bytes(uploaded_file.getvalue())
    return p


# =========================
# VIDEO
# =========================
def safe_filename(title: str, max_len: int = 60) -> str:
    s = "".join(ch for ch in (title or "") if ch.isalnum() or ch in " _-").strip()
    s = s[:max_len].strip()
    return s or "video"


def get_audio_duration(path: Path) -> float | None:
    if not path or not path.exists():
        return None
    clip = None
    try:
        clip = AudioFileClip(str(path))
        return float(clip.duration)
    except Exception:
        return None
    finally:
        try:
            if clip:
                clip.close()
        except Exception:
            pass


def add_logo_overlay(main_clip, logo_png_path: Path, position: str, scale: float, margin: int):
    """
    Overlay PNG con transparencia sobre el video principal.
    Se aplica SOLO al main_clip (no al cierre).
    """
    if not logo_png_path or not logo_png_path.exists():
        return main_clip

    logo = (
        ImageClip(str(logo_png_path))
        .set_duration(main_clip.duration)
        .resize(scale)
        .set_position(position)
    )

    # margin: ajustamos usando posiciones tipo ("right","top") con padding manual
    # MoviePy permite funciones para posición.
    def pos_func(t):
        x, y = 0, 0
        if position == "top-left":
            x, y = margin, margin
        elif position == "top-right":
            x = main_clip.w - logo.w - margin
            y = margin
        elif position == "bottom-left":
            x = margin
            y = main_clip.h - logo.h - margin
        elif position == "bottom-right":
            x = main_clip.w - logo.w - margin
            y = main_clip.h - logo.h - margin
        else:
            x = main_clip.w - logo.w - margin
            y = margin
        return (x, y)

    logo = logo.set_position(pos_func)
    return CompositeVideoClip([main_clip, logo], size=(main_clip.w, main_clip.h))


def crear_video(
    textos_slides: list[str],
    imagenes: list[Path],
    titulo: str,
    overlay_text: bool,
    voice_path: Path | None,
    music_path: Path | None,
    voice_volume: float,
    music_volume: float,
    music_fade: float,
    base_slide_duration: float,
    fps: int,
    work_dir: Path,
    cierre_video_path: Path | None,
    logo_png_path: Path | None,
    logo_position: str,
    logo_scale: float,
    logo_margin: int,
) -> Path:
    slides_dir = work_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    # 1) Render slides
    slide_imgs: list[Path] = []
    for idx, txt in enumerate(textos_slides):
        img_path = imagenes[idx % len(imagenes)]
        slide_imgs.extend(render_slide(img_path, txt if overlay_text else "", idx, slides_dir, font_size=FONT_SIZE))

    if not slide_imgs:
        raise ValueError("No se generaron slides (slide_imgs vacío).")

    # 2) Duración
    voice_duration = get_audio_duration(voice_path) if voice_path else None

    if voice_duration and voice_duration > 0:
        per_slide = max(MIN_SLIDE_DURATION_WITH_VOICE, voice_duration / len(slide_imgs))
        main_duration = voice_duration
    else:
        per_slide = float(base_slide_duration)
        main_duration = per_slide * len(slide_imgs)

    # 3) Main video
    image_clips = [ImageClip(str(p)).set_duration(per_slide) for p in slide_imgs]
    main_video = concatenate_videoclips(image_clips, method="compose")

    # 4) Audio main
    target_audio_duration = float(voice_duration) if voice_duration and voice_duration > 0 else float(main_duration)
    audio_clip = build_final_audio_clip(
        voice_path=voice_path if (voice_path and voice_path.exists()) else None,
        music_path=music_path if (music_path and music_path.exists()) else None,
        target_duration=target_audio_duration,
        voice_volume=voice_volume,
        music_volume=music_volume,
        fade_sec=music_fade,
    )
    if audio_clip:
        main_video = main_video.set_audio(audio_clip).set_duration(target_audio_duration)
    else:
        main_video = main_video.set_duration(main_duration)

    # 5) Logo overlay SOLO en el video principal (hasta antes del cierre)
    if logo_png_path and logo_png_path.exists():
        main_video = add_logo_overlay(
            main_video,
            logo_png_path=logo_png_path,
            position=logo_position,
            scale=logo_scale,
            margin=logo_margin,
        )

    # 6) Append CIERRE (sin logo)
    final_clips = [main_video]
    cierre_clip = None

    if cierre_video_path and cierre_video_path.exists():
        cierre_clip = VideoFileClip(str(cierre_video_path))
        cierre_clip = fit_clip_to_1080p(cierre_clip)
        final_clips.append(cierre_clip)

    final = concatenate_videoclips(final_clips, method="compose")

    out = work_dir / f"{safe_filename(titulo)}.mp4"

    # 7) Encoding optimizado para Streamlit Cloud
    final.write_videofile(
        str(out),
        fps=int(fps),
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=2,
        bitrate="1500k",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        logger=None,
    )

    # cleanup
    try:
        final.close()
    except Exception:
        pass
    try:
        main_video.close()
    except Exception:
        pass
    for c in image_clips:
        try:
            c.close()
        except Exception:
            pass
    try:
        if audio_clip:
            audio_clip.close()
    except Exception:
        pass
    try:
        if cierre_clip:
            cierre_clip.close()
    except Exception:
        pass

    return out


# =========================
# HELPERS
# =========================
def build_textos_slides(include_title: bool, titulo: str, selected_pars: list[str]) -> list[str]:
    textos_slides: list[str] = []
    if include_title:
        t = normalizar_texto(titulo)
        if t:
            textos_slides.extend(segmentar_para_slides(t, SLIDE_TEXT_MAX_CHARS, SLIDE_TEXT_MAX_SENTENCES))
    for p in selected_pars:
        p = normalizar_texto(p)
        if p:
            textos_slides.extend(segmentar_para_slides(p, SLIDE_TEXT_MAX_CHARS, SLIDE_TEXT_MAX_SENTENCES))
    return [t for t in textos_slides if t.strip()]


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Video + Voz", layout="wide")
st.title("Generador de Video con Narración (ElevenLabs)")

require_pin_if_configured()

with st.sidebar:
    st.header("ElevenLabs")

    secret_key = st.secrets.get("ELEVENLABS_API_KEY", "")
    use_secrets = st.checkbox("Usar API Key de Secrets", value=bool(secret_key))
    if use_secrets and secret_key:
        api_key = secret_key
        st.caption("Usando API Key desde Secrets.")
    else:
        api_key = st.text_input("API Key", type="password")

    model_id = st.selectbox(
        "Modelo",
        ["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"],
        index=0,
    )

    output_format = st.selectbox("Formato (MP3)", ["mp3_44100_128", "mp3_44100_192"], index=0)

    # ✅ Voice ID en desplegable
    voice_name = st.selectbox("Voz", list(VOICE_OPTIONS.keys()), index=1)  # default JC News
    voice_id_direct = VOICE_OPTIONS[voice_name]
    st.caption(f"voice_id = {voice_id_direct}")

    st.subheader("Voice settings")
    stability = st.slider("stability", 0.0, 1.0, 0.45, 0.01)
    similarity = st.slider("similarity_boost", 0.0, 1.0, 0.75, 0.01)
    style = st.slider("style", 0.0, 1.0, 0.20, 0.01)
    speed = st.slider("speed", 0.7, 1.3, 1.0, 0.01)
    use_boost = st.checkbox("use_speaker_boost", value=True)

voice_settings = {
    "stability": stability,
    "similarity_boost": similarity,
    "style": style,
    "use_speaker_boost": use_boost,
    "speed": speed,
}

st.divider()

modo = st.radio(
    "¿Cómo quieres ingresar el contenido?",
    ["Desde URL de El Tiempo", "Texto e imágenes manual"],
    horizontal=True,
)

output_mode = st.radio(
    "¿Cómo quieres el video?",
    ["Texto + Voz", "Solo Texto", "Solo Voz", "Solo Texto + Música"],
    horizontal=True,
)

want_voice = output_mode in ("Texto + Voz", "Solo Voz")
want_text = output_mode in ("Texto + Voz", "Solo Texto", "Solo Texto + Música")
want_music_required = output_mode == "Solo Texto + Música"
overlay_text = want_text

# Música
st.subheader("Música (Audio Network)")
bgm_file = st.file_uploader(
    "Sube música de fondo (MP3/WAV)",
    type=["mp3", "wav"],
    accept_multiple_files=False,
    key="bgm_upload",
)

use_bgm = False
if want_music_required:
    use_bgm = True
    st.info("En **Solo Texto + Música**, la música es obligatoria.")
else:
    if bgm_file is not None:
        use_bgm = st.checkbox("Usar esta música como fondo", value=False, key="use_bgm_checkbox")

music_volume = st.slider("Volumen música", 0.0, 1.0, 0.18, 0.01)
voice_volume = st.slider("Volumen voz", 0.0, 2.0, 1.0, 0.05)
music_fade = st.slider("Fade música (seg)", 0.0, 3.0, 1.0, 0.1)

fps = st.selectbox("FPS (recomendado 15 en Streamlit Cloud)", [15, 24], index=0)
base_slide_duration = st.slider("Duración base por slide (si NO hay voz)", 2.0, 12.0, DEFAULT_SLIDE_DURATION, 0.5)

# ✅ Cierre (un solo archivo)
st.subheader("Cierre (opcional)")
cierre_video_upload = st.file_uploader(
    "Video de cierre (opcional) - MP4/MOV/WEBM",
    type=["mp4", "mov", "webm"],
    accept_multiple_files=False,
    key="cierre_video",
)

# ✅ Mosca/logo overlay
st.subheader("Mosca / Logo (opcional)")
logo_png_upload = st.file_uploader(
    "Sube logo PNG transparente (opcional)",
    type=["png"],
    accept_multiple_files=False,
    key="logo_png",
)

logo_position = st.selectbox("Posición del logo", ["top-right", "top-left", "bottom-right", "bottom-left"], index=0)
logo_scale = st.slider("Tamaño del logo (escala)", 0.05, 0.60, 0.18, 0.01)
logo_margin = st.slider("Margen del logo (px)", 0, 80, 24, 1)

st.info(f"✅ Requisito: mínimo {MIN_IMAGES_REQUIRED} imágenes para generar.")


# Estados
if "extracted" not in st.session_state:
    st.session_state.extracted = None
if "manual_paragraphs" not in st.session_state:
    st.session_state.manual_paragraphs = []


def run_generate(
    *,
    titulo_final: str,
    textos_slides: list[str],
    imagenes_paths: list[Path],
    work_dir: Path,
    cierre_path: Path | None,
    logo_path: Path | None,
) -> Path:
    texto_narracion = normalizar_texto(" ".join(textos_slides))

    # Voz
    voice_path = None
    if want_voice:
        if not api_key:
            raise ValueError("Falta API Key de ElevenLabs.")
        if not voice_id_direct:
            raise ValueError("Voice ID inválido.")
        if not texto_narracion.strip():
            raise ValueError("No hay texto para narrar.")
        voice_path = eleven_tts_long_to_mp3(
            text=texto_narracion,
            api_key=api_key,
            voice_id=voice_id_direct,
            model_id=model_id,
            output_format=output_format,
            voice_settings=voice_settings,
            out_mp3=work_dir / "voice.mp3",
            work_dir=work_dir,
        )

    # Música
    music_path = None
    if use_bgm:
        if bgm_file is None:
            raise ValueError("Seleccionaste un modo con música, pero no subiste archivo de música.")
        music_path = save_uploaded_audio(bgm_file, work_dir / "audio")
        if not music_path or not music_path.exists():
            raise ValueError("No pude guardar el archivo de música.")

    out_video = crear_video(
        textos_slides=textos_slides,
        imagenes=imagenes_paths,
        titulo=titulo_final,
        overlay_text=overlay_text,
        voice_path=voice_path,
        music_path=music_path,
        voice_volume=voice_volume,
        music_volume=music_volume,
        music_fade=music_fade,
        base_slide_duration=base_slide_duration,
        fps=int(fps),
        work_dir=work_dir,
        cierre_video_path=cierre_path,
        logo_png_path=logo_path,
        logo_position=logo_position,
        logo_scale=logo_scale,
        logo_margin=logo_margin,
    )
    return out_video


# =========================
# MODO URL
# =========================
if modo == "Desde URL de El Tiempo":
    url = st.text_input("URL del artículo", placeholder="https://www.eltiempo.com/...")
    uploaded_extra = st.file_uploader(
        "Sube imágenes extra (opcional)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="url_extra_imgs",
    )

    if st.button("1) Extraer contenido", type="primary", disabled=not bool(url)):
        try:
            with st.spinner("Extrayendo..."):
                titulo, parrafos, img_urls = extraer_contenido_articulo(url)
            st.session_state.extracted = {"titulo": titulo, "parrafos": parrafos, "img_urls": img_urls}
            st.success(f"Listo. Párrafos: {len(parrafos)} | Imágenes encontradas: {len(img_urls)}")
        except Exception as e:
            st.session_state.extracted = None
            st.error(f"No pude extraer: {e}")

    data = st.session_state.extracted

    if data:
        st.divider()
        st.subheader("2) Selecciona y edita textos")

        include_title = st.checkbox("Incluir título", value=True, key="url_include_title")
        titulo_in = st.text_input("Título", value=data["titulo"], key="url_title")

        st.write("Párrafos (marca los que quieres incluir):")
        selected_pars = []
        for i, p in enumerate(data["parrafos"]):
            col_chk, col_txt = st.columns([0.12, 0.88], vertical_alignment="top")
            with col_chk:
                ck = st.checkbox("Usar", value=True, key=f"url_par_ck_{i}")
            with col_txt:
                txt = st.text_area(label=f"Párrafo {i+1}", value=p, height=90, key=f"url_par_txt_{i}")
            if ck:
                selected_pars.append(txt)

        st.divider()
        st.subheader("3) Generar video")

        if st.button("Generar video", type="primary", key="url_generate"):
            if want_voice and not api_key:
                st.error("Falta API Key de ElevenLabs.")
                st.stop()
            if want_music_required and bgm_file is None:
                st.error("En 'Solo Texto + Música' debes subir música.")
                st.stop()
            if include_title and not (titulo_in or "").strip():
                st.error("Marcaste 'Incluir título' pero el título está vacío.")
                st.stop()
            if not selected_pars and not include_title:
                st.error("No hay textos seleccionados (ni título ni párrafos).")
                st.stop()

            work_dir = Path(tempfile.mkdtemp(prefix="url_video_"))
            imgs_dir = work_dir / "imgs"
            imgs_dir.mkdir(exist_ok=True)

            progress = st.progress(0, text="Iniciando...")
            try:
                progress.progress(10, text="Preparando textos (slides)...")
                titulo_final = normalizar_texto(titulo_in) if include_title else "video"
                textos_slides = build_textos_slides(include_title, titulo_in, selected_pars)
                if not textos_slides:
                    raise ValueError("No quedaron textos para slides tras segmentar.")

                progress.progress(30, text="Descargando imágenes del artículo...")
                # descargamos todas las que podamos; el mínimo se valida después
                imgs = descargar_imagenes(data["img_urls"], imgs_dir, max_workers=10)

                if uploaded_extra:
                    imgs.extend(guardar_imagenes_subidas(uploaded_extra, imgs_dir))

                # ✅ validar mínimo de imágenes
                if len(imgs) < MIN_IMAGES_REQUIRED:
                    st.error(f"Necesitas mínimo {MIN_IMAGES_REQUIRED} imágenes. Actualmente tienes {len(imgs)}. Sube más fotos para continuar.")
                    st.stop()

                cierre_path = save_uploaded_video(cierre_video_upload, work_dir / "endclips", "cierre")
                logo_path = save_uploaded_png(logo_png_upload, work_dir / "overlay", "logo") if logo_png_upload else None

                progress.progress(60, text="Generando voz/música y renderizando video...")
                out_video = run_generate(
                    titulo_final=titulo_final,
                    textos_slides=textos_slides,
                    imagenes_paths=imgs,
                    work_dir=work_dir,
                    cierre_path=cierre_path,
                    logo_path=logo_path,
                )

                progress.progress(100, text="Listo ✅")
                st.success("Video creado.")
                video_bytes = out_video.read_bytes()
                st.video(video_bytes)
                st.download_button("Descargar MP4", data=video_bytes, file_name=out_video.name, mime="video/mp4")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass
    else:
        st.info("Pega una URL y presiona **Extraer contenido** para empezar.")


# =========================
# MODO MANUAL
# =========================
else:
    st.subheader("Texto e imágenes manual")

    colL, colR = st.columns([1, 1])
    with colL:
        include_title_m = st.checkbox("Incluir título", value=True, key="manual_include_title")
        titulo_m = st.text_input("Título", value="", key="manual_title")
    with colR:
        uploaded_manual_imgs = st.file_uploader(
            "Sube imágenes (mínimo 5)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="manual_imgs",
        )

    texto_manual = st.text_area(
        "Pega aquí el texto completo (se separa por párrafos usando líneas en blanco)",
        height=220,
        key="manual_text",
    )

    if st.button("1) Cargar texto", type="primary", key="manual_load_text"):
        pars = split_paragraphs_from_manual(texto_manual)
        st.session_state.manual_paragraphs = pars
        st.success(f"Texto cargado. Párrafos detectados: {len(pars)}")

    if st.session_state.manual_paragraphs:
        st.divider()
        st.subheader("2) Selecciona y edita textos")

        selected_pars = []
        for i, p in enumerate(st.session_state.manual_paragraphs):
            col_chk, col_txt = st.columns([0.12, 0.88], vertical_alignment="top")
            with col_chk:
                ck = st.checkbox("Usar", value=True, key=f"manual_par_ck_{i}")
            with col_txt:
                txt = st.text_area(label=f"Párrafo {i+1}", value=p, height=90, key=f"manual_par_txt_{i}")
            if ck:
                selected_pars.append(txt)

        st.divider()
        st.subheader("3) Generar video")

        if st.button("Generar video", type="primary", key="manual_generate"):
            if want_voice and not api_key:
                st.error("Falta API Key de ElevenLabs.")
                st.stop()
            if want_music_required and bgm_file is None:
                st.error("En 'Solo Texto + Música' debes subir música.")
                st.stop()
            if include_title_m and not (titulo_m or "").strip():
                st.error("Marcaste 'Incluir título' pero el título está vacío.")
                st.stop()
            if not selected_pars and not include_title_m:
                st.error("No hay textos seleccionados (ni título ni párrafos).")
                st.stop()
            if not uploaded_manual_imgs:
                st.error(f"En modo manual debes subir al menos {MIN_IMAGES_REQUIRED} imágenes.")
                st.stop()

            work_dir = Path(tempfile.mkdtemp(prefix="manual_video_"))
            imgs_dir = work_dir / "imgs"
            imgs_dir.mkdir(exist_ok=True)

            progress = st.progress(0, text="Iniciando...")
            try:
                progress.progress(10, text="Preparando textos (slides)...")
                titulo_final = normalizar_texto(titulo_m) if include_title_m else "video"
                textos_slides = build_textos_slides(include_title_m, titulo_m, selected_pars)
                if not textos_slides:
                    raise ValueError("No quedaron textos para slides tras segmentar.")

                progress.progress(30, text="Guardando imágenes subidas...")
                imgs = guardar_imagenes_subidas(uploaded_manual_imgs, imgs_dir)

                # ✅ validar mínimo de imágenes
                if len(imgs) < MIN_IMAGES_REQUIRED:
                    st.error(f"Necesitas mínimo {MIN_IMAGES_REQUIRED} imágenes. Actualmente tienes {len(imgs)}. Sube más fotos para continuar.")
                    st.stop()

                cierre_path = save_uploaded_video(cierre_video_upload, work_dir / "endclips", "cierre")
                logo_path = save_uploaded_png(logo_png_upload, work_dir / "overlay", "logo") if logo_png_upload else None

                progress.progress(60, text="Generando voz/música y renderizando video...")
                out_video = run_generate(
                    titulo_final=titulo_final,
                    textos_slides=textos_slides,
                    imagenes_paths=imgs,
                    work_dir=work_dir,
                    cierre_path=cierre_path,
                    logo_path=logo_path,
                )

                progress.progress(100, text="Listo ✅")
                st.success("Video creado.")
                video_bytes = out_video.read_bytes()
                st.video(video_bytes)
                st.download_button("Descargar MP4", data=video_bytes, file_name=out_video.name, mime="video/mp4")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass
    else:
        st.info("Pega texto y presiona **Cargar texto** para generar los párrafos seleccionables.")
