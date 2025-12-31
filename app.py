import os
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

from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
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
FPS = 24
DEFAULT_SLIDE_DURATION = 7.0
MIN_SLIDE_DURATION_WITH_AUDIO = 3.5

ELEVEN_BASE = "https://api.elevenlabs.io"

# Slides: 30 chars/line, 2 lines/slide => ~60 chars por bloque
SLIDE_TEXT_MAX_CHARS = 60
SLIDE_TEXT_MAX_SENTENCES = 1

# Render: 30 chars line wrap, 2 lines per image
RENDER_WRAP_WIDTH = 30
RENDER_LINES_PER_IMAGE = 2


# =========================
# PIN gate
# =========================
def require_pin_if_configured():
    """
    Pide PIN si st.secrets tiene APP_PIN.
    """
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
QUOTES_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "´": "'",
    "…": "...",
    "\u00A0": " ",
})

ABREVIATURAS = [
    "Sr.", "Sra.", "Dr.", "Dra.", "Ing.", "Lic.", "No.", "Nro.", "Art.", "Pág.",
    "p. ej.", "p.ej.", "etc.", "EE. UU.", "U.S.", "vs."
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
    """
    Segmenta en bloques cortos para slides (antes del wrap 30/2).
    No corta palabras; si una frase excede max_chars, se parte por palabras.
    """
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


# =========================
# FONT
# =========================
def load_font(size: int) -> ImageFont.FreeTypeFont:
    # En Streamlit Cloud suele existir DejaVu
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

    for media in cuerpo.find_all(["figure", "div"], class_=["c-cuerpo__media__thumb", "c-cuerpo__media", "c-detail__media"]):
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

def render_slide(imagen_path: Path, texto: str, idx: int, out_dir: Path, font_size: int = 65) -> list[Path]:
    """
    - Máx 30 chars por línea (wrap_width=30)
    - Máx 2 líneas por imagen (si hay más, crea más imágenes)
    - No corta palabras
    - Sin barra azul: texto blanco con borde negro
    """
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
        break_on_hyphens=False
    )

    bloques = [lineas[i:i+RENDER_LINES_PER_IMAGE] for i in range(0, len(lineas), RENDER_LINES_PER_IMAGE)]
    outs = []

    for j, bloque in enumerate(bloques):
        base = fondo.copy()
        d = ImageDraw.Draw(base)

        widths, heights = [], []
        for linea in bloque:
            bbox = d.textbbox((0, 0), linea, font=fuente)
            widths.append(bbox[2] - bbox[0])
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
                stroke_fill="#000000"
            )
            y += h + 10

        out = out_dir / f"slide_{idx:04d}_{j:02d}.jpg"
        base.save(out, quality=92)
        outs.append(out)

    return outs


# =========================
# ElevenLabs: voices + long TTS
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def eleven_get_voices(api_key: str) -> list[dict]:
    r = requests.get(f"{ELEVEN_BASE}/v1/voices", headers={"xi-api-key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("voices", data)

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def pick_voice(voices: list[dict], name_contains: str, gender: str, language: str, voice_id_direct: str) -> tuple[str, dict]:
    if voice_id_direct:
        for v in voices:
            if v.get("voice_id") == voice_id_direct:
                return voice_id_direct, v
        raise ValueError("VOICE_ID no encontrado en tu cuenta.")

    name_contains_n = _norm(name_contains)
    gender_n = _norm(gender)
    lang_n = _norm(language)

    def score(v: dict) -> int:
        s = 0
        name = _norm(v.get("name"))
        labels = v.get("labels") or {}
        labels_n = {_norm(k): _norm(str(val)) for k, val in labels.items()}

        if name_contains_n:
            s += 50 if name_contains_n in name else -10

        if gender_n:
            g = labels_n.get("gender", "")
            if g:
                s += 15 if gender_n in g else -5

        if lang_n:
            l = labels_n.get("language", "")
            if l:
                s += 10 if (lang_n in l or l in lang_n) else -3

        return s

    best = sorted(voices, key=score, reverse=True)[0]
    return best["voice_id"], best

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

def eleven_tts_long_to_mp3(text: str, api_key: str, voice_id: str, model_id: str,
                           output_format: str, voice_settings: dict, out_mp3: Path, work_dir: Path) -> Path:
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
            url, params=params,
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120
        )
        r.raise_for_status()

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
        cmd2 = [ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)]
        proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc2.returncode != 0:
            raise RuntimeError("ffmpeg falló concatenando audio:\n" + (proc2.stderr[-1500:] or proc.stderr[-1500:]))

    return out_mp3


# =========================
# VIDEO
# =========================
def safe_filename(title: str, max_len: int = 60) -> str:
    s = "".join(ch for ch in (title or "") if ch.isalnum() or ch in " _-").strip()
    s = s[:max_len].strip()
    return s or "video"

def crear_video(textos_slides: list[str], imagenes: list[Path], titulo: str, audio_path: Path | None, work_dir: Path) -> Path:
    slides_dir = work_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    slide_imgs = []
    for idx, txt in enumerate(textos_slides):
        img_path = imagenes[idx % len(imagenes)]
        slide_imgs.extend(render_slide(img_path, txt, idx, slides_dir))

    audio_clip = None
    if audio_path and audio_path.exists():
        audio_clip = AudioFileClip(str(audio_path))
        total = max(1.0, float(audio_clip.duration))
        dur = max(MIN_SLIDE_DURATION_WITH_AUDIO, total / max(1, len(slide_imgs)))
    else:
        dur = DEFAULT_SLIDE_DURATION

    clips = [ImageClip(str(p)).set_duration(dur) for p in slide_imgs]
    final = concatenate_videoclips(clips, method="compose")

    if audio_clip:
        final = final.set_audio(audio_clip).set_duration(audio_clip.duration)

    out = work_dir / f"{safe_filename(titulo)}.mp4"
    final.write_videofile(str(out), fps=FPS, audio_codec="aac")

    try:
        final.close()
    except:
        pass
    try:
        if audio_clip:
            audio_clip.close()
    except:
        pass

    return out


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Nota → Video + Voz", layout="wide")
st.title("Nota de El Tiempo → Video con narración (ElevenLabs)")

# PIN gate
require_pin_if_configured()

with st.sidebar:
    st.header("ElevenLabs")
    secret_key = st.secrets.get("ELEVENLABS_API_KEY", "")
    api_key = secret_key or st.text_input("API Key (si no está en Secrets)", type="password")
    st.caption("Recomendado: configúrala en Secrets.")

    model_id = st.selectbox("Modelo", ["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"], index=0)
    output_format = st.selectbox("Formato", ["mp3_44100_128", "mp3_44100_192", "mp3_24000_48", "pcm_44100"], index=0)

    voice_id_direct = st.text_input("Voice ID (opcional)", value="")
    name_contains = st.text_input("Nombre de voz contiene (opcional)", value="")
    gender = st.selectbox("Preferencia género (si labels)", ["", "female", "male", "neutral"], index=1)
    language = st.selectbox("Preferencia idioma (si labels)", ["", "es", "spanish", "en", "english"], index=1)

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

st.caption("Solo modo avanzado (selección por marcación).")

url = st.text_input("URL del artículo", placeholder="https://www.eltiempo.com/...")
uploaded = st.file_uploader("Sube imágenes extra (opcional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

c1, c2 = st.columns(2)
with c1:
    max_imgs = st.slider("Máx imágenes del artículo", 1, 30, 12)
with c2:
    gen_audio = st.checkbox("Generar narración con ElevenLabs", value=True)

# Estado
if "extracted" not in st.session_state:
    st.session_state.extracted = None

if st.button("1) Extraer contenido", type="primary", disabled=not bool(url)):
    try:
        with st.spinner("Extrayendo..."):
            titulo, parrafos, img_urls = extraer_contenido_articulo(url)
        st.session_state.extracted = {"titulo": titulo, "parrafos": parrafos, "img_urls": img_urls}
        st.success(f"Listo. Párrafos: {len(parrafos)} | Imágenes: {len(img_urls)}")
    except Exception as e:
        st.session_state.extracted = None
        st.error(f"No pude extraer: {e}")

data = st.session_state.extracted

if data:
    st.divider()
    st.subheader("2) Selecciona y edita textos (sin desplegables)")

    include_title = st.checkbox("Incluir título", value=True)
    titulo_in = st.text_input("Título", value=data["titulo"])

    st.write("Párrafos (marca los que quieres incluir):")
    selected_pars = []

    for i, p in enumerate(data["parrafos"]):
        col_chk, col_txt = st.columns([0.12, 0.88], vertical_alignment="top")
        with col_chk:
            ck = st.checkbox("Usar", value=True, key=f"par_ck_{i}")
        with col_txt:
            txt = st.text_area(label=f"Párrafo {i+1}", value=p, height=90, key=f"par_txt_{i}")
        if ck:
            selected_pars.append(txt)

    st.divider()
    st.subheader("3) Generar video")

    if st.button("Generar video", type="primary"):
        if gen_audio and not api_key:
            st.error("Falta API Key (ponla en Secrets o pégala en sidebar).")
            st.stop()

        textos_seleccionados = []
        if include_title:
            textos_seleccionados.append(titulo_in)
        textos_seleccionados.extend(selected_pars)

        if not textos_seleccionados:
            st.error("No hay textos seleccionados.")
            st.stop()

        work_dir = Path(tempfile.mkdtemp(prefix="nota_video_"))
        imgs_dir = work_dir / "imgs"
        imgs_dir.mkdir(exist_ok=True)

        progress = st.progress(0, text="Iniciando...")

        try:
            progress.progress(10, text="Preparando textos...")

            titulo_final = normalizar_texto(titulo_in) if include_title else "video"
            texto_narracion = normalizar_texto("\n\n".join(textos_seleccionados))

            # Slides (≈ 60 chars por bloque, luego render_slide aplica 30/2)
            textos_slides = []
            if include_title:
                t = normalizar_texto(titulo_in)
                textos_slides.extend(segmentar_para_slides(t, max_chars=SLIDE_TEXT_MAX_CHARS, max_sentences=SLIDE_TEXT_MAX_SENTENCES))

            for p in selected_pars:
                textos_slides.extend(segmentar_para_slides(p, max_chars=SLIDE_TEXT_MAX_CHARS, max_sentences=SLIDE_TEXT_MAX_SENTENCES))

            if not textos_slides:
                raise ValueError("No quedaron textos para slides tras segmentar.")

            progress.progress(30, text="Descargando imágenes...")
            downloaded_imgs = descargar_imagenes(data["img_urls"][:max_imgs], imgs_dir, max_workers=10)

            if uploaded:
                for uf in uploaded:
                    img = Image.open(BytesIO(uf.read()))
                    img = ajustar_imagen(img).convert("RGB")
                    pth = imgs_dir / f"upload_{uf.name}"
                    img.save(pth, quality=92)
                    downloaded_imgs.append(pth)

            if not downloaded_imgs:
                raise ValueError("No hay imágenes disponibles (ni del artículo ni subidas).")

            audio_path = None
            if gen_audio:
                progress.progress(55, text="Listando voces...")
                voices = eleven_get_voices(api_key)
                voice_id, voice_obj = pick_voice(voices, name_contains, gender, language, voice_id_direct)
                st.info(f"Voz: {voice_obj.get('name')} | labels={voice_obj.get('labels')}")

                progress.progress(70, text="Generando narración...")
                audio_path = eleven_tts_long_to_mp3(
                    text=texto_narracion,
                    api_key=api_key,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format=output_format,
                    voice_settings=voice_settings,
                    out_mp3=work_dir / "narracion.mp3",
                    work_dir=work_dir,
                )

            progress.progress(85, text="Renderizando video...")
            out_video = crear_video(textos_slides, downloaded_imgs, titulo_final, audio_path, work_dir)

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
            except:
                pass
else:
    st.info("Pega una URL y presiona **Extraer contenido** para empezar.")
