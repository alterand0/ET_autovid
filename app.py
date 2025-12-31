import os
import re
import json
import math
import shutil
import hashlib
import tempfile
import subprocess
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
import imageio_ffmpeg


# =========================
# Config general
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

# =========================
# Texto: normalización y segmentación
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
# Fuente (opcional)
# =========================
def load_font(size: int) -> ImageFont.FreeTypeFont:
    # Si quieres una fuente custom, súbela al repo (ej: assets/Calibri-Bold.ttf)
    # y cárgala aquí. Para simplificar, intentamos DejaVu (suele existir) y fallback.
    for candidate in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


# =========================
# Scraper eltiempo
# =========================
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
# Imágenes: descarga y render
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
    d0 = ImageDraw.Draw(fondo)

    # wrap aproximado
    max_width_px = 1800
    wrap_width = max(18, max_width_px // max(1, (font_size // 2)))
    lineas = textwrap.wrap(texto, width=wrap_width, break_long_words=False, break_on_hyphens=False)

    # 2 líneas por imagen, si hay más -> varias imágenes
    bloques = [lineas[i:i+2] for i in range(0, len(lineas), 2)]
    outs = []

    for j, bloque in enumerate(bloques):
        base = fondo.copy()
        d = ImageDraw.Draw(base)

        widths, heights = [], []
        for linea in bloque:
            bbox = d.textbbox((0, 0), linea, font=fuente)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])

        total_h = sum(heights) + 8 * (len(bloque) - 1)
        y0 = RES_H - total_h - 100
        max_w = max(widths) if widths else 0
        x0 = (RES_W - max_w) // 2

        rect = (x0 - 18, y0 - 18, x0 + max_w + 18, RES_H)
        d.rectangle(rect, fill="#0066ae")

        y = y0
        for linea, h in zip(bloque, heights):
            bbox = d.textbbox((0, 0), linea, font=fuente)
            w = bbox[2] - bbox[0]
            d.text(((RES_W - w) // 2, y), linea, fill="#ffffff", font=fuente)
            y += h + 8

        out = out_dir / f"slide_{idx:04d}_{j:02d}.jpg"
        base.save(out, quality=92)
        outs.append(out)

    return outs


# =========================
# ElevenLabs: voces y TTS largo
# =========================
def eleven_get_voices(api_key: str) -> list[dict]:
    r = requests.get(f"{ELEVEN_BASE}/v1/voices", headers={"xi-api-key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("voices", data)

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def pick_voice(voices: list[dict], name_contains: str, gender: str, language: str) -> tuple[str, dict]:
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
    """
    Genera mp3 por chunks y concatena con ffmpeg (imageio-ffmpeg, sin depender del sistema).
    """
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
        payload = {
            "text": chunk,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        r = requests.post(
            url,
            params=params,
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120
        )
        r.raise_for_status()
        p = parts_dir / f"part_{i:04d}.mp3"
        p.write_bytes(r.content)
        part_files.append(p)

    # concat list con rutas absolutas (fix del error que te salió)
    list_file = (parts_dir / "concat_list.txt").resolve()
    with list_file.open("w", encoding="utf-8") as f:
        for p in part_files:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    out_mp3 = out_mp3.resolve()
    cmd = [ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_mp3)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # fallback: re-encode (más robusto)
        cmd2 = [ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)]
        proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc2.returncode != 0:
            raise RuntimeError("ffmpeg falló concatenando audio:\n" + (proc2.stderr[-1500:] or proc.stderr[-1500:]))

    return out_mp3


# =========================
# Video
# =========================
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
    except Exception:
        pass
    try:
        if audio_clip:
            audio_clip.close()
    except Exception:
        pass

    return out

def safe_filename(title: str, max_len: int = 60) -> str:
    s = "".join(ch for ch in (title or "") if ch.isalnum() or ch in " _-").strip()
    s = s[:max_len].strip()
    return s or "video"


# =========================
# UI Streamlit
# =========================
st.set_page_config(page_title="Nota → Video + Voz", layout="wide")
st.title("Nota de El Tiempo → Video con narración (ElevenLabs)")

with st.expander("⚠️ Nota", expanded=False):
    st.write("Este tipo de automatización debe usarse respetando derechos/condiciones del medio y con permisos cuando aplique.")

col1, col2 = st.columns([1.2, 1])

with col1:
    url = st.text_input("URL del artículo", placeholder="https://www.eltiempo.com/...")
    btn_extract = st.button("1) Extraer contenido", type="primary")

with col2:
    st.subheader("Voz (ElevenLabs)")
    api_key = st.secrets.get("ELEVENLABS_API_KEY", "")  # En Cloud se configura en Secrets UI.  [oai_citation:3‡Streamlit Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management?utm_source=chatgpt.com)
    if not api_key:
        st.info("Configura ELEVENLABS_API_KEY en Secrets (Streamlit Cloud).")

    model_id = st.selectbox("Modelo", ["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"], index=0)
    output_format = st.selectbox("Formato", ["mp3_44100_128", "mp3_44100_192", "mp3_24000_48", "pcm_44100"], index=0)

    gender = st.selectbox("Preferencia de género (si labels existen)", ["", "female", "male", "neutral"], index=1)
    language = st.selectbox("Preferencia de idioma (si labels existen)", ["", "es", "spanish", "en", "english"], index=1)
    name_contains = st.text_input("Nombre de voz contiene (opcional)", value="")

    st.write("Ajustes de voz")
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

if "extracted" not in st.session_state:
    st.session_state.extracted = None

if btn_extract and url:
    with st.spinner("Extrayendo..."):
        try:
            titulo, parrafos, img_urls = extraer_contenido_articulo(url)
            st.session_state.extracted = {
                "titulo": titulo,
                "parrafos": parrafos,
                "img_urls": img_urls,
            }
            st.success(f"Listo. Párrafos: {len(parrafos)} | Imágenes: {len(img_urls)}")
        except Exception as e:
            st.session_state.extracted = None
            st.error(f"No pude extraer: {e}")

data = st.session_state.extracted

if data:
    st.divider()
    st.subheader("2) Revisa/edita lo que se va a usar")

    # Textos seleccionables/editables
    st.write("### Textos")
    titulo_in = st.text_input("Título", value=data["titulo"])
    include_title = st.checkbox("Incluir título", value=True)

    st.write("Párrafos (marca lo que quieras narrar/usar):")
    selected_pars = []
    for i, p in enumerate(data["parrafos"]):
        with st.expander(f"Párrafo {i+1}", expanded=False):
            ck = st.checkbox("Incluir", value=True, key=f"par_ck_{i}")
            txt = st.text_area("Texto", value=p, height=110, key=f"par_txt_{i}")
            if ck:
                selected_pars.append(txt)

    # Imágenes: descarga + uploader extra
    st.write("### Imágenes")
    uploaded = st.file_uploader("Sube imágenes extra (opcional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    max_imgs = st.slider("Máximo de imágenes del artículo a usar", 1, 30, 12)
    title_max_chars = st.slider("Máx caracteres título (slide)", 80, 220, 140)
    body_max_chars = st.slider("Máx caracteres cuerpo (slide)", 90, 220, 160)
    body_max_sent = st.slider("Máx frases por slide", 1, 3, 2)

    gen_audio = st.checkbox("Generar narración con ElevenLabs", value=True)

    btn_generate = st.button("3) Generar video", type="primary")

    if btn_generate:
        if not include_title and not selected_pars:
            st.error("No hay textos seleccionados.")
            st.stop()

        if gen_audio and not api_key:
            st.error("Falta ELEVENLABS_API_KEY en Secrets.")
            st.stop()

        work_dir = Path(tempfile.mkdtemp(prefix="nota_video_"))
        imgs_dir = work_dir / "imgs"
        imgs_dir.mkdir(exist_ok=True)

        progress = st.progress(0, text="Iniciando...")

        try:
            # 1) juntar texto final (narración)
            textos_seleccionados = []
            if include_title:
                textos_seleccionados.append(titulo_in)
            textos_seleccionados.extend(selected_pars)

            texto_narracion = normalizar_texto("\n\n".join(textos_seleccionados))

            # 2) generar slides (texto segmentado)
            textos_slides = []
            if include_title:
                t = normalizar_texto(titulo_in)[:title_max_chars]
                textos_slides.extend(segmentar_para_slides(t, max_chars=title_max_chars, max_sentences=1))

            for p in selected_pars:
                textos_slides.extend(segmentar_para_slides(p, max_chars=body_max_chars, max_sentences=body_max_sent))

            if not textos_slides:
                raise ValueError("No quedaron textos para slides tras segmentar.")

            progress.progress(10, text="Descargando imágenes...")

            # 3) descargar imágenes del artículo
            img_urls = data["img_urls"][:max_imgs]
            downloaded_imgs = descargar_imagenes(img_urls, imgs_dir, max_workers=10)

            # 4) agregar imágenes subidas
            if uploaded:
                for uf in uploaded:
                    img = Image.open(BytesIO(uf.read()))
                    img = ajustar_imagen(img).convert("RGB")
                    p = imgs_dir / f"upload_{uf.name}"
                    img.save(p, quality=92)
                    downloaded_imgs.append(p)

            if not downloaded_imgs:
                raise ValueError("No hay imágenes disponibles (ni del artículo ni subidas).")

            progress.progress(35, text="Eligiendo voz...")

            # 5) elegir voz y generar audio
            audio_path = None
            if gen_audio:
                voices = eleven_get_voices(api_key)
                voice_id, voice_obj = pick_voice(voices, name_contains=name_contains, gender=gender, language=language)
                st.info(f"Voz elegida: {voice_obj.get('name')} | labels={voice_obj.get('labels')}")
                progress.progress(55, text="Generando narración (ElevenLabs)...")

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

            progress.progress(75, text="Renderizando video...")

            # 6) crear video
            out_video = crear_video(
                textos_slides=textos_slides,
                imagenes=downloaded_imgs,
                titulo=titulo_in or "video",
                audio_path=audio_path,
                work_dir=work_dir,
            )

            progress.progress(100, text="Listo ✅")

            st.success("Video creado.")
            video_bytes = out_video.read_bytes()
            st.video(video_bytes)
            st.download_button("Descargar MP4", data=video_bytes, file_name=out_video.name, mime="video/mp4")

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            # En Cloud, el FS es efímero; igual limpiamos.
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
