import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from string import ascii_lowercase
from gtts import gTTS
import io
from PIL import Image
import speech_recognition as sr
import tempfile
import json
from urllib import request, parse
from pydub import AudioSegment

IMG_SIZE = 50
DATADIR = 'dataset'
MODEL_PATH = 'CNN.model'
TEST_DIR = 'test'

@st.cache_resource
def load_model():
    m = tf.keras.models.load_model(MODEL_PATH)
    return m

@st.cache_data
def get_categories():
    cats = sorted([d for d in os.listdir(DATADIR) if not d.startswith('.')])
    if not cats:
        st.warning("No categories found in dataset/ directory")
    return cats

try:
    model = load_model()
    CATEGORIES = get_categories()
    st.sidebar.success(f"Model loaded: {len(CATEGORIES)} categories")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)}

DET_PHRASES = [
    'hi how are you', 'i dont know', 'what is your name', 'who are you',
    'what is this', 'where are you', 'how are you', 'i am hungry',
    'i am ironman', 'i love you', 'i hate you', 'i am sick',
    'i am sleeping', 'i am thirsty', 'i am in home', 'thankyou'
] * 2

def alphabet_position(text):
    text = text.lower()
    return [LETTERS[char] for char in text if char in LETTERS]

def translate_text(text, dest='ta'):
    try:
        data = json.dumps({'q': text, 'source': 'en', 'target': dest}).encode()
        req = request.Request(
            'https://libretranslate.de/translate',
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        resp = request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode())['translatedText']
    except:
        return f"[{text}]"

def get_sign_images(text):
    images = []
    for char in text:
        if char == " ":
            img_path = os.path.join(TEST_DIR, 'space.png')
        elif char.lower() in LETTERS:
            pos = alphabet_position(char)[0] if alphabet_position(char) else None
            if pos is None:
                continue
            img_path = os.path.join(TEST_DIR, f'{int(pos)}.jpg')
        else:
            continue
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((150, 150))
            images.append(img)
    return images

st.set_page_config(page_title="Sign & Speech Translator", layout="centered")
st.markdown("<h1 style='color:#3498db;'>Sign & Speech Translator</h1>", unsafe_allow_html=True)

mode = st.radio("Choose Mode:", ['Sign Language to Text', 'Speech to Sign'], horizontal=True)

if mode == 'Sign Language to Text':
    img_file = st.camera_input("Capture your hand sign")

    if img_file is not None:
        with st.spinner("Analyzing sign..."):
            bytes_data = img_file.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            box_size = min(300, h, w)
            hand = gray[0:box_size, 0:box_size]
            resized = cv2.resize(hand, (IMG_SIZE, IMG_SIZE))
            img_array = resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

            prediction = model.predict(img_array, verbose=0)
            index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Captured Image", width=250)

        with col2:
            if "unknown" not in CATEGORIES[index] and index < len(DET_PHRASES):
                text = DET_PHRASES[index]
                st.success(f"**Sign:** {CATEGORIES[index]}")
                st.info(f"**Detected:** {text}")
                st.metric("Confidence", f"{confidence:.1f}%")

                with st.spinner("Translating to Tamil..."):
                    try:
                        translated = translate_text(text)
                        st.write(f"**Tamil:** {translated}")

                        tts = gTTS(text=translated, lang='ta')
                        audio_bytes = io.BytesIO()
                        tts.write_to_fp(audio_bytes)
                        audio_bytes.seek(0)
                        st.audio(audio_bytes, format='audio/mp3')
                    except Exception as e:
                        st.error(f"Translation/audio error: {e}")
            else:
                st.info(f"**Sign detected:** {CATEGORIES[index]}")
                st.metric("Confidence", f"{confidence:.1f}%")

else:
    audio_file = st.audio_input("Speak now...")

    if audio_file is not None:
        with st.spinner("Processing speech..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file.getvalue())
                tmp_path = tmp.name

            try:
                audio = AudioSegment.from_file(tmp_path)
                audio.export(tmp_path, format='wav')

                r = sr.Recognizer()
                with sr.AudioFile(tmp_path) as source:
                    audio_data = r.record(source)

                recog = r.recognize_google(audio_data)
                st.success(f"**You said:** {recog}")

                images = get_sign_images(recog)
                if images:
                    st.write("**Sign representation:**")
                    cols = st.columns(min(len(images), 10))
                    for i, img in enumerate(images):
                        with cols[i % 10]:
                            st.image(img, width=75)
                else:
                    st.warning("No sign images available for the recognized text.")
            except sr.UnknownValueError:
                st.error("Could not understand the audio. Please speak clearly.")
            except sr.RequestError as e:
                st.error(f"Speech recognition service error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)
