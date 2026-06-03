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
import hashlib

IMG_SIZE = 50
DATADIR = 'dataset'
MODEL_PATH = 'CNN.model'
TEST_DIR = 'test'
USERS_FILE = 'users.json'

st.set_page_config(page_title="Sign & Speech Translator", layout="centered")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ''

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def get_categories():
    cats = sorted([d for d in os.listdir(DATADIR) if not d.startswith('.')])
    return cats

if not st.session_state.authenticated:
    st.markdown("<h1 style='color:#3498db;'>Sign & Speech Translator</h1>", unsafe_allow_html=True)
    st.markdown("### Login or Register to continue")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                users = load_users()
                if username in users and users[username] == hash_password(password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("register_form"):
            reg_user = st.text_input("Choose Username")
            reg_pass = st.text_input("Choose Password", type="password")
            reg_confirm = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register"):
                if not reg_user or not reg_pass:
                    st.error("Please fill all fields")
                elif reg_pass != reg_confirm:
                    st.error("Passwords do not match")
                elif len(reg_pass) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    users = load_users()
                    if reg_user in users:
                        st.error("Username already exists")
                    else:
                        users[reg_user] = hash_password(reg_pass)
                        save_users(users)
                        st.success("Registration successful! Please login.")

    st.stop()

try:
    model = load_model()
    CATEGORIES = get_categories()
    st.sidebar.success(f"Model loaded: {len(CATEGORIES)} categories")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.sidebar.write(f"**{st.session_state.username}**")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ''
    st.rerun()

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

st.markdown("<h1 style='color:#3498db;'>Sign & Speech Translator</h1>", unsafe_allow_html=True)

mode = st.radio("Choose Mode:", ['Sign Language to Text', 'Speech to Sign'], horizontal=True)

if mode == 'Sign Language to Text':
    if 'sign_buffer' not in st.session_state:
        st.session_state.sign_buffer = ''
    if 'last_capture' not in st.session_state:
        st.session_state.last_capture = None

    st.text_area("Constructed Text", value=st.session_state.sign_buffer, height=100)

    col_cam, col_side = st.columns([2, 1])

    with col_cam:
        img_file = st.camera_input("Capture hand sign", key='cam')

    with col_side:
        if st.button("Speak", use_container_width=True):
            if st.session_state.sign_buffer.strip():
                tts = gTTS(text=st.session_state.sign_buffer, lang='en')
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                st.audio(audio_bytes, format='audio/mp3')
        if st.button("Space", use_container_width=True):
            st.session_state.sign_buffer += ' '
            st.rerun()
        if st.button("Delete Last", use_container_width=True):
            st.session_state.sign_buffer = st.session_state.sign_buffer[:-1]
            st.rerun()
        if st.button("Clear", use_container_width=True):
            st.session_state.sign_buffer = ''
            st.rerun()

    if img_file is not None:
        current = img_file.getvalue()
        if current != st.session_state.last_capture:
            st.session_state.last_capture = current
            with st.spinner("Analyzing sign..."):
                nparr = np.frombuffer(current, np.uint8)
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

            st.image(img_file, width=150)
            if "unknown" not in CATEGORIES[index]:
                letter = CATEGORIES[index]
                st.success(f"**{letter}** ({confidence:.0f}%)")
                st.session_state.sign_buffer += letter
                st.rerun()
            else:
                st.info(f"Unknown sign ({confidence:.0f}%)")

else:
    typed_text = st.text_input("Or type text to convert to sign language:", placeholder="e.g. hello world")

    if typed_text:
        images = get_sign_images(typed_text)
        if images:
            st.write("**Sign representation:**")
            cols = st.columns(min(len(images), 10))
            for i, img in enumerate(images):
                with cols[i % 10]:
                        st.image(img, width=75)
        else:
            st.warning("No sign images available for that text.")

    audio_file = st.audio_input("Or speak now...")

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
