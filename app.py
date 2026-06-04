import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import time
from string import ascii_lowercase
from gtts import gTTS
import io
from PIL import Image
import speech_recognition as sr
import tempfile
import json
import base64
from urllib import request, parse
from urllib.error import HTTPError
from pydub import AudioSegment
import hashlib

IMG_SIZE = 50
DATADIR = 'dataset'
MODEL_PATH = 'CNN.model'
TEST_DIR = 'test'
USERS_FILE = 'users.json'

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY', '')
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_ANON_KEY)

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_REPO = os.environ.get('GITHUB_REPO', '')
USE_GITHUB = bool(GITHUB_TOKEN and GITHUB_REPO)

st.set_page_config(page_title="Sign & Speech Translator", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stButton>button {
        background: linear-gradient(135deg, #4FC3F7, #0288D1);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.4);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117, #1E2230);
        border-right: 1px solid #2A2F3E;
    }
    div[data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #FF5252, #D32F2F);
    }
    div[data-testid="stSidebar"] .stButton>button:hover {
        box-shadow: 0 4px 12px rgba(255, 82, 82, 0.4);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1E2230 !important;
        color: #E0E0E0 !important;
        border: 1px solid #2A2F3E !important;
        border-radius: 6px;
    }
    div.stTabs button {
        color: #888 !important;
        font-weight: 600;
    }
    div.stTabs button[aria-selected="true"] {
        color: #4FC3F7 !important;
        border-bottom: 2px solid #4FC3F7;
    }
    .stRadio>label {
        color: #E0E0E0 !important;
    }
    div[data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2A2F3E;
    }
    .stSuccess, .stError, .stInfo {
        border-radius: 8px !important;
    }
    video {
        width: 100%;
        border-radius: 8px;
        border: 2px solid #2A2F3E;
        background: #1E2230;
    }
</style>
""", unsafe_allow_html=True)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ''

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def _supabase_get(username=None):
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/users?select=username,password_hash"
    if username:
        url += f"&username=eq.{parse.quote(username)}"
    req = request.Request(url, headers={
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
    }, method='GET')
    try:
        resp = request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode())
    except Exception:
        return None

def _supabase_post(data):
    req = request.Request(
        f"{SUPABASE_URL.rstrip('/')}/rest/v1/users",
        data=json.dumps(data).encode(),
        headers={
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation',
        },
        method='POST',
    )
    try:
        resp = request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode())
    except Exception:
        return None

def _github_read_users():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/users.json"
    req = request.Request(url, headers={
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    })
    try:
        resp = request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
        content = base64.b64decode(data['content']).decode()
        return json.loads(content), data['sha']
    except HTTPError as e:
        if e.code == 404:
            return {}, None
        return None, None

def _github_write_users(users, sha):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/users.json"
    content = base64.b64encode(json.dumps(users).encode()).decode()
    body = json.dumps({'message': 'update users', 'content': content, 'sha': sha})
    req = request.Request(url, data=body.encode(), headers={
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.github.v3+json',
    }, method='PUT')
    try:
        request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False

def _get_users():
    if USE_SUPABASE:
        result = _supabase_get()
        if result is not None:
            return {u['username']: u['password_hash'] for u in result}
    if USE_GITHUB:
        users, _ = _github_read_users()
        if users is not None:
            return users
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def _save_user(username, password_hash):
    if USE_SUPABASE:
        existing = _supabase_get(username)
        if existing and len(existing) > 0:
            return False
        result = _supabase_post([{'username': username, 'password_hash': password_hash}])
        return result is not None
    if USE_GITHUB:
        users, sha = _github_read_users()
        if users is None:
            return False
        if username in users:
            return False
        users[username] = password_hash
        return _github_write_users(users, sha)
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            users = json.load(f)
    if username in users:
        return False
    users[username] = password_hash
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)
    return True

def authenticate_user(username, password):
    if not username or not password:
        return False
    users = _get_users()
    return username in users and users[username] == hash_password(password)

def register_user(username, password):
    if not username or not password:
        return False, "Please fill all fields"
    if len(password) < 4:
        return False, "Password must be at least 4 characters"
    if _save_user(username, hash_password(password)):
        return True, "Registration successful! Please login."
    return False, "Username already exists"

_MODEL_CACHE = None
_CATEGORIES_CACHE = None

def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = tf.keras.models.load_model(MODEL_PATH)
    return _MODEL_CACHE

def get_categories():
    global _CATEGORIES_CACHE
    if _CATEGORIES_CACHE is None:
        _CATEGORIES_CACHE = sorted([d for d in os.listdir(DATADIR) if not d.startswith('.')])
    return _CATEGORIES_CACHE

if not st.session_state.authenticated:
    st.markdown("<h1 style='color:#4FC3F7;'>Sign & Speech Translator</h1>", unsafe_allow_html=True)
    st.markdown("### Login or Register to continue")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if authenticate_user(username, password):
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
                if reg_pass != reg_confirm:
                    st.error("Passwords do not match")
                else:
                    ok, msg = register_user(reg_user, reg_pass)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

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
    blank = Image.new('RGB', (150, 150), color=(255, 255, 255))
    for char in text:
        if char == " ":
            images.append(blank)
            continue
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

st.markdown("<h1 style='color:#4FC3F7;'>Sign & Speech Translator</h1>", unsafe_allow_html=True)

mode = st.radio("Choose Mode:", ['Sign Language to Text', 'Speech to Sign'], horizontal=True)

if mode == 'Sign Language to Text':
    import mediapipe as mp
    import threading
    import av
    from collections import Counter
    from streamlit_webrtc import webrtc_streamer, WebRtcMode

    BUFFER_SIZE = 8
    STABLE_THRESHOLD = 6
    COOLDOWN_FRAMES = 8
    NO_HAND_RESET = 5

    class SignVideoProcessor:
        def __init__(self):
            self.lock = threading.Lock()
            self._hands = None
            self._pred_class = None
            self._confidence = 0.0
            self._hand_count = 0

        def _get_hands(self):
            if self._hands is None:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False, max_num_hands=2,
                    min_detection_confidence=0.6, min_tracking_confidence=0.5,
                )
            return self._hands

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            try:
                if not hasattr(self, '_frame_count'):
                    self._frame_count = 0
                self._frame_count += 1

                h, w, _ = img.shape
                pred_class = None
                conf = 0.0
                num_hands = 0

                if self._frame_count % 3 == 1:
                    hands = self._get_hands()
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self._last_result = hands.process(rgb)
                    self._last_img = img.copy()

                result = getattr(self, '_last_result', None)
                if result is None:
                    with self.lock:
                        self._hand_count = 0
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

                if not result.multi_hand_landmarks:
                    with self.lock:
                        self._hand_count = 0
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

                num_hands = len(result.multi_hand_landmarks)
                hand_boxes = []
                all_x, all_y = [], []
                mp_drawing = mp.solutions.drawing_utils
                for lm in result.multi_hand_landmarks:
                    xs = [l.x * w for l in lm.landmark]
                    ys = [l.y * h for l in lm.landmark]
                    all_x.extend(xs); all_y.extend(ys)
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    hand_boxes.append((min(xs), min(ys), max(xs), max(ys), area, lm))
                    mp_drawing.draw_landmarks(
                        img, lm, mp.solutions.hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(79, 195, 247), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(2, 136, 209), thickness=1),
                    )
                pad = 25
                bx1 = max(0, int(min(all_x) - pad))
                by1 = max(0, int(min(all_y) - pad))
                bx2 = min(w, int(max(all_x) + pad))
                by2 = min(h, int(max(all_y) + pad))
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (79, 195, 247), 3)
                label = f"Detected ({num_hands})"
                cv2.putText(img, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (79, 195, 247), 2)

                if self._frame_count % 5 == 1:
                    dominant = max(hand_boxes, key=lambda b: b[4])
                    dom = dominant[5]
                    dxs = [l.x * w for l in dom.landmark]
                    dys = [l.y * h for l in dom.landmark]
                    hx1 = max(0, int(min(dxs) - pad))
                    hy1 = max(0, int(min(dys) - pad))
                    hx2 = min(w, int(max(dxs) + pad))
                    hy2 = min(h, int(max(dys) + pad))
                    roi = img[hy1:hy2, hx1:hx2]
                    if roi.size > 0:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                        arr = resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                        m = load_model()
                        cats = get_categories()
                        pred = m.predict(arr, verbose=0)
                        idx = np.argmax(pred)
                        conf = float(np.max(pred) * 100)
                        if "unknown" not in cats[idx]:
                            pred_class = cats[idx]

                with self.lock:
                    self._hand_count = num_hands
                    if pred_class is not None:
                        self._pred_class = pred_class
                        self._confidence = conf
            except Exception:
                pass
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def get_state(self):
            with self.lock:
                return self._pred_class, self._confidence, self._hand_count

    for key, default in [
        ('sign_buffer', ''), ('pred_buffer', []), ('conf_buffer', []),
        ('stable_count', 0), ('cooldown', 0), ('cur_sign', ''),
        ('cur_conf', 0.0), ('cur_stability', 0.0), ('hand_status', 'Waiting'),
        ('no_hand_frames', 0), ('audio_bytes', None), ('last_accepted', ''),
        ('cam_status', 'Off'),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown("### Live Sign Recognition")
    col_feed, col_info = st.columns([2, 1])
    with col_feed:
        st.markdown("Click **Start** below to turn on your camera")
        ctx = webrtc_streamer(
            key="isl-realtime",
            video_processor_factory=SignVideoProcessor,
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480},
                    "height": {"ideal": 360},
                    "frameRate": {"ideal": 15},
                },
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            },
        )
    with col_info:
        st.markdown("**Recognition Info**")
        is_on = ctx.state.playing if hasattr(ctx.state, 'playing') else False
        col = '🟢' if is_on else '🔴'
        cam_label = "Running" if is_on else "Off"
        st.markdown(f"**Camera:** {col} {cam_label}")
        st.markdown(f"**Hands:** {st.session_state.hand_status}")
        s = st.session_state.cur_sign if st.session_state.cur_sign else '—'
        st.markdown(f"**Sign:** {s}")
        c = f"{st.session_state.cur_conf:.0f}%" if st.session_state.cur_conf > 0 else '—'
        st.markdown(f"**Confidence:** {c}")
        if st.session_state.cur_stability > 0:
            st.markdown(f"**Stability:** {st.session_state.cur_stability:.0%}")

    st.text_area("Generated Text", value=st.session_state.sign_buffer, height=100, disabled=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.sign_buffer = ''
            st.session_state.audio_bytes = None
            st.rerun()
    with col_b2:
        if st.button("🔊 Speak", use_container_width=True):
            if st.session_state.sign_buffer.strip():
                tts = gTTS(text=st.session_state.sign_buffer, lang='en')
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                st.session_state.audio_bytes = buf.getvalue()

    if st.session_state.audio_bytes:
        st.audio(st.session_state.audio_bytes, format='audio/mp3')

    if ctx.state.playing:
        st.session_state.cam_status = "Running"
        if ctx.video_processor:
            pred_class, conf, num_hands = ctx.video_processor.get_state()
            st.session_state.hand_status = f"Detected ({num_hands})" if num_hands > 0 else "No hands"

            if num_hands == 0:
                st.session_state.no_hand_frames += 1
                st.session_state.pred_buffer = []
                st.session_state.conf_buffer = []
                st.session_state.cur_sign = ''
                st.session_state.cur_conf = 0.0
                st.session_state.cur_stability = 0.0
                st.session_state.stable_count = 0
                if st.session_state.no_hand_frames >= NO_HAND_RESET:
                    st.session_state.last_accepted = ''
                    st.session_state.cooldown = 0
            else:
                st.session_state.no_hand_frames = 0

            if pred_class is not None:
                st.session_state.pred_buffer.append(pred_class)
                st.session_state.conf_buffer.append(conf)
                if len(st.session_state.pred_buffer) > BUFFER_SIZE:
                    st.session_state.pred_buffer.pop(0)
                    st.session_state.conf_buffer.pop(0)

                buf = st.session_state.pred_buffer
                if len(buf) >= max(3, BUFFER_SIZE // 2):
                    counter = Counter(buf)
                    top_class, top_count = counter.most_common(1)[0]
                    stability = top_count / len(buf)
                    idxs = [i for i, c in enumerate(buf) if c == top_class]
                    avg_conf = float(np.mean([st.session_state.conf_buffer[i] for i in idxs]))
                    st.session_state.cur_sign = top_class
                    st.session_state.cur_conf = avg_conf
                    st.session_state.cur_stability = float(stability)

                    if st.session_state.cooldown > 0:
                        st.session_state.cooldown -= 1
                    elif top_class == st.session_state.last_accepted:
                        st.session_state.stable_count = 0
                    elif stability >= 0.6 and avg_conf > 50:
                        st.session_state.stable_count += 1
                        if st.session_state.stable_count >= STABLE_THRESHOLD:
                            st.session_state.sign_buffer += top_class
                            st.session_state.last_accepted = top_class
                            st.session_state.cooldown = COOLDOWN_FRAMES
                            st.session_state.stable_count = 0
                    else:
                        st.session_state.stable_count = 0

            now = time.time()
            sig = (num_hands, pred_class, int(conf), len(st.session_state.sign_buffer))
            prev_sig = st.session_state.get('_prev_sig')
            last_rerun = st.session_state.get('_last_rerun', 0)
            st.session_state._prev_sig = sig
            needs_rerun = (sig != prev_sig) or (now - last_rerun > 1.0)
            if needs_rerun:
                st.session_state._last_rerun = now
                time.sleep(0.2)
                st.rerun()

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
