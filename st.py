import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from string import ascii_lowercase
from googletrans import Translator
from gtts import gTTS
import pygame
from PIL import Image
import time
import speech_recognition as sr

# ================== CONFIG ==================
IMG_SIZE = 50
DATADIR = 'dataset'
AUDIO_FOLDER = 'test'
MODEL_PATH = 'CNN.model'

# ================== LOAD MODEL ==================
model = tf.keras.models.load_model(MODEL_PATH)
CATEGORIES = os.listdir(DATADIR)

if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)}

# ================== UTILS ==================
def alphabet_position(text):
    text = text.lower()
    return [LETTERS[char] for char in text if char in LETTERS]

# ================== SIGN TO TEXT ==================
def predict_sign_language():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    box_size = 300
    det = [
        'hi how are you', 'i dont know', 'what is your name', 'who are you',
        'what is this', 'where are you', 'how are you', 'i am hungry',
        'i am ironman', 'i love you', 'i hate you', 'i am sick',
        'i am sleeping', 'i am thirsty', 'i am in home', 'thankyou'
    ] * 2

    st.info("Press *Q* to stop detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== COLOR ROI ONLY =====
        hand = frame[0:box_size, 0:box_size]
        # Convert to grayscale
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)

        # Resize
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        # Normalize
        img_array = resized / 255.0

        # Reshape for CNN (IMPORTANT: 1 channel)
        img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


        prediction = model.predict(img_array, verbose=0)
        index = np.argmax(prediction)
        category = CATEGORIES[index]

        # Draw bounding box
        cv2.rectangle(frame, (0, 0), (box_size, box_size), (0, 255, 0), 2)
        cv2.putText(frame, category, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show COLOR frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels='RGB')

        if "unknown" not in category:
            time.sleep(0.3)

            text = det[index]
            st.success(f"Detected: {text}")

            translated = Translator().translate(text, dest='ta').text
            st.write("Tamil:", translated)

            tts = gTTS(text=translated, lang='ta')
            audio_path = os.path.join(AUDIO_FOLDER, "output_audio.mp3")
            tts.save(audio_path)

            try:
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                pygame.quit()
            except Exception as e:
                st.error(f"Audio error: {e}")

            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# ================== SPEECH TO SIGN ==================
def speech_to_sign():
    r = sr.Recognizer()
    mic = sr.Microphone()

    st.write("🎤 Speak now...")
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        recog = r.recognize_google(audio)
        st.success(f"You said: {recog}")

        images = []
        for char in recog:
            if char == " ":
                img_path = 'test/space.png'
            elif char.lower() in LETTERS:
                l = alphabet_position(char)[0]
                img_path = f'test/{int(l)}.jpg'
            else:
                continue

            if os.path.exists(img_path):
                img = Image.open(img_path).resize((150, 150))
                images.append(img)

        if images:
            st.image(images, width=75)
        else:
            st.warning("No valid characters")

    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"API error: {e}")

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Sign & Speech Translator", layout="centered")

st.markdown("<h1 style='color:#3498db;'>🧠 AI Sign & Speech Translator</h1>", unsafe_allow_html=True)

option = st.radio(
    "Choose Mode:",
    ['🖐️ Sign Language to Text (Deaf Mode)', '🗣️ Speech to Sign (Dumb Mode)']
)

if option == '🖐️ Sign Language to Text (Deaf Mode)':
    if st.button("Start Camera"):
        predict_sign_language()

else:
    if st.button("Start Recording"):
        speech_to_sign()
