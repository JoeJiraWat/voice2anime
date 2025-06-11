import os
import tempfile
import torch
import torchaudio
from transformers import pipeline
import streamlit as st

# ตั้งค่าที่ปลอดภัย
os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"
os.makedirs("/app/hf_cache", exist_ok=True)

st.title("🎧 AI แปลงเสียงเป็น Anime Voice")

@st.cache_resource
def load_model():
    return pipeline(
        "audio-to-audio",
        model="B4by/Test",
        framework="pt",
        cache_dir="/app/hf_cache"
    )

model = load_model()

uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียง (.wav, .mp3)", type=["wav", "mp3"])
if uploaded_file:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded_file.read())
        input_path = f.name

    st.info("กำลังแปลงเสียงเป็น Anime... 🎶")
    output = model(input_path)
    audio_out = output["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f_out:
        torchaudio.save(f_out.name, torch.tensor(audio_out).unsqueeze(0), 16000)
        output_path = f_out.name

    st.audio(output_path)
    with open(output_path, "rb") as f:
        st.download_button("📥 ดาวน์โหลดเสียง Anime", f, "anime_voice.wav")

    os.remove(input_path)
    os.remove(output_path)
