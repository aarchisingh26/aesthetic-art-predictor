import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color
from io import BytesIO
import requests

st.set_page_config(page_title="Aesthetic Success Predictor", layout="wide")
st.title("Aesthetic Art/Graphics Predictor")

st.markdown("""
Upload an image and get insights into how aesthetically pleasing it is, based on principles like:
- Color harmony (using LAB color space and clustering)
- Composition (rule of thirds, symmetry)
- Emotional tone (color psychology)
""")

with st.expander("📷 Upload or Paste Image URL"):
    upload_col, url_col = st.columns(2)
    image = None

    with upload_col:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    with url_col:
        url = st.text_input("Or paste an image URL")
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
            except:
                st.error("Could not load image from URL")

if image:
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    img_np = np.array(image)
    if img_np.shape[-1] == 4:  # RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    
    img_resized = cv2.resize(img_np, (400, 400))

   
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Colors", "📐 Composition", "😊 Emotion", "📊 Final Score"])

    
    with tab1:
        st.subheader("Color Harmony")
        lab_img = color.rgb2lab(img_resized / 255.0)
        flat_lab = lab_img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=5).fit(flat_lab)
        colors = color.lab2rgb(kmeans.cluster_centers_.reshape(1, 5, 3)).reshape(5, 3)

        st.markdown("**Dominant Colors:**")
        color_cols = st.columns(5)
        for i in range(5):
            rgb = (colors[i] * 255).astype(int)
            hex_color = '#%02x%02x%02x' % tuple(rgb)
            with color_cols[i]:
                st.color_picker("", hex_color, label_visibility="collapsed")

        harmony_score = 1 - np.std(kmeans.cluster_centers_) / 100  # heuristic
        st.metric("🎨 Color Harmony Score", f"{harmony_score * 100:.1f} / 100")

    
    with tab2:
        st.subheader("Rule of Thirds & Balance")

        thirds_img = img_resized.copy()
        h, w = thirds_img.shape[:2]
        for i in [1, 2]:
            cv2.line(thirds_img, (0, i*h//3), (w, i*h//3), (255, 255, 255), 1)
            cv2.line(thirds_img, (i*w//3, 0), (i*w//3, h), (255, 255, 255), 1)

        st.image(thirds_img, caption="Rule of Thirds Grid", use_container_width=True)

        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        center_mass = np.array(np.unravel_index(np.argmax(gray), gray.shape))
        center_mass_norm = center_mass / np.array(gray.shape)

        distance_to_center = np.linalg.norm(center_mass_norm - np.array([0.5, 0.5]))
        composition_score = max(0, 1 - distance_to_center)

        st.metric("📐 Composition Score", f"{composition_score * 100:.1f} / 100")

   
    with tab3:
        st.subheader("Color Psychology & Emotion")

       
        avg_color = img_resized.mean(axis=(0, 1)) / 255
        r, g, b = avg_color
        emotion = ""
        if r > 0.6 and b < 0.4:
            emotion = "Energetic / Passionate"
        elif b > 0.5:
            emotion = "Calm / Cool"
        elif g > 0.5:
            emotion = "Natural / Relaxing"
        elif r > 0.5 and g > 0.5:
            emotion = "Warm / Cheerful"
        else:
            emotion = "Neutral / Subtle"

        st.metric("😊 Emotional Tone", emotion)

    
    with tab4:
        st.subheader("📊 Aesthetic Score")
        final_score = (harmony_score + composition_score) / 2 * 100
        st.success(f"🌟 Aesthetic Score: {final_score:.1f} / 100")

        if final_score > 80:
            st.markdown("Great aesthetic! Balanced, pleasing, and harmonious.")
        elif final_score > 60:
            st.markdown("Looks good, though there may be room for minor improvements in color or composition.")
        else:
            st.markdown("Consider adjusting color balance or following the rule of thirds for better visual appeal.")
