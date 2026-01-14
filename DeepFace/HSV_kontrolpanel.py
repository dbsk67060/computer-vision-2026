import streamlit as st
import cv2
import numpy as np

st.subheader("HSV kontrolpanel")

if "screenshot" in st.session_state:
    bgr = st.session_state["screenshot"]
    bgr = cv2.resize(bgr, (400, 600))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # --- SLIDERS (erstatter trackbars) ---
    col1, col2 = st.columns(2)

    with col1:
        h_min = st.slider("H min", 0, 179, 0)
        s_min = st.slider("S min", 0, 255, 40)
        v_min = st.slider("V min", 0, 255, 40)

    with col2:
        h_max = st.slider("H max", 0, 179, 25)
        s_max = st.slider("S max", 0, 255, 255)
        v_max = st.slider("V max", 0, 255, 230)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # --- HSV mask ---
    mask = cv2.inRange(hsv, lower, upper)

    # --- Morfologi ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- Apply mask ---
    result = cv2.bitwise_and(bgr, bgr, mask=mask)

    # --- VISNING ---
    v1, v2, v3 = st.columns(3)

    v1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original")
    v2.image(mask, caption="HSV mask", clamp=True)
    v3.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Fremhævet")

else:
    st.info("Tag et screenshot for at aktivere HSV-kontrol.")
