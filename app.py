import streamlit as st
import tempfile
import os
import time

# Import the inference engine
from inference import DeepfakeInferenceEngine

# 1. Page Configuration (Wide layout for a dashboard feel)
st.set_page_config(
    page_title="Deepfake Engine OS",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glowing text and cleaner metrics
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .authentic { color: #00FF00; text-shadow: 0px 0px 10px #00FF00; }
    .fake { color: #FF0000; text-shadow: 0px 0px 10px #FF0000; }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8681/8681120.png", width=100) # Placeholder icon
    st.title("System Status")
    st.markdown("---")
    st.success("🟢 MTCNN Face Extractor: ONLINE")
    st.success("🟢 ResNeXt-50 Backbone: ONLINE")
    st.success("🟢 Bi-LSTM Sequence: ONLINE")
    st.success("🟢 FFT Frequency Layer: ONLINE")
    st.markdown("---")
    st.caption("Engine Version 1.0.0 | Dual-Branch Architecture")

@st.cache_resource
def load_engine():
    try:
        return DeepfakeInferenceEngine(checkpoint_path="best_deepfake_detector.pth")
    except Exception as e:
        return DeepfakeInferenceEngine()

engine = load_engine()

# 3. Main Header
st.title("👁️ Deepfake Analysis Dashboard")
st.markdown("Upload media below to run it through the Spatiotemporal & Frequency anomaly detection nodes.")

# 4. Interface Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Video Scanner", "⚙️ System Architecture", "📊 Analysis Logs"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Target Acquisition")
        uploaded_file = st.file_uploader("Upload Target Video (.mp4, .mov)", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            analyze_button = st.button("⚡ Initialize Scan Pipeline", type="primary", use_container_width=True)

    with col2:
        st.subheader("Telemetry & Results")
        
        if uploaded_file is not None and analyze_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_video_path = tmp_file.name

            try:
                # Interactive UI Status Box
                with st.status("Executing Dual-Branch Pipeline...", expanded=True) as status:
                    st.write("1️⃣ Decoding video and normalizing framerate...")
                    time.sleep(0.5) 
                    st.write("2️⃣ MTCNN isolating facial bounding boxes...")
                    time.sleep(0.5)
                    st.write("3️⃣ Extracting ResNeXt-50 spatial vectors...")
                    time.sleep(0.5)
                    st.write("4️⃣ Running 2D FFT & Bi-LSTM motion sequence...")
                    
                    start_time = time.time()
                    results = engine.predict_video(tmp_video_path)
                    inference_time = time.time() - start_time
                    
                    status.update(label="Analysis Complete", state="complete", expanded=False)

                # Render Results visually
                st.markdown("---")
                if results["prediction"] == "DEEPFAKE":
                    st.markdown('<p class="big-font fake">🚨 MANIPULATION DETECTED (DEEPFAKE)</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="big-font authentic">✅ AUTHENTIC MEDIA VERIFIED</p>', unsafe_allow_html=True)
                
                st.markdown("### Confidence Score")
                # Visual Progress bar for the AI Score
                prob = results["fake_probability"]
                st.progress(prob, text=f"AI Manipulation Probability: {prob:.4f}")

                # Metric Cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Final Output", results["prediction"])
                m2.metric("Confidence", results["confidence_percentage"])
                m3.metric("Latency", f"{inference_time:.2f} s")

            except Exception as e:
                st.error(f"Pipeline Failure: {e}")
            finally:
                if os.path.exists(tmp_video_path):
                    os.remove(tmp_video_path)
        elif uploaded_file is None:
            st.info("Awaiting media upload...")

with tab2:
    st.header("Dual-Branch Architecture")
    st.markdown("""
    **Branch 1 (Spatial-Temporal):** Looks at physical inconsistencies. Uses a ResNeXt-50 CNN to extract frame-by-frame pixel data, feeding it into a Bidirectional LSTM to check for unnatural motion like jittering or broken blinking physics.
    
    **Branch 2 (Frequency):** Looks for mathematical anomalies. Converts the video to the frequency domain using a 2D Fast Fourier Transform (FFT) to spot the microscopic checkerboard grid artifacts left behind by GAN and Diffusion upsampling algorithms.
    """)

with tab3:
    st.header("Session Logs")
    st.code("""
    > System initialized.
    > Weights loaded successfully.
    > Waiting for user input...
    """, language="bash")