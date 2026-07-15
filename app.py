import os
import tempfile
import streamlit as st
import matplotlib.pyplot as plt

from src.inference.video_predictor import VideoPredictor

# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="DeepShield",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

</style>
""",
unsafe_allow_html=True)

# ==================================================
# Sidebar
# ==================================================

st.sidebar.success("DeepShield v1.0")

st.sidebar.divider()

st.sidebar.markdown("### 🤖 Model")
st.sidebar.info("EfficientNetB0\n\nFine-Tuned CNN")

st.sidebar.markdown("### 📂 Dataset")
st.sidebar.info("Celeb-DF")

st.sidebar.markdown("### 📹 Supported Formats")
st.sidebar.code("MP4\nAVI\nMOV")

st.sidebar.markdown("### 🎯 Prediction")
st.sidebar.success("Real / Fake")

st.sidebar.divider()

st.sidebar.caption("Developed by Tanmay Tawade")

# ==================================================
# Title
# ==================================================

st.title("🛡️ DeepShield")

st.subheader("AI Powered Deepfake Detection")

st.caption(
    "Real-Time Video Deepfake Detection using EfficientNetB0 Fine-Tuned CNN"
)

st.info(
"""
🛡 **DeepShield** is an AI-powered system for detecting manipulated facial videos.

Workflow

📹 Upload Video
➡ Extract Frames
➡ Detect Faces
➡ CNN Classification
➡ Majority Voting
➡ Final Prediction
"""
)

st.divider()

# ==================================================
# Upload
# ==================================================

st.markdown("## 📤 Upload Video")

st.caption("Supported formats: MP4 • AVI • MOV")

uploaded_video = st.file_uploader(
    "",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is None:

    st.info("Upload a video to begin.")

    st.stop()

# ==================================================
# Save Uploaded Video
# ==================================================

temp_video = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
)

temp_video.write(uploaded_video.getbuffer())
temp_video.flush()
temp_video.close()

video_path = temp_video.name

# ==================================================
# Preview
# ==================================================

st.markdown("## 🎥 Video Preview")

st.video(uploaded_video.getvalue())

# ==================================================
# Analyze Button
# ==================================================

if st.button(
    "🚀 Analyze Video",
    use_container_width=True
):

    MODEL_PATH = "models/finetune/best_cnn_finetuned.keras"

    # Create progress widgets

    progress_bar = st.progress(0)

    status = st.empty()


    def update_progress(progress, message):

        progress_bar.progress(int(progress * 100))

        status.info(message)


    predictor = VideoPredictor(MODEL_PATH)

    result = predictor.predict_video(

        video_path,

        progress_callback=update_progress

    )

    progress_bar.empty()

    status.success("✅ AI Analysis Completed Successfully")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        
        st.markdown("## 🤖 AI Prediction")

        if result["prediction"] == "Fake":

            st.error(
                "🚨 Fake Video Detected\n\nThe uploaded video shows strong signs of manipulation."
            )

        else:

            st.success(
                "✅ Authentic Video\n\nNo significant manipulation was detected."
            )

        confidence = result["confidence"] * 100

        st.markdown("### Model Confidence")

        st.progress(int(confidence))

        st.markdown(
            f"<h3 style='text-align:center'>{confidence:.2f}%</h3>",
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("## 📊 Analysis Summary")

    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.metric(
            "📷 Frames extracted",
            result["frames"]
        )

    with row1_col2:
        st.metric(
            "😀 Faces detected",
            result["faces"]
        )

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.metric(
            "🟢 Real Frames",
            result["real_frames"]
        )

    with row2_col2:
        st.metric(
            "🔴 Fake Frames",
            result["fake_frames"]
        )

    with row2_col3:
        st.metric(
            "⏱ Processing Time",
            f"{result['time']:.2f} sec"
        )
        
    st.divider()

    left_col, right_col = st.columns([1, 1])

    # ==================================================
    # Donut Chart
    # ==================================================

    with left_col:

        st.subheader("📊 Frame Prediction Distribution")

        real = result["real_frames"]
        fake = result["fake_frames"]

        fig, ax = plt.subplots(figsize=(3.4, 3.4))

        colors = ["#4CAF50", "#F44336"]

        wedges, texts, autotexts = ax.pie(
            [real, fake],
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="white"),
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            pctdistance=0.78,
            textprops={
                "fontsize":9,
                "fontweight":"bold",
                "color":"white"
            }
        )

        ax.legend(
            wedges,
            [f"🟢 Real ({real})", f"🔴 Fake ({fake})"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False,
            fontsize=9
        )

        ax.set(aspect="equal")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # ==================================================
    # Sample Faces
    # ==================================================

    with right_col:

        st.markdown("## 🖼 Sample Analyzed Faces")

        st.caption(
            "Representative face crops used by the model during analysis."
        )

        frames = result["sample_frames"]

        if len(frames) == 0:

            st.info("No sample frames available.")

        else:

            cols = st.columns(2)

            for index, frame in enumerate(frames):

                with cols[index % 2]:

                    st.image(
                        frame["path"],
                        use_container_width=True
                    )

                    confidence = frame["confidence"] * 100

                    if frame["label"] == "Fake":

                        st.error(
                            f"🔴 Fake ({confidence:.2f}%)"
                        )

                    else:

                        st.success(
                            f"🟢 Real ({confidence:.2f}%)"
                        )
                        
    st.divider()

    st.markdown(
        """
    <div style="text-align:center;color:gray;font-size:13px;opacity:0.8;">

    DeepShield v1.0

    Powered by TensorFlow • EfficientNetB0 • OpenCV • MTCNN • Streamlit

    © 2026 Tanmay Tawade

    </div>
    """,
    unsafe_allow_html=True
    )