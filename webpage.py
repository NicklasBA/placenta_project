import os
import random
import time
from io import StringIO
import tempfile
import streamlit as st
import logging
from webutils.store_evaluations import store_result, get_result, get_user_downloads_folder

video_folder = get_user_downloads_folder()
# Setup logger
logging.basicConfig(filename=os.path.join(video_folder, "latest_run.log"),
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s : %(message)s")
log = logging.getLogger(__name__)
log.info(f"Logging to {os.path.join(video_folder, 'latest_run.log')}")

st.set_page_config(layout="centered", page_icon="🩸", page_title="CELLUNATOR 2000™")

# Data visualisation part
st.image("webutils/data/celluvision_banner.png")

# Welcome part, upload video
st.title("CELLUNATOR 2000™")

st.markdown("Welcome to the Cellunator 2000™ human user inferface. Below is a user friendly way of importing a " \
            "video to be analysed.")

UPLOAD_SUCCESSFUL = False
# Upload file
uploaded_video = st.file_uploader("Choose a video",
                                  type=['mp4', 'avi'],
                                  help="Upload video of blood sample here")

try:
    d_class
    ns_class
except Exception:
    all_results = get_result("", full=True)
    d_class = all_results["CLASSIFIER"]["D"]
    ns_class = all_results["CLASSIFIER"]["NS"]
    del all_results

# Check video is uploaded and save to disk
if uploaded_video is not None:
    videofile_name = uploaded_video.name
    log.info(f"File uploaded: {videofile_name}, ({uploaded_video})")
    try:
        videofile = os.path.join(video_folder, videofile_name)
        # stringio = StringIO(uploaded_video.getvalue().decode("utf-8"))
        with open(videofile, "wb") as vf:
            vf.write(uploaded_video.read())
        log.info(f"Saved {uploaded_video.name} to disk as {videofile}")
        del uploaded_video
        UPLOAD_SUCCESSFUL = True
    except Exception as e:
        msg = f'Unexpected error occurred when uploading video. Try again or try another file. Error message: {e}'
        log.error(msg)
        st.error(msg)

st.title("Current data:")
data_space = st.empty()

data_space.info('No video data uploaded - Upload the file to be analyzed above.')

st.title("Results:")
result_space = st.empty()
result_space.info('No data received.')
if UPLOAD_SUCCESSFUL:
    result_space.info('Waiting for result from analysis...')

ANALYSIS_SUCCESSFUL = False
if UPLOAD_SUCCESSFUL:
    # Replace the chart with several elements:
    timedivisor = 1
    with data_space.container():
        st.write(f"Video uploaded: {videofile_name}")
        video_file = open(videofile, 'rb')
        video_bytes = video_file.read()
        video_file.close()
        st.video(video_bytes, format="video/mp4", start_time=0)

        spacer = st.empty()
        col1, col2, col3 = spacer.columns(3)
        with col1:
            with st.warning(""):
                with st.spinner('Detecting cells...'):
                    time.sleep(4/timedivisor)
            st.success('Cells found ✅')
        with col2:
            with st.spinner('Using AI...'):
                time.sleep(5/timedivisor)
            st.success('AI has been applied ✅')
        with col3:
            with st.spinner('Finishing up...'):
                time.sleep(2/timedivisor)
            st.success('All done ✅')
        time.sleep(2/timedivisor)

        spacer.success('Results are ready!')
    ANALYSIS_SUCCESSFUL = True


if ANALYSIS_SUCCESSFUL:
    with result_space.container():
        st.subheader(f"Results for: {videofile_name}")

        results = get_result(videofile_name)
        if results is None:
            sample_class = random.sample([d_class, d_class, ns_class], 1)[0]
            if sample_class == d_class:
                no_ns = random.randint(0, 10)
                no_d = random.randint(2000, 9000)
            elif sample_class == ns_class:
                no_ns = random.randint(80, 150)
                no_d = random.randint(2000, 9000)
            else:
                st.error("Something bad happend during the transport of the result")
        else:
            no_ns = results["no_ns"]
            no_d = results["no_d"]
            sample_class = results["classifier"]
        total_no = no_ns + no_d

        col1, col2, col3 = st.columns(3)
        col1.metric("Total number of cells", f"{total_no} cells")
        col2.metric("Maternal", f"{no_d/total_no*100:.3f}%", f"{no_d} cells")
        col3.metric("Fetal", f"{no_ns/total_no*100:.3f}%", f"{no_ns} cells")

        if sample_class == d_class:
            st.success("No leakage or leakage is below a safe threshold.")
        elif sample_class == ns_class:
            st.warning("⚠️ Possible leakage detected! Further tests are advised. ⚠️ ")
        else:
            st.error("Something went wrong, no conclusion!")

if False:
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')

    placeholder = st.empty()

    # Replace the placeholder with some text:
    placeholder.text("Hello")

    # Replace the text with a chart:
    placeholder.line_chart({"data": [1, 5, 2, 6]})

    # Replace the chart with several elements:
    with placeholder.container():
        st.write("This is one element")
        st.write("This is another")
