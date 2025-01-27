import streamlit as st
from imageai.Detection import VideoObjectDetection
import os
import time

def main():
    st.title("Video Object Detection with ImageAI")
    st.write("Upload a video file and perform object detection using TinyYOLOv3.")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    frames_per_second = st.sidebar.slider("Frames per second for processing", 1, 30, 10)
    min_probability = st.sidebar.slider("Minimum probability for detection (%)", 10, 100, 30)

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        with open("temp_input_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        st.video("temp_input_video.mp4")

        if st.button("Run Object Detection"):
            st.write("Running object detection...")

            # Initialize detector
            video_detector = VideoObjectDetection()
            video_detector.setModelTypeAsTinyYOLOv3()

            model_path = os.path.join(os.getcwd(), "D:\\Git\\Infosys-springboard-internship-5.0\\Project\\models\\tiny-yolov3.pt")

            if not os.path.isfile(model_path):
                st.error(f"Model file not found at {model_path}. Please upload the correct model file.")
                return

            try:
                video_detector.setModelPath(model_path)
                video_detector.loadModel()
            except Exception as e:
                st.error(f"Error loading the model: {e}")
                return

        # Define placeholders for dynamic updates
            frame_placeholder = st.empty()
            second_placeholder = st.empty()
            minute_placeholder = st.empty()

            def forFrame(frame_number, output_array, output_count):
                # Update the frame-related details in the placeholder
                frame_placeholder.markdown(f"""
                    ### Frame: {frame_number}
                    **Objects Detected:** {output_array}  
                    **Unique Object Counts:** {output_count}  
                """)

            def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
                # Update the second-related details in the placeholder
                second_placeholder.markdown(f"""
                    ### Second: {second_number}
                    **Objects Detected Per Frame:** {output_arrays}  
                    **Unique Object Counts Per Frame:** {count_arrays}  
                    **Average Unique Objects:** {average_output_count}  
                """)

            def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
                # Update the minute-related details in the placeholder
                minute_placeholder.markdown(f"""
                    ### Minute: {minute_number}
                    **Objects Detected Per Frame:** {output_arrays}  
                    **Unique Object Counts Per Frame:** {count_arrays}  
                    **Average Unique Objects Per Minute:** {average_output_count}  
                """)

            # Ensure output file is correctly specified
            output_video_path = "D:\\Git\\Infosys-springboard-internship-5.0\\Project\\output\\output_video.mp4"

            start_time = time.time()

            try:
                video_detector.detectObjectsFromVideo(
                    input_file_path="temp_input_video.mp4",
                    output_file_path=output_video_path,
                    frames_per_second=frames_per_second,
                    per_second_function=forSeconds,
                    per_frame_function=forFrame,
                    per_minute_function=forMinute,
                    minimum_percentage_probability=min_probability,
                    save_detected_video=True
                )

                end_time = time.time()
                st.success(f"Object detection completed in {end_time - start_time:.2f} seconds.")
                st.video(output_video_path)

            except Exception as e:
                st.error(f"Error during object detection: {e}")

if __name__ == "__main__":
    main()
