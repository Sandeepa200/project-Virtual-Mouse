import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import platform
import os
from pyvirtualdisplay import Display


def setup_environment():
    """Set up environment based on operating system."""
    if platform.system() == "Linux":
        try:
            # Set up virtual display for Linux (cloud deployment)
            if 'DISPLAY' not in os.environ:
                os.environ['DISPLAY'] = ':99'
            display = Display(visible=0, size=(1024, 768))
            display.start()
            return display
        except Exception as e:
            st.error("Failed to initialize display server. Please check if Xvfb is installed.")
            st.stop()
    else:
        # Windows/MacOS (local development)
        return None


def main():
    # Initialize environment
    display = setup_environment()

    try:
        st.title("Virtual Mouse Control")
        st.write("Control your mouse using hand gestures!")

        # Initialize session state variables
        if 'is_dragging' not in st.session_state:
            st.session_state.is_dragging = False
        if 'running' not in st.session_state:
            st.session_state.running = True

        # Sidebar controls
        st.sidebar.header("Controls")
        detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.7)

        if st.sidebar.button('Stop'):
            st.session_state.running = False
            st.experimental_rerun()

        # Initialize components
        hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        drawing_utils = mp.solutions.drawing_utils
        screen_width, screen_height = pyautogui.size()

        # Create placeholders
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # Start webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not access webcam! Please check your camera settings.")
            st.stop()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video frame!")
                break

            # Process frame
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            output = hand_detector.process(rgb_frame)
            hands = output.multi_hand_landmarks

            if hands:
                for hand in hands:
                    drawing_utils.draw_landmarks(frame, hand)
                    landmarks = hand.landmark

                    index_finger = landmarks[8]
                    thumb = landmarks[4]

                    index_x = int(index_finger.x * frame_width)
                    index_y = int(index_finger.y * frame_height)
                    screen_x = min(screen_width, max(0, screen_width / frame_width * index_x))
                    screen_y = min(screen_height, max(0, screen_height / frame_height * index_y))

                    cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255))
                    thumb_x = int(thumb.x * frame_width)
                    thumb_y = int(thumb.y * frame_height)
                    cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255))

                    vertical_distance = abs(index_y - thumb_y)

                    try:
                        pyautogui.moveTo(screen_x, screen_y, duration=0.1)

                        if vertical_distance < 40 and not st.session_state.is_dragging:
                            pyautogui.mouseDown()
                            st.session_state.is_dragging = True
                            status_placeholder.info("Dragging")
                        elif vertical_distance >= 40 and st.session_state.is_dragging:
                            pyautogui.mouseUp()
                            st.session_state.is_dragging = False
                            status_placeholder.info("Released")

                        metrics_placeholder.metric(
                            label="Hand Distance",
                            value=f"{vertical_distance:.1f}",
                            delta="Dragging" if st.session_state.is_dragging else "Released"
                        )
                    except pyautogui.FailSafeException:
                        st.warning("Mouse movement prevented - failsafe triggered")

            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Cleanup
        if 'is_dragging' in st.session_state and st.session_state.is_dragging:
            pyautogui.mouseUp()
        if cap is not None:
            cap.release()
        if hand_detector is not None:
            hand_detector.close()
        if display is not None:
            display.stop()


if __name__ == "__main__":
    main()