import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from PIL import Image
import os


def initialize_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )


def main():
    st.title("Virtual Mouse Control")
    st.write("Control your mouse using hand gestures!")

    # Initialize session state variables
    if 'is_dragging' not in st.session_state:
        st.session_state.is_dragging = False
    if 'running' not in st.session_state:
        st.session_state.running = True

    # Sidebar controls
    st.sidebar.header("Controls")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.7, key="detection_slider")

    # Stop button in sidebar with unique key
    if st.sidebar.button('Stop', key='stop_button'):
        st.session_state.running = False
        st.experimental_rerun()

    # Initialize components
    hand_detector = initialize_hand_detector()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()

    # Create placeholders
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Start webcam
    cap = cv2.VideoCapture(0)

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not access webcam turn off other applications that use webcam and retry!")
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

                    # Extract index finger and thumb positions
                    index_finger = landmarks[8]
                    thumb = landmarks[4]

                    # Convert coordinates
                    index_x = int(index_finger.x * frame_width)
                    index_y = int(index_finger.y * frame_height)
                    screen_x = min(screen_width, max(0, screen_width / frame_width * index_x))
                    screen_y = min(screen_height, max(0, screen_height / frame_height * index_y))

                    # Draw markers
                    cv2.circle(img=frame, center=(index_x, index_y),
                               radius=10, color=(0, 255, 255))

                    thumb_x = int(thumb.x * frame_width)
                    thumb_y = int(thumb.y * frame_height)
                    cv2.circle(img=frame, center=(thumb_x, thumb_y),
                               radius=10, color=(0, 255, 255))

                    # Calculate distance
                    vertical_distance = abs(index_y - thumb_y)

                    # Update mouse position and handle dragging
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)

                    if vertical_distance < 40 and not st.session_state.is_dragging:
                        pyautogui.mouseDown()
                        st.session_state.is_dragging = True
                        status_placeholder.info("Dragging")

                    elif vertical_distance >= 40 and st.session_state.is_dragging:
                        pyautogui.mouseUp()
                        st.session_state.is_dragging = False
                        status_placeholder.info("Released")

                    # Update metrics without key parameter
                    metrics_placeholder.metric(
                        label="Hand Distance",
                        value=f"{vertical_distance:.1f}",
                        delta="Dragging" if st.session_state.is_dragging else "Released"
                    )

            # Convert frame to RGB for Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if st.session_state.is_dragging:
            pyautogui.mouseUp()
        cap.release()
        hand_detector.close()


if __name__ == "__main__":
    main()