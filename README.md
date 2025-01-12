# Virtual Mouse Control with Streamlit and Mediapipe

This project demonstrates a virtual mouse control system using hand gestures, developed with Streamlit, Mediapipe, and OpenCV. It provides flexibility for both local development and deployment on Streamlit Cloud.

---

## Features
- **Hand Gesture Detection**: Uses Mediapipe for real-time hand tracking and gesture recognition.
- **Mouse Control**: Controls mouse movement and dragging actions based on finger positions.
- **Streamlit Interface**: Includes an interactive web-based interface for running the application.
- **Local Running Support**: A non-Streamlit version is available for complete local execution.

---

## Project Structure
```plaintext
.
├── app.py                  # Main file for Streamlit Cloud deployment do not use this.
├── local_streamlit_app.py  # For running the Streamlit app locally.
├── local_code.py           # Standalone local script without Streamlit.
├── requirements.txt        # Python dependencies.
└── README.md               # Project documentation.
