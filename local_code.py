import cv2
import mediapipe as mp
import pyautogui
import sys

# Initialize PyAutoGUI settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Added small delay between PyAutoGUI commands


def initialize_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    return cap


def initialize_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )


def main():
    # Initialize components
    cap = initialize_capture()
    hand_detector = initialize_hand_detector()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    index_x = index_y = 0
    is_dragging = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Process frame
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                output = hand_detector.process(rgb_frame)
                hands = output.multi_hand_landmarks

                if hands:
                    for hand in hands:
                        drawing_utils.draw_landmarks(frame, hand)
                        landmarks = hand.landmark

                        # Extract index finger and thumb positions
                        index_finger = landmarks[8]
                        thumb = landmarks[4]

                        # Convert index finger coordinates
                        index_x = int(index_finger.x * frame_width)
                        index_y = int(index_finger.y * frame_height)
                        screen_x = min(screen_width, max(0, screen_width / frame_width * index_x))
                        screen_y = min(screen_height, max(0, screen_height / frame_height * index_y))

                        # Draw index finger position
                        cv2.circle(img=frame, center=(index_x, index_y),
                                   radius=10, color=(0, 255, 255))

                        # Convert thumb coordinates
                        thumb_x = int(thumb.x * frame_width)
                        thumb_y = int(thumb.y * frame_height)

                        # Draw thumb position
                        cv2.circle(img=frame, center=(thumb_x, thumb_y),
                                   radius=10, color=(0, 255, 255))

                        # Calculate vertical distance between thumb and index
                        vertical_distance = abs(index_y - thumb_y)

                        # Move mouse pointer
                        pyautogui.moveTo(screen_x, screen_y, duration=0.1)

                        # Handle dragging
                        if vertical_distance < 40 and not is_dragging:
                            pyautogui.mouseDown()
                            is_dragging = True
                            cv2.circle(img=frame, center=(thumb_x, thumb_y),
                                       radius=20, color=(0, 0, 255))
                            print('Started dragging', vertical_distance)

                        elif vertical_distance >= 40 and is_dragging:
                            pyautogui.mouseUp()
                            is_dragging = False
                            print('Released drag', vertical_distance)

            except Exception as e:
                print(f"Error processing hand: {str(e)}")
                continue

            # Display frame
            cv2.imshow('Virtual Mouse', frame)

            # Check for exit command (ESC key)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up resources
        if is_dragging:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()


if __name__ == "__main__":
    main()