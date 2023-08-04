import cv2


def open_camera():
    # Use index 0 for the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            print("Failed to read frame from camera.")
            break

        # Display the frame in a window
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    open_camera()
