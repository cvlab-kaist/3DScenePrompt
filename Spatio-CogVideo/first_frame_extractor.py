import sys
import cv2

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("./first_frame.jpg", frame)
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python first_frame_extractor.py <video_path>")
        sys.exit(1)
    extract_first_frame(sys.argv[1])