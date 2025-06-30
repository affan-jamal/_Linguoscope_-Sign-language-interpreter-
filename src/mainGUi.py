import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3
import tkinter as tk
from tkinter import ttk


class SignLanguageApp:
    def __init__(self):
        self.cap = None
        self.detector = HandDetector(maxHands=2)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 120)

        # Constants
        self.offset = 20
        self.imgSize = 300
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U",
                       "V", "W", "X", "Y", "Z","Hi","Thank You","I Love You","This","Machine","Food"]
        self.STABILITY_THRESHOLD = 1.0
        self.INITIAL_DELAY = 3

        # UI Colors
        self.COLORS = {
            'background': (30, 30, 30),
            'overlay': (50, 50, 50),
            'accent': (0, 173, 181),
            'text': (240, 240, 240),
            'highlight': (252, 163, 17),
            'stable': (0, 255, 0)
        }

        # State variables
        self.reset_state()

    def reset_state(self):
        """Reset all state variables"""
        self.accumulated_text = ""
        self.last_prediction = ""
        self.prediction_time = time.time()
        self.is_reading = False
        self.stable_gesture_time = 0
        self.current_stable_prediction = None
        self.startup_time = time.time()
        self.is_startup = True

    def initialize_camera(self):
        """Initialize camera capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        return self.cap.isOpened()

    def release_camera(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def create_minimal_ui(self, img, text, accumulated, mode):
        """Create UI overlay based on the current mode"""
        h, w = img.shape[:2]
        overlay = img.copy()

        # Background overlay
        cv2.rectangle(overlay, (0, h - 100), (w, h), self.COLORS['overlay'], cv2.FILLED)
        img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)

        # Display detected gesture
        if text:
            cv2.putText(img, f"Gesture: {text}", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['highlight'],
                        2)

        # Mode-specific UI elements
        if mode == "practice":
            cv2.putText(img, "Practice Mode - Press 'Q' to quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.COLORS['accent'], 2)
        elif mode == "predict":
            cv2.putText(img, "Q: Quit | Space: Add Space | C: Clear", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.COLORS['accent'], 2)
            cv2.putText(img, accumulated[-40:], (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['text'], 2)
        elif mode == "predict_read":
            cv2.putText(img, "Q: Quit | Space: Add Space | R: Read | C: Clear", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.COLORS['accent'], 2)
            cv2.putText(img, accumulated[-40:], (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['text'], 2)

        return img

    def process_hand(self, img):
        hands, img = self.detector.findHands(img)

        if hands:
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0

            for hand in hands:  # Process all detected hands
                x, y, w, h = hand['bbox']
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            # Define the bounding box for both hands combined
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y_min - self.offset:y_max + self.offset, x_min - self.offset:x_max + self.offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = (y_max - y_min) / (x_max - x_min)

                if aspectRatio > 1:
                    k = self.imgSize / (y_max - y_min)
                    wCal = math.ceil(k * (x_max - x_min))
                    imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                    wGap = math.ceil((self.imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = self.imgSize / (x_max - x_min)
                    hCal = math.ceil(k * (y_max - y_min))
                    imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                    hGap = math.ceil((self.imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                return self.labels[index], (x_min, y_min)

        return None, None

    def practice_mode(self):
        """Run the practice mode"""
        if not self.initialize_camera():
            return

        while True:
            success, img = self.cap.read()
            if not success:
                break

            imgOutput = img.copy()
            prediction, coords = self.process_hand(img)

            if prediction and coords:
                x, y = coords
                cv2.putText(imgOutput, f"Sign: {prediction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            self.COLORS['stable'], 2)

            imgOutput = self.create_minimal_ui(imgOutput, prediction, "", "practice")
            cv2.imshow("Sign Language Practice", imgOutput)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_camera()

    def prediction_mode(self, with_reading=False):
        """Run the prediction mode with optional reading feature"""
        if not self.initialize_camera():
            return

        self.reset_state()

        while True:
            success, img = self.cap.read()
            if not success:
                break

            imgOutput = img.copy()
            current_time = time.time()

            if self.is_startup and (current_time - self.startup_time) < self.INITIAL_DELAY:
                imgOutput = self.create_minimal_ui(imgOutput, self.last_prediction, self.accumulated_text,
                                                   "predict_read" if with_reading else "predict")
                cv2.imshow("Sign Language Interpreter", imgOutput)
                cv2.waitKey(1)
                continue

            prediction, coords = self.process_hand(img)

            if prediction and coords:
                x, y = coords

                if prediction == self.current_stable_prediction:
                    if self.stable_gesture_time == 0:
                        self.stable_gesture_time = current_time
                    elif current_time - self.stable_gesture_time >= self.STABILITY_THRESHOLD:
                        self.accumulated_text += prediction
                        self.last_prediction = prediction
                        self.stable_gesture_time = current_time
                else:
                    self.current_stable_prediction = prediction
                    self.stable_gesture_time = 0

                color = self.COLORS[
                    'stable'] if current_time - self.stable_gesture_time >= self.STABILITY_THRESHOLD else self.COLORS[
                    'accent']
                cv2.putText(imgOutput, f"Detecting: {prediction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            imgOutput = self.create_minimal_ui(imgOutput, self.last_prediction, self.accumulated_text,
                                               "predict_read" if with_reading else "predict")
            cv2.imshow("Sign Language Interpreter", imgOutput)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.accumulated_text = ""
                self.last_prediction = ""
            elif key == ord(' '):
                self.accumulated_text += " "
            elif with_reading and key == ord('r') and not self.is_reading:
                if self.accumulated_text.strip():
                    self.is_reading = True
                    self.engine.say(self.accumulated_text)
                    self.engine.runAndWait()
                    self.is_reading = False

        self.release_camera()


class MenuGUI:
    def __init__(self):
        self.app = SignLanguageApp()
        self.root = tk.Tk()
        self.root.title("Sign Language Interpreter")
        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI elements"""
        # Configure the main window
        self.root.geometry("400x500")
        self.root.configure(bg='#1e1e1e')

        # Style configuration
        style = ttk.Style()
        style.configure('TButton',
                        padding=10,
                        font=('Helvetica', 12),
                        background='#2e2e2e')

        # Title
        title = tk.Label(self.root,
                         text="Sign Language Interpreter",
                         font=('Helvetica', 20, 'bold'),
                         fg='white',
                         bg='#1e1e1e')
        title.pack(pady=40)

        # Buttons
        btn_practice = ttk.Button(self.root,
                                  text="Practice Hand Signs",
                                  command=self.practice_mode)
        btn_practice.pack(pady=20, padx=50, fill='x')

        btn_predict = ttk.Button(self.root,
                                 text="Prediction Mode",
                                 command=self.predict_mode)
        btn_predict.pack(pady=20, padx=50, fill='x')

        btn_predict_read = ttk.Button(self.root,
                                      text="Prediction with Reading",
                                      command=self.predict_read_mode)
        btn_predict_read.pack(pady=20, padx=50, fill='x')

        btn_exit = ttk.Button(self.root,
                              text="Exit",
                              command=self.root.quit)
        btn_exit.pack(pady=20, padx=50, fill='x')

    def practice_mode(self):
        """Launch practice mode"""
        self.root.withdraw()  # Hide the menu
        self.app.practice_mode()
        self.root.deiconify()  # Show the menu again

    def predict_mode(self):
        """Launch prediction mode"""
        self.root.withdraw()
        self.app.prediction_mode(with_reading=False)
        self.root.deiconify()

    def predict_read_mode(self):
        """Launch prediction mode with reading"""
        self.root.withdraw()
        self.app.prediction_mode(with_reading=True)
        self.root.deiconify()

    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    menu = MenuGUI()
    menu.run()