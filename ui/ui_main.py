import customtkinter as ctk
from PIL import Image
import cv2
import datetime
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ui_components import VideoFrame, TextOutputBox, ControlPanel

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language Translator AI")
        self.geometry("1100x700")

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.is_running = False
        self.cap = None

        self.video_frame = VideoFrame(self)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.right_sidebar = ctk.CTkFrame(self, fg_color="transparent")
        self.right_sidebar.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.controls = ControlPanel(
            self.right_sidebar,
            start_cb=self.start_detection,
            stop_cb=self.stop_detection,
            clear_cb=self.clear_log,
            save_cb=self.save_log
        )
        self.controls.pack(side="bottom", fill="x", pady=10)

        self.text_box = TextOutputBox(self.right_sidebar)
        self.text_box.pack(side="top", fill="both", expand=True)

    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not self.cap.isOpened():
                print("Не удалось открыть камеру")
                self.is_running = False
                return
            self.update_loop()

    def stop_detection(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

        self.video_frame.label.configure(image='', text="Camera Paused")

    def clear_log(self):
        self.text_box.clear()

    def save_log(self):
        text = self.text_box.get_content()
        filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {filename}")

    def update_loop(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
                import random
                if random.randint(0, 50) == 0:
                    self.text_box.add_text("Word")


                viewport_width = self.video_frame.winfo_width()
                viewport_height = self.video_frame.winfo_height()

                if viewport_width < 10 or viewport_height < 10:
                    viewport_width = 640
                    viewport_height = 480

                pil_image = Image.fromarray(frame)

                pil_image = pil_image.resize((viewport_width, viewport_height))

                self.video_frame.update_image(pil_image)

            self.after(10, self.update_loop)


if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()