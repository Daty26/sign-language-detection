import customtkinter as ctk
from PIL import Image
import cv2
import datetime
import sys
import os

# Get the directory of the current file (ui folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add current directory to Python path to allow local imports
sys.path.append(current_dir)

# Import custom UI components
from ui_components import VideoFrame, TextOutputBox, ControlPanel

# Configure global appearance settings for CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window title and size
        self.title("Sign Language Translator AI")
        self.geometry("1100x700")

        # Configure grid layout (main video area + right sidebar)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Camera / detection state
        self.is_running = False
        self.cap = None

        # Video display frame (left side)
        self.video_frame = VideoFrame(self)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Right sidebar container
        self.right_sidebar = ctk.CTkFrame(self, fg_color="transparent")
        self.right_sidebar.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # Control buttons panel
        self.controls = ControlPanel(
            self.right_sidebar,
            start_cb=self.start_detection,
            stop_cb=self.stop_detection,
            clear_cb=self.clear_log,
            save_cb=self.save_log
        )
        self.controls.pack(side="bottom", fill="x", pady=10)

        # Text output box for detected words
        self.text_box = TextOutputBox(self.right_sidebar)
        self.text_box.pack(side="top", fill="both", expand=True)

    def start_detection(self):
        """Start camera capture and detection loop."""
        if not self.is_running:
            self.is_running = True

            # Open default camera (index 0)
            self.cap = cv2.VideoCapture(0)

            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # Check if camera opened successfully
            if not self.cap.isOpened():
                print("Failed to open camera")
                self.is_running = False
                return

            # Start frame update loop
            self.update_loop()

    def stop_detection(self):
        """Stop camera capture and pause the video feed."""
        self.is_running = False

        if self.cap:
            self.cap.release()

        # Clear the video frame
        self.video_frame.label.configure(image='', text="Camera Paused")

    def clear_log(self):
        """Clear text output box."""
        self.text_box.clear()

    def save_log(self):
        """Save detected text into a timestamped log file."""
        text = self.text_box.get_content()

        # Move from ui directory to project root
        project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

        # Create /logs directory if it does not exist
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Generate filename with current date and time
        filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(logs_dir, filename)

        # Write log content to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved to {filepath}")

    def update_loop(self):
        """Main loop: read camera frame, process it, and update UI."""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()

            if ret:
                # Convert frame from BGR (OpenCV) to RGB (PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Draw a placeholder rectangle (future detection area)
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

                # Temporary random word generation (mock recognition)
                import random
                if random.randint(0, 50) == 0:
                    self.text_box.add_text("Word")

                # Get current size of the video frame widget
                viewport_width = self.video_frame.winfo_width()
                viewport_height = self.video_frame.winfo_height()

                # Fallback size if widget is not ready yet
                if viewport_width < 10 or viewport_height < 10:
                    viewport_width = 640
                    viewport_height = 480

                # Convert NumPy array to PIL image
                pil_image = Image.fromarray(frame)

                # Resize image to fit UI container
                pil_image = pil_image.resize((viewport_width, viewport_height))

                # Update image in the UI
                self.video_frame.update_image(pil_image)

            # Schedule next frame update (≈100 FPS)
            self.after(10, self.update_loop)


if __name__ == "__main__":
    # Create and run the application
    app = SignLanguageApp()
    app.mainloop()
