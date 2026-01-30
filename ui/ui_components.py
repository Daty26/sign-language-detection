import customtkinter as ctk
import tkinter as tk
from PIL import ImageTk


class VideoFrame(ctk.CTkFrame):
    """Frame widget used to display the camera/video feed."""

    def __init__(self, master, width=640, height=480):
        # Initialize a CustomTkinter frame with fixed size and rounded corners
        super().__init__(master, width=width, height=height, corner_radius=15)

        # Prevent the frame from shrinking to fit its children
        self.pack_propagate(False)

        # Use a standard Tkinter Label to display images (ImageTk.PhotoImage)
        self.label = tk.Label(
            self,
            text="Camera Feed Off",   # Placeholder text when no image is shown
            bg="#2b2b2b",             # Dark background to match the UI theme
            fg="white"               # Text color
        )
        self.label.pack(fill="both", expand=True)

    def update_image(self, pil_image):
        """
        Accepts a PIL image, converts it into a Tkinter-compatible image,
        and displays it inside the label.

        Note: We use ImageTk.PhotoImage here (not CTkImage), because Tkinter
        labels expect PhotoImage objects.
        """
        # Convert PIL image to Tkinter PhotoImage
        imgtk = ImageTk.PhotoImage(image=pil_image)

        # Update label: show the image and remove placeholder text
        self.label.configure(image=imgtk, text="")

        # Keep a reference to prevent the image from being garbage-collected
        self.label.image = imgtk


class TextOutputBox(ctk.CTkFrame):
    """A text output panel that displays recognized words/phrases."""

    def __init__(self, master, title="Recognized Stream"):
        # Transparent frame to blend into sidebar background
        super().__init__(master, fg_color="transparent")

        # Title label above the textbox
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.pack(anchor="w", pady=(0, 5))

        # Textbox where recognized output is appended
        self.text_area = ctk.CTkTextbox(
            self,
            height=200,
            width=300,
            font=ctk.CTkFont(family="Consolas", size=14),
            corner_radius=10
        )
        self.text_area.pack(side="left", fill="both", expand=True)

    def add_text(self, text):
        """Append text to the output area and auto-scroll to the end."""
        self.text_area.insert("end", text + " ")
        self.text_area.see("end")

    def clear(self):
        """Clear all content from the text area."""
        self.text_area.delete('1.0', "end")

    def get_content(self):
        """Return the full text content (useful for saving logs)."""
        return self.text_area.get('1.0', "end")


class ControlPanel(ctk.CTkFrame):
    """Panel with action buttons (start/stop/clear/save)."""

    def __init__(self, master, start_cb, stop_cb, clear_cb, save_cb):
        # Transparent frame so buttons sit nicely on the sidebar
        super().__init__(master, fg_color="transparent")

        # Start button triggers the provided callback
        self.btn_start = ctk.CTkButton(
            self,
            text="▶ Start Detection",
            command=start_cb,
            fg_color="#2CC985",
            hover_color="#229A65",
            height=40,
            font=ctk.CTkFont(weight="bold")
        )

        # Stop button triggers the provided callback
        self.btn_stop = ctk.CTkButton(
            self,
            text="⏹ Stop",
            command=stop_cb,
            fg_color="#E74C3C",
            hover_color="#C0392B",
            height=40
        )

        # Clear and Save buttons
        self.btn_clear = ctk.CTkButton(self, text="🗑 Clear Text", command=clear_cb)
        self.btn_save = ctk.CTkButton(self, text="💾 Save Log", command=save_cb)

        # Stack buttons vertically with spacing
        self.btn_start.pack(pady=(0, 10), fill="x")
        self.btn_stop.pack(pady=(0, 10), fill="x")
        self.btn_clear.pack(pady=(0, 10), fill="x")
        self.btn_save.pack(pady=(0, 10), fill="x")
