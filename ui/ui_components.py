import customtkinter as ctk
import tkinter as tk  
from PIL import ImageTk


class VideoFrame(ctk.CTkFrame):
    """Рамка для видеопотока"""

    def __init__(self, master, width=640, height=480):
        super().__init__(master, width=width, height=height, corner_radius=15)
        self.pack_propagate(False)

        self.label = tk.Label(
            self,
            text="Camera Feed Off",
            bg="#2b2b2b", 
            fg="white"
        )
        self.label.pack(fill="both", expand=True)

    def update_image(self, pil_image):
        """
        Принимает обычное PIL изображение.
        Конвертирует его в стандартный ImageTk (не CTkImage).
        """
        imgtk = ImageTk.PhotoImage(image=pil_image)

        self.label.configure(image=imgtk, text="")

        self.label.image = imgtk


class TextOutputBox(ctk.CTkFrame):

    def __init__(self, master, title="Recognized Stream"):
        super().__init__(master, fg_color="transparent")

        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.pack(anchor="w", pady=(0, 5))

        self.text_area = ctk.CTkTextbox(
            self,
            height=200,
            width=300,
            font=ctk.CTkFont(family="Consolas", size=14),
            corner_radius=10
        )
        self.text_area.pack(side="left", fill="both", expand=True)

    def add_text(self, text):
        self.text_area.insert("end", text + " ")
        self.text_area.see("end")

    def clear(self):
        self.text_area.delete('1.0', "end")

    def get_content(self):
        return self.text_area.get('1.0', "end")


class ControlPanel(ctk.CTkFrame):

    def __init__(self, master, start_cb, stop_cb, clear_cb, save_cb):
        super().__init__(master, fg_color="transparent")

        self.btn_start = ctk.CTkButton(
            self, text="▶ Start Detection", command=start_cb,
            fg_color="#2CC985", hover_color="#229A65", height=40,
            font=ctk.CTkFont(weight="bold")
        )

        self.btn_stop = ctk.CTkButton(
            self, text="⏹ Stop", command=stop_cb,
            fg_color="#E74C3C", hover_color="#C0392B", height=40
        )

        self.btn_clear = ctk.CTkButton(self, text="🗑 Clear Text", command=clear_cb)
        self.btn_save = ctk.CTkButton(self, text="💾 Save Log", command=save_cb)

        self.btn_start.pack(pady=(0, 10), fill="x")
        self.btn_stop.pack(pady=(0, 10), fill="x")
        self.btn_clear.pack(pady=(0, 10), fill="x")
        self.btn_save.pack(pady=(0, 10), fill="x")