import tkinter as tk
from tkinter import filedialog
import os

class VideoFileSelector:
    def __init__(self):
        self.video_path = None
        self.img2_path = None

    def select_video_file(self):
        # Create a Tkinter root window and hide it
        root = tk.Tk()
        root.withdraw()

        # Open a file dialog to select a video file
        self.video_path = filedialog.askopenfilename(title='Select Video File', filetypes=[('Video files', '*.mp4;*.avi;*.mov')])

        if self.video_path:  # If a file was selected
            # Extract the directory of the selected video
            video_directory = os.path.dirname(self.video_path)
            # Extract the base name of the video file (without extension)
            base_name = os.path.basename(self.video_path).rsplit('.', 1)[0]
            # Construct the img2 path to be in the "map" directory within the same directory as the video file
            img2_directory = os.path.join(video_directory, "map")
            self.img2_path = os.path.join(img2_directory, f"{base_name}.png")

            # Check if the "map" directory exists, if not, create it
            if not os.path.exists(img2_directory):
                os.makedirs(img2_directory)

            # Now you have both paths
            print(f"Video Path: {self.video_path}")
            print(f"Img2 Path: {self.img2_path}")

            return self.video_path, self.img2_path
        else:
            print("No file selected.")
            return None, None