import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import time
import logging
from tracking_demo import build_tracker, build_video_capture


class TrackerDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tracker Demo GUI")
        
        # Set window size and make it non-resizable
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        # Configure grid weights to center the content
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Configure style for larger elements
        style = ttk.Style()
        style.configure('Large.TRadiobutton', font=('Arial', 11))
        style.configure('Large.TCheckbutton', font=('Arial', 11))
        style.configure('Large.TButton', font=('Arial', 11), padding=5)
        style.configure('Large.TLabel', font=('Arial', 11))
        style.configure('Large.TLabelframe', font=('Arial', 11))
        
        # Variables
        self.tracker_type = tk.StringVar(value="kcf")
        self.show_fps = tk.BooleanVar(value=True)
        self.data_path = None
        self.capture = None
        self.tracker = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Initially disable the run button
        self.run_button.state(['disabled'])
        
    def create_widgets(self):
        # Create a centered container frame
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Main content frame with padding
        main_frame = ttk.Frame(container, padding="20")
        main_frame.grid(row=0, column=0)

        # Tracker selection
        tracker_frame = ttk.LabelFrame(main_frame, text="Tracker Type", padding="10",
                                       style='Large.TLabelframe')
        tracker_frame.grid(row=0, column=0, padx=10, pady=10)
        tracker_frame.columnconfigure(0, weight=1)

        trackers = [("KCF", "kcf"), ("NanoTrack", "nano"), ("TLD", "tld"), ("SiamFC", "siamfc")]
        for i, (text, value) in enumerate(trackers):
            ttk.Radiobutton(tracker_frame, text=text, value=value,
                             variable=self.tracker_type, style='Large.TRadiobutton').grid(
                                row=0, column=i, padx=20)

        # Show FPS checkbox in its own frame for centering
        fps_frame = ttk.Frame(main_frame)
        fps_frame.grid(row=1, column=0, pady=15)
        ttk.Checkbutton(fps_frame, text="Show FPS", variable=self.show_fps,
                        style='Large.TCheckbutton').pack()

        # Path selection buttons and label
        path_frame = ttk.Frame(main_frame)
        path_frame.grid(row=2, column=0, padx=10, pady=10)
        path_frame.columnconfigure((0, 1), weight=1)

        ttk.Button(path_frame, text="Select Video", style='Large.TButton',
                   command=lambda: self.select_path("video")).grid(
                      row=0, column=0, padx=10)
        ttk.Button(path_frame, text="Select Folder", style='Large.TButton',
                   command=lambda: self.select_path("folder")).grid(
                      row=0, column=1, padx=10)

        # Path display label
        self.path_label = ttk.Label(main_frame, text="No file or folder selected",
                                   wraplength=450, justify="center", style='Large.TLabel')
        self.path_label.grid(row=3, column=0, padx=10, pady=20)

        # Run button
        self.run_button = ttk.Button(main_frame, text="Run", command=self.run_demo,
                                    style='Large.TButton', width=20)
        self.run_button.grid(row=4, column=0, padx=10, pady=20)
        
    def select_path(self, path_type):
        if path_type == "video":
            path = filedialog.askopenfilename(
                filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")])
        else:
            path = filedialog.askdirectory()
            
        if path:
            self.data_path = path
            self.path_label.config(text=f"Selected: {path}")
            self.run_button.state(['!disabled'])  # Enable the run button
        else:
            self.data_path = None
            self.path_label.config(text="No file or folder selected")
            self.run_button.state(['disabled'])  # Disable the run button
            
    def run_demo(self):
        if self.data_path is None:
            messagebox.showerror("Error", "Please select a video file or folder first")
            return
            
        try:
            self.tracking_loop()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def tracking_loop(self):
        self.capture = build_video_capture(self.data_path)
        
        if not self.capture.isOpened():
            messagebox.showerror("Error", f"Could not open video source: {self.data_path}")
            return
            
        self.tracker = build_tracker(self.tracker_type.get())
        
        status, frame = self.capture.read()
        if not status:
            messagebox.showerror("Error", "Could not read the first frame")
            return

        try:
            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
            bbox = cv2.selectROI("Tracking", frame, printNotice=True)
        except Exception:
            bbox = (0, 0, 0, 0)

        if not bbox or (isinstance(bbox, tuple) and sum(bbox) == 0):
            # user cancelled ROI selection
            messagebox.showerror("Error", "ROI selection cancelled or invalid")
            return
        
        bbox_color = (255, 0, 0)
        self.tracker.init(frame, bbox)
        while True:
            status, frame = self.capture.read()
            
            if not status:
                logging.warning("Video sequence ends")
                break
                
            start_time = time.time()
            tracking_result, new_bbox = self.tracker.update(frame)
            end_time = time.time()
            
            # Calculate FPS
            if end_time != start_time:
                fps = 1.0 / (end_time - start_time)
                
            x, y, w, h = map(int, new_bbox)
            
            if tracking_result:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2, 1)
            else:
                frame = cv2.putText(frame, "Could not detect object",
                                    (100, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (0, 0, 255))
                
            # Show FPS if enabled
            if self.show_fps.get():
                fps_text = f"FPS: {fps:.2f}"
                frame = cv2.putText(frame, fps_text, (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 1)
            
            cv2.imshow("Tracking", frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        if self.capture:
            self.capture.release()
            
def main():
    root = tk.Tk()
    app = TrackerDemoGUI(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()