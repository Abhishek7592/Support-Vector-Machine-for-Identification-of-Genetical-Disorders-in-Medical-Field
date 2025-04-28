import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import joblib
import numpy as np
from PIL import Image, ImageTk
from data_preparation import preprocess_image
from config import DISORDER_MAP, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, SCALER_FILENAME, MODEL_SVM_FILENAME, CONFIG_FILENAME

def launch_prediction_gui(scaler, svm_model, config):
    if not scaler or not svm_model or not config:
        messagebox.showerror("Error", "Model components (scaler, svm, config) not available.")
        return

    img_width = config.get('img_width', 64)
    img_height = config.get('img_height', 64)
    img_channels = config.get('img_channels', 1)
    grayscale_mode = (img_channels == 1)
    target_size = (img_width, img_height)

    selected_image_path = None
    tk_image = None

    def browse_image():
        nonlocal selected_image_path, tk_image
        try:
            f_types = [('Image Files', '*.jpg *.jpeg *.png *.bmp')]
            filename = filedialog.askopenfilename(filetypes=f_types)
            if filename:
                selected_image_path = filename
                path_var.set(os.path.basename(filename))
                result_var.set("Prediction: Image selected. Click Predict.")
                result_label.config(foreground='gray')

                img = Image.open(filename)
                img.thumbnail((150, 150))
                tk_image = ImageTk.PhotoImage(img)
                image_label.config(image=tk_image)
                image_label.image = tk_image
            else:
                 selected_image_path = None
                 path_var.set("No image selected")
                 image_label.config(image=None)

        except Exception as e:
            messagebox.showerror("Image Load Error", f"Could not load or display image:\n{e}")
            selected_image_path = None
            path_var.set("Error loading image")
            image_label.config(image=None)

    def perform_prediction():
        if not selected_image_path:
            messagebox.showwarning("No Image", "Please select an image file first.")
            return

        try:
            flattened_img = preprocess_image(selected_image_path, target_size=target_size, grayscale=grayscale_mode)
            if flattened_img is None:
                messagebox.showerror("Preprocessing Error", "Failed to preprocess the selected image.")
                return

            input_vector = flattened_img.reshape(1, -1)
            scaled_vector = scaler.transform(input_vector)
            prediction_num = svm_model.predict(scaled_vector)[0]
            prediction_text = DISORDER_MAP.get(prediction_num, f"Unknown Code ({prediction_num})")
            result_var.set(f"Prediction: {prediction_text}")
            result_label.config(foreground='navy' if prediction_num == 0 else 'red')

        except Exception as e:
             messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")
             result_var.set("Prediction: Error")
             result_label.config(foreground='black')

    window = tk.Tk()
    window.title("Genetic Disorder Prediction (SVM - Image Input)")
    window.geometry("400x450")

    style = ttk.Style(window)
    try: style.theme_use('clam')
    except tk.TclError: pass

    main_frame = ttk.Frame(window, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    select_frame = ttk.Frame(main_frame)
    select_frame.pack(pady=10, fill=tk.X)
    browse_button = ttk.Button(select_frame, text="Browse Image...", command=browse_image)
    browse_button.pack(side=tk.LEFT, padx=(0, 10))
    path_var = tk.StringVar(value="No image selected")
    path_label = ttk.Label(select_frame, textvariable=path_var, relief=tk.SUNKEN, width=30)
    path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    image_frame = ttk.Frame(main_frame, relief=tk.GROOVE, borderwidth=1)
    image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    image_label = ttk.Label(image_frame, text="Image Thumbnail")
    image_label.pack(padx=5, pady=5)

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=(10,0), fill=tk.X)
    predict_button = ttk.Button(button_frame, text="Predict Disorder Risk", command=perform_prediction)
    predict_button.pack(pady=5)
    result_var = tk.StringVar()
    result_var.set("Prediction: Select an image and click Predict")
    result_label = ttk.Label(button_frame, textvariable=result_var, font=('Helvetica', 11, 'italic'), foreground='gray')
    result_label.pack(pady=(0, 10))

    window.mainloop()
