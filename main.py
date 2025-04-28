import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from PIL import Image, ImageTk 
import os
import glob 
import warnings
import random 

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MODEL_FILENAME = 'svm_image_disorder_pipeline.joblib'
SCALER_FILENAME = 'svm_image_scaler.joblib'
MODEL_SVM_FILENAME = 'svm_image_svm_model.joblib' 
CONFIG_FILENAME = 'svm_image_config.joblib' 

FORCE_RETRAIN = False

IMG_WIDTH = 64 
IMG_HEIGHT = 64 
IMG_CHANNELS = 1 

DISORDER_MAP = {
    0: 'Healthy/No Disorder',
    1: 'Hereditary Breast and Ovarian Cancer (HBOC)',
    2: 'Lynch Syndrome',
    3: 'Familial Hypercholesterolemia (FH)'
}
CLASSES = list(DISORDER_MAP.keys()) 
CLASS_NAMES = list(DISORDER_MAP.values()) 

def preprocess_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT), grayscale=(IMG_CHANNELS == 1)):
    """Loads an image, resizes, converts to grayscale (optional), and normalizes."""
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        if grayscale:
            img = img.convert('L') 
        else:
            img = img.convert('RGB') 

        img_array = np.array(img)

        img_array = img_array / 255.0

       
        flattened_array = img_array.flatten()
        return flattened_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def prepare_image_data(base_dir="image_data_synthetic", num_samples_per_class=None, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Loads images from structured folders OR creates synthetic placeholder data.
    Returns flattened image data (X) and labels (y).
    """
    if num_samples_per_class is None:
        num_samples_per_class = {0: 100, 1: 20, 2: 15, 3: 25} 

    all_images_flattened = []
    all_labels = []

    if not os.path.exists(base_dir) or len(glob.glob(os.path.join(base_dir, "*", "*.png"))) == 0 :
        print(f"Synthetic data directory '{base_dir}' not found or empty. Creating placeholders...")
        if not os.path.exists(base_dir): os.makedirs(base_dir)
        for class_idx, class_name_key in enumerate(CLASSES):
             safe_folder_name = f"class_{class_idx}_{DISORDER_MAP[class_name_key].split(' ')[0].lower()}"
             class_dir = os.path.join(base_dir, safe_folder_name)
             if not os.path.exists(class_dir): os.makedirs(class_dir)

             num_images = num_samples_per_class.get(class_idx, 10) 
             print(f"  Creating {num_images} placeholder images for class {class_idx} in '{class_dir}'")
             for i in range(num_images):
                 img_array = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * (50 + class_idx * 50)
                 img = Image.fromarray(img_array)
                 img_path = os.path.join(class_dir, f"placeholder_{class_idx}_{i}.png")
                 img.save(img_path)
        print("Placeholder image structure created.")

    print(f"Loading image data from '{base_dir}'...")
    found_images = False
    for class_idx, class_name_key in enumerate(CLASSES):
        safe_folder_name = f"class_{class_idx}_{DISORDER_MAP[class_name_key].split(' ')[0].lower()}"
        class_dir = os.path.join(base_dir, safe_folder_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found for class {class_idx}: {class_dir}")
            continue

        image_files = glob.glob(os.path.join(class_dir, "*.png")) + \
                      glob.glob(os.path.join(class_dir, "*.jpg")) + \
                      glob.glob(os.path.join(class_dir, "*.jpeg"))

        print(f"  Found {len(image_files)} images for class {class_idx} in '{os.path.basename(class_dir)}'")
        for img_path in image_files:
            flattened_img = preprocess_image(img_path, target_size=target_size, grayscale=(IMG_CHANNELS==1))
            if flattened_img is not None:
                all_images_flattened.append(flattened_img)
                all_labels.append(class_idx)
                found_images = True

    if not found_images:
         raise FileNotFoundError(f"No valid images found in the subdirectories of '{base_dir}'. Please check the structure and image files.")

    X = np.array(all_images_flattened)
    y = np.array(all_labels)
    print(f"Loaded data shape: X={X.shape}, y={y.shape}")
    print("Class distribution in loaded data:\n", pd.Series(y).value_counts().sort_index())

    return X, y

def train_and_save_model():
    """
    Loads image data, trains SVM with SMOTE on flattened image vectors,
    evaluates, and saves necessary components. Returns True on success.
    """
    print("--- Starting Model Training (Image Input) ---")
    try:
        data_base_dir = "image_data_genetic_disorders"
        X, y = prepare_image_data(
            base_dir=data_base_dir,
            num_samples_per_class = {0: 300, 1: 40, 2: 30, 3: 50}, target_size=(IMG_WIDTH, IMG_HEIGHT)
        )
    except Exception as e:
        print(f"Error preparing image data: {e}")
        return False

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Error during train/test split - potentially insufficient samples for a class: {e}")
        print("Value counts in y:\n", pd.Series(y).value_counts())
        print("Trying split without stratify (may lead to skewed test set)...")
        try:
             X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size=0.25, random_state=42 
             )
             print("Warning: Proceeding without stratify due to small class size.")
        except Exception as e_nostrat:
            print(f"Still cannot split data: {e_nostrat}. Aborting training.")
            return False

    print("\n--- Data Splitting ---")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape:  {X_test.shape}")
    print("-" * 30)

   
    scaler = StandardScaler()

    min_class_count_train = min(np.bincount(y_train))
    k_neighbors = max(1, min(5, min_class_count_train - 1)) 
    print(f"Using k_neighbors={k_neighbors} for SMOTE (min class count in train: {min_class_count_train})")
    if min_class_count_train <= 1:
         print("Warning: Smallest class in training set has <= 1 sample. SMOTE cannot be applied.")
         smote = None 
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)


    svm = SVC(probability=True, random_state=42)

    param_grid = {
        'svm__C': [0.1, 1, 10],       
        'svm__gamma': ['scale', 'auto'], 
        'svm__kernel': ['rbf'] 
    }

   
    print("Fitting scaler...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

    if smote:
        print("Applying SMOTE...")
        try:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            print("SMOTE applied. New training shape:", X_train_resampled.shape)
            print("Class distribution after SMOTE:\n", pd.Series(y_train_resampled).value_counts().sort_index())
        except Exception as e:
            print(f"Error applying SMOTE: {e}. Proceeding without oversampling.")
            X_train_resampled, y_train_resampled = X_train_scaled, y_train 
    else:
         print("Skipping SMOTE step.")
         X_train_resampled, y_train_resampled = X_train_scaled, y_train 

   
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = 'accuracy'

    print("\n--- Starting SVM Hyperparameter Tuning (GridSearchCV) ---")
    grid_search_svm = GridSearchCV(svm, param_grid={'C': param_grid['svm__C'],
                                                    'gamma': param_grid['svm__gamma'],
                                                    'kernel': param_grid['svm__kernel']},
                                  cv=cv_strategy, scoring=scoring, n_jobs=-1, verbose=1)
    try:
        grid_search_svm.fit(X_train_resampled, y_train_resampled) 
    except ValueError as e:
         print(f"Error during GridSearchCV fitting (likely due to CV splits and class imbalance if SMOTE failed): {e}")
         print("Trying GridSearchCV without stratified CV...")
         try:
            from sklearn.model_selection import KFold
            grid_search_svm.cv = KFold(n_splits=3, shuffle=True, random_state=42)
            grid_search_svm.fit(X_train_resampled, y_train_resampled)
            print("Warning: Using KFold for CV due to stratification issues.")
         except Exception as e_cv:
             print(f"Still failed to fit GridSearchCV: {e_cv}. Using default SVM parameters.")
             best_svm = SVC(probability=True, random_state=42).fit(X_train_resampled, y_train_resampled)
             print("Using default SVM.")


    if 'best_svm' not in locals(): 
        best_svm = grid_search_svm.best_estimator_
        print("\n--- Tuning Complete ---")
        print(f"Best SVM parameters: {grid_search_svm.best_params_}")
        print(f"Best CV Accuracy score: {grid_search_svm.best_score_:.4f}")
    print("-" * 30)

    print("\n--- Model Evaluation on Test Set ---")
    y_pred = best_svm.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report (Test Set):")
    report_target_names = [DISORDER_MAP.get(i, f"Unknown {i}") for i in sorted(list(set(y_test) | set(y_pred)))]
    try:
        print(classification_report(y_test, y_pred, target_names=report_target_names, digits=3, zero_division=0))
    except ValueError as e:
         print(f"Could not generate classification report (possibly classes missing in predictions): {e}")
         print("Test Labels:", np.unique(y_test, return_counts=True))
         print("Pred Labels:", np.unique(y_pred, return_counts=True))


    try:
        joblib.dump(scaler, SCALER_FILENAME)
        print(f"\nScaler saved to: {SCALER_FILENAME}")
        joblib.dump(best_svm, MODEL_SVM_FILENAME)
        print(f"Trained SVM model saved to: {MODEL_SVM_FILENAME}")
        config = {'img_width': IMG_WIDTH, 'img_height': IMG_HEIGHT, 'img_channels': IMG_CHANNELS}
        joblib.dump(config, CONFIG_FILENAME)
        print(f"Configuration (image dims) saved to: {CONFIG_FILENAME}")

        print("-" * 30)
        return True 
    except Exception as e:
        print(f"\nError saving model components: {e}")
        return False

def launch_prediction_gui(scaler, svm_model, config):
    """Launches the Tkinter GUI for making predictions from image files."""

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
        """Preprocesses selected image, predicts, and updates GUI."""
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

if __name__ == "__main__":
    loaded_scaler = None
    loaded_svm = None
    loaded_config = None

    if not FORCE_RETRAIN and os.path.exists(SCALER_FILENAME) and os.path.exists(MODEL_SVM_FILENAME) and os.path.exists(CONFIG_FILENAME):
        print("--- Attempting to load existing model components ---")
        try:
            loaded_scaler = joblib.load(SCALER_FILENAME)
            loaded_svm = joblib.load(MODEL_SVM_FILENAME)
            loaded_config = joblib.load(CONFIG_FILENAME)
            print("Scaler, SVM model, and Config loaded successfully.")
            if loaded_config.get('img_width') != IMG_WIDTH or loaded_config.get('img_height') != IMG_HEIGHT:
                 print("Warning: Loaded image dimensions differ from code constants. Using loaded dimensions.")
                 IMG_WIDTH = loaded_config.get('img_width')
                 IMG_HEIGHT = loaded_config.get('img_height')
                 IMG_CHANNELS = loaded_config.get('img_channels')

            print("-" * 30)
        except Exception as e:
            print(f"Error loading existing components: {e}. Forcing retrain.")
            loaded_scaler, loaded_svm, loaded_config = None, None, None
            FORCE_RETRAIN = True 
    if loaded_scaler is None or loaded_svm is None or loaded_config is None:
         print("\n--- Training new image model... ---")
         success = train_and_save_model()
         if success:
             print("Reloading components after training...")
             try: 
                 loaded_scaler = joblib.load(SCALER_FILENAME)
                 loaded_svm = joblib.load(MODEL_SVM_FILENAME)
                 loaded_config = joblib.load(CONFIG_FILENAME)
                 print("Components reloaded successfully.")
             except Exception as e:
                 print(f"ERROR: Failed to reload components after training: {e}")
                 loaded_scaler, loaded_svm, loaded_config = None, None, None 
         else:
             print("!!! Model training failed. Cannot launch GUI. Exiting. !!!")
             exit()

    if loaded_scaler and loaded_svm and loaded_config:
        print("\n--- Launching Prediction GUI (Image Input) ---")
        launch_prediction_gui(loaded_scaler, loaded_svm, loaded_config)
    else:
        print("!!! Critical Error: Model components unavailable after train/load. Exiting. !!!")

    print("\nScript finished.")
