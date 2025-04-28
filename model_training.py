import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from data_preparation import prepare_image_data
from config import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, DISORDER_MAP, SCALER_FILENAME, MODEL_SVM_FILENAME, CONFIG_FILENAME

def train_and_save_model():
    print("--- Starting Model Training (Image Input) ---")
    try:
        data_base_dir = "image_data_genetic_disorders"
        X, y = prepare_image_data(
            base_dir=data_base_dir,
            num_samples_per_class = {0: 300, 1: 40, 2: 30, 3: 50},
            target_size=(IMG_WIDTH, IMG_HEIGHT)
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
