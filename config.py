MODEL_FILENAME = 'svm_image_disorder_pipeline.joblib'
SCALER_FILENAME = 'models/svm_image_scaler.joblib'
MODEL_SVM_FILENAME = 'models/svm_image_svm_model.joblib'
CONFIG_FILENAME = 'models/svm_image_config.joblib'

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
