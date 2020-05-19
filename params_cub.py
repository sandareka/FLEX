TRAIN_NOUNS_ADJECTIVES = 'data/cub/train_noun_adjectives.npy'
TRAIN_DESC_FILE = 'data/cub/train_descriptions.txt'
VAL_DESC_FILE = 'data/cub/val_descriptions.txt'
TEST_DESC_FILE = 'data/cub/test_descriptions.txt'
DATA_DIC_NAME = "data/cub/data_dict.pkl"
TRAIN_NO_SENT_PER_IMAGE = "data/cub/train_no_sent_per_image.npy"
VAL_NO_SENT_PER_IMAGE = "data/cub/val_no_sent_per_image.npy"
TEST_NO_SENT_PER_IMAGE = "data/cub/test_no_sent_per_image.npy"

# -------- Classifier parameters --------
FEATURE_VECTOR_SIZE = 8192  # size of all features in the desired layer
CLASSIFIER_PROTOTXT_FILE = 'classifiers/cmpt-bilinear/ft_all.prototxt'  # path to prototxt file
CLASSIFIER_WEIGHT_FILE = 'classifiers/cmpt-bilinear/ft_all_iter_20000.caffemodel'  # path to weights file
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
LAYER_NAME = "bilinear_l2"  # layer name from which features are extracted
FINAL_LAYER_NAME = "fc8_cub"  # 'final layer name'
DATA_LAYER_NAME = 'data'
LAYERS = ["conv5_3", "pool4", "pool3", "pool2", "pool1"]  # Different layers from which we extract features
NO_OF_CHANNELS = [512, 512, 256, 128, 64]  # Number of channels in the each layer we declared above

# -------- Image dataset related parameters --------
IMAGE_FOLDER = "data/cub/images"  # path to images folder
ALL_IMAGE_NAME_FILE = "data/images.txt"  # path to images name file
IMAGE_FEATURE_SAVE_FOLDER = "data/cub/visual_features"  # folder to save activations and gradients
TEXT_DESC_FOLDER = 'data/cub/text_c10'
CLASS_NAMES_FILE = 'data/cub/classes.txt'
TRAIN_IMAGE_NAMES = "data/cub/train_no_cub.txt"
VAL_IMAGE_NAMES = "data/cub/val_no_cub.txt"
TEST_IMAGE_NAMES = "data/cub/test_no_cub.txt"


# -------- Relevance score calculation related information --------
NEETA = 0.80
NO_OF_TOP_FEATURE_MAPS_FROM_LAYER = 5
CO_OCCURRENCE_FILE_NAME_TRAIN = 'co_occurrence/co_occurrence_train.npy'
CO_OCCURRENCE_FILE_NAME_VAL = 'co_occurrence/co_occurrence_val.npy'
CO_OCCURRENCE_STAT_FILE_NAME_TRAIN = 'co_occurrence/co_occurrence_stat_train.npy'
CO_OCCURRENCE_STAT_FILE_NAME_VAL = 'co_occurrence/co_occurrence_stat_val.npy'
CO_OCCURRENCE_INFO_FOLDER = 'co_occurrence_info'
CO_OCCURRENCE_STAT_INFO_FOLDER = 'co_occurrence_stat_info'
DECISION_RELEVANT_WORDS_TRAIN = 'data/cub/decision_relevant_words.txt'
DECISION_RELEVANT_WORDS_VAL = 'data/cub/decision_relevant_words_val.txt'
DECISION_RELEVANCE_SCORE_TRAIN = 'data/cub/train_decision_relevance_scores.npy'
DECISION_RELEVANCE_SCORE_VAL = 'data/cub/val_decision_relevance_scores.npy'
DECISION_RELEVANCE_SCORE_TRAIN_EXPANDED = 'data/cub/train_decision_relevance_scores_expanded.npy'
DECISION_RELEVANCE_SCORE_VAL_EXPANDED = 'data/cub/val_decision_relevance_scores_expanded.npy'
DECISION_RELEVANCE_SCORE_DICT_TRAIN = 'data/cub/train_decision_relevance_scores_dict.npy'
DECISION_RELEVANCE_SCORE_DICT_VAL = 'data/cub/val_decision_relevance_scores_dict.npy'


# -------- Visual feature information --------
TRAIN_VISUAL_FEATURE_FILE = "data/cub/cub_bilinear_train_visual_feature.h5"  # file name of the visual features
VAL_VISUAL_FEATURE_FILE = "data/cub/cub_bilinear_val_visual_feature.h5"  # file name of the visual features
TEST_VISUAL_FEATURE_FILE = "data/cub/cub_bilinear_test_visual_feature.h5"  # file name of the visual features
IMG_FEATURE_SAVE_FOLDER_PATH = "data/cub/visual_features"


# -------- FLEX training information --------
LSTM_HIDDEN_NUM = 1000
LSTM_EMBEDDING_SIZE = 1000
IMG_EMBED_SIZE = 512
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
LAMBDA_VALUE = 1e-1
KEEP_PROB = 0.5
DROPOUT = True
MODEL_SAVE_FOLDER = 'trained_models/cub/'  # path to save the trained model
