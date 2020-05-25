DATA_DIRECTORY = 'data/cub/'

DATA_DIC_NAME = DATA_DIRECTORY+"data_dict.pkl"

TRAIN_DESC_FILE = DATA_DIRECTORY+'train_descriptions.txt'
VAL_DESC_FILE = DATA_DIRECTORY+'val_descriptions.txt'
TEST_DESC_FILE = DATA_DIRECTORY+'test_descriptions.txt'
TRAIN_NO_SENT_PER_IMAGE = DATA_DIRECTORY+"train_no_sent_per_image.npy"
VAL_NO_SENT_PER_IMAGE = DATA_DIRECTORY+"val_no_sent_per_image.npy"
TEST_NO_SENT_PER_IMAGE = DATA_DIRECTORY+"test_no_sent_per_image.npy"

# -------- Classifier parameters --------
FEATURE_VECTOR_SIZE = 8192  # dimension of visual features in the desired layer
CLASSIFIER_PROTOTXT_FILE = 'classifiers/cmpt-bilinear/ft_all.prototxt'  # path to prototxt file
CLASSIFIER_WEIGHT_FILE = 'classifiers/cmpt-bilinear/ft_all_iter_20000.caffemodel'  # path to weights file
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
LAYER_NAME = "bilinear_l2"  # layer name from which features are extracted
FINAL_LAYER_NAME = "fc8_cub"  # name of the final layer of the classifier
DATA_LAYER_NAME = 'data'
LAYERS = ["conv5_3", "pool4", "pool3", "pool2", "pool1"]  # Different layers from which we extract features
NO_OF_CHANNELS = [512, 512, 256, 128, 64]  # Number of channels in the each layer we declared above

# -------- Image dataset related parameters --------
IMAGE_FOLDER = DATA_DIRECTORY+"images"  # path to images folder
ALL_IMAGE_NAME_FILE = DATA_DIRECTORY+"images.txt"  # path to images name file
IMAGE_FEATURE_SAVE_FOLDER = DATA_DIRECTORY+"visual_features"  # folder to save activations and gradients
TEXT_DESC_FOLDER = DATA_DIRECTORY+'text_c10'  # folder containing text descriptions
CLASS_NAMES_FILE = DATA_DIRECTORY+'classes.txt'
TRAIN_IMAGE_NAMES = DATA_DIRECTORY+"train_no_cub.txt"  # Name of the file containing train images
VAL_IMAGE_NAMES = DATA_DIRECTORY+"val_no_cub.txt"
TEST_IMAGE_NAMES = DATA_DIRECTORY+"test_no_cub.txt"


# -------- Relevance score calculation related information --------
NEETA = 0.80
NO_OF_TOP_FEATURE_MAPS_FROM_LAYER = 5
CO_OCCURRENCE_FILE_NAME_TRAIN = 'co_occurrence/cub/co_occurrence_train.npy'
CO_OCCURRENCE_FILE_NAME_VAL = 'co_occurrence/cub/co_occurrence_val.npy'
CO_OCCURRENCE_STAT_FILE_NAME_TRAIN = 'co_occurrence/cub/co_occurrence_stat_train.npy'
CO_OCCURRENCE_STAT_FILE_NAME_VAL = 'co_occurrence/cub/co_occurrence_stat_val.npy'
CO_OCCURRENCE_INFO_FOLDER = 'co_occurrence_info/cub'
CO_OCCURRENCE_STAT_INFO_FOLDER = 'co_occurrence_stat_info/cub'
DECISION_RELEVANT_WORDS_TRAIN = DATA_DIRECTORY+'decision_relevant_words.txt'
DECISION_RELEVANT_WORDS_VAL = DATA_DIRECTORY+'decision_relevant_words_val.txt'
DECISION_RELEVANCE_SCORE_TRAIN = DATA_DIRECTORY+'train_decision_relevance_scores.npy'
DECISION_RELEVANCE_SCORE_VAL = DATA_DIRECTORY+'val_decision_relevance_scores.npy'
DECISION_RELEVANCE_SCORE_TRAIN_EXPANDED = DATA_DIRECTORY+'train_decision_relevance_scores_expanded.npy'
DECISION_RELEVANCE_SCORE_VAL_EXPANDED = DATA_DIRECTORY+'val_decision_relevance_scores_expanded.npy'
DECISION_RELEVANCE_SCORE_DICT_TRAIN = DATA_DIRECTORY+'train_decision_relevance_scores_dict.npy'
DECISION_RELEVANCE_SCORE_DICT_VAL = DATA_DIRECTORY+'val_decision_relevance_scores_dict.npy'

TRAIN_NOUNS_ADJECTIVES = DATA_DIRECTORY+'train_noun_adjectives_words.npy'
VAL_NOUNS_ADJECTIVES = DATA_DIRECTORY+'val_noun_adjectives_words.npy'


# -------- Visual feature information --------
TRAIN_VISUAL_FEATURE_FILE = DATA_DIRECTORY+"cub_bilinear_train_visual_feature.h5"  # file name of the train visual features
VAL_VISUAL_FEATURE_FILE = DATA_DIRECTORY+"cub_bilinear_val_visual_feature.h5"  # file name of the val visual features
TEST_VISUAL_FEATURE_FILE = DATA_DIRECTORY+"cub_bilinear_test_visual_feature.h5"  # file name of the test visual features
IMG_FEATURE_SAVE_FOLDER_PATH = DATA_DIRECTORY+"visual_features"  # the folder to save activations and gradients


# -------- FLEX training information --------
LSTM_HIDDEN_NUM = 1000
LSTM_EMBEDDING_SIZE = 1000
IMG_EMBED_SIZE = 512
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4
LAMBDA_VALUE = 1e-1
KEEP_PROB = 0.5
DROPOUT = True
MODEL_SAVE_FOLDER = 'trained_models/cub/'  # path to save the trained model
MODEL_VERSION = 'flex_v1'  # model version
