import os
from flex import FLEX
import tables
import argparse
from nltk.tokenize.moses import MosesDetokenizer
import numpy as np
import pickle
import params_cub as params


def generate_explanations_for_one_image(image_name):
    """
    Generate explanations for one image in the test set
    :param image_name:
    :return:
    """
    # -------- LOAD DATA --------
    data_dict = pickle.load(open(params.DATA_DIC_NAME, 'rb'))

    vocab_size = data_dict['num_words']  # no. words in vocab
    id_to_word = data_dict['id_to_word']  # dict: integer id to english word
    word_to_id = data_dict['word_to_id']  # dict: english word to id
    max_length = data_dict['max_length']  # max length of sentence, or X_sentence.shape[-1]

    with open(params.TEST_IMAGE_NAMES, 'rb') as f:
        test = [line.rstrip().decode('utf-8') for line in f.readlines()]
        test_names = [str(f).split("/")[-1] for f in test]

    print('Done loading data_dict')

    # -------- LOAD MODEL --------
    model_path = os.path.join(params.MODEL_SAVE_FOLDER, opt.model_version, 'flex')

    model_params = {
        'img_embed_size': params.IMG_EMBED_SIZE,
        'lstm_embedding_size': params.LSTM_EMBEDDING_SIZE,
        'num_hidden_lstm': params.LSTM_HIDDEN_NUM,
        'vocab_size': vocab_size,
        'max_length': max_length,
        'learning_rate': params.LEARNING_RATE,
        'dropout': params.DROPOUT,
        'lambda_value': params.LAMBDA_VALUE,
        'img_feature_size': params.FEATURE_VECTOR_SIZE}

    flex_model = FLEX(model_params)
    flex_model.load(model_path)
    print("Done loading model")

    detokenizer = MosesDetokenizer()

    data_f = tables.open_file(params.TEST_VISUAL_FEATURE_FILE, mode='r')

    # -------- WRITE EXPLANATIONS --------

    index = test_names.index(image_name)
    feat = data_f.root.data[index, :]
    feat = np.expand_dims(feat, axis=0)
    sentence, indices = flex_model.get_explanation(feat, id_to_word, word_to_id)
    sent = [id_to_word.get(s) for s in indices]
    sent = detokenizer.detokenize(sent, return_str=True)
    print(test[index] + ": ", sent)

    class_name = " ".join(i.capitalize() for i in image_name.lower().split('/')[-1].split("_")[:-2])

    exp = "This is a <b>" + class_name + "</b> because, <br>" + sent
    return exp


def generate_explanations():
    """
    Generate explanations for the complete cub test set
    :return:
    """
    # -------- LOAD DATA --------
    data_dict = pickle.load(open(params.DATA_DIC_NAME, 'rb'))
    print('Done loading data_dict')

    vocab_size = data_dict['num_words']  # no. words in vocab
    id_to_word = data_dict['id_to_word']  # dict: integer id to english word
    word_to_id = data_dict['word_to_id']  # dict: english word to id
    max_length = data_dict['max_length']  # max length of sentence, or X_sentence.shape[-1]
    print('Done extracting objects from data_dict')

    with open(params.TEST_IMAGE_NAMES, 'rb') as f:
        test = [line.rstrip().decode('utf-8') for line in f.readlines()]
        test_names = [str(f).split("/")[-1] for f in test]

    # -------- LOAD MODEL --------
    model_path = os.path.join(params.MODEL_SAVE_FOLDER, opt.model_version, 'flex')

    model_params = {
        'img_embed_size': params.IMG_EMBED_SIZE,
        'lstm_embedding_size': params.LSTM_EMBEDDING_SIZE,
        'num_hidden_lstm': params.LSTM_HIDDEN_NUM,
        'vocab_size': vocab_size,
        'max_length': max_length,
        'learning_rate': params.LEARNING_RATE,
        'dropout': params.DROPOUT,
        'lambda_value': params.LAMBDA_VALUE,
        'img_feature_size': params.FEATURE_VECTOR_SIZE}

    flex_model = FLEX(model_params)
    flex_model.load(model_path)
    print("Done loading Model")

    detokenizer = MosesDetokenizer()

    data_f = tables.open_file(params.TEST_VISUAL_FEATURE_FILE, mode='r')
    # -------- WRITE EXPLANATIONS --------
    explanations = []
    for i, img_name in enumerate(test_names):
        index = test_names.index(img_name)
        feat = data_f.root.data[index, :]
        feat = np.expand_dims(feat, axis=0)
        sentence, indices = flex_model.get_explanation(feat, id_to_word, word_to_id)
        sent = [id_to_word.get(s) for s in indices]
        sent = detokenizer.detokenize(sent, return_str=True)
        explanation = test_names[index] + ": " + sent
        print(str(i)+':'+explanation)
        explanations.append(explanation)

    target = open(os.path.join(params.MODEL_SAVE_FOLDER, opt.model_version, 'explanations_' + opt.model_version+".txt"), 'w')
    target.write('\n'.join(explanations))
    target.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='generate_exp', help='Which method you want to execute')
    parser.add_argument('--model_version', default='flex_v1', help='Name of the trained model')
    parser.add_argument('--image_name', default='Cerulean_Warbler_0005_797206,jpg', help='The test image for which you want to generate the explanation')

    opt = parser.parse_args()

    if opt.method == 'generate_exp':
        generate_explanations()
    elif opt.method == 'generate_one_exp':
        generate_explanations_for_one_image(opt.image_name)


