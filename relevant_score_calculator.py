import numpy as np
import os
import pickle
import params_cub as params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def score_words_wrt_image(vocab, descriptions_file):
    """
    Calculate score of each word in the vocab w.r.t. the image
    :param vocab:
    :param descriptions_file:
    :return:
    """

    # -------- A text pre-processing step --------
    lemmas = {}
    fw = open("data/cub/lemma.txt", 'r')
    for line in fw.readlines():
        lemmas[line.rstrip("\n\r").split("#")[0]] = line.rstrip("\n\r").split("#")[1]

    # -------- Indicate word occurrence for each class and each image --------
    lemma_dict = {}
    class_word_occurrence = {}
    image_word_occurrence = {}

    f = open(descriptions_file, 'r')
    for line in f.readlines():
        image_name = line.rstrip("\n").split("#")[0]
        class_name = '_'.join(image_name.split("_")[:-2])

        if class_name not in class_word_occurrence.keys():
            class_word_occurrence[class_name] = np.zeros(shape=len(vocab))

        if image_name not in image_word_occurrence.keys():
            image_word_occurrence[image_name] = np.zeros(shape=len(vocab))
            lemma_dict[image_name] = {}

        words = line.rstrip("\n").split("#")[1].split(" ")
        for word in words:
            if word not in vocab:
                continue

            if word in lemmas.keys():
                lemma_key = lemmas[word]
                if lemma_key not in lemma_dict[image_name].keys():
                    lemma_dict[image_name][lemma_key] = set()
                lemma_dict[image_name][lemma_key].add(word)

            index_of_word = vocab.index(word)
            class_word_occurrence[class_name][index_of_word] = 1
            image_word_occurrence[image_name][index_of_word] = 1

    # -------- Calculate the score for words w.r.t. class as cls_score = -log(number of words relevant to the class/len of vocab) --------
    for key, value in class_word_occurrence.items():
        no_relevant_words = np.sum([x for x in class_word_occurrence[key] if x == 1])
        weight = (np.log(no_relevant_words / len(vocab))) * (-1)
        for i in range(len(vocab)):
            if class_word_occurrence[key][i] == 1:
                class_word_occurrence[key][i] = weight

    # -------- Calculate the score for words w.r.t. image as img_score = -log(number of words relevant to the image/len of vocab) --------
    # -------- Total score of a word becomes cls_score+img_score
    for key, value in image_word_occurrence.items():
        no_relevant_words = np.sum([x for x in image_word_occurrence[key] if x == 1])
        weight = (np.log(no_relevant_words / len(vocab))) * (-1)

        class_name = '_'.join(key.split("_")[:-2])

        for i in range(len(vocab)):
            if image_word_occurrence[key][i] == 1:
                image_word_occurrence[key][i] = weight + class_word_occurrence[class_name][i]
            elif class_word_occurrence[class_name][i] > 0:
                image_word_occurrence[key][i] = class_word_occurrence[class_name][i]  # Words which are not related to this image but related to this class of images

    return image_word_occurrence, lemma_dict


def calculate_decision_relevance_scores(data_dictionary_name, img_names, descriptions_relevant_words, decision_relevance_score_file, decision_relevance_score_dict_file, descriptions_file):
    """
    Calculate decision relevance scores as described in the FLEX paper
    :param data_dictionary_name:
    :param img_names:
    :param descriptions_relevant_words:
    :param decision_relevance_score_file:
    :param decision_relevance_score_dict_file:
    :param descriptions_file:
    :return:
    """
    # -------- Load data dictionary and get vocab --------
    f = open(data_dictionary_name, 'rb')
    data_dict = pickle.load(f)
    f.close()

    vocab = data_dict['vocab']

    # -------- Read image names --------
    image_names = []
    f = open(img_names, 'r')
    for line in f.readlines():
        image_names.append(line.rstrip("\n").split("/")[1])

    image_word_relevance, lemma_dict = score_words_wrt_image(vocab, descriptions_file)
    print("Score for each word w.r.t image and class is found...")

    # print("Finding important and more important attributes and their values.....")
    image_decision_relevant_words = {}
    with open(descriptions_relevant_words, 'r') as f:
        decision_relevant_words = f.readlines()
    for k in range(len(decision_relevant_words)):
        image_name = decision_relevant_words[k].rstrip("\n\r").split("#")[0]
        decision_relevant_words[k] = decision_relevant_words[k].rstrip("\n\r").split("#")[1].replace("-", ",").replace(" ", ",")
        words = decision_relevant_words[k] .split(',')

        lemma_keys = lemma_dict[image_name].keys()
        included_lemmas = set(lemma_keys).intersection(set(words))

        for l in list(included_lemmas):
            words = words + list(lemma_dict[image_name][l])

        if image_name not in image_decision_relevant_words.keys():
            image_decision_relevant_words[image_name] = np.zeros(shape=len(vocab))

        for w in range(len(words)):
            if words[w] == "":
                continue
            index_of_word = vocab.index(words[w])
            image_decision_relevant_words[image_name][index_of_word] = 1

    # -------- Calculate the score for words w.r.t. decision as decision_score = -log(number of words relevant to the decision/len of vocab) --------
    # -------- Total score of a word becomes decision_score+img_score
    for key, value in image_decision_relevant_words.items():
        no_relevant_words = np.sum([x for x in image_decision_relevant_words[key] if x == 1]) + 0.00001
        weight = (np.log(no_relevant_words / len(vocab))) * (-1)
        for i in range(len(vocab)):
            if image_decision_relevant_words[key][i] == 1:
                image_decision_relevant_words[key][i] = weight + image_word_relevance[key][i]
            elif image_word_relevance[key][i] > 0:
                image_decision_relevant_words[key][i] = image_word_relevance[key][i]

    decision_relevance_scores_dictionary = {}
    for key, value in image_decision_relevant_words.items():
        max_value = np.max(value)
        min_value = np.min(value)
        value = np.subtract(value, [min_value])
        value = value / (max_value - min_value)
        decision_relevance_scores_dictionary[key] = value

    # -------- Create decision relevant score vectors --------
    decision_relevance_scores = np.zeros(shape=(len(image_names), len(vocab)))
    decision_relevance_scores_all_dict = {}
    for i in range(len(image_names)):
        image_name = image_names[i]
        value = decision_relevance_scores_dictionary[image_name]

        decision_relevance_scores[i, :] = value
        decision_relevance_scores_all_dict[image_name] = value

    # --------- Save decision relevant score vectors --------
    np.save(decision_relevance_score_file, decision_relevance_scores)
    np.save(decision_relevance_score_dict_file, decision_relevance_scores_all_dict)


def expand_words_weight_vectors(no_sent_per_image, img_names, decision_relevance_score, decision_relevance_score_expanded):
    """ Expand decision relevance scores for each image for the number of sentences describing that image
    :return:
    """

    # -------- Load number of sentences per image --------
    num_for_each_image = np.load(no_sent_per_image, allow_pickle=True).item()

    image_names = []
    f = open(img_names, 'r')
    for line in f.readlines():
        image_name = line.rstrip("\n\r").split("/")[1]
        image_names.append(image_name)

    calculated_relevance = np.load(decision_relevance_score)
    final_relevance_scores = []
    for i in range(len(calculated_relevance)):
        image_name = image_names[i]

        # -------- Repeat visual feature for 'num' of times
        num = num_for_each_image[image_name]
        for k in range(num):
            final_relevance_scores.append(calculated_relevance[i])

    np.save(decision_relevance_score_expanded, final_relevance_scores)


def main():

    data_dictionary_name = params.DATA_DIC_NAME

    # -------- For train data --------
    descriptions_relevant_words = params.DECISION_RELEVANT_WORDS_TRAIN
    no_sent_per_image = params.TRAIN_NO_SENT_PER_IMAGE
    img_names = params.TRAIN_IMAGE_NAMES
    decision_relevance_score = params.DECISION_RELEVANCE_SCORE_TRAIN
    decision_relevance_score_expanded = params.DECISION_RELEVANCE_SCORE_TRAIN_EXPANDED
    decision_relevance_score_dict = params.DECISION_RELEVANCE_SCORE_DICT_TRAIN
    descriptions_file = params.TRAIN_DESC_FILE

    print('calculating train decision relevance scores...')

    calculate_decision_relevance_scores(data_dictionary_name, img_names, descriptions_relevant_words, decision_relevance_score, decision_relevance_score_dict, descriptions_file)
    expand_words_weight_vectors(no_sent_per_image, img_names, decision_relevance_score, decision_relevance_score_expanded)

    # -------- For val data --------
    descriptions_relevant_words = params.DECISION_RELEVANT_WORDS_VAL
    no_sent_per_image = params.VAL_NO_SENT_PER_IMAGE
    img_names = params.VAL_IMAGE_NAMES
    decision_relevance_score = params.DECISION_RELEVANCE_SCORE_VAL
    decision_relevance_score_expanded = params.DECISION_RELEVANCE_SCORE_VAL_EXPANDED
    decision_relevance_score_dict = params.DECISION_RELEVANCE_SCORE_DICT_VAL
    descriptions_file = params.VAL_DESC_FILE

    print('calculating val decision relevance scores...')

    calculate_decision_relevance_scores(data_dictionary_name, img_names, descriptions_relevant_words, decision_relevance_score, decision_relevance_score_dict, descriptions_file)
    expand_words_weight_vectors(no_sent_per_image, img_names, decision_relevance_score, decision_relevance_score_expanded)


if __name__ == '__main__':
    main()
