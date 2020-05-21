import os
import numpy as np
import nltk
import pickle
from keras.preprocessing import sequence
from utils import pre_pro_build_word_vocab, get_captions
import params_cub as params


def prepare_text_descriptions(text_desc_folder, img_names_file, descriptions_file, no_sent_per_image_file):
    """
    Preparing train, val and test text descriptions
    Download text data for CUB dataset from here https://github.com/reedscot/cvpr2016
    Place text_c10 folder inside 'data' folder

    :param text_desc_folder:
    :param img_names_file:
    :param descriptions_file:
    :param no_sent_per_image_file:
    :return: descriptions
    """

    folder_name = text_desc_folder

    image_names = []
    with open(img_names_file, 'r') as f:
        for line in f.readlines():
            image_name = line.rstrip("\n\r").split('/')[1]
            image_names.append(image_name)

    descriptions = []

    # -------- To calculate number of sentences per each image --------
    no_sent_per_image = {}

    for sub_folder_name in os.listdir(folder_name):
        for file_name in os.listdir(folder_name + '/' + sub_folder_name):
            if '.txt' in file_name:
                fw = open(folder_name + '/' + sub_folder_name + '/' + file_name, 'r')
                image_name = file_name.split('.')[0] + '.jpg'
                for line in fw.readlines():
                    if image_name in image_names:
                        if image_name not in no_sent_per_image.keys():
                            no_sent_per_image[image_name] = 0
                        no_sent_per_image[image_name] += 1
                        tokenized_words = nltk.word_tokenize(line.rstrip("\n\r"))
                        if tokenized_words[-1] != '.':
                            tokenized_words = tokenized_words+['.']
                        sentence = " ".join(tokenized_words)
                        descriptions.append(image_name + "#" + sentence)

    target = open(descriptions_file, 'w')
    target.write('\n'.join(descriptions))
    target.close()

    np.save(no_sent_per_image_file, no_sent_per_image)

    return descriptions


def prepare_data(text_desc_folder, train_img_names, train_descriptions_file, train_no_sent_per_image_file, val_img_names, val_descriptions_file, val_no_sent_per_image_file, test_img_names,
                 test_descriptions_file, test_no_sent_per_image_file, data_dictionary_name):
    """
    Prepare all the text data required to train FLEX
    :param text_desc_folder:
    :param train_img_names:
    :param train_descriptions_file:
    :param train_no_sent_per_image_file:
    :param val_img_names:
    :param val_descriptions_file:
    :param val_no_sent_per_image_file:
    :param test_img_names:
    :param test_descriptions_file:
    :param test_no_sent_per_image_file:
    :param data_dictionary_name:
    :return:
    """

    print("Preparing text data...")

    # -------- Prepare text descriptions ---------
    print("Preparing text descriptions...")

    # -------- For train images --------
    train_descriptions = prepare_text_descriptions(text_desc_folder, train_img_names, train_descriptions_file, train_no_sent_per_image_file)
    # -------- For val images --------
    val_descriptions = prepare_text_descriptions(text_desc_folder, val_img_names, val_descriptions_file, val_no_sent_per_image_file)
    # -------- For test images --------
    test_descriptions = prepare_text_descriptions(text_desc_folder, test_img_names, test_descriptions_file, test_no_sent_per_image_file)

    # -------- Prepare word_to_id and id_to_word dictionaries as well as vocab list --------
    captions, images = get_captions(train_descriptions_file)
    max_length = np.max([x for x in map(lambda x: len(x.split(' ')), captions)]).astype(np.int32)
    word_to_id, id_to_word, vocab = pre_pro_build_word_vocab(captions, word_count_threshold=3)

    train_sentences_dict = {}
    for train_desc in train_descriptions:
        image_name = train_desc.rstrip().split("#")[0]
        sentence = train_desc.rstrip().split("#")[1]
        if image_name not in train_sentences_dict.keys():
            train_sentences_dict[image_name] = []
        train_sentences_dict[image_name].append(sentence)

    val_sentences_dict = {}
    for val_desc in val_descriptions:
        image_name = val_desc.rstrip().split("#")[0]
        sentence = val_desc.rstrip().split("#")[1]
        if image_name not in val_sentences_dict.keys():
            val_sentences_dict[image_name] = []
        val_sentences_dict[image_name].append(sentence)

    train_images = []
    with open(train_img_names, 'r') as f:
        for line in f.readlines():
            image_name = line.rstrip("\n\r").split('/')[1]
            train_images.append(image_name)

    val_images = []
    with open(val_img_names, 'r') as f:
        for line in f.readlines():
            image_name = line.rstrip("\n\r").split('/')[1]
            val_images.append(image_name)

    x_train_word_sequences = []
    y_train_word_sequences = []
    train_lengths = []

    for k in range(len(train_images)):
        image_name = train_images[k]
        sentences = train_sentences_dict[image_name]

        for sens in sentences:
            words = sens.split(" ")
            x_train_word_sequences.append([word_to_id['<s>']] + [word_to_id[x] for x in words if x in word_to_id])
            y_train_word_sequences.append([word_to_id[x] for x in words if x in word_to_id])
            train_lengths.append(len([word_to_id[x] for x in words if x in word_to_id]) + 1)

    x_train_word_sequences = sequence.pad_sequences(x_train_word_sequences, padding='post', maxlen=max_length)
    y_train_word_sequences = sequence.pad_sequences(y_train_word_sequences, padding='post', maxlen=max_length)

    x_val_word_sequences = []
    y_val_word_sequences = []
    val_lengths = []

    for k in range(len(val_images)):
        image_name = val_images[k]
        sentences = val_sentences_dict[image_name]

        for sens in sentences:
            words = sens.split(" ")
            x_val_word_sequences.append([word_to_id['<s>']] + [word_to_id[x] for x in words if x in word_to_id])
            y_val_word_sequences.append([word_to_id[x] for x in words if x in word_to_id])
            val_lengths.append(len([word_to_id[x] for x in words if x in word_to_id]) + 1)

    x_val_word_sequences = sequence.pad_sequences(x_val_word_sequences, padding='post', maxlen=max_length)
    y_val_word_sequences = sequence.pad_sequences(y_val_word_sequences, padding='post', maxlen=max_length)

    data_dict = {}
    data_dict['max_length'] = max_length
    data_dict['id_to_word'] = id_to_word
    data_dict['word_to_id'] = word_to_id
    data_dict['num_words'] = len(word_to_id)
    data_dict['vocab'] = vocab
    data_dict["x_train_word_sequences"] = x_train_word_sequences
    data_dict['x_val_word_sequences'] = x_val_word_sequences
    data_dict['y_train_word_sequences'] = y_train_word_sequences
    data_dict['y_val_word_sequences'] = y_val_word_sequences
    data_dict['train_lengths'] = train_lengths
    data_dict['val_lengths'] = val_lengths

    pickle.dump(data_dict, open(data_dictionary_name, "wb"))


def main():
    text_desc_folder = params.TEXT_DESC_FOLDER
    train_img_names = params.TRAIN_IMAGE_NAMES
    train_descriptions_file = params.TRAIN_DESC_FILE
    train_no_sent_per_image_file = params.TRAIN_NO_SENT_PER_IMAGE
    val_img_names = params.VAL_IMAGE_NAMES
    val_descriptions_file = params.VAL_DESC_FILE
    val_no_sent_per_image_file = params.VAL_NO_SENT_PER_IMAGE
    test_img_names = params.TEST_IMAGE_NAMES
    test_descriptions_file = params.TEST_DESC_FILE
    test_no_sent_per_image_file = params.TEST_NO_SENT_PER_IMAGE
    data_dictionary_name = params.DATA_DIC_NAME

    if not os.path.exists(params.DATA_DIRECTORY):
        os.makedirs(params.DATA_DIRECTORY)

    prepare_data(text_desc_folder, train_img_names, train_descriptions_file, train_no_sent_per_image_file, val_img_names, val_descriptions_file, val_no_sent_per_image_file, test_img_names,
                 test_descriptions_file, test_no_sent_per_image_file, data_dictionary_name)


if __name__ == '__main__':
    main()
