import caffe
import cv2
import numpy as np
from copy import deepcopy
import pickle
import os
from extract_visual_features import grad_cam
import params_cub as params
from os import makedirs
from utils import pre_pro_build_word_vocab, get_captions


def transform_img(img, img_width, img_height):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def normalize(x):
    return x / (np.sqrt(np.mean(np.square(x))) + 1e-5)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calculate_co_occurrence(image_relevant_words, vocab_size, word_to_ix, image_folder, classifier_prototxt_file, classifier_weights_file, class_file, image_names_file, final_layer,
                            data_layer, layers, neeta, no_of_channels, image_width, image_height, co_occurrence_file_name, co_occurrence_info_folder):
    """
    Calculate co-occurrence among visual features and words (nouns and adjectives) for a given classifier
    :param image_relevant_words:
    :param vocab_size:
    :param word_to_ix:
    :param image_folder:
    :param classifier_prototxt_file:
    :param classifier_weights_file:
    :param class_file:
    :param image_names_file:
    :param final_layer:
    :param data_layer:
    :param layers:
    :param neeta:
    :param no_of_channels:
    :param image_width:
    :param image_height:
    :param co_occurrence_file_name:
    :param co_occurrence_info_folder:
    :return:
    """

    total_number_of_channels = np.sum(np.asanyarray(no_of_channels))
    co_occurrence_matrix = np.zeros(shape=(total_number_of_channels, vocab_size))

    # -------- Read img names --------
    with open(image_names_file, 'rb') as f:
        image_names = [line.rstrip().decode('utf-8') for line in f.readlines()]

    nouns_adjectives = {}
    for key, value in image_relevant_words.items():
        nouns_adjectives[key] = [word_to_ix[x] for x in value if x in word_to_ix]

    # -------- Initialize the classifier --------
    net = caffe.Net(classifier_prototxt_file, classifier_weights_file, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1, 3, image_width, image_height)

    # -------- Read_class labels --------
    with open(class_file, 'r') as f:
        class_info = f.readlines()
    classes = []
    for w in range(len(class_info)):
        classes.append(class_info[w].split(".")[1].rstrip("\n\r").lower())

    for i in range(len(image_names)):

        if i % 100 == 0:
            print('Completed : %d / %d' % (i, len(image_names)))

        image_name = image_names[i].split('/')[1]

        co_occurrence_info = {}

        image_path = os.path.join(image_folder, image_names[i])
        class_name = '_'.join(image_name.split("_")[:-2])
        class_id = classes.index(class_name.lower())

        # -------- Forward pass the image --------
        pre_processed_image = caffe.io.load_image(image_path)
        net.blobs['data'].data[...] = transformer.preprocess('data', pre_processed_image)
        net.forward()

        image_noun_adjectives = nouns_adjectives[image_name]
        values = np.ones(shape=len(image_noun_adjectives)).astype(np.int32)  # Find indices of nouns and adjectives of this image
        vocab_instance = np.zeros(shape=vocab_size).astype(np.int32)
        indices = np.asanyarray(image_noun_adjectives).astype(np.int32)
        np.put(vocab_instance, indices, values)  # Mark nouns and adjectives of this image in a vocab instance

        for k in range(len(layers)):

            layer_name = layers[k]
            co_occurrence_info[layer_name] = {}
            offset = 0
            for n in range(k):
                offset = offset + no_of_channels[n]

            activations = net.blobs[layer_name].data[0, :, :, :]
            activations = np.maximum(0, activations)
            mean_activations = np.mean(activations, axis=(1, 2))
            co_occurrence_info[layer_name]['mean_activations'] = mean_activations

            # -------- Find weight of each feature map in the layer --------
            weights = grad_cam(net, class_id, layer_name, final_layer, data_layer)
            co_occurrence_info[layer_name]['weights'] = weights

            # -------- Normalize feature maps and find most important feature maps --------
            sm_weights = softmax(weights)
            sm_weights_sorted = np.sort(sm_weights)[::-1]
            index_sm = np.argmin(sm_weights_sorted.cumsum() < neeta)
            if index_sm == 0:
                index_sm = 1

            most_important_ms = sm_weights.argsort()[::-1][:index_sm]

            index_values = most_important_ms + offset

            change = np.tile(vocab_instance, (len(index_values), 1))
            co_occurrence_matrix[index_values, :] = co_occurrence_matrix[index_values, :] + change

        output_file_path = os.path.join(co_occurrence_info_folder, image_name + '.npz')
        np.savez_compressed(output_file_path, x=co_occurrence_info)

    np.save(co_occurrence_file_name, co_occurrence_matrix)


def find_co_occurrence_statistics(co_occurrence_matrix, co_occurrence_stat_matrix):
    """ Find co-occurrence statistics
    :param co_occurrence_matrix:
    :param co_occurrence_stat_matrix:
    :return: co_occurrence_stat_matrix
    """
    column_sum = np.sum(co_occurrence_matrix, axis=0)
    row_sum = np.sum(co_occurrence_matrix, axis=1)

    for i in range(co_occurrence_matrix.shape[0]):
        for k in range(co_occurrence_matrix.shape[1]):
            if row_sum[i] + column_sum[k] != 0:
                co_occurrence_stat_matrix[i][k] = co_occurrence_matrix[i][k] / (row_sum[i] + column_sum[k])

    return co_occurrence_stat_matrix


def find_decision_relevant_words(image_relevant_words, vocab_size, ix_to_word, co_occurrence_file, layers, no_of_channels, image_names_file, classifier_prototxt_file, classifier_weights_file,
                                 final_layer, class_file, image_folder, co_occurrence_stat_file, no_feats_from_layer, image_width, image_height, co_occurrence_stat_info_folder,
                                 decision_relevant_file):
    """Given the model decision find decision-relevant words
    :param image_relevant_words:
    :param vocab_size:
    :param ix_to_word:
    :param co_occurrence_file:
    :param layers:
    :param no_of_channels:
    :param image_names_file:
    :param classifier_prototxt_file:
    :param classifier_weights_file:
    :param final_layer:
    :param class_file:
    :param image_folder:
    :param co_occurrence_stat_file:
    :param no_feats_from_layer:
    :param image_width:
    :param image_height:
    :param co_occurrence_stat_info_folder:
    :param decision_relevant_file:
    :return:
    """

    # Load all the co_occurrence metrics calculated
    co_occurrence_matrix = np.load(co_occurrence_file)
    co_occurrence_stat_matrix = np.zeros(shape=(1472, vocab_size))

    # -------- Read img names --------
    with open(image_names_file, 'rb') as f:
        image_names = [line.rstrip().decode('utf-8') for line in f.readlines()]

    # -------- Calculate co-occurrence scores --------
    co_occurrence_stat_matrix = find_co_occurrence_statistics(co_occurrence_matrix, co_occurrence_stat_matrix)
    np.save(co_occurrence_stat_file, co_occurrence_stat_matrix)
    print("Co-occurrence statistics found!")

    # -------- Load the classifier --------
    net = caffe.Net(classifier_prototxt_file, classifier_weights_file, caffe.TEST)

    # -------- Load input and configure pre-processing --------
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1, 3, image_width, image_height)

    with open(class_file, 'r') as f:
        class_info = f.readlines()
    classes = []
    for w in range(len(class_info)):
        classes.append(class_info[w].split(".")[1].rstrip("\n\r").lower())

    decision_relevant_words_per_image = []

    if not os.path.exists(co_occurrence_stat_info_folder):
        makedirs(os.path.join(co_occurrence_stat_info_folder))

    for i in range(len(image_names)):

        co_occurrence_stat_info = {}
        all_words = []
        image_name = image_names[i].split('/')[1]

        if i % 100 == 0:
            print('Completed : %d / %d' % (i, len(image_names)))

        image_path = os.path.join(image_folder, image_names[i])

        pre_processed_image = caffe.io.load_image(image_path)
        net.blobs['data'].data[...] = transformer.preprocess('data', pre_processed_image)
        predictions = net.forward()
        class_id = np.argmax(predictions[final_layer])

        index_list = []
        layer_weights = []

        # -------- Find important features w.r.t to predicted class in each layer --------
        for k in range(len(layers)):
            layer_name = layers[k]
            co_occurrence_stat_info[layer_name] = {}
            weights_l = grad_cam(net, class_id, layer_name, final_layer)
            co_occurrence_stat_info[layer_name]['layer_weights'] = weights_l
            layer_weights.append(weights_l)

        for k in range(len(layers)):
            weights_list = []
            layer_name = layers[k]
            offset = 0
            for n in range(k):
                offset = offset + no_of_channels[n]

            weights = layer_weights[k]

            if k == 0:
                weights_sm = softmax(weights)
                most_impt_feat_map = weights_sm.argsort()[::-1][:no_feats_from_layer]
                net_copy = net
                diffs = deepcopy(net.blobs[layer_name].diff[0, :, :, :])

                for l in range(no_feats_from_layer):
                    loss = np.zeros(shape=net.blobs[layer_name].data[...].shape)
                    loss[0, most_impt_feat_map[l], :, :] = diffs[most_impt_feat_map[l], :, :]
                    net_copy.forward(end=layer_name)

                    # -------- Back propagate to find out which feature maps in the layer beneath caused max activation --------
                    net_copy.blobs[layer_name].diff[...] = loss
                    imdiff = net_copy.backward(start=layer_name, end=layers[k + 1], diffs=[layers[k + 1]])
                    gradients = imdiff[layers[k + 1]]
                    gradients = normalize(gradients)
                    gradients = gradients[0, :, :, :]
                    weights_list.append(np.mean(gradients, axis=(1, 2)))
                prev_weights = weights_list

                for l in range(no_feats_from_layer):
                    index_list.append(most_impt_feat_map[l])
                    index = most_impt_feat_map[l] + offset
                    matching_word_index = np.argmax(co_occurrence_stat_matrix[index])
                    word = ix_to_word[matching_word_index]
                    all_words.append(word)

            else:
                internal_i_list = []
                net_copy = net
                diffs = deepcopy(net.blobs[layer_name].diff[0, :, :, :])

                for l in range(no_feats_from_layer):
                    weights_l_sm = softmax(prev_weights[l])
                    internal_i_list.append(np.argmax(weights_l_sm))

                if k + 1 < len(layers):  # The last convolutional layer should not be considered
                    for l in range(no_feats_from_layer):

                        index = internal_i_list[l]

                        loss = np.zeros(shape=net.blobs[layer_name].data[...].shape)
                        loss[0, index, :, :] = diffs[index, :, :]
                        net_copy.forward(end=layer_name)
                        # -------- Back propagate to find out what feature map in the layer beneath caused max activation
                        net_copy.blobs[layer_name].diff[...] = loss
                        imdiff = net_copy.backward(start=layer_name, end=layers[k + 1], diffs=[layers[k + 1]])
                        gradients = imdiff[layers[k + 1]]
                        gradients = normalize(gradients)
                        gradients = gradients[0, :, :, :]
                        weights_list.append(np.mean(gradients, axis=(1, 2)))
                    prev_weights = weights_list

                for l in range(no_feats_from_layer):
                    index = (internal_i_list[l]) + offset
                    matching_word_index = np.argmax(co_occurrence_stat_matrix[index])
                    word = ix_to_word[matching_word_index]
                    all_words.append(word)

            co_occurrence_stat_info[layer_name]['pre_weights'] = prev_weights

        output_file_path = os.path.join(co_occurrence_stat_info_folder, image_name.split('.')[0] + '.npz')
        np.savez_compressed(output_file_path, x=co_occurrence_stat_info)

        # From the identified words select only image relevant words
        filtered_words = list(set(all_words).intersection(set(image_relevant_words[image_name])))
        decision_relevant_words_per_image.append(image_name + "#" + ','.join(filtered_words))

        if i % 100 == 0:  # Save intermediate_results
            target = open(decision_relevant_file, 'w')
            target.write('\n'.join(decision_relevant_words_per_image))
            target.close()

    target = open(decision_relevant_file, 'w')
    target.write('\n'.join(decision_relevant_words_per_image))
    target.close()


def main():
    image_folder = params.IMAGE_FOLDER
    classifier_prototxt_file = params.CLASSIFIER_PROTOTXT_FILE
    classifier_weights_file = params.CLASSIFIER_WEIGHT_FILE
    class_file = params.CLASS_NAMES_FILE
    final_layer = params.FINAL_LAYER_NAME
    data_layer = params.DATA_LAYER_NAME
    layers = params.LAYERS
    neeta = params.NEETA
    no_of_channels = params.NO_OF_CHANNELS
    image_width = params.IMAGE_WIDTH
    image_height = params.IMAGE_HEIGHT
    co_occurrence_info_folder = params.CO_OCCURRENCE_INFO_FOLDER
    co_occurrence_stat_info_folder = params.CO_OCCURRENCE_STAT_INFO_FOLDER

    if not os.path.exists(co_occurrence_stat_info_folder):
        os.makedirs(co_occurrence_stat_info_folder)

    if not os.path.exists(co_occurrence_info_folder):
        makedirs(os.path.join(co_occurrence_info_folder))

    # -------- For train dataset --------

    image_names_file = params.TRAIN_IMAGE_NAMES
    co_occurrence_file_name = params.CO_OCCURRENCE_FILE_NAME_TRAIN
    co_occurrence_stat_file = params.CO_OCCURRENCE_STAT_FILE_NAME_TRAIN
    no_feats_from_layer = params.NO_OF_TOP_FEATURE_MAPS_FROM_LAYER
    decision_relevant_file = params.DECISION_RELEVANT_WORDS_TRAIN
    train_noun_adjectives = np.load(params.TRAIN_NOUNS_ADJECTIVES, allow_pickle=True).item()

    noun_adjective_list = []
    for key, value in train_noun_adjectives.items():
        noun_adjective_list.append(' '.join(list(value)))
    word_to_id, id_to_word, _ = pre_pro_build_word_vocab(noun_adjective_list, word_count_threshold=3)
    vocab_size = len(word_to_id)

    co_occurrence_folder = co_occurrence_file_name.split('/')[0]

    if not os.path.exists(co_occurrence_folder):
        makedirs(os.path.join(co_occurrence_folder))

    calculate_co_occurrence(train_noun_adjectives, vocab_size, word_to_id, image_folder, classifier_prototxt_file, classifier_weights_file, class_file, image_names_file,
                             final_layer, data_layer, layers, neeta, no_of_channels, image_width, image_height, co_occurrence_file_name, co_occurrence_info_folder)
    find_decision_relevant_words(train_noun_adjectives, vocab_size, id_to_word, co_occurrence_file_name, layers, no_of_channels, image_names_file, classifier_prototxt_file, classifier_weights_file,
                                 final_layer, class_file, image_folder, co_occurrence_stat_file, no_feats_from_layer, image_width, image_height, co_occurrence_stat_info_folder, decision_relevant_file)

    # -------- For val dataset --------

    image_names_file = params.VAL_IMAGE_NAMES
    co_occurrence_file_name = params.CO_OCCURRENCE_FILE_NAME_VAL
    co_occurrence_stat_file = params.CO_OCCURRENCE_STAT_FILE_NAME_VAL
    no_feats_from_layer = params.NO_OF_TOP_FEATURE_MAPS_FROM_LAYER
    decision_relevant_file = params.DECISION_RELEVANT_WORDS_VAL
    val_noun_adjectives = np.load(params.VAL_NOUNS_ADJECTIVES, allow_pickle=True).item()

    calculate_co_occurrence(val_noun_adjectives, vocab_size, word_to_id, image_folder, classifier_prototxt_file, classifier_weights_file, class_file, image_names_file,
                            final_layer, data_layer, layers, neeta, no_of_channels, image_width, image_height, co_occurrence_file_name, co_occurrence_info_folder)
    find_decision_relevant_words(val_noun_adjectives, vocab_size, id_to_word, co_occurrence_file_name, layers, no_of_channels, image_names_file, classifier_prototxt_file, classifier_weights_file,
                                 final_layer, class_file, image_folder, co_occurrence_stat_file, no_feats_from_layer, image_width, image_height, co_occurrence_stat_info_folder, decision_relevant_file)


if __name__ == '__main__':
    main()
