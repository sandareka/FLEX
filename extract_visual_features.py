import caffe
import numpy as np
import os
import tables
import params_cub as params

caffe.set_mode_cpu()


def normalize(x):
    return x / (np.sqrt(np.mean(np.square(x))) + 1e-5)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def grad_cam(input_model, category_index, layer_name, final_layer="fc8_cub", data_layer='data'):
    label = np.zeros(input_model.blobs[final_layer].shape)
    label[0, category_index] = 1

    imdiff = input_model.backward(diffs=[data_layer, layer_name], **{input_model.outputs[0]: label})
    gradients = imdiff[layer_name]
    gradients = normalize(gradients)
    gradients = gradients[0, :, :, :]

    weights = np.mean(gradients, axis=(1, 2))

    return weights


def get_gradients_and_weights(input_model, category_index, layer_name, final_layer="fc8_cub", data_layer_name='data'):
    label = np.zeros(input_model.blobs[final_layer].shape)
    label[0, category_index] = 1

    im_diff = input_model.backward(diffs=[data_layer_name, layer_name], **{input_model.outputs[0]: label})
    gradients = im_diff[layer_name]
    gradients = normalize(gradients)
    gradients = gradients[0, :, :, :]

    weights = np.mean(gradients, axis=(1, 2))

    return gradients, weights


def extract_visual_feats_and_gradients(all_image_name_file, prototxt_file, weight_file, image_width, image_height, image_folder, layer_name, save_folder_path, final_layer, data_layer):
    """Extract visual features from a given layer of a Caffe model
    :param all_image_name_file:
    :param prototxt_file:
    :param weight_file:
    :param image_width:
    :param image_height:
    :param image_folder:
    :param layer_name:
    :param save_folder_path:
    :param final_layer:
    :param data_layer:
    :return:
    """
    predicted_class_list = []

    # ------- Get image names --------
    with open(os.path.join(all_image_name_file), 'rb') as f:
        images = [line.rstrip().decode('utf-8') for line in f.readlines()]

    # -------- Load the Caffe model --------
    net = caffe.Net(prototxt_file, weight_file, caffe.TEST)

    # -------- Configure pre-processing --------
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1, 3, image_width, image_height)

    # -------- Extract image features --------
    for k in range(len(images)):

        image_name = images[k].split("/")[1]
        image_path = os.path.join(image_folder, image_name)

        pre_processed_image = caffe.io.load_image(image_path)
        net.blobs['data'].data[...] = transformer.preprocess('data', pre_processed_image)
        out = net.forward()
        predicted_class = out[layer_name].argmax()
        predicted_class_list.append(image_name + "\t" + str(predicted_class))

        if k % 100 == 0:
            print('Completed : %d / %d' % (k, len(images)))

        activations = net.blobs[layer_name].data[0, :, :, :]

        output_file_path = os.path.join(save_folder_path, "activations", image_name.split('.')[0] + '.npz')
        np.savez_compressed(output_file_path, x=activations)

        # -------- Get gradients for the predicted class --------
        gradients, weights = get_gradients_and_weights(net, predicted_class, layer_name, final_layer, data_layer)

        output_file_path = os.path.join(save_folder_path, "grad_pred", image_name.split('.')[0] + '.npz')
        np.savez_compressed(output_file_path, x=gradients)


def create_final_features(visual_feature_file_name, feature_vector_size, image_name_file, no_sent_per_image, save_folder_path, is_testing=False):
    """ Create weighted image features
    :param visual_feature_file_name:
    :param feature_vector_size:
    :param image_name_file:
    :param no_sent_per_image:
    :param save_folder_path:
    :param is_testing:
    :return:
    """
    f = tables.open_file(visual_feature_file_name, mode='w')
    atom = tables.Float64Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0, feature_vector_size))

    with open(os.path.join(image_name_file), 'rb') as f:
        images = [line.rstrip().decode('utf-8') for line in f.readlines()]

    # -------- Load number of sentences per image --------
    num_for_each_image = np.load(no_sent_per_image, allow_pickle=True).item()

    # -------- Create image features --------
    for k in range(len(images)):

        image_name = images[k].split("/")[1]

        if k % 100 == 0:
            print('Completed : %d / %d' % (k, len(images)))

        activations = np.load(os.path.join(save_folder_path, "activations", image_name + '.npz'))['x']

        gradients = np.load(os.path.join(save_folder_path, "grad_pred", image_name + '.npz'))['x']

        weights = np.mean(gradients, axis=(1, 2))
        weights = np.maximum(weights, 0)

        # --------- Created weighted visual features --------
        for i, w in enumerate(weights):
            activations[i, :, :] = w * activations[i, :, :]

        # -------- Reshape --------
        feature_vector = activations.reshape(feature_vector_size)

        if not is_testing:
            # -------- Repeat visual feature for 'num' of times
            num = num_for_each_image[image_name]
            for i in range(num):
                array_c.append(np.expand_dims(feature_vector, axis=0))
        else:
            array_c.append(np.expand_dims(feature_vector, axis=0))


def main():
    feature_vector_size = params.FEATURE_VECTOR_SIZE
    save_folder_path = params.IMG_FEATURE_SAVE_FOLDER_PATH
    all_image_name_file = params.ALL_IMAGE_NAME_FILE
    prototxt_file = params.CLASSIFIER_PROTOTXT_FILE
    weight_file = params.CLASSIFIER_WEIGHT_FILE
    image_width = params.IMAGE_WIDTH
    image_height = params.IMAGE_HEIGHT
    image_folder = params.IMAGE_FOLDER
    layer_name = params.LAYER_NAME
    final_layer = params.FINAL_LAYER_NAME
    data_layer = params.DATA_LAYER_NAME

    extract_visual_feats_and_gradients(all_image_name_file, prototxt_file, weight_file, image_width, image_height, image_folder, layer_name, save_folder_path, final_layer, data_layer)

    # -------- For train images --------
    visual_feature_file_name = params.TRAIN_VISUAL_FEATURE_FILE
    image_name_file = params.TRAIN_IMAGE_NAMES
    no_sent_per_image = params.TRAIN_NO_SENT_PER_IMAGE
    create_final_features(visual_feature_file_name, feature_vector_size, image_name_file, no_sent_per_image, save_folder_path)

    # -------- For val images --------
    visual_feature_file_name = params.VAL_VISUAL_FEATURE_FILE
    image_name_file = params.VAL_IMAGE_NAMES
    no_sent_per_image = params.VAL_NO_SENT_PER_IMAGE
    create_final_features(visual_feature_file_name, feature_vector_size, image_name_file, no_sent_per_image, save_folder_path)

    # -------- For test images --------
    visual_feature_file_name = params.TEST_VISUAL_FEATURE_FILE
    image_name_file = params.TEST_IMAGE_NAMES
    no_sent_per_image = params.TEST_NO_SENT_PER_IMAGE
    create_final_features(visual_feature_file_name, feature_vector_size, image_name_file, no_sent_per_image, save_folder_path, is_testing=True)


if __name__ == '__main__':
    main()
