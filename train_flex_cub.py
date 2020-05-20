import os
from flex import FLEX
import numpy as np
import pickle
import params_cub as params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_flex():
    """
    Train flex model using parameters specified in 'param_cub.py' file
    :return:
    """
    data_dictionary_name = params.DATA_DIC_NAME
    train_relevance_scores_file = params.DECISION_RELEVANCE_SCORE_TRAIN_EXPANDED
    val_relevance_scores_file = params.DECISION_RELEVANCE_SCORE_VAL_EXPANDED
    learning_rate = params.LEARNING_RATE
    num_epochs = params.NUM_EPOCHS
    train_visual_feat_file = params.TRAIN_VISUAL_FEATURE_FILE
    val_visual_feat_file = params.VAL_VISUAL_FEATURE_FILE
    batch_size = params.BATCH_SIZE
    keep_prob = params.KEEP_PROB
    model_save_folder = os.path.join(params.MODEL_SAVE_FOLDER, params.MODEL_VERSION)
    os.makedirs(model_save_folder, exist_ok=True)
    model_path = os.path.join(model_save_folder, 'flex')

    # -------- Load Data ---------
    data_dict = pickle.load(open(data_dictionary_name, 'rb'))

    vocab_size = data_dict['num_words']
    max_length = data_dict['max_length']

    train_relevance_scores = np.load(train_relevance_scores_file)
    val_relevance_scores = np.load(val_relevance_scores_file)

    x_sentence_train = np.array(data_dict['x_train_word_sequences'], dtype=np.int32)
    x_sentence_val = np.array(data_dict['x_val_word_sequences'], dtype=np.int32)

    y_sentence_train = np.array(data_dict['y_train_word_sequences'], dtype=np.int32)
    y_sentence_val = np.array(data_dict['y_val_word_sequences'], dtype=np.int32)

    lengths_train = np.array(data_dict['train_lengths'], dtype=np.int32)
    lengths_val = np.array(data_dict['val_lengths'], dtype=np.int32)

    print('Done loading data')

    num_train = x_sentence_train.shape[0]
    num_validation = x_sentence_val.shape[0]

    print("Number of train instances:", num_train)
    print("Number of val instances:", num_validation, "\n")

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
    print(model_params)
    print("Done Initializing FLEX")

    logs = {
        'epoch': [],
        'val_loss': [],
        'train_loss': [],
    }

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print("\n------Epoch %d------" % (epoch + 1))

        train_loss = flex_model.train_epoch(visual_feat_file=train_visual_feat_file,
                                            x_word_seq=x_sentence_train,
                                            y_word_seq=y_sentence_train,
                                            lengths=lengths_train,
                                            words_relevance_scores=train_relevance_scores,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size,
                                            keep_prob=keep_prob)

        val_loss = flex_model.validate_epoch(visual_feat_file=val_visual_feat_file,
                                             x_word_seq=x_sentence_val,
                                             y_word_seq=y_sentence_val,
                                             lengths=lengths_val,
                                             words_relevance_scores=val_relevance_scores,
                                             batch_size=batch_size,
                                             keep_prob=1)

        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        logs['epoch'].append(epoch)

        print("Train loss: %.4f" % train_loss)
        print("Validation loss: %.4f" % val_loss)

        if val_loss < best_val_loss:
            print("Saving model")
            flex_model.save(model_path)
            best_val_loss = val_loss
        else:
            learning_rate /= 2  # anneal learning rate
            pickle.dump(logs, open(os.path.join(model_save_folder, 'logs.pkl'), 'wb'))

    print("Writing Metrics to file...")
    pickle.dump(logs, open(os.path.join(model_path, 'logs.pkl'), 'wb'))


if __name__ == '__main__':
    train_flex()
