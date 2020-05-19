import pandas as pd


def pre_pro_build_word_vocab(sentence_iterator, word_count_threshold=1):
    """ Pre -process and build vocab, word_to_id and id_to_word dictionaries
    function from Andre Karpathy's NeuralTalk
    :param sentence_iterator:
    :param word_count_threshold:
    :return:
    """
    print('Pre-processing %d word vocab' % (word_count_threshold,))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    vocab.sort()
    print('Preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<s>'
    ixtoword[nsents] = '.'
    wordtoix = {}
    wordtoix['<s>'] = 0
    word_counts['.'] = nsents
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    vocab = ['<s>'] + vocab
    return wordtoix, ixtoword, vocab


def get_captions(annotation_path):
    with open(annotation_path) as f:
        annotations = pd.read_table(f, sep='#', header=None, names=['image', 'caption'])
    return annotations['caption'].values, annotations['image'].values
