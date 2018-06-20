#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Translates a source file using a translation model that fill future input from the same sentence with past predictions
(i.e. it eliminates teacher forcing)
Because our task consists of joint lemmatization and pos tagging, we want to use pos tags (without lemmas) we have
predicted previously within the same sentence.
'''
__author__ = "Bogomil Gospodinov"
__email__ = "s1312650@sms.ed.ac.uk"
__status__ = "dev"

if __name__ == "__main__":
    import logging
    import argparse
    import time
    import numpy as np

    from util import load_config, seq2words, prepare_data
    from compat import fill_options
    from settings import TranslationSettings
    from nmt import create_model, load_dictionaries
    import inference
    import tensorflow as tf

    # parse console arguments
    settings = TranslationSettings(from_console_arguments=True)
    input_file = settings.input
    output_file = settings.output

    # start logging
    level = logging.DEBUG if settings.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')

    sess = tf.Session()

    options = []
    for model in settings.models:
        config = load_config(model)
        # backward compatibility
        fill_options(config)
        config['reload'] = model
        options.append(argparse.Namespace(**config))

    config = options[0]
    _, _, _, num_to_target = load_dictionaries(config)

    sentences = input_file.readlines()

    logging.info('Translating {0} segments...\n'.format(len(sentences)))
    start_time = time.time()

    models = []

    for i, opts in enumerate(options):
        with tf.name_scope("model%d" % i) as scope:
            model, saver = create_model(opts, sess, ensemble_scope=scope)
            models.append(model)

    logging.info("NOTE: Length of translations is capped to {}".format(config.translation_maxlen))

    source_to_num, _, _, _ = load_dictionaries(config)
    lines = []
    for sent in sentences:
        line = []
        for w in sent.strip().split():
            w = [source_to_num[0][w] if w in source_to_num[0] else 1]
            line.append(w)
        lines.append(line)
    lines = np.array(lines)

    sentences = lines
    n_sent = len(sentences)

    for sentence in sentences:
        y_dummy = np.zeros(shape=(len(sentence), 1))
        x, x_mask, _, _ = prepare_data(np.expand_dims(np.array(sentence), axis=0), y_dummy, maxlen=None)
        hypotheses = inference.beam_search(models, sess, x, x_mask, settings.beam_width)[0]
        if settings.normalization_alpha:
            hypotheses = map(lambda sent_cost: (sent_cost[0], sent_cost[1] / len(sent_cost[0]) ** settings.normalization_alpha), hypotheses)
        hypotheses = sorted(hypotheses, key=lambda sent_cost: sent_cost[1])
        translation = seq2words(hypotheses[0][0], num_to_target, join=False)
        output_file.write(" ".join(translation) + "\n")
        output_file.flush()

    duration = time.time() - start_time
    logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(n_sent, duration, n_sent / duration))
    input_file.close()
    output_file.close()
    logging.info('Done')
