import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf


def create_character_mapping():

    character_map = {' ': 0}

    for i in range(97, 123):
        character_map[chr(i)] = len(character_map)

    return character_map


def get_data_details(filename):

    result = {
        'max_input_length': 0,
        'max_label_length': 0,
        'num_samples': 0
    }

    # Get max lengths
    with open(filename, 'r') as metadata:
        metadata_reader = csv.DictReader(
            metadata, fieldnames=['filename', 'spec_length', 'labels_length', 'labels'])
        next(metadata_reader)
        for row in metadata_reader:
            if int(row['spec_length']) > result['max_input_length']:
                result['max_input_length'] = int(row['spec_length'])
            if int(row['labels_length']) > result['max_label_length']:
                result['max_label_length'] = int(row['labels_length'])
            result['num_samples'] += 1

    return result


def create_data_generator(directory, max_input_length, max_label_length, batch_size=64):

    while True:
        x, y, input_lengths, label_lengths = [], [], [], []

        metadata_csv = pd.read_csv(os.path.join(directory, 'metadata.csv'))
        # total_samples = len(metadata_csv)
        # num_to_select = (total_samples // batch_size) * batch_size
        rows = metadata_csv.sample(n=batch_size)
        # print('lenrows', len(rows))

        for _, row in rows.iterrows():
            # print(row)
            audio = np.load(os.path.join(directory, row['filename'] + '.npy'))
            x.append(audio)
            y.append([int(i) for i in row['labels'].split(' ')])
            input_lengths.append(row['spec_length'])
            label_lengths.append(row['labels_length'])

            # print(len(x))
            # if len(x) == batch_size:
        yield {
            'inputs': tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_input_length, padding='post'),
            'labels': tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_label_length, padding='post'),
            'input_lengths': np.asarray(input_lengths, dtype=np.int16),
            'label_lengths': np.asarray(label_lengths, dtype=np.int16)
        }, {
            'ctc': np.zeros([batch_size])
        }
        # x, y, input_lengths, label_lengths = [], [], [], []

        # print(metadata_csv, batch_size, total_samples, num_to_select)
        # print(rows)
        # exit()

        # with open(os.path.join(directory, 'metadata.csv'), 'r') as metadata:
        #     metadata_reader = csv.DictReader(
        #         metadata, fieldnames=['filename', 'spec_length', 'labels_length', 'labels'])
        #     next(metadata_reader)
        #     for row in metadata_reader:
        #         audio = np.load(os.path.join(
        #             directory, row['filename'] + '.npy'))
        #         x.append(audio)
        #         y.append([int(i) for i in row['labels'].split(' ')])
        #         input_lengths.append(row['spec_length'])
        #         label_lengths.append(row['labels_length'])

        #         if len(x) == batch_size:
        #             yield {
        #                 'inputs': tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_input_length, padding='post'),
        #                 'labels': tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_label_length, padding='post'),
        #                 'input_lengths': np.asarray(input_lengths, dtype=np.int16),
        #                 'label_lengths': np.asarray(label_lengths, dtype=np.int16)
        #             }, {
        #                 'ctc': np.zeros([batch_size])
        #             }
        #             x, y, input_lengths, label_lengths = [], [], [], []
