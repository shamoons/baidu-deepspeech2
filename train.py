import wandb
from model import SpeechModel
import config
import os
import utils
import math


if __name__ == '__main__':

    wandb.init(project="baidu-deepspeech2")

    character_mapping = utils.create_character_mapping()
    data_details = utils.get_data_details(filename=os.path.join(
        config.preprocessing['data_dir'], 'metadata.csv'))

    print('data_details', data_details)

    config.training['steps_per_epoch'] = math.floor(
        data_details['num_samples'] / config.training['batch_size'])
    config.model['max_input_length'] = data_details['max_input_length']
    config.model['max_label_length'] = data_details['max_label_length']
    config.model['vocab_size'] = len(character_mapping)

    data_generator = utils.create_data_generator(directory=config.preprocessing['data_dir'],
                                                 max_input_length=config.model['max_input_length'],
                                                 max_label_length=config.model['max_label_length'],
                                                 batch_size=config.training['batch_size'])

    model = SpeechModel(hparams=config.model)

    model.train_generator(data_generator, config.training)
