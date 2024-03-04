import os
import argparse
import shutil

import saka_message_saver
from saka_message_saver import logger

from logging import FileHandler


def get_parsed_args():
    parser = argparse.ArgumentParser()
    # params directory --params_directory
    parser.add_argument('--params_directory', type=str, default=os.path.join(
        saka_message_saver.PROJECT_ROOT_PATH, 'config'))
    # params filename --params_filename
    parser.add_argument('--params_filename', type=str, default='params.yaml')
    # save params --save_params or -s
    parser.add_argument('--save_params', '-s', action='store_true')
    # load params --load_params or -l
    parser.add_argument('--load_params', '-l', action='store_true')
    # photo mode --photo
    parser.add_argument('--photo', action='store_true')
    # movie mode --movie
    parser.add_argument('--movie', action='store_true')
    # setting --setting
    parser.add_argument('--setting', action='store_true')

    # screen shot save directory --directory or -d
    parser.add_argument('--directory', '-d', type=str,
                        default=os.path.join(saka_message_saver.PROJECT_ROOT_PATH, 'images'))
    # screen shot save sub directory --sub_directory
    parser.add_argument('--sub_directory', type=str, default='test')
    # screen shot save filename base --filename_base or -f
    parser.add_argument('--filename_base', '-f', type=str, default='')

    # test times --test_times
    parser.add_argument('--loop_times', type=int, default=-1)
    # reverse --reverse
    parser.add_argument('--reverse', action='store_true')

    return parser.parse_args()


def main():
    args = get_parsed_args()

    params_file_path = os.path.join(
        args.params_directory, args.params_filename)
    if args.load_params:
        logger.info(f'load params from {params_file_path}')
        if not os.path.exists(params_file_path):
            raise FileNotFoundError(f'{params_file_path} is not found.')

        params = saka_message_saver.Parameters.load_from_yaml(params_file_path)
    else:
        logger.info('use default params')
        params = saka_message_saver.Parameters(
            directory=args.directory,
            sub_directory=args.sub_directory,
            filename_base=args.filename_base,
            loop_times=args.loop_times,
            reverse=args.reverse
        )

    params.directory = os.path.join(
        params.base_directory, params.sub_directory,
        saka_message_saver.ImageSaver.get_datetime()
    )

    if args.photo and args.movie:
        raise ValueError(
            'photo and movie cannot be specified at the same time.')

    logger.info(f'save to "{params.directory}"')

    if args.photo:
        logger.info('photo mode')
        saver = saka_message_saver.SakaMessagePhotoSaver(params=params)
    elif args.movie:
        logger.info('movie mode')
        saver = saka_message_saver.SakaMessageMovieSaver(params=params)
    elif args.setting:
        logger.info('setting mode')
        saver = saka_message_saver.SettingUI(params=params)
    else:
        logger.info('message mode')
        saver = saka_message_saver.SakaMessageSaver(params=params)
    saver.run()

    if args.save_params:
        saver.params.save_to_yaml(params_file_path)
        logger.info(f'save params to {params_file_path}')

    # copy config directory to save directory
    shutil.copytree(args.params_directory, os.path.join(
        params.directory, 'config'))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(e)
        raise e
