import os
import argparse

import saka_message_saver
from saka_message_saver import logger

from logging import FileHandler


def get_parsed_args():
    parser = argparse.ArgumentParser()
    # params file --params_file or -p
    parser.add_argument('--params_file', '-p', type=str, default=os.path.join(
        saka_message_saver.PROJECT_ROOT_PATH, 'config', 'params.yaml'))
    # save params --save_params or -s
    parser.add_argument('--save_params', '-s', action='store_true')
    # load params --load_params or -l
    parser.add_argument('--load_params', '-l', action='store_true')
    # photo mode --photo
    parser.add_argument('--photo', action='store_true')
    # screen shot save directory --directory or -d
    parser.add_argument('--directory', '-d', type=str,
                        default=os.path.join(saka_message_saver.PROJECT_ROOT_PATH, 'images'))
    # screen shot save sub directory --sub_directory
    parser.add_argument('--sub_directory', type=str, default='test')
    # screen shot save filename base --filename_base or -f
    parser.add_argument('--filename_base', '-f', type=str, default='')

    # reference data setting --reference_data or -rf
    parser.add_argument('--reference_data', '-rf', action='store_true')

    return parser.parse_args()


def main():
    args = get_parsed_args()

    params = saka_message_saver.Parameters()
    if args.load_params:
        if not os.path.exists(args.params_file):
            raise FileNotFoundError(f'{args.params_file} is not found.')

        params.load_from_yaml(args.params_file)

    directory = os.path.join(args.directory, args.sub_directory,
                             saka_message_saver.ImageSaver.get_datetime())

    if args.photo:
        saver = saka_message_saver.SakaMessagePhotoSaver(directory=directory,
                                                         filename_base=args.filename_base,
                                                         params=params)
    elif args.reference_data:
        pass
    else:
        saver = saka_message_saver.SakaMessageSaver(directory=directory,
                                                    filename_base=args.filename_base,
                                                    params=params)
    saver.run()

    if args.save_params:
        saver.params.save_to_yaml(args.params_file)
        logger.info(f'save params to {args.params_file}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(e)
        raise e
