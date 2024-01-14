import os
import argparse

import saka_message_saver


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
parser.add_argument('--filename_base', '-f', type=str, default='image')

args = parser.parse_args()

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
else:
    saver = saka_message_saver.SakaMessageSaver(directory=directory,
                                                filename_base=args.filename_base,
                                                params=params)
saver.run()

if args.save_params:
    params.save_to_yaml(args.params_file)
    print(f'save params to {args.params_file}')
