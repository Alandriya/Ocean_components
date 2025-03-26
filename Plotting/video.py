import os
import subprocess
import time

import matplotlib.colors as colors
import numpy as np


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """
    creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a matplotlib sequential colormap from [0, 1] segment to [minval, maxval]
    :param cmap: matplotlib colormap
    :param minval:
    :param maxval:
    :param n:
    :return:
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def create_video(files_path_prefix: str, tmp_dir: str, pic_prefix: str, name: str, speed: int = 20, start: int = 0):
    """
    Creates an .mp4 video from pictures in tmp subdirectory of full_prefix path with pic_prefix in filenames

    :param speed: coefficient of video speed, the more - the slower
    :param files_path_prefix: path to the working directory
    :param tmp_dir: directory with pictures
    :param pic_prefix: selects pictures with this prefix, e.g. A_ for A_00001.png, A_00002.png, ...
    :param name: short name of videofile to create
    :param start: start number of pictures
    :return:
    """
    video_name = files_path_prefix + f'videos/{name}.mp4'
    print(video_name)
    # print(files_path_prefix + f'videos/{flux_type}/tmp/%05d.png')
    # print(files_path_prefix + tmp_dir + f"{pic_prefix}%5d.png")
    # print(os.path.exists(files_path_prefix + tmp_dir))
    if os.path.exists(video_name):
        os.remove(video_name)
    subprocess.call([
        'ffmpeg', '-itsscale', str(speed), '-start_number', str(start), '-i', files_path_prefix + tmp_dir + f"{pic_prefix}%5d.png",
        '-r', '5', '-pix_fmt', 'yuv420p', video_name,
    ])
    time.sleep(5)
    # subprocess.call([
    #     './ffmpeg-7.0-amd64-static/ffmpeg', '-itsscale', str(speed), '-start_number', str(start), '-i', files_path_prefix + tmp_dir + f"{pic_prefix}%5d.png",
    #     '-r', '5', '-pix_fmt', 'yuv420p', video_name,
    # ])

    return
