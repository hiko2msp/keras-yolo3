#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
前提条件:
    label_me_jsonにjsonファイルを配置する

使い方:
    python labelme_annotation.py\
        -i <input_dir_path_of_json>\
        -o <output_path_of_image>\
        -t <annotation_data_for_train>\
        -c <class_label_path>

実行後:
    data/imageに画像ファイルが配置される
    train.txtにアノテーションのデータが記録される

'''
import argparse
import os
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from glob import iglob
import traceback

# Args
argparser = argparse.ArgumentParser(
    description="Convert Label me json to YOLO dataset.")

argparser.add_argument(
    '-i',
    '--input_path',
    help="path to directory that contains label-me-formed json data(.json)",
    default='label_me_json')

argparser.add_argument(
    '-o',
    '--output_path',
    help='output_image_path',
    default='data/image')

argparser.add_argument(
    '-t',
    '--output_train_data_path',
    help="path to train data path",
    default='train.txt')

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('model_data', 'coco_classes.txt'))

def get_box(points):
    points = np.asarray(points)
    x_min = np.floor(np.min(points[:, 0]))
    y_min = np.floor(np.min(points[:, 1]))
    x_max = np.floor(np.max(points[:, 0]))
    y_max = np.floor(np.max(points[:, 1]))
    return list(map(int, [x_min, y_min, x_max, y_max]))

def parse_annotation_json(annotation_json_path, class_names):
    with open(annotation_json_path) as f:
        data = json.load(f)

    pil_img = Image.open(BytesIO(base64.b64decode(data['imageData'])))
    boxes = [get_box(shape['points']) for shape in data['shapes']]
    classes = [class_names.index(shape['label'].replace(' ', '')) for shape in data['shapes']]

    return boxes, classes, pil_img

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def ensure_path(path):
    os.makedirs(path, exist_ok=True)

def main():
    args = argparser.parse_args()
    annotation_dir_path = args.input_path
    class_path = args.classes_path
    output_image_dir = args.output_path
    train_data_path = args.output_train_data_path

    class_names = get_classes(class_path)
    ensure_path(output_image_dir)

    img_box_class_list = []
    for i, annotation_json_path in enumerate(iglob(annotation_dir_path+'/*')):
        try:
            filename = annotation_json_path.split('/')[-1]
            boxes, classes, pil_img = parse_annotation_json(annotation_json_path, class_names)
            img_file_path = os.path.join(output_image_dir, filename + '.png')
            print(img_file_path)
            pil_img.save(img_file_path)
            img_box_class_list.append((img_file_path, boxes, classes))
        except:
            print(traceback.format_exc())
            continue


    for img_file_path, boxes, classes in img_box_class_list:
        if len(boxes) == 0:
            continue
        try:
            box_classes = [','.join(map(str, [*box, class_id])) for box, class_id in zip(boxes, classes)]
            line = f'{img_file_path} {" ".join(box_classes)}\n'
            with open(train_data_path, 'w') as f:
                f.write(line)
        except:
            print(traceback.format_exc())
            continue

if __name__ == '__main__':
    main()
