# path alian
DATASET_PATH = '/usr/src/dataset/'
PROJECT_PATH = '/usr/src/app/project/'
IMAGES_PATH = '/usr/src/app/project/images/'
LABELS_PATH = '/usr/src/app/project/labels/'
LABELS_TXT_PATH = '/usr/src/app/project/labels/labels.txt'

ANCHORS_PATH = '/usr/src/app/RK_anchors.txt'
RUNS_PATH = '/usr/src/app/runs/'
RESULTS_CSV_PATH = '/usr/src/app/runs/train/exp/results.csv'
PT_PATH = '/usr/src/app/runs/train/exp/weights/best.pt'
ONNX_PATH = '/usr/src/app/runs/train/exp/weights/best.onnx'
RKNN_PATH = '/usr/src/app/runs/train/exp/weights/best.rknn'

YOLO_TAR_PATH = '/usr/src/app/project/yolov5.tar'

import singtownai
try:

    import logging
    # logging.basicConfig(filename='log.txt', encoding='utf-8', level=logging.INFO, 
    #     format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    import os, sys, time, shutil, json
    import requests
    from datetime import datetime

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tid", help="Train id on host")
    args = parser.parse_args()
    tid = args.tid
    if tid == '-1':
        event = json.loads(os.getenv('FC_CUSTOM_CONTAINER_EVENT'))
        tid = event["tid"]

    singtownai.start(tid)
    singtownai.message("Prepare")
    def ensure_empty(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    dataset = singtownai.dataset()
    train_info = dataset['train']
    labels = dataset['train']['labels']
    singtownai.download(dataset['annotations'], DATASET_PATH)

    # prepare dataset format
    ensure_empty(PROJECT_PATH)
    os.mkdir(IMAGES_PATH)
    os.mkdir(LABELS_PATH)
    singtownai.convert_yolo(dataset, LABELS_PATH)

    for anno in dataset['annotations']:
        os.symlink(os.path.join(DATASET_PATH, anno['name']), os.path.join(IMAGES_PATH, anno['name']))

    from utils.dataloaders import autosplit
    autosplit(IMAGES_PATH)

    import yaml
    with open('data/dataset.yaml', 'w') as f:
        f.write(yaml.dump({
            'path': PROJECT_PATH,
            'train': './autosplit_train.txt',
            'val': './autosplit_val.txt',
            'nc': len(labels),
            'names': dict(enumerate(labels)),
        }))


    # train
    from watchdog.observers import Observer
    from watchdog.events import *

    class FileEventHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.src_path in [RESULTS_CSV_PATH]:
                singtownai.log(event.src_path)

    ensure_empty('runs')
    observer = Observer()
    event_handler = FileEventHandler()
    observer.schedule(event_handler, RUNS_PATH, True)
    observer.start()

    singtownai.message("Training")

    cmd = f"python3 train.py --data dataset.yaml \
            --weights /usr/src/app/weights/{train_info['model']}_{train_info['weights']}.pt --img 640 \
            --epochs {train_info['epochs']} --batch-size 32" #--patience 5

    if train_info["freeze_backbone"]:
        cmd += " --freeze 10"

    ret = os.system(cmd)
    if ret != 0:
        singtownai.failed()
        exit(-1)

    observer.stop()


    # export
    singtownai.message("Exporting")

    width = 640

    if train_info['cpu'] == 'rv1103':
        height = 480
    elif train_info['cpu'] == 'rv1106':
        height = 384

    ret = os.system(f"python3 export.py --rknpu rv1103 --img {height} {width} --weights {PT_PATH} --include onnx")
    if ret != 0:
        singtownai.failed()
        exit(-1)

    # convert rknn
    singtownai.message("Converting")

    from rknn.api import RKNN
    rknn = RKNN()

    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rv1106")
    ret = rknn.load_onnx(ONNX_PATH)
    if ret != 0:
        singtownai.failed()
        exit(-1)

    ret = rknn.build(do_quantization=True, dataset=os.path.join(PROJECT_PATH, 'autosplit_val.txt'))
    if ret != 0:
        singtownai.failed()
        exit(-1)

    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        singtownai.failed()
        exit(-1)

    rknn.release()

    # package and upload
    singtownai.message("Uploading")
    import tarfile
    with tarfile.open(YOLO_TAR_PATH, "w") as tar:
        tar.add(RKNN_PATH, 'yolov5.rknn')
        tar.add(ANCHORS_PATH, 'anchors.txt')
        tar.add(LABELS_TXT_PATH, 'labels.txt')

    singtownai.result(YOLO_TAR_PATH)
    logging.info(f'p:{tid}, succeed')
    singtownai.message("Finished")
    singtownai.succeed()

except:
    singtownai.failed()
    raise