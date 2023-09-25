# path alian
TRAIN_JSON_PATH = '/usr/src/app/project/train.json'
IMAGE_LIST_PATH = '/usr/src/app/project/list.txt'
DATASET_PATH = '/usr/src/dataset/'
PROJECT_PATH = '/usr/src/app/project/'
IMAGES_PATH = '/usr/src/app/project/images/'
LABELS_PATH = '/usr/src/app/project/labels/'
LABELS_ZIP_PATH = '/usr/src/app/project/labels.zip'
LABELS_TXT_PATH = '/usr/src/app/project/labels/labels.txt'

ANCHORS_PATH = '/usr/src/app/RK_anchors.txt'
RUNS_PATH = '/usr/src/app/runs/'
RESULTS_CSV_PATH = '/usr/src/app/runs/train/exp/results.csv'
PT_PATH = '/usr/src/app/runs/train/exp/weights/best.pt'
ONNX_PATH = '/usr/src/app/runs/train/exp/weights/best.onnx'
RKNN_PATH = '/usr/src/app/runs/train/exp/weights/best.rknn'

YOLO_TAR_PATH = '/usr/src/app/project/yolov5.tar'

host = 'https://ai.singtown.com/'

try:

    import logging
    # logging.basicConfig(filename='log.txt', encoding='utf-8', level=logging.INFO, 
    #     format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    import os, sys, time, shutil, json
    import requests
    from datetime import datetime
    from tqdm import tqdm


    import oss2
    auth = oss2.Auth(os.getenv('OSS_ACCESS_KEY_ID'), os.getenv('OSS_ACCESS_KEY_SECRET'))
    bucket = oss2.Bucket(auth, os.getenv('OSS_ENDPOINT'), os.getenv('OSS_BUCKET_NAME'))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", help="Project id on host")
    args = parser.parse_args()
    pid = args.pid

    base_auth = requests.auth.HTTPBasicAuth(os.getenv('USERNAME'), os.getenv('PASSWORD'))
    def post_failed():
        requests.get(host+f'project/{pid}/train/finish/?status=FAILED', auth=base_auth)

    def post_succeed():
        requests.get(host+f'project/{pid}/train/finish/?status=SUCCEED', auth=base_auth)

    def ensure_empty(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    # download project
    ensure_empty(PROJECT_PATH)
    for f in bucket.list_objects('project/'+pid).object_list:
        bucket.get_object_to_file(f.key, f.key.replace('project/'+pid, PROJECT_PATH))

    with open(TRAIN_JSON_PATH) as f:
        train_info = json.load(f)

    if train_info['device'] not in ['rv1103', 'rv1106'] or train_info['model'] not in ['yolov5s640']:
        post_failed()

    with open(IMAGE_LIST_PATH) as f:
        img_list = [l for l in f.read().splitlines()]

    # download images to dataset folder
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    def download_image(f):
        img_path = os.path.join(DATASET_PATH, f)
        if not os.path.exists(img_path):
            bucket.get_object_to_file('dataset/'+f, img_path)

    from multiprocessing import Pool
    from tqdm import tqdm
    pbar = tqdm(total=len(img_list))
    def update(*a):
        pbar.update()
    pool = Pool(8)
    for f in img_list:
        pool.apply_async(download_image, (f,), callback=update)
    pool.close()
    pool.join()


    # prepare dataset format
    ensure_empty(IMAGES_PATH)
    for f in img_list:
        os.symlink(os.path.join(DATASET_PATH, f), os.path.join(IMAGES_PATH, f))

    from utils.dataloaders import autosplit
    autosplit(IMAGES_PATH)

    import zipfile
    with zipfile.ZipFile(LABELS_ZIP_PATH,"r") as zip_ref:
        zip_ref.extractall(LABELS_PATH)
    with open(LABELS_TXT_PATH) as f:
        labels = [l for l in f.read().splitlines()]

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
                bucket.put_object_from_file(f'project/{pid}/results.csv', event.src_path)

    ensure_empty('runs')
    observer = Observer()
    event_handler = FileEventHandler()
    observer.schedule(event_handler, RUNS_PATH, True)
    observer.start()


    cmd = f"python3 train.py --data dataset.yaml \
            --weights /usr/src/app/weights/{train_info['model']}_{train_info['weights']}.pt --img 640 \
            --epochs {train_info['epochs']} --batch-size 32 --patience 5"

    if train_info["freeze_backbone"]:
        cmd += " --freeze 10"

    ret = os.system(cmd)
    if ret != 0:
        post_failed()
        exit(-1)

    observer.stop()


    # export

    width = 640

    if train_info['device'] == 'rv1103':
        height = 480
    elif train_info['device'] == 'rv1106':
        height = 384

    ret = os.system(f"python3 export.py --rknpu rv1103 --img {height} {width} --weights {PT_PATH} --include onnx")
    if ret != 0:
        post_failed()
        exit(-1)

    # convert rknn

    from rknn.api import RKNN
    rknn = RKNN()

    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rv1106")
    ret = rknn.load_onnx(ONNX_PATH)
    if ret != 0:
        post_failed()
        exit(-1)

    ret = rknn.build(do_quantization=True, dataset=os.path.join(PROJECT_PATH, 'autosplit_val.txt'))
    if ret != 0:
        post_failed()
        exit(-1)

    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        post_failed()
        exit(-1)

    rknn.release()


    # package and upload

    import tarfile
    with tarfile.open(YOLO_TAR_PATH, "w") as tar:
        tar.add(RKNN_PATH, 'yolov5.rknn')
        tar.add(ANCHORS_PATH, 'anchors.txt')
        tar.add(LABELS_TXT_PATH, 'labels.txt')

    bucket.put_object_from_file(f'project/{pid}/results.zip', YOLO_TAR_PATH)
    logging.info(f'p:{pid}, succeed')
    post_succeed()

except:
    post_failed()
    raise