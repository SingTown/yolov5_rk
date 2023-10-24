import os
import requests
import random
from multiprocessing import Pool

host = os.getenv('HOST')
trainer_username = os.getenv('TRAINER_USERNAME')
trainer_key = os.getenv('TRAINER_KEY')

tid = '-1'

def start(id:str):
    global tid
    tid = id,
    response = requests.post(
        host+'trainer/start/', 
        data = {
            'id': tid,
            'trainer_username' : trainer_username,
            'trainer_key' : trainer_key,
        }
    ).json()

    if not response['succeed']:
        raise RuntimeError(response['message'])
    print("SingTown AI trainer start", id)

def failed():
    requests.post(
        host+'trainer/failed/',
        data = {
            'id': tid,
            'trainer_username' : trainer_username,
            'trainer_key' : trainer_key,
        }
    )
    print("SingTown AI trainer failed")

def succeed():
    requests.post(
        host+'trainer/succeed/', 
        data = {
            'id': tid,
            'trainer_username' : trainer_username,
            'trainer_key' : trainer_key,
        }
    )
    print("SingTown AI trainer succeed")

def dataset():
    response = requests.post(
        host+'trainer/dataset/', 
        data = {
            'id': tid,
            'trainer_username' : trainer_username,
            'trainer_key' : trainer_key,
        }
    ).json()
    if not response['succeed']:
        raise RuntimeError(response['message'])
    print("SingTown AI trainer dataset")
    return response['value']

def result(file:str):
    response = requests.post(
        host+'trainer/result/', 
        files = {'file': open(file, 'rb')},
        data = {
            'id': tid,
            'trainer_username' : trainer_username,
            'trainer_key' : trainer_key,
        }
    ).json()
    if not response['succeed']:
        raise RuntimeError(response['message'])
    print("SingTown AI trainer result")
    return response['value']


def log(file:str):
    response = requests.post(
        host+'trainer/log/', 
        files = {'file': open(file, 'rt')},
        data = {
            'id': tid,
            'trainer_username' : trainer_username,
            'trainer_key' : trainer_key,
        }
    ).json()
    if not response['succeed']:
        raise RuntimeError(response['message'])
    print("SingTown AI trainer log")
    return response['value']

def validate(file:str):
    response = requests.post(
        host+'trainer/validate/', 
        files = {'file': file},
        data = {
            'id': tid,
            'trainer_username' : trainer_username,
            'trainer_key' : trainer_key,
        }
    ).json()
    if not response['succeed']:
        raise RuntimeError(response['message'])
    print("SingTown AI trainer validate")
    return response['value']

def message(message:str):
    response = requests.post(
        host+'trainer/message/', 
        data = {
            'id': tid,
            'message': message,
            'trainer_username' : trainer_username,
            'trainer_key' : trainer_key,
        }
    ).json()
    if not response['succeed']:
        raise RuntimeError(response['message'])
    print("SingTown AI trainer message", message)
    return response['value']

def _download_img(url:str, name:str, path:str):
    if not os.path.exists(path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, "wb") as file:
                file.write(response.content)
            print(f'download {name}')
        else:
            print(f'Warning: failed download {url}')

def download(annotations:list, path:str):
    os.makedirs(path, exist_ok=True)

    with Pool(processes=4) as pool:
        for anno in annotations:
            img_path = os.path.join(path, anno['name'])
            pool.apply_async(_download_img, args=(anno['url'], anno['name'], img_path))
        pool.close()
        pool.join()

def convert_yolo(dataset:dict, path:str):
    labels = dataset['train']['labels']
    with open(os.path.join(path, "labels.txt"), 'wt') as f:
        f.write("\n".join(labels))

    for img in dataset['annotations']:
        anno = img['anno']
        if not anno:
            continue
        if not isinstance(anno, list):
            continue
            
        txt = ''
        for b in anno:
            idx = labels.index(b['class'])
            cx = b['x'] + b['w']/2
            cy = b['y'] + b['h']/2
            w = b['w']
            h = b['h']
            txt += ("%d %f %f %f %f\n" % (idx, cx, cy, w, h))
        
        name = os.path.splitext(img['name'])[0]
        with open(os.path.join(path, name+".txt"), 'wt') as f:
            f.write(txt)

'''
Create class folder:
train
 - cat
  - cat1.jpg
  - cat2.jpg
 - dog
   - dog1.jpg
   - dog2.jpg
val
 - cat
  - cat3.jpg
  - cat4.jpg
 - dog
   - dog3.jpg
   - dog4.jpg
'''
def convert_classfolder(dataset:dict, dataset_path:str, path:str):
    labels = dataset['train']['labels']
    os.mkdir(os.path.join(path, 'train'))
    os.mkdir(os.path.join(path, 'val'))
    for l in labels:
        os.mkdir(os.path.join(path, 'train', l))
        os.mkdir(os.path.join(path, 'val', l))
    
    random.shuffle(dataset['annotations'])
    validation_num = int(len(dataset['annotations'])*0.1)

    validation = dataset['annotations'][0:validation_num]
    training = dataset['annotations'][validation_num:]

    for anno in training:
        os.symlink(os.path.join(dataset_path, anno['name']), os.path.join(path, 'train', anno['anno'], anno['name']))

    for anno in validation:
        os.symlink(os.path.join(dataset_path, anno['name']), os.path.join(path, 'val', anno['anno'], anno['name']))

    return training, validation