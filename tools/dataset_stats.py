import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
emo_dict = {0: 'surprise',
            1: 'fear',
            2: 'disgust',
            3: 'happiness',
            4: 'sadness',
            5: 'anger',
            6: 'neutral'}

raf_path = r'D:\datasets\RafDB'
base_dataset = pd.read_csv(os.path.join(raf_path, 'list_patition_label.txt'),
                           sep=' ', header=None,
                           names=['img', 'label'])
add_align = lambda x: str(x).split('.')[0] + '_aligned.jpg'
base_dataset['img'] = base_dataset['img'].apply(add_align)
base_dataset['label'] = base_dataset['label'] - 1
train = base_dataset[base_dataset['img'].str.startswith('train')]
print(f'Number of train images: {len(train)}')
test = base_dataset[base_dataset['img'].str.startswith('test')]
print(f'Number of test images: {len(test)}')
emotions = list(emo_dict.values())

labels_train = train.groupby('label').count().to_dict()['img']
emo_count = list(labels_train.values())
plt.figure()
plt.bar(emotions, emo_count, width=0.4, align='edge', label='RAF-DB')
fer_train_root_path = Path(r'D:\datasets\fer2013_superresolution\train')
emotion_dirs_train = fer_train_root_path.glob('**/*')
emo2idx = {emo_dict[key]: key for key in emo_dict.keys()}
for emotion_dir in emotion_dirs_train:
    if emotion_dir.is_dir():
        if emotion_dir.name == 'contempt':
            continue
        emo_idx = emo2idx[emotion_dir.name]
        images = list(emotion_dir.glob('*.png'))
        nr_of_samples = len(images)
        labels_train[emo_idx] += nr_of_samples

emo_count = list(labels_train.values())
print(f'RAF-DB + FER2013 nr of train samples: {sum(emo_count)}')
# plt.figure()
plt.bar(emotions, emo_count, width=-0.4, align='edge', label='RAF-DB+FER2013')
plt.xticks(np.arange(len(emo_count)), rotation=0)
plt.legend()
plt.grid("on")
plt.title('Train dataset emotion distribution')
plt.show()




