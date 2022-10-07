import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms
import math
import numpy as np

from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import load_rgb_frames_from_video, video_to_tensor
from decoder import get_gloss
import dask

NUM_CLASSES = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = './checkpoints/nslt_2000_065846_0.447803.pt'
test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
VIDEO_ROOT = "./asl_video/"


def pad(imgs, total_frames=64):
    if imgs.shape[0] < total_frames:
        num_padding = total_frames - imgs.shape[0]

        if num_padding:
            prob = np.random.random_sample()
            if prob > 0.5:
                pad_img = imgs[0]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
            else:
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
    else:
        padded_imgs = imgs

    return padded_imgs


def predict_word_from_sign(index, tmp_img):

    transcription = {index:''}

    print('Paddding images.....')
    tmp_img = pad(tmp_img)
    print(len(tmp_img))
    print('Images padded...')

    # Run through the data augmentation
    # 64 x 224 x 224 x 3
    tmp_img = test_transforms(tmp_img)
    ret_img = video_to_tensor(tmp_img)
    inputs = ret_img[np.newaxis, ...]
    print(inputs.shape)

# ----------------* END VIDEO PROCESSING *-----------------


    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    i3d.replace_logits(NUM_CLASSES)
    i3d.load_state_dict(torch.load(
        weights,
        map_location=device))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    per_frame_logits = i3d(inputs)
    print(per_frame_logits[0].shape)
    ## 1 x num_classes
    predictions = torch.max(per_frame_logits, dim=2)[0]
    # predictions[0] --> num_classes tensor
    # lowest as the first element - highest as the last element
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    out_probs = np.sort(predictions.cpu().detach().numpy()[0])

    gloss = get_gloss(out_labels[-1])
    print("gloss predicted: ", gloss)
    print(out_probs)

    z = 1/(1 + np.exp(-out_probs))
    print(z)

    if z[-1] >= 0.0000005:
        transcription[index] = gloss
    
    return transcription

def batch(seq):
    sub_results = []
    for x in seq:
        sub_results.append(predict_word_from_sign(x['index'], x['img']))
    return sub_results

def get_final_phrase(index, word_list):
    word_dict = {}
    for word in word_list:
        for i in word:
            word_dict.update(i)
    
    final_string = []
    for i in range(index):
        if len(final_string)==0:
            final_string.append(word_dict.get(i, ''))
        elif final_string[-1]!=word_dict.get(i, ''):
            final_string.append(word_dict.get(i, ''))

    return ' '.join(final_string)

def video_to_text(vid="combined"):

    # ----------------* VIDEO PROCESSING *-------------------
    print('Loading images.....')
    imgs = load_rgb_frames_from_video(vid_root=VIDEO_ROOT, vid=vid, start=0, num=64)
    print('Images loaded')

    all_images = []
    i=0
    j=0
    tmp_img = imgs[i:i+64]
    while len(tmp_img) >= 64:
        all_images.append({'img':tmp_img, 'index':j})
        i+=10
        j+=1
        tmp_img = imgs[i:i+64]

    all_images.append({'img':tmp_img, 'index':j})

    string_predict = []


    for i in range(0, len(all_images), 2):
        result_batch = dask.delayed(batch)(all_images[i:i+2])
        string_predict.append(result_batch)

    temp_dask = dask.delayed(get_final_phrase)(j, string_predict)
    print('ok')

    final_phrase = temp_dask.compute()

    print(final_phrase)

    return final_phrase


if __name__=='__main__':
    video_to_text()