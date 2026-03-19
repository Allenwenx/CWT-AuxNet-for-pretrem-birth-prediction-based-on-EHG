import os
import numpy as np
import mat73
import shutil
import random
from scipy.signal import welch,stft
import pywt
import antropy as ant

# --------------------------- Configuration parameters ---------------------------
window_size = 512
stride = 256
train_ratio = 1.0  
val_ratio = 0.0
test_ratio = 1.0 - train_ratio - val_ratio
fs = 20  # sampling rate

# --------------------------- functions ---------------------------
def normalize(x):
    max_x = x.max(axis=1, keepdims=True)
    min_x = x.min(axis=1, keepdims=True)
    return (x - min_x) / (max_x - min_x + 1e-6)

def compute_cwt(signal, wavelet='morl', scales=np.arange(3, 204)):
    coefs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    return np.abs(coefs)

def compute_stft(signal, fs=20, nperseg=2400, noverlap=1200):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)  # shape: [freq_bins, time_bins]

def compute_normalized_psd(signal, fs=20):
    freqs, psd = welch(signal, fs=fs, nperseg=len(signal))
    ref_band_mask = (freqs >= 0.08) & (freqs <= 1.0)
    psd_sum = np.sum(psd[ref_band_mask])
    psd = psd / psd_sum if psd_sum != 0 else psd
    return freqs, psd

def extract_pa(freqs, psd):
    band_mask = (freqs >= 0.125) & (freqs <= 0.425)
    return np.max(psd[band_mask]) if np.any(band_mask) else 0.0


def extract_feature(segment):
    s2, s3 = segment[0], segment[1]
    freqs2, psd2 = compute_normalized_psd(s2, fs)
    freqs3, psd3 = compute_normalized_psd(s3, fs)
    pa = extract_pa(freqs2, psd2)
    return np.array([pa], dtype=np.float32)

# --------------------------- Data Retrieval and Preparation ---------------------------
data = mat73.loadmat('./data/com_Data.mat')['com_Data']
num_people = len(data[0]['EHG'])

people_class_0, people_class_1 = [], []
for i in range(num_people):
    label = int(data[0]['Label'][i][1])
    (people_class_0 if label == 0 else people_class_1).append(i)

random.shuffle(people_class_0)
random.shuffle(people_class_1)

num_train_0 = int(len(people_class_0) * train_ratio)
num_val_0 = int(len(people_class_0) * val_ratio)
num_train_1 = int(len(people_class_1) * train_ratio)
num_val_1 = int(len(people_class_1) * val_ratio)

train_people = people_class_0[:num_train_0] + people_class_1[:num_train_1]
val_people = people_class_0[num_train_0:num_train_0 + num_val_0] + people_class_1[num_train_1:num_train_1 + num_val_1]
test_people = people_class_0[num_train_0 + num_val_0:] + people_class_1[num_train_1 + num_val_1:]

# --------------------------- Initialisation of the file directory ---------------------------
shutil.rmtree('./data/samples', ignore_errors=True)
os.makedirs('./data/samples', exist_ok=True)
train_file = open('./data/train.txt', 'w')
val_file = open('./data/val.txt', 'w')
test_file = open('./data/test.txt', 'w')

for i in range(num_people):
    ehg_signal = normalize(data[0]['EHG'][i])
    ehg_signal = ehg_signal[1:3, :]  # Keep only the S2 and S3
    label = str(int(data[0]['Label'][i][1]))
    user_id = i

    stat_features = extract_feature(ehg_signal)  
   
    for k in range(0, ehg_signal.shape[1] - window_size + 1, stride):
        segment = ehg_signal[:, k:k + window_size]
        sample_name = f"{i}_{k}"

       
        cwt_s2 = compute_cwt(segment[0])
        cwt_s3 = compute_cwt(segment[1])
       
        
        np.save(f'./data/samples/{sample_name}_S2.npy', cwt_s2.astype(np.float32))
        np.save(f'./data/samples/{sample_name}_S3.npy', cwt_s3.astype(np.float32))
        
        np.save(f'./data/samples/{sample_name}_statfeat.npy', stat_features)

        row = f"{sample_name} {label}\n"
        if user_id in train_people:
            train_file.write(row)
        elif user_id in val_people:
            val_file.write(row)
        else:
            test_file.write(row)

train_file.close()
val_file.close()
test_file.close()
