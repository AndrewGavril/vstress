import cv2
import torch as tr
import heartpy as hp
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, SGD
from scipy.interpolate import PchipInterpolator
from scipy import signal
from sklearn import preprocessing
from matplotlib import pyplot as plt
from pathlib import Path
import biosppy.signals as human_signals
from torch.utils.tensorboard import SummaryWriter
from ignite.contrib.handlers.clearml_logger import (
    ClearMLLogger,
    ClearMLSaver,
    GradsHistHandler,
    GradsScalarHandler,
    WeightsHistHandler,
    WeightsScalarHandler,
    global_step_from_engine,
)
from clearml import Logger
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, EpochOutputStore
from ignite.metrics import Loss
from torchmetrics import PearsonCorrCoef
from ignite.utils import setup_logger
from ignite.contrib.handlers import PiecewiseLinear
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter
import scipy
import numpy
import mediapipe as mp
import resampy
from sklearn.preprocessing import MinMaxScaler, Normalizer
from biosppy.signals import bvp, ppg, ecg
from typing import Optional
from torch.nn import functional as F
UBFC_DATASET_PATH = Path('./input/FormatedUBFC')
HANDS_DATASET_PATH = Path('./input/DNNSimpleData/dataset')
TRAIN_DATASET_PATH = UBFC_DATASET_PATH


LOW = 42 / 60
HIGH = 220 / 60
TO = 1000
L = 64
WINDOW_LENGTH = 64


def load_ubfc():
    videos = []
    with open(UBFC_DATASET_PATH / 'annotation.txt') as annotation_file:
        for line in annotation_file.readlines():
            video_info = line.split()
            video_info[1] = 6000
            video_info[2] = float(video_info[2])
            video_info[3] = float(video_info[3])
            video_info[4] = float(video_info[4])
            video_info[5] = float(video_info[5])
            videos.append(tuple(video_info))
        return videos




def load_hands():
    videos = []
    with open(HANDS_DATASET_PATH / 'annotation.txt') as annotation_file:
        for line in annotation_file.readlines():
            video_info = line.split()
            video_info[1] = int(video_info[1]) - 1
            video_info[2] = float(video_info[2])
            video_info.append(130)
            videos.append(tuple(video_info))
        return videos




def shift_to_function(function_to_shift: 'numpy.ndarray', ref_function: 'numpy.ndarray') -> 'numpy.ndarray':
    function_to_shift_fft = scipy.fft.fft(function_to_shift)
    ref_function_fft = scipy.fft.fft(ref_function)
    shift = numpy.argmax(scipy.fft.ifft(function_to_shift_fft * numpy.conj(ref_function_fft)))
    return shift

# Pulse preprocessing
def ecg_preprocessing(pulse, from_fps, to_fps, frames_count):
    # Remove all data except R-peaks data
    pulse[pulse < 120] = 0
    filtered = pulse
    # Apply bandpass filter to signal
    b, a = signal.butter(2, (LOW, HIGH), 'bandpass', fs=from_fps)
    filtered = signal.lfilter(b, a, pulse)

    filtered = uniform_filter(filtered, 10)

    shift = shift_to_function(np.array(filtered), np.array(pulse))
    filtered = numpy.roll(filtered, -shift)

    # Resample to FPS freq
    len_in_sec = len(filtered) / from_fps
    y_raw = np.arange(0, len_in_sec, 1 / from_fps)
    y_res = np.arange(0, len_in_sec + 1, 1 / to_fps)
    decimated = PchipInterpolator(y_raw, filtered)(y_res, extrapolate=True)
    decimated = decimated[:frames_count]
    normalized = MinMaxScaler().fit_transform(np.array(decimated).reshape(-1, 1))
    enhanced = hp.enhance_peaks(normalized, iterations=2)
    enhanced = MinMaxScaler().fit_transform(np.array(enhanced).reshape(-1, 1))
    
    return enhanced.flatten()

def bvp_preprocessing(pulse, from_fps, to_fps, frames_count):
    filtered = pulse
    
    # Resample to FPS freq
    len_in_sec = len(filtered) / from_fps
    y_raw = np.arange(0, len_in_sec, 1 / from_fps)
    y_res = np.arange(0, len_in_sec + 1, 1 / to_fps)
    decimated = PchipInterpolator(y_raw, filtered)(y_res, extrapolate=True)
    decimated = decimated[:frames_count]
    normalized = MinMaxScaler().fit_transform(np.array(decimated).reshape(-1, 1))
    enhanced = hp.enhance_peaks(normalized, iterations=2) * 10e10
    enhanced = MinMaxScaler().fit_transform(np.array(enhanced).reshape(-1, 1))
    return enhanced.flatten() #/ np.linalg.norm(enhanced.flatten())

def preprocess_ground_truth(v_info, dataset_type='ubfc'):
    if dataset_type == 'ubfc':
        pulse_signal = np.load(UBFC_DATASET_PATH / v_info[0] / 'pulse.npy')
        processed_data = bvp_preprocessing(pulse_signal, v_info[3], v_info[2], v_info[1])
    else:
        pulse_signal, _ = np.load(HANDS_DATASET_PATH / v_info[0] / 'pulse.npy')
        processed_data = ecg_preprocessing(pulse_signal, v_info[3], v_info[2], v_info[1])
    assert len(processed_data) == v_info[1], f"Len (before: {len(pulse_signal)}; after: {len(processed_data)}) of resampled data does not equal to len of frames in vedeo ({v_info[1]}) {TRAIN_DATASET_PATH / v_info[0] / 'pulse.npy'}"
    return processed_data

videos = load_ubfc()
ubfc_pulse_signal= np.load(UBFC_DATASET_PATH / videos[0][0] / 'pulse.npy')
interpolated = preprocess_ground_truth(videos[0])
ubfc_pulse_signal = MinMaxScaler().fit_transform(np.array(ubfc_pulse_signal).reshape(-1, 1)).flatten()
ubfc_pulse_signal = ubfc_pulse_signal / np.linalg.norm(ubfc_pulse_signal)

len_in_sec = len(ubfc_pulse_signal) / videos[0][3]
ts_raw = np.arange(0, len_in_sec, 1 / videos[0][3])
ts_res = np.arange(0, len_in_sec + 1, 1 / videos[0][2])[:videos[0][1]]
plt.plot(ts_raw[:TO], ubfc_pulse_signal[:TO])

interp_to = int(TO * videos[0][2] // videos[0][3])
plt.plot(ts_res[:interp_to], interpolated[:interp_to])


pred_data, pred_measures = hp.process(interpolated[:interp_to], 35)
hp.plotter(pred_data, pred_measures)
out = bvp.bvp(interpolated[:interp_to], 35)
out['onsets']

videos = load_hands()
v_info = videos[0]
hands_pulse_signal, _ = np.load(HANDS_DATASET_PATH / videos[0][0] / 'pulse.npy')
interpolated = ecg_preprocessing(hands_pulse_signal, v_info[3], v_info[2], v_info[1])
hands_pulse_signal = MinMaxScaler().fit_transform(np.array(hands_pulse_signal).reshape(-1, 1))
interpolated = MinMaxScaler().fit_transform(np.array(interpolated).reshape(-1, 1))

len_in_sec = len(ubfc_pulse_signal) / videos[0][3]
ts_raw = np.arange(0, len_in_sec, 1 / videos[0][3])
ts_res = np.arange(0, len_in_sec + 1, 1 / videos[0][2])[:videos[0][1]]
plt.plot(ts_raw[:TO], hands_pulse_signal[:TO])

interp_to = int(TO * videos[0][2] // videos[0][3])
plt.plot(ts_res[:interp_to], interpolated[:interp_to])

bb_data = {'./subject10/video_1': (64, 220, 756, 865), './subject10/video_2': (33, 86, 796, 775), './subject10/video_3': (34, 55, 942, 944), './subject27/video_1': (323, 357, 755, 744), './subject27/video_2': (118, 235, 881, 815), './subject27/video_3': (257, 278, 764, 867), './subject3/video_1': (221, 274, 665, 714), './subject3/video_2': (135, 251, 790, 793), './subject3/video_3': (121, 179, 721, 791), './subject2/video_1': (238, 271, 760, 793), './subject2/video_2': (107, 218, 896, 965), './subject2/video_3': (138, 230, 893, 1023), './subject1/video_1': (257, 276, 784, 858), './subject1/video_2': (227, 173, 895, 826), './subject1/video_3': (182, 228, 946, 927), './subject5/video_1': (287, 143, 819, 749), './subject5/video_2': (334, 127, 930, 763), './subject5/video_3': (288, 225, 919, 993), './subject4/video_1': (236, 272, 853, 796), './subject4/video_2': (288, 132, 896, 799), './subject4/video_3': (0, 127, 1023, 1023), './subject6/video_1': (388, 263, 817, 704), './subject6/video_2': (310, 91, 1023, 757), './subject6/video_3': (138, 71, 962, 879), './subject8/video_1': (155, 353, 682, 886), './subject8/video_2': (110, 229, 745, 841), './subject8/video_3': (0, 251, 1023, 1023), './subject7/video_1': (238, 297, 721, 777), './subject7/video_2': (128, 186, 944, 1012), './subject7/video_3': (216, 181, 1023, 917), './subject9/video_1': (304, 200, 1023, 922), './subject9/video_2': (0, 162, 1023, 792), './subject9/video_3': (0, 171, 1023, 1023), './subject11/video_1': (59, 0, 660, 729), './subject11/video_2': (0, 0, 853, 705), './subject11/video_3': (39, 29, 881, 759), './subject12/video_1': (280, 189, 806, 720), './subject12/video_2': (221, 151, 915, 707), './subject12/video_3': (265, 146, 822, 730), './subject14/video_1': (387, 317, 1014, 823), './subject14/video_2': (225, 146, 990, 904), './subject14/video_3': (247, 169, 1023, 932), './subject13/video_1': (212, 243, 762, 762), './subject13/video_2': (63, 43, 869, 784), './subject13/video_3': (239, 87, 858, 779), './subject15/video_1': (300, 260, 725, 669), './subject15/video_2': (307, 138, 878, 780), './subject15/video_3': (174, 157, 919, 957), './subject16/video_1': (287, 241, 765, 741), './subject16/video_2': (317, 196, 938, 794), './subject16/video_3': (270, 192, 911, 848), './subject18/video_1': (148, 245, 664, 836), './subject18/video_2': (202, 275, 705, 785), './subject18/video_3': (159, 297, 678, 898), './subject17/video_1': (104, 292, 659, 831), './subject17/video_2': (0, 176, 721, 1023), './subject17/video_3': (0, 202, 712, 840), './subject19/video_1': (280, 224, 883, 934), './subject19/video_2': (174, 138, 1023, 903), './subject19/video_3': (54, 97, 976, 850), './subject20/video_1': (296, 357, 774, 805), './subject20/video_2': (185, 314, 861, 858), './subject20/video_3': (128, 283, 914, 1023), './subject21/video_1': (0, 121, 893, 753), './subject21/video_2': (107, 204, 846, 762), './subject22/video_1': (236, 260, 743, 762), './subject22/video_2': (83, 195, 823, 817), './subject22/video_3': (200, 115, 769, 744), './subject23/video_1': (259, 299, 762, 796), './subject23/video_2': (48, 169, 877, 841), './subject23/video_3': (67, 227, 833, 904), './subject24/video_1': (305, 187, 797, 723), './subject24/video_2': (199, 174, 1023, 837), './subject24/video_3': (142, 91, 1023, 828), './subject28/video_1': (336, 294, 889, 887), './subject28/video_2': (0, 130, 1011, 757), './subject28/video_3': (209, 90, 1023, 932), './subject26/video_1': (249, 347, 766, 879), './subject26/video_2': (149, 346, 763, 795), './subject26/video_3': (246, 329, 762, 793), './subject25/video_1': (133, 238, 663, 736), './subject25/video_2': (63, 182, 763, 861), './subject25/video_3': (40, 161, 797, 846), './subject29/video_1': (278, 324, 845, 813), './subject29/video_2': (231, 216, 1023, 931), './subject29/video_3': (95, 202, 1005, 946), './subject30/video_1': (321, 354, 982, 1023), './subject30/video_2': (352, 244, 1003, 856), './subject30/video_3': (305, 193, 1023, 943), './subject31/video_1': (250, 213, 742, 800), './subject31/video_2': (136, 103, 836, 799), './subject31/video_3': (0, 102, 869, 909), './subject34/video_1': (367, 298, 936, 765), './subject34/video_2': (463, 162, 1023, 803), './subject34/video_3': (524, 189, 1023, 769), './subject32/video_1': (482, 125, 1001, 636), './subject32/video_2': (406, 0, 1023, 598), './subject32/video_3': (445, 98, 984, 704), './subject33/video_1': (321, 271, 1023, 871), './subject33/video_2': (383, 103, 1023, 740), './subject33/video_3': (230, 182, 1023, 1023), './subject36/video_1': (219, 246, 656, 683), './subject36/video_2': (43, 176, 827, 891), './subject36/video_3': (0, 173, 644, 786), './subject35/video_1': (221, 263, 687, 757), './subject35/video_2': (125, 170, 670, 731), './subject35/video_3': (126, 229, 776, 740), './subject37/video_1': (141, 286, 638, 911), './subject37/video_2': (145, 58, 745, 698), './subject37/video_3': (0, 112, 603, 849), './subject39/video_1': (417, 330, 927, 838), './subject39/video_2': (414, 248, 879, 771), './subject39/video_3': (343, 176, 955, 838), './subject38/video_1': (279, 240, 721, 639), './subject38/video_2': (209, 110, 989, 971), './subject38/video_3': (162, 108, 918, 994), './subject41/video_1': (257, 214, 775, 724), './subject41/video_2': (276, 226, 798, 804), './subject41/video_3': (178, 190, 771, 708), './subject42/video_1': (163, 272, 784, 934), './subject42/video_2': (151, 123, 1023, 1023), './subject42/video_3': (90, 32, 902, 1002), './subject43/video_1': (386, 247, 787, 651), './subject43/video_2': (245, 162, 844, 663), './subject43/video_3': (210, 155, 697, 679), './subject45/video_1': (281, 306, 811, 860), './subject45/video_2': (243, 131, 766, 772), './subject45/video_3': (76, 173, 823, 967), './subject46/video_1': (294, 316, 720, 740), './subject46/video_2': (186, 278, 817, 911), './subject46/video_3': (132, 292, 798, 894), './subject47/video_1': (32, 185, 693, 833), './subject47/video_2': (38, 53, 840, 834), './subject47/video_3': (0, 22, 946, 1023), './subject48/video_1': (0, 212, 586, 754), './subject48/video_2': (0, 54, 919, 792), './subject48/video_3': (0, 148, 1023, 787), './subject49/video_1': (97, 278, 596, 780), './subject49/video_2': (0, 64, 813, 796), './subject49/video_3': (0, 109, 667, 992), './subject44/video_1': (379, 111, 886, 806), './subject44/video_2': (299, 88, 878, 664), './subject44/video_3': (243, 205, 943, 1023), './subject50/video_1': (247, 346, 651, 763), './subject50/video_2': (68, 227, 737, 704), './subject50/video_3': (102, 253, 656, 740), './subject52/video_1': (293, 314, 709, 763), './subject52/video_2': (208, 157, 954, 900), './subject52/video_3': (236, 217, 991, 1023), './subject51/video_1': (267, 311, 786, 794), './subject51/video_2': (294, 103, 816, 789), './subject51/video_3': (246, 214, 803, 796), './subject54/video_1': (252, 290, 753, 790), './subject54/video_2': (237, 291, 843, 809), './subject54/video_3': (246, 276, 753, 801), './subject55/video_1': (327, 224, 1023, 933), './subject55/video_2': (89, 0, 1023, 762), './subject55/video_3': (0, 138, 1023, 993), './subject53/video_1': (257, 233, 729, 695), './subject53/video_2': (212, 164, 932, 851), './subject53/video_3': (33, 209, 957, 937), './subject40/video_1': (355, 258, 821, 771), './subject40/video_2': (302, 190, 923, 737), './subject40/video_3': (345, 124, 981, 755), './subject56/video_1': (227, 238, 931, 1023), './subject56/video_2': (323, 165, 1023, 830), './subject56/video_3': (316, 99, 1023, 815)}



class SkinVideoDataset(Dataset):
    def __init__(self, dataset_path, device, annotation='annotation.txt', dataset_type='ubfc'):
        self.window_length = WINDOW_LENGTH
        self.device = device
        
        super(SkinVideoDataset).__init__()
        self.path = Path(dataset_path)
        
        self.videos = load_ubfc() if dataset_type == 'ubfc' else load_hands()
                
        self.max_bb = bb_data if dataset_type == 'ubfc' else None
        if bb_data is None:
            self.max_bb = {}
        self.pulse_signals = []
        for video_info in tqdm(self.videos):
            pulse = preprocess_ground_truth(video_info, dataset_type)
            
            self.pulse_signals.append(pulse)
            if bb_data is None and dataset_type == 'ubfc':
                res_x_left, res_y_top, res_x_right, res_y_bot = None, None, None, None

                for i in range(0, video_info[1], 5):
                    
                    image = cv2.imread(str(self.path / video_info[0] / 'video' / f'img_{i:05d}.jpg'))
                    bb = self.cut_face(image)
                    if bb is not None:
                        x_left, y_top, x_right, y_bot = bb
                        if res_x_left is None:
                            res_x_left, res_y_top, res_x_right, res_y_bot = bb
                        if x_left < res_x_left:
                            res_x_left = x_left
                        if x_right > res_x_right:
                            res_x_right = x_right
                        if y_bot > res_y_bot:
                            res_y_bot = y_bot
                        if y_top < res_y_top:
                            res_y_top = y_top
                if res_x_left is None:
                    res_x_left, res_y_top, res_x_right, res_y_bot = 0, 1024, 1024, 0
                self.max_bb[video_info[0]] = (res_x_left, res_y_top, res_x_right, res_y_bot)
        # Count total frames
        self.total_frames = 0
        print(self.max_bb)
        for video_info in self.videos:
            self.total_frames += video_info[1] - self.window_length

    def cut_face(self, image):
        with mp.solutions.face_detection.FaceDetection(model_selection="full", min_detection_confidence=0.2) as face_detection:
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.detections:
                detection=results.detections[0]
                location = detection.location_data

                relative_bounding_box = location.relative_bounding_box
                x_left = np.floor(relative_bounding_box.xmin * image.shape[1])
                y_top = np.floor(relative_bounding_box.ymin * image.shape[0])
                x_right = np.floor((relative_bounding_box.xmin + relative_bounding_box.width) * image.shape[1])
                y_bot = np.floor((relative_bounding_box.ymin + relative_bounding_box.height) * image.shape[0])
                
                x_left = int(max(0, x_left))
                y_top = int(max(0, y_top))
                x_right = int(min(x_right, image.shape[1] - 1))
                y_bot = int(min(y_bot, image.shape[0] - 1))
                return x_left, y_top, x_right, y_bot
            else:
                return None

    def __len__(self):
        return self.total_frames
                
    def __find_video_index(self, idx):
        # handle negative index
        if idx < 0:
            idx = self.__len__() + idx
            if idx < 0:
                raise IndexError('Frame index out of range')
        video_index = None
        for i, video_info in enumerate(self.videos):
            if (video_info[1] - self.window_length) > idx:
                video_index = i
                break
            idx -= video_info[1] - self.window_length
        
        if video_index is None:
            raise IndexError('Frame index out of range')   
        return video_index, idx
    
    
    def get_fs(self):
        return self.videos[0][2]
    

    def __getitem__(self, idx):
        video_index, frame_idx = self.__find_video_index(idx)
        
        target = self.pulse_signals[video_index][frame_idx:frame_idx+self.window_length]
        assert len(target) == self.window_length, f"Wrong bvp length: {len(target)} on idx: {frame_idx}"
        
        target = np.array(target).astype(np.float32)
        target = tr.from_numpy(target).float().squeeze()
        # Read frame images from disk
        frames = []
        for i in range(self.window_length):
            image = cv2.imread(str(self.path / self.videos[video_index][0] / 'video' / f'img_{frame_idx+i:05d}.jpg'))
            frames.append(image)
        if self.max_bb is not None and len(self.max_bb.keys()) > 0:
            x_left, y_top, x_right, y_bot = self.max_bb[self.videos[video_index][0]]
        for i, image in enumerate(frames):
            # Resize images to  LxL
            try:
                if self.max_bb is not None and len(self.max_bb.keys()) > 0:
                    image = image[y_top:y_bot, x_left:x_right]
                image = cv2.resize(image, (L, L), interpolation=cv2.INTER_CUBIC)
                frames[i] = image
            except: 
                print(f"Cannot resize image to ({L}x{L}): {str(self.path / self.videos[video_index][0] / 'video' / f'img_{frame_idx+i:05d}.jpg')} from {image.shape} with bb {self.max_bb[self.videos[video_index][0]]}")
        frames = np.array(frames)
        assert frames.shape == (len(frames), L, L, 3), f"Wrong frames patch dim: {frames.shape} on idx: {frame_idx}"
        frames = frames.reshape((3,len(frames), L, L))
        frames = tr.from_numpy(frames.astype(np.float32))
                
        sample = (frames, target)
        return sample
        




class ConvBlock2D(tr.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock2D, self).__init__()
        self.conv_block_2d = tr.nn.Sequential(
            tr.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            tr.nn.BatchNorm2d(out_channel),
            tr.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block_2d(x)


class cnn_blocks(tr.nn.Module):
    def __init__(self):
        super(cnn_blocks, self).__init__()
        self.cnn_blocks = tr.nn.Sequential(
            ConvBlock2D(3, 16, [5, 5], [1, 1], [2, 2]),
            tr.nn.MaxPool2d((2, 2), stride=(2, 2)),
            ConvBlock2D(16, 32, [3, 3], [1, 1], [1, 1]),
            ConvBlock2D(32, 64, [3, 3], [1, 1], [1, 1]),
            tr.nn.MaxPool2d((2, 2), stride=(2, 2)),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            tr.nn.MaxPool2d((2, 2), stride=(2, 2)),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            tr.nn.MaxPool2d((2, 2), stride=(2, 2)),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            tr.nn.AdaptiveMaxPool2d(1)
        )

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        x = x.view(batch * length, channel, width, height)
        x = self.cnn_blocks(x)
        x = x.view(batch,length,-1,1,1)

        return x

class ConvLSTMCell(tr.nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = tr.nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=4 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = tr.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = tr.split(combined_conv, self.hidden_dim, dim=1)
        i = tr.sigmoid(cc_i)
        f = tr.sigmoid(cc_f)
        o = tr.sigmoid(cc_o)
        g = tr.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tr.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (tr.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                tr.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(tr.nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = tr.nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = tr.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
class PhysNet_2DCNN_LSTM(tr.nn.Module):
    def __init__(self, frame=32):
        super(PhysNet_2DCNN_LSTM, self).__init__()
        self.physnet_lstm = tr.nn.ModuleDict({
            'cnn_blocks': cnn_blocks(),
            'cov_lstm': ConvLSTM(64, [1, 1, 64], (1, 1), num_layers=3, batch_first=True, bias=True,
                                 return_all_layers=False),
            'cnn_flatten': tr.nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        })

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        x = self.physnet_lstm['cnn_blocks'](x)
        x, _ = self.physnet_lstm['cov_lstm'](x)
        x = tr.permute(x[0], (0, 2, 1, 3, 4))
        x = self.physnet_lstm['cnn_flatten'](x)
        return x.view(-1, length)


class NegPearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(NegPearson, self).__init__()
        return

    def forward(self, preds, labels):  # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = tr.sum(preds[i])  # x
            sum_y = tr.sum(labels[i])  # y
            sum_xy = tr.sum(preds[i] * labels[i])  # xy
            sum_x2 = tr.sum(tr.pow(preds[i], 2))  # x^2
            sum_y2 = tr.sum(tr.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                tr.sqrt((N * sum_x2 - tr.pow(sum_x, 2)) * (N * sum_y2 - tr.pow(sum_y, 2))))

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss
    
class OnsetsMAE(nn.Module):
    def __init__(self, fs):
        super(OnsetsMAE, self).__init__()
        self.mae = nn.L1Loss()
        self.fs = fs

    def find_peaks_intervals(self, batch):
        width = int(self.fs // 4)
        i = 0
        batch = batch.to("cpu")
        pred_onsets_vectors = tr.zeros(batch.shape[0], batch.shape[1], dtype=tr.float32)
        for i, vector in enumerate(batch):
            window_maxima = tr.nn.functional.max_pool1d_with_indices(vector.view(1,1,-1), width, 1, padding=width//2)[1].squeeze()
            candidates = window_maxima.unique()
            peaks = candidates[(window_maxima[candidates]==candidates).nonzero()].flatten()
            pred_onsets_vectors[i][:peaks.shape[0]] = peaks
            i += 1
        return tr.sort(pred_onsets_vectors)[0]


    def forward(self, preds, labels):
        pred_onsets = self.find_peaks_intervals(preds)
        gt_onsets = self.find_peaks_intervals(labels)
        return self.mae(pred_onsets, gt_onsets)
    

class PulseLoss(nn.Module):
    def __init__(self, fs):
        super(PulseLoss,self).__init__()
        self.fs = fs
        self.onsets_mae = OnsetsMAE(fs)
        self.pearson = NegPearson()
    

    def forward(self, preds, labels):
        return self.pearson(preds, labels)


USE_PRETRAINED = False
def run(train_batch_size, val_batch_size, epochs, lr):

    num_workers = 6
    device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    Path("./output/checkpoints").mkdir(parents=True, exist_ok=True)

    dataset = SkinVideoDataset(TRAIN_DATASET_PATH, device)

    train_set, test_set = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set,
                            batch_size=train_batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_set,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    
    loss_function = PulseLoss(dataset.get_fs())
    net = PhysNet_2DCNN_LSTM()
    if USE_PRETRAINED:
        checkpoint = tr.load("/tmp/ignite_checkpoints_2023_05_02_09_34_54_d9mk3pa2/best_model_13_val_loss=8.0236.pt")
        net.load_state_dict(checkpoint)
    print(f"Valbatches: {len(test_loader)}, TrainBatches: {len(train_loader)}")
    net.to(device)
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    step_scheduler = StepLR(optimizer, step_size=1000, gamma=1)
    lr_scheduler = LRScheduler(step_scheduler)

    trainer = create_supervised_trainer(net, optimizer, loss_function, device=device)#, amp_mode='amp')
    trainer.logger = setup_logger("Trainer")
    metrics = {"TrainLoss": Loss(loss_function), "lossNegPearson": Loss(NegPearson()), "lossPulseOnset": Loss(OnsetsMAE(dataset.get_fs()))}

    train_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device)#, amp_mode='amp')
    train_evaluator.logger = setup_logger("Train Evaluator")
    validation_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device)#, amp_mode='amp')
    validation_evaluator.logger = setup_logger("Val Evaluator")
    eos = EpochOutputStore()

    eos.attach(validation_evaluator, 'output')
    
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)


    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        def plot_res(output, iter):
            pr_output = list(output[0][0].cpu())
            gt_output = list(output[0][1].cpu())
            for i in range(0, len(pr_output), 10):
                cur_gt = gt_output[i]
                cur_pr = pr_output[i]
                pulse = MinMaxScaler().fit_transform(np.array(cur_gt).reshape(-1, 1))
                gt_pulse = MinMaxScaler().fit_transform(np.array(cur_pr).reshape(-1, 1))
                plt.plot(pulse)
                plt.plot(gt_pulse)
                Logger.current_logger().report_matplotlib_figure(f"Interation: {iter} Batch: {i}", f"Batch: {i}", plt)
                plt.cla()
        validation_evaluator.run(test_loader)
        output = validation_evaluator.state.output
        plot_res(output, engine.state_dict()["iteration"])

    
    clearml_logger = ClearMLLogger(project_name="rPPG", task_name="ignite") 

    clearml_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=50),
        tag="training",
        output_transform=lambda loss: {"batchloss": loss},
    )

    for tag, evaluator in [("training metrics", train_evaluator), ("validation metrics", validation_evaluator)]:
        clearml_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["TrainLoss", "lossNegPearson"],
            global_step_transform=global_step_from_engine(trainer),
        )

    clearml_logger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=100), optimizer=optimizer
    )

    clearml_logger.attach(
        trainer, log_handler=WeightsScalarHandler(net), event_name=Events.ITERATION_COMPLETED(every=100)
    )

    clearml_logger.attach(trainer, log_handler=WeightsHistHandler(net), event_name=Events.EPOCH_COMPLETED(every=100))

    clearml_logger.attach(
        trainer, log_handler=GradsScalarHandler(net), event_name=Events.ITERATION_COMPLETED(every=100)
    )

    clearml_logger.attach(trainer, log_handler=GradsHistHandler(net), event_name=Events.EPOCH_COMPLETED(every=100))

    handler = Checkpoint(
        {"model": net},
        ClearMLSaver(),
        n_saved=1,
        score_function=lambda e: 1 / e.state.metrics["TrainLoss"],
        score_name="val_loss",
        filename_prefix="best",
        global_step_transform=global_step_from_engine(trainer),
    )
    
    
        
    validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    trainer.run(train_loader, max_epochs=epochs)

    clearml_logger.close()



batch_size = 16
number_of_epochs = 60
lr = 0.001


run(batch_size, batch_size, number_of_epochs, lr)


device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')
print(f"Device: {device}")

video_1 = SkinVideoDataset(UBFC_DATASET_PATH, device)
video_loader = DataLoader(video_1,
                         batch_size=1,
                         num_workers=6,
                         shuffle=False,
                         pin_memory=True)
PLOT_LIMIT = 6000
net = PhysNet_2DCNN_LSTM()



checkpoint = tr.load("/home/andrei/work/vpulse/64_64_LSTM_val_loss=5.9632.pt")
net.load_state_dict(checkpoint)
net.to(device)

ground_truth = []
result = []
counter = 0
for inputs, targets in tqdm(video_loader):
    with tr.no_grad():
        outputs = net(inputs.to(device)).squeeze()
        result.append(outputs.data.cpu().numpy().tolist())
        ground_truth.append(targets.data.cpu().numpy().tolist())
        counter += 1
        if counter == PLOT_LIMIT:
            break




gt_res = [] + ground_truth[0][0]
for i in range(1, len(ground_truth)):
    gt_res.append(ground_truth[i][0][-1])

res = [[] for _ in range(len(result) + WINDOW_LENGTH - 1)]
for i in range(0, len(result)):
    for j in range(len(result[i])):
        if i + j < len(res):
            res[i + j].append(result[i][j])

for i in range(len(res)):
    res[i] = np.mean(res[i])
pulse = np.array(res).flatten()




# For single video analysis
for i in range(0, len(gt_res), 6000):
    cur_gt = MinMaxScaler().fit_transform(np.array(gt_res[i:i+6000]).reshape(-1, 1)).flatten()[:6000]
    cur_pulse = MinMaxScaler().fit_transform(np.array(pulse[i:i+6000]).reshape(-1, 1)).flatten()[:6000]
    cur_pulse = hp.filter_signal(cur_pulse, (20/60, 180/60), 35.0, filtertype='bandpass')
    cur_gt /= np.linalg.norm(cur_gt)
    cur_pulse /= np.linalg.norm(cur_pulse)
    pred_data, pred_measures = hp.process(cur_pulse, 35.0, calc_freq= True)
    hp.plotter(pred_data, pred_measures)
    gt_data, gt_measures = hp.process(cur_gt, 35.0, calc_freq=True)
    hp.plotter(gt_data, gt_measures)
    print(f'GT:\n{gt_measures}\nPred:\n{pred_measures}')
    bvp.bvp(cur_pulse, 35)
    bvp.bvp(cur_gt, 35)

gt_res = MinMaxScaler().fit_transform(np.array(gt_res).reshape(-1, 1)).flatten()[:6000]
pulse = MinMaxScaler().fit_transform(np.array(pulse).reshape(-1, 1)).flatten()[:6000]
plt.plot(np.array(pulse[:PLOT_LIMIT]))
plt.plot(np.array(gt_res[:PLOT_LIMIT]))
plt.show()
print(len(pulse), len(gt_res))

# Apply bandpass filter
LOW = 42 / 60
HIGH = 180 / 60
MEASUREMENT_FREQUENCY = 60
filtered = pulse

b, a = signal.butter(2, (LOW, HIGH), 'bandpass', fs=MEASUREMENT_FREQUENCY,)
filtered = scipy.signal.filtfilt(b, a, filtered)
shift = shift_to_function(filtered, gt_res)
filtered = numpy.roll(filtered, -shift)

filtered = MinMaxScaler().fit_transform(np.array(filtered).reshape(-1, 1)).flatten()
filtered_res = filtered[0:PLOT_LIMIT]
ground_truth_res = gt_res[0:PLOT_LIMIT]



plt.plot(filtered_res)
plt.plot(ground_truth_res)
plt.show()

print(len(filtered_res), len(ground_truth_res))

plt.psd(filtered_res, Fs=60)
plt.psd(ground_truth_res, Fs=60)
gt_spd = signal.welch(ground_truth_res, fs=60)
res_spd = signal.welch(filtered_res, fs=60)
print(f'gt: {gt_spd[0][np.argmax(gt_spd[1])]*60}; res:{res_spd[0][np.argmax(res_spd[1])]*60}')
plt.show()
