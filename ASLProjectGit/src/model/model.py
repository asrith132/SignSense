import os
import json
import time
import random
import requests

# Numerical & Math Libraries
from jax import nn
from sympy.integrals.meijerint_doc import category

# MediaPipe for Hand Tracking
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerResult

# Computer Vision & Video Processing
import cv2
from pytube import YouTube
from yt_dlp import YoutubeDL
from PIL import Image

# PyTorch & Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

# torchvision for Image Processing
import torchvision
from torchvision import transforms

# Progress Bar Utility
from tqdm import tqdm
import numpy as np

# Global Vars
MASK_VALUE = -9999  # Sentinel value for missing data
EPOCHS = 30  # Number of epochs
LEARNING_RATE = 0.001  # Learning rate for optimizer
DATA_PATH = "/Users/nvelnambi/PycharmProjects/PythonProject/processed_data"
BATCH_SIZE = 8  # Define batch size before using it

used_words = {"hungry", "yes", "no", "thirsty", "prefer/favorite"}

def main():
    live()

class LSTMModel(nn.Module):
    def __init__(self, input_size=127, hidden_size=512, num_layers=3, output_size=5, bidirectional=True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (hidden, _) = self.lstm(packed_x)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.fc2(out)

        return out


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                param.data.fill_(0)  # Initialize biases to zero


class FocalLoss(nn.Module):
    def __init__(self, gamma=3, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]
            focal_loss = alpha_factor * focal_loss
        return focal_loss.mean()

def model_initialize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=127, hidden_size=256, num_layers=2, output_size=5)
    model.apply(initialize_weights)  # Now correctly initializes LSTM weights
    model.to(device)
    print(model)
    return model

def train(model):
    print("Training started")
    dataset_path = DATA_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2, alpha=torch.tensor([1.0, 1.0, 1.5, 1.5, 1.5]).to(device))

    sequences, labels, sequence_lengths = [], [], []
    label_map, label_counter = {}, 0

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        if label not in label_map:
            label_map[label] = label_counter
            label_counter += 1

    num_classes = len(label_map)
    assert num_classes == 5, f"Expected 5 classes, found {num_classes}"

    for label in label_map:
        label_path = os.path.join(dataset_path, label)
        video_folders = os.listdir(label_path)

        for video_folder in video_folders:
            video_path = os.path.join(label_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            json_files = sorted([f for f in os.listdir(video_path) if f.endswith('.json')])
            sequence = []

            for json_file in json_files:
                with open(os.path.join(video_path, json_file), 'r') as f:
                    frame_data = json.load(f)

                frame_features = []
                finger_state_vector = []

                if frame_data["hands"] != "NO HANDS DETECTED":
                    for hand in frame_data["hands"]:
                        for landmark in hand["landmarks"]:
                            frame_features.extend([
                                landmark["rel_x"], landmark["rel_y"], landmark["rel_z"],
                                landmark["vx"], landmark["vy"], landmark["vz"]
                            ])

                        # Convert finger state into binary values (1 for open, 0 for bent)
                        for finger in ["Index", "Middle", "Ring", "Pinky"]:
                            finger_state_vector.append(1 if hand["finger_states"].get(finger) == "open" else 0)

                frame_features.extend(finger_state_vector)

                if len(frame_features) > 131:
                    frame_features = frame_features[:131]
                while len(frame_features) < 131:
                    frame_features.append(MASK_VALUE)

                sequence.append(torch.tensor(frame_features, dtype=torch.float32))

            sequences.append(torch.stack(sequence))
            labels.append(label_map[label])
            sequence_lengths.append(len(sequence))

    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=MASK_VALUE)
    labels = torch.tensor(labels, dtype=torch.long)
    sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)

    dataset = TensorDataset(sequences_padded, labels, sequence_lengths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    epoch_count = 0

    while True:
        for epoch in range(5):
            epoch_count += 1
            total_loss = 0.0

            for batch_sequences, batch_labels, batch_lengths in dataloader:
                batch_sequences, batch_labels, batch_lengths = (
                    batch_sequences.to(device), batch_labels.to(device), batch_lengths.to(device)
                )

                optimizer.zero_grad()
                output = model(batch_sequences, batch_lengths)
                loss = criterion(output, batch_labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch_count} | Loss: {avg_loss:.4f}")

        save_model(model)
        print("\n🔹 **Checking Accuracy on Training Data...**")
        test_model_on_training_data("sign_language_model.pth")
        print("\n🔹 **You can now test the model live.**")
        live("sign_language_model.pth")

        user_input = input("\nDo you want to train for 5 more epochs? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("\n✅ Training stopped by user.")
            break


def save_model(model, path="sign_language_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

import torch
import os
from collections import defaultdict

def load_model(model_path="sign_language_model.pth"):
    """Load the trained model from a file."""
    model = LSTMModel()  # Ensure the model architecture matches training
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def annotate(to_train):
    processed_dir = "processed_data"
    os.makedirs(processed_dir, exist_ok=True)
    for words in used_words:
        word_dir = os.path.join(processed_dir, words)
        os.makedirs(word_dir, exist_ok=True)

    datapath = "/Users/nvelnambi/Desktop/ASL_Signing/"

    for folder in os.listdir(datapath):
        print(folder)
        if folder == ".DS_Store": continue
        for file in os.listdir(os.path.join(datapath, folder)):
            print(file)
            if file == ".DS_Store": continue
            pathname = os.path.join(datapath, folder, file)
            directory_path = os.path.dirname(pathname)
            folder_name = os.path.basename(directory_path)
            vid_dir = os.path.join(processed_dir, folder_name, os.path.splitext(os.path.basename(pathname))[0])
            os.makedirs(vid_dir, exist_ok=True)
            annotate_hands(pathname, vid_dir)


def annotate_hands(pathname, vid_dir):
    """Extracts hand landmarks per frame, computes velocity & finger states, and saves processed data."""
    vidcap = cv2.VideoCapture(pathname)
    success, image = vidcap.read()
    hands = mp.solutions.hands.Hands()

    time_start = time.time()
    frame_count = 0
    previous_hands = {}

    while success:
        time_stamp = time.time() - time_start
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image_rgb)

        frame_landmarks = {
            "timestamp": time_stamp,
            "frame": frame_count,
            "hands": [],
        }

        current_hands = {}

        if hand_results.multi_hand_landmarks:
            for hand_id, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = "Left" if hand_results.multi_handedness[hand_id].classification[0].label == "Left" else "Right"
                hand_key = f"{handedness}_{hand_id}"

                hand_data = {
                    "hand_id": hand_id,
                    "handedness": handedness,
                    "landmarks": []
                }

                hand_dict = {i: (lm.x, lm.y, lm.z) for i, lm in enumerate(hand_landmarks.landmark)}
                finger_states = calculate_finger_angles(hand_dict)

                # Directly store the finger states without reprocessing
                hand_data["finger_states"] = finger_states
                # Compute velocity
                prev_landmarks = previous_hands.get(hand_key)
                velocity_data = compute_velocity(list(hand_dict.values()), prev_landmarks)

                # Store in JSON output
                for i in range(len(hand_dict)):
                    landmark_data = {
                        "landmark_index": i,
                        "rel_x": hand_dict[i][0],
                        "rel_y": hand_dict[i][1],
                        "rel_z": hand_dict[i][2],
                        "vx": velocity_data[i]["vx"],
                        "vy": velocity_data[i]["vy"],
                        "vz": velocity_data[i]["vz"],
                    }

                    hand_data["landmarks"].append(landmark_data)

                frame_landmarks["hands"].append(hand_data)
                current_hands[hand_key] = list(hand_dict.values())

        else:
            frame_landmarks["hands"] = "NO HANDS DETECTED"

        # Save to JSON
        json_path = os.path.join(vid_dir, f"frame_{frame_count:04d}.json")
        with open(json_path, "w") as f:
            json.dump(frame_landmarks, f, indent=4)

        previous_hands = current_hands
        success, image = vidcap.read()
        frame_count += 1


def normalize_value(value, min_val, max_val):
    """ Normalize a value to [-1,1] using per-frame min/max. """
    return 2 * (value - min_val) / (max_val - min_val) - 1 if max_val > min_val else 0

def compute_velocity(current_landmarks, prev_landmarks):
    """Computes velocity (vx, vy, vz) using absolute x, y, z values and normalizes to [-1,1] per frame."""
    num_landmarks = 21
    velocity = [{"landmark_index": i, "vx": 0, "vy": 0, "vz": 0} for i in range(num_landmarks)]

    if prev_landmarks is None or len(prev_landmarks) != num_landmarks:
        return velocity  # Return zero velocities if no previous frame

    # Compute raw velocities
    raw_vx = []
    raw_vy = []
    raw_vz = []

    for i in range(num_landmarks):
        vx = current_landmarks[i][0] - prev_landmarks[i][0]  # Δx using tuple indexing
        vy = current_landmarks[i][1] - prev_landmarks[i][1]  # Δy
        vz = current_landmarks[i][2] - prev_landmarks[i][2]  # Δz

        raw_vx.append(vx)
        raw_vy.append(vy)
        raw_vz.append(vz)

    # Compute per-frame min/max for normalization
    min_vx, max_vx = min(raw_vx), max(raw_vx)
    min_vy, max_vy = min(raw_vy), max(raw_vy)
    min_vz, max_vz = min(raw_vz), max(raw_vz)

    for i in range(num_landmarks):
        velocity[i]["vx"] = normalize_value(raw_vx[i], min_vx, max_vx)
        velocity[i]["vy"] = normalize_value(raw_vy[i], min_vy, max_vy)
        velocity[i]["vz"] = normalize_value(raw_vz[i], min_vz, max_vz)

    return velocity



def downloader():
    # Define the target words and output directory
    output_dir = "training_data_unclean"
    os.makedirs(output_dir, exist_ok=True)

    # Path to the JSON file containing annotations
    annotations_file = "/Users/nvelnambi/Downloads/MS-ASL/MSASL_train.json"

    # Load annotations from the JSON file
    with open(annotations_file, "r") as f:
        data = json.load(f)

    # Process entries to filter and download URLs
    for entry in tqdm(data, desc="Processing and Downloading Videos"):
        clean_word = entry.get("text", "").strip().lower()

        # Process only if the word is in the target list
        if clean_word in used_words:
            # Create a subdirectory for the clean word
            word_dir = os.path.join(output_dir, clean_word)
            os.makedirs(word_dir, exist_ok=True)

            # Extract the video URL
            video_url = entry["url"]

            # Configure yt-dlp options dynamically for each video
            ydl_opts = {
                'format': 'mp4',
                'outtmpl': os.path.join(word_dir, '%(title)s.%(ext)s'),  # Save in the corresponding folder
                'quiet': True,  # Suppress output
                'noprogress': True,  # Suppress progress bar
            }

            # Download the video
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
            except Exception as e:
                print(f"Failed to download {video_url}: {e}")
    print(f"Videos have been successfully downloaded into the '{output_dir}' directory.")

import time
import numpy as np

def load_model(path="sign_language_model.pth"):
    model = model_initialize()  # Recreate the model structure
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")
    return model

import time
import numpy as np

import torch
import cv2
import mediapipe as mp
import time

from collections import deque

# Parameters for exponential scaling
CONFIDENCE_BOOST_THRESHOLD = 10  # Frames before boost starts becoming strong
SCALING_FACTOR = 5  # Lower values make boost grow faster

def exponential_boost(t, T_0=CONFIDENCE_BOOST_THRESHOLD, K=SCALING_FACTOR):
    """ Exponential confidence boost based on duration of pattern match. """
    return 1 + min(5, (2.718 ** ((t - T_0) / K)))  # Limit max boost factor to 5x

def exponential_decay(t, T_0=CONFIDENCE_BOOST_THRESHOLD, K=SCALING_FACTOR):
    """ Exponential confidence decay when pattern is broken. """
    return max(0.1, (2.718 ** (-(t - T_0) / K)))  # Ensure confidence doesn't drop to 0


import time
import torch
import cv2
import numpy as np
import mediapipe as mp


import time

import time
import torch
import cv2
import numpy as np
import mediapipe as mp

import time


import time

def live(model_path="sign_language_model.pth", min_frames=30, max_frames=200, confidence_threshold=0.75):
    # Load trained model
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    previous_hands = {}

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Confidence boost tracking
    confidence_boost = {word: 1.0 for word in used_words}
    last_hand_seen_time = time.time()
    last_print_time = time.time()
    finger_state_history = {word: [] for word in used_words}  # Stores last few frames of finger states

    # Define expected finger positions with core fingers
    expected_finger_states = {
        "prefer/favorite": {"Index": "open", "Middle": "bent", "Ring": "open", "Pinky": "open", "Thumb": "open"},
        "hungry": {"Index": "bent", "Middle": "bent", "Ring": "bent", "Pinky": "bent", "Thumb": "open"},
        "thirsty": {"Index": "open", "Middle": "bent", "Ring": "bent", "Pinky": "bent", "Thumb": "bent"},
        "yes": {"Index": "bent", "Middle": "bent", "Ring": "bent", "Pinky": "bent", "Thumb": "bent"},
        "no": {"Index": "open", "Middle": "open", "Ring": "bent", "Pinky": "bent", "Thumb": "open"}
    }

    core_fingers = {
        "prefer/favorite": "Middle",
        "thirsty": "Index",
    }

    print("\n🔹 Live Sign Language Recognition Started (Press 'q' to Stop)\n")

    # Sequence buffer
    sequence = []
    last_prediction_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)

        frame_landmarks = []
        current_hands = {}
        detected_finger_states = None

        if hand_results.multi_hand_landmarks:
            last_hand_seen_time = time.time()
            all_x, all_y, all_z = [], [], []

            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_x.append(lm.x)
                    all_y.append(lm.y)
                    all_z.append(lm.z)

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            min_z, max_z = min(all_z), max(all_z)

            for hand_id, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = "Left" if hand_results.multi_handedness[hand_id].classification[0].label == "Left" else "Right"
                hand_key = f"{handedness}_{hand_id}"

                hand_data = []
                absolute_landmarks = []

                hand_dict = {i: (lm.x, lm.y, lm.z) for i, lm in enumerate(hand_landmarks.landmark)}
                detected_finger_states = calculate_finger_angles(hand_dict)  # Ensure this is always computed

                for idx, lm in enumerate(hand_landmarks.landmark):
                    abs_x, abs_y, abs_z = lm.x, lm.y, lm.z
                    absolute_landmarks.append((abs_x, abs_y, abs_z))

                    rel_x = normalize_value(abs_x, min_x, max_x)
                    rel_y = normalize_value(abs_y, min_y, max_y)
                    rel_z = normalize_value(abs_z, min_z, max_z)

                    hand_data.extend([rel_x, rel_y, rel_z])

                prev_landmarks = previous_hands.get(hand_key)
                velocity_data = compute_velocity(absolute_landmarks, prev_landmarks)

                for i in range(len(velocity_data)):
                    hand_data.extend([
                        velocity_data[i]["vx"],
                        velocity_data[i]["vy"],
                        velocity_data[i]["vz"]
                    ])

                frame_landmarks.extend(hand_data)
                current_hands[hand_key] = absolute_landmarks

        else:
            # Decay confidence boost when no hands are seen
            time_since_last_hand = time.time() - last_hand_seen_time
            decay_factor = 0.85 ** time_since_last_hand
            for word in confidence_boost:
                confidence_boost[word] *= decay_factor
                if confidence_boost[word] < 1.0:
                    confidence_boost[word] = 1.0

        if len(frame_landmarks) > 127:
            frame_landmarks = frame_landmarks[:127]
        while len(frame_landmarks) < 127:
            frame_landmarks.append(MASK_VALUE)

        frame_tensor = torch.tensor(frame_landmarks, dtype=torch.float32)
        sequence.append(frame_tensor)

        if len(sequence) > max_frames:
            sequence.pop(0)

        previous_hands = current_hands

        if len(sequence) >= min_frames:
            input_sequence = torch.stack(sequence)
            sequence_length = torch.tensor([len(sequence)], dtype=torch.long)

            input_sequence = input_sequence.unsqueeze(0).to(device).to(torch.float32)
            sequence_length = sequence_length.to(device)

            with torch.no_grad():
                output = model(input_sequence, sequence_length)
                probabilities = torch.softmax(output, dim=1)

                sign_labels = list(used_words)
                predicted_index = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_index].item()
                predicted_label = sign_labels[predicted_index]

                # Track past finger states
                if detected_finger_states and predicted_label in expected_finger_states:
                    expected_states = expected_finger_states[predicted_label]

                    finger_state_history[predicted_label].append(detected_finger_states)
                    if len(finger_state_history[predicted_label]) > 15:
                        finger_state_history[predicted_label].pop(0)

                    # Compute how often fingers matched the expected state in the last few seconds
                    match_ratios = [
                        sum(detected_finger_states.get(finger, "unknown") == expected_states.get(finger, "unknown")
                            for finger in expected_states) / len(expected_states)
                        for detected_finger_states in finger_state_history[predicted_label]
                    ]

                    match_threshold = sum(r >= 0.6 for r in match_ratios) >= 9  # At least 3/5 window must match
                    core_finger_match = detected_finger_states.get(core_fingers.get(predicted_label), "unknown") == \
                                        expected_states.get(core_fingers.get(predicted_label), "unknown")

                    if match_threshold and core_finger_match:
                        final_confidence = confidence * confidence_boost[predicted_label]

                        if final_confidence >= confidence_threshold:
                            print(f"\nPredicted Sign: {predicted_label} (Confidence: {final_confidence:.2f})")
                            sequence.clear()

                # Print probabilities and finger states every 2 seconds
                if time.time() - last_print_time >= 2.0:
                    last_print_time = time.time()
                    print("\n--- Sign Probabilities ---")
                    for i, label in enumerate(sign_labels):
                        adjusted_confidence = probabilities[0, i].item() * confidence_boost[label]
                        print(f"{label}: {adjusted_confidence:.2f}")

                    if detected_finger_states:
                        print("\n--- Finger States ---")
                        for finger, state in detected_finger_states.items():
                            print(f"{finger}: {state}")

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🔹 Live testing stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


def test_model_on_training_data(model_path="sign_language_model.pth"):
    """Tests the trained model on the training dataset and prints accuracy per class."""

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path)  # Load the trained model
    model.to(device)
    model.eval()  # Set to evaluation mode

    dataset_path = DATA_PATH  # Define the dataset path
    sequences = []
    labels = []
    sequence_lengths = []
    label_map = {}
    label_counter = 0

    # Load dataset
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        if label not in label_map:
            label_map[label] = label_counter
            label_counter += 1

    num_classes = len(label_map)
    assert num_classes == 5, f"Expected 5 classes, found {num_classes}"

    correct_per_class = {label: 0 for label in label_map}
    total_per_class = {label: 0 for label in label_map}

    for label in label_map:
        label_path = os.path.join(dataset_path, label)
        video_folders = os.listdir(label_path)

        for video_folder in video_folders:
            video_path = os.path.join(label_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            json_files = sorted([f for f in os.listdir(video_path) if f.endswith('.json')])
            sequence = []

            for json_file in json_files:
                with open(os.path.join(video_path, json_file), 'r') as f:
                    frame_data = json.load(f)

                frame_features = []
                if frame_data["hands"] != "NO HANDS DETECTED":
                    for hand in frame_data["hands"]:
                        for landmark in hand["landmarks"]:
                            frame_features.extend([
                                landmark["rel_x"], landmark["rel_y"], landmark["rel_z"],
                                landmark["vx"], landmark["vy"], landmark["vz"]
                            ])

                # Ensure exactly 127 features per frame
                if len(frame_features) > 127:
                    frame_features = frame_features[:127]
                while len(frame_features) < 127:
                    frame_features.append(MASK_VALUE)

                sequence.append(torch.tensor(frame_features, dtype=torch.float32))

            sequences.append(torch.stack(sequence))
            labels.append(label_map[label])
            sequence_lengths.append(len(sequence))

    # Pad sequences so they are all the same length
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=MASK_VALUE)
    labels = torch.tensor(labels, dtype=torch.long)
    sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)

    # Create DataLoader for batch processing
    dataset = TensorDataset(sequences_padded, labels, sequence_lengths)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Test the model
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_sequences, batch_labels, batch_lengths in dataloader:
            batch_sequences, batch_labels, batch_lengths = batch_sequences.to(device), batch_labels.to(
                device), batch_lengths.to(device)

            output = model(batch_sequences, batch_lengths)
            predictions = torch.argmax(output, dim=1)

            for i in range(len(predictions)):
                pred_label = list(label_map.keys())[predictions[i].item()]
                true_label = list(label_map.keys())[batch_labels[i].item()]

                if pred_label == true_label:
                    correct_per_class[true_label] += 1
                total_per_class[true_label] += 1

            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    print("\n🔹 **Training Accuracy Per Class:**")
    for label in label_map:
        accuracy = 100 * correct_per_class[label] / max(1, total_per_class[label])
        print(f"  {label}: {accuracy:.2f}% ({correct_per_class[label]}/{total_per_class[label]})")

    overall_accuracy = 100 * correct / total
    print(f"\n🔹 **Overall Training Accuracy: {overall_accuracy:.2f}% ({correct}/{total})**")


def max_json(directory):
    max_count = 0

    for main_folder in os.listdir(directory):  # Iterate through main folders
        main_folder_path = os.path.join(directory, main_folder)

        if os.path.isdir(main_folder_path):  # Ensure it's a directory
            for video_folder in os.listdir(main_folder_path):  # Iterate through video folders
                video_folder_path = os.path.join(main_folder_path, video_folder)

                if os.path.isdir(video_folder_path):  # Ensure it's a folder
                    json_files = [f for f in os.listdir(video_folder_path) if f.endswith('.json')]
                    print(f"Video folder: {video_folder_path}, JSON count: {len(json_files)}")

                    max_count = max(max_count, len(json_files))

    return max_count


# import cv2
# import mediapipe as mp
# import time
#
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
#
# def compute_angle(p1, p2, p3):
#     """
#     Computes the angle between vectors formed by three points (p1, p2, p3).
#     p1, p2, p3 are tuples of (x, y) coordinates.
#     Returns the angle in degrees.
#     """
#     A = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
#     B = np.array([p3[0] - p2[0], p3[1] - p2[1] , p3[2] - p2[2]])
#
#     dot_product = np.dot(A, B)
#     mag_A = np.linalg.norm(A)
#     mag_B = np.linalg.norm(B)
#
#     if mag_A == 0 or mag_B == 0:  # Avoid division by zero
#         return None
#
#     angle_rad = np.arccos(np.clip(dot_product / (mag_A * mag_B), -1.0, 1.0))
#     angle_deg = np.degrees(angle_rad)
#
#     return angle_deg
#
# def calculate_finger_angles_verbose(hand_landmarks):
#     """
#     Computes the angles for each finger and prints the results in degrees.
#     """
#     finger_segments = {
#         "Thumb": [(2, 3, 4)],
#         "Index": [(5, 6, 7)],
#         "Middle": [(9, 10, 11)],
#         "Ring": [(13, 14, 15)],
#         "Pinky": [(17, 18, 19)]
#     }
#
#     finger_angles = {}
#     for finger, segments in finger_segments.items():
#         angles = [
#             compute_angle(hand_landmarks[a], hand_landmarks[b], hand_landmarks[c])
#             for a, b, c in segments
#         ]
#         finger_angles[finger] = angles
#         finger_states = {}
#
#         if angles[0] is not None and isinstance(angles[0], (int, float)):
#             finger_states[finger] = "open" if angles[0] < 30 else "bent"
#         else:
#             finger_states[finger] = "unknown"
#         print(f"{finger} states: {finger_states}")
#
#     return finger_angles
#
# def track_and_compute_finger_angles():
#     """
#     Captures video from the webcam, detects hands using MediaPipe Hands,
#     extracts hand landmarks, and computes angles every second.
#     """
#     cap = cv2.VideoCapture(0)
#     hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
#     last_time = time.time()
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Convert frame to RGB (MediaPipe requires RGB)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)
#
#         # If a hand is detected, process it
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw hand landmarks
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#                 # Extract landmark coordinates
#                 hand_dict = {i: (lm.x, lm.y, lm.z) for i, lm in enumerate(hand_landmarks.landmark)}
#
#                 # Process every second
#                 if time.time() - last_time >= 3.0:
#                     print("\nComputing angles...")
#                     calculate_finger_angles_verbose(hand_dict)
#                     last_time = time.time()
#
#         # Display the frame
#         cv2.imshow("Hand Tracking", frame)
#
#         # Exit with 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

# Run the hand tracking function
#track_and_compute_finger_angles()



def compute_angle(p1, p2, p3):
    """
    Computes the angle between vectors formed by three points (p1, p2, p3).
    p1, p2, p3 are tuples of (x, y, z) coordinates.
    Returns the angle in degrees.
    """
    A = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
    B = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

    dot_product = np.dot(A, B)
    mag_A = np.linalg.norm(A)
    mag_B = np.linalg.norm(B)

    if mag_A == 0 or mag_B == 0:  # Avoid division by zero
        return None

    angle_rad = np.arccos(np.clip(dot_product / (mag_A * mag_B), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_finger_angles(hand_landmarks):
    """
    Computes the angles for each finger using pre-extracted hand landmarks.
    Returns a dictionary where each finger has an "open" or "bent" state.
    """
    finger_segments = {
        "Thumb": [(2, 3, 4)],  # Thumb segment for angle calculation
        "Index": [(5, 6, 7)],
        "Middle": [(9, 10, 11)],
        "Ring": [(13, 14, 15)],
        "Pinky": [(17, 18, 19)]
    }

    finger_states = {}

    for finger, segments in finger_segments.items():
        angles = [
            compute_angle(hand_landmarks[a], hand_landmarks[b], hand_landmarks[c])
            for a, b, c in segments
        ]

        if angles[0] is not None and angles[0] < 30:  # Angle threshold for open/bent
            finger_states[finger] = "open"
        else:
            finger_states[finger] = "bent"

    return finger_states  # Returns dictionary: {'Thumb': 'bent', 'Index': 'open', ...}


if __name__ == "__main__":
    main()
