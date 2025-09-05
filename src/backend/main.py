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



# Global Vars
MASK_VALUE = -9999  # Sentinel value for missing data
EPOCHS = 30  # Number of epochs
LEARNING_RATE = 0.001  # Learning rate for optimizer
DATA_PATH = "/Users/nvelnambi/PycharmProjects/PythonProject/processed_data"
BATCH_SIZE = 8  # Define batch size before using it

used_words = {"hungry", "yes", "no", "thirsty", "prefer/favorite"}

def main():
    track_and_compute_finger_angles()

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

    # Load dataset & balance classes
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
                if frame_data["hands"] != "NO HANDS DETECTED":
                    for hand in frame_data["hands"]:
                        for landmark in hand["landmarks"]:
                            frame_features.extend([
                                landmark["rel_x"], landmark["rel_y"], landmark["rel_z"],
                                landmark["vx"], landmark["vy"], landmark["vz"]
                            ])

                if len(frame_features) > 127:
                    frame_features = frame_features[:127]
                while len(frame_features) < 127:
                    frame_features.append(MASK_VALUE)

                sequence.append(torch.tensor(frame_features, dtype=torch.float32))

            sequences.append(torch.stack(sequence))
            labels.append(label_map[label])
            sequence_lengths.append(len(sequence))

    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=MASK_VALUE)
    labels = torch.tensor(labels, dtype=torch.long)
    sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)

    # Create DataLoader with **batch size defined**
    dataset = TensorDataset(sequences_padded, labels, sequence_lengths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # Now BATCH_SIZE is defined

    model.train()
    epoch_count = 0

    while True:  # Keep training until user decides to stop
        for epoch in range(5):  # Train in increments of 5 epochs
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

        # Save model after every 5 epochs
        save_model(model)

        # **TEST THE MODEL ON TRAINING DATA**
        print("\n🔹 **Checking Accuracy on Training Data...**")
        test_model_on_training_data("sign_language_model.pth")

        # **LET USER TEST LIVE**
        print("\n🔹 **You can now test the model live.**")
        live("sign_language_model.pth")  # Run real-time testing

        # **ASK USER IF THEY WANT TO CONTINUE TRAINING**
        user_input = input("\nDo you want to train for 5 more epochs? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("\n✅ Training stopped by user.")
            break  # Stop training


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
    """Extracts hand landmarks per frame, computes velocity, and saves processed data."""
    vidcap = cv2.VideoCapture(pathname)
    success, image = vidcap.read()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    time_start = time.time()
    frame_count = 0
    previous_hands = {}  # Store absolute positions of previous frame's hands

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
            all_x, all_y, all_z = [], [], []

            # Extract absolute landmark coordinates to compute per-frame min/max
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_x.append(lm.x)
                    all_y.append(lm.y)
                    all_z.append(lm.z)

            # Compute per-frame min/max for [-1,1] normalization
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            min_z, max_z = min(all_z), max(all_z)

            for hand_id, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = "Left" if hand_results.multi_handedness[hand_id].classification[
                                           0].label == "Left" else "Right"
                hand_key = f"{handedness}_{hand_id}"

                hand_data = {
                    "hand_id": hand_id,
                    "handedness": handedness,
                    "landmarks": []
                }

                current_hand_landmarks = []
                absolute_landmarks = []  # Store absolute coordinates for velocity calculation

                for idx, lm in enumerate(hand_landmarks.landmark):
                    abs_x, abs_y, abs_z = lm.x, lm.y, lm.z  # Absolute position
                    absolute_landmarks.append((abs_x, abs_y, abs_z))  # Store for velocity

                    # Normalize within [-1,1] range
                    rel_x = normalize_value(abs_x, min_x, max_x)
                    rel_y = normalize_value(abs_y, min_y, max_y)
                    rel_z = normalize_value(abs_z, min_z, max_z)

                    landmark_data = {
                        "landmark_index": idx,
                        "rel_x": rel_x,
                        "rel_y": rel_y,
                        "rel_z": rel_z,
                    }
                    current_hand_landmarks.append(landmark_data)

                # Compute velocity using absolute coordinates before discarding them
                prev_landmarks = previous_hands.get(hand_key)
                velocity_data = compute_velocity(absolute_landmarks, prev_landmarks)

                # Store velocity data in normalized landmark dictionary
                for i in range(len(current_hand_landmarks)):
                    current_hand_landmarks[i]["vx"] = velocity_data[i]["vx"]
                    current_hand_landmarks[i]["vy"] = velocity_data[i]["vy"]
                    current_hand_landmarks[i]["vz"] = velocity_data[i]["vz"]

                hand_data["landmarks"] = current_hand_landmarks
                frame_landmarks["hands"].append(hand_data)

                # Store absolute coordinates for velocity calculation in the next frame
                current_hands[hand_key] = absolute_landmarks

        else:
            frame_landmarks["hands"] = "NO HANDS DETECTED"
            current_hands = {}

        # Save the JSON output (excluding absolute positions)
        json_path = os.path.join(vid_dir, f"frame_{frame_count:04d}.json")
        with open(json_path, "w") as f:
            json.dump(frame_landmarks, f, indent=4)

        # Update previous hand landmarks
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

def live(model_path="sign_language_model.pth", min_frames=30, max_frames=200, confidence_threshold=0.75):
    # Load trained model
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    previous_hands = {}

    # Initialize Mediapipe for hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Dynamic sequence buffer
    sequence = []
    last_prediction_time = time.time()

    print("\n🔹 Live Sign Language Recognition Started (Press 'q' to Stop)\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        hand_results = hands.process(frame_rgb)

        # Extract hand landmarks if detected
        frame_landmarks = []
        current_hands = {}

        if hand_results.multi_hand_landmarks:
            all_x, all_y, all_z = [], [], []

            # Extract raw coordinates for normalization
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_x.append(lm.x)
                    all_y.append(lm.y)
                    all_z.append(lm.z)

            # Compute per-frame min/max for normalization
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            min_z, max_z = min(all_z), max(all_z)

            for hand_id, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = "Left" if hand_results.multi_handedness[hand_id].classification[0].label == "Left" else "Right"
                hand_key = f"{handedness}_{hand_id}"

                hand_data = []
                absolute_landmarks = []  # Store absolute coordinates for velocity calculation

                for idx, lm in enumerate(hand_landmarks.landmark):
                    abs_x, abs_y, abs_z = lm.x, lm.y, lm.z  # Absolute position
                    absolute_landmarks.append((abs_x, abs_y, abs_z))  # Store for velocity calculation

                    # Normalize within [-1,1] range
                    rel_x = normalize_value(abs_x, min_x, max_x)
                    rel_y = normalize_value(abs_y, min_y, max_y)
                    rel_z = normalize_value(abs_z, min_z, max_z)

                    hand_data.extend([rel_x, rel_y, rel_z])  # Store for model input

                # Compute velocity
                prev_landmarks = previous_hands.get(hand_key)
                velocity_data = compute_velocity(absolute_landmarks, prev_landmarks)

                for i in range(len(velocity_data)):
                    hand_data.extend([
                        velocity_data[i]["vx"],
                        velocity_data[i]["vy"],
                        velocity_data[i]["vz"]
                    ])

                frame_landmarks.extend(hand_data)
                current_hands[hand_key] = absolute_landmarks  # Store for next frame velocity

        # Ensure exactly 127 features per frame
        if len(frame_landmarks) > 127:
            frame_landmarks = frame_landmarks[:127]  # Truncate extra features
        while len(frame_landmarks) < 127:
            frame_landmarks.append(MASK_VALUE)  # Pad missing features

        # Convert to tensor and add to sequence
        frame_tensor = torch.tensor(frame_landmarks, dtype=torch.float32)

        # Dynamically adjust buffer size
        sequence.append(frame_tensor)
        if len(sequence) > max_frames:
            sequence.pop(0)  # Remove oldest frame

        # Store absolute positions for next frame's velocity calculation
        previous_hands = current_hands

        # Trigger prediction when sequence length is sufficient
        if len(sequence) >= min_frames:
            input_sequence = torch.stack(sequence)  # Shape: (seq_len, 127)
            sequence_length = torch.tensor([len(sequence)], dtype=torch.long)

            # Add batch dimension and move to device
            input_sequence = input_sequence.unsqueeze(0).to(device).to(torch.float32)  # Convert to float32
            sequence_length = sequence_length.to(device)

            # Get model prediction
            with torch.no_grad():
                model.hidden = None  # Reset LSTM hidden state between predictions
                output = model(input_sequence, sequence_length)
                probabilities = torch.softmax(output, dim=1)  # Convert to probabilities

                # Get corresponding sign labels
                sign_labels = list(used_words)

                # Print all probabilities every second for debugging
                if time.time() - last_prediction_time >= 1.0:
                    last_prediction_time = time.time()
                    print("\n--- Sign Predictions ---")
                    for i, label in enumerate(sign_labels):
                        print(f"{label}: {probabilities[0, i].item():.2f}")

                # Get most confident prediction
                predicted_index = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_index].item()
                predicted_label = sign_labels[predicted_index]

                # Only show high-confidence predictions
                if confidence >= confidence_threshold:
                    print(f"\nPredicted Sign: {predicted_label} (Confidence: {confidence:.2f})")
                    sequence.clear()  # Reset buffer after confident prediction

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow("Sign Language Recognition", frame)

        # Press 'q' to exit
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

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def compute_angle(p1, p2, p3):
    """
    Computes the angle between vectors formed by three points (p1, p2, p3).
    p1, p2, p3 are tuples of (x, y) coordinates.
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

def calculate_finger_angles_verbose(hand_landmarks):
    """
    Computes the angles for each finger and prints the results in degrees.
    """
    finger_segments = {
        "Thumb": [(2, 3, 4)],
        "Index": [(5, 6, 7)],
        "Middle": [(9, 10, 11)],
        "Ring": [(13, 14, 15)],
        "Pinky": [(17, 18, 19)]
    }

    finger_angles = {}
    for finger, segments in finger_segments.items():
        angles = [
            compute_angle(hand_landmarks[a], hand_landmarks[b], hand_landmarks[c])
            for a, b, c in segments
        ]
        finger_angles[finger] = angles
        print(f"{finger} angles: {angles}")

    return finger_angles

def track_and_compute_finger_angles():
    """
    Captures video from the webcam, detects hands using MediaPipe Hands,
    extracts hand landmarks, and computes angles every second.
    """
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # If a hand is detected, process it
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark coordinates
                hand_dict = {i: (lm.x, lm.y, lm.z) for i, lm in enumerate(hand_landmarks.landmark)}

                # Process every second
                if time.time() - last_time >= 1.0:
                    print("\nComputing angles...")
                    calculate_finger_angles_verbose(hand_dict)
                    last_time = time.time()

        # Display the frame
        cv2.imshow("Hand Tracking", frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the hand tracking function
track_and_compute_finger_angles()


if __name__ == "__main__":
    main()
