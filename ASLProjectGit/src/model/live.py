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