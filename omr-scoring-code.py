import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# -------------------------
# Utilities / IO
# -------------------------
def load_answer_key(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    answer_dict = {}
    for _, row in df.iterrows():
        q = int(row["question"])
        ans = int(row["answer"]) - 1  # convert to 0-indexed
        answer_dict[q] = ans
    return answer_dict


# -------------------------
# Model builder & synthetic data
# -------------------------
def build_bubble_classifier():
    model = models.Sequential([
        layers.Input(shape=(40, 40, 1)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def generate_synthetic_data(n_samples=2000, size=40):
    X, y = [], []
    for _ in range(n_samples // 2):
        # Empty bubble
        empty = np.ones((size, size), dtype=np.uint8) * 250
        center = size // 2
        radius = size // 3
        cv2.circle(empty, (center, center), radius, 50, 2)
        noise = np.random.normal(0, 8, empty.shape).astype(np.int16)
        empty = np.clip(empty.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        X.append(empty)
        y.append(0)

        # Filled bubble
        filled = np.ones((size, size), dtype=np.uint8) * 250
        cv2.circle(filled, (center, center), radius, 50, 2)
        cv2.circle(filled, (center, center), radius - 3, 80, -1)
        for _ in range(150):
            px = center + np.random.randint(-radius + 4, radius - 4)
            py = center + np.random.randint(-radius + 4, radius - 4)
            if (px - center) ** 2 + (py - center) ** 2 < (radius - 3) ** 2:
                cv2.circle(filled, (px, py), 1, np.random.randint(40, 100), -1)
        noise = np.random.normal(0, 8, filled.shape).astype(np.int16)
        filled = np.clip(filled.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        X.append(filled)
        y.append(1)

    X = np.array(X, dtype=np.float32) / 255.0
    X = np.expand_dims(X, -1)
    y = np.array(y, dtype=np.int32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def train_model(epochs=15):
    print("Training CNN (synthetic data)...")
    model = build_bubble_classifier()
    X, y = generate_synthetic_data(n_samples=2000)
    split = int(0.85 * len(X))
    model.fit(X[:split], y[:split],
              validation_data=(X[split:], y[split:]),
              epochs=epochs,
              batch_size=32,
              verbose=1)
    return model


# -------------------------
# Image preprocessing & detection
# -------------------------
def preprocess_image(img_bgr):
    h, w = img_bgr.shape[:2]
    if h > 1500:
        scale = 1500 / h
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return img_bgr, gray


def detect_bubbles_hough(gray, debug=False):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    all_circles = []

    circles1 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=20, minRadius=10, maxRadius=25)
    if circles1 is not None:
        all_circles.extend(circles1[0])

    circles2 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=40, param2=25, minRadius=12, maxRadius=22)
    if circles2 is not None:
        all_circles.extend(circles2[0])

    bubbles = []
    for circle in all_circles:
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        is_duplicate = False
        for existing in bubbles:
            dist = np.sqrt((x - existing['x']) ** 2 + (y - existing['y']) ** 2)
            if dist < 15:
                is_duplicate = True
                break
        if not is_duplicate:
            bubbles.append({'x': x, 'y': y, 'r': r, 'area': np.pi * r * r})

    # header cutoff
    h = gray.shape[0]
    header_cutoff = int(h * 0.15)
    bubbles = [b for b in bubbles if b['y'] > header_cutoff]

    bubbles = sorted(bubbles, key=lambda b: (b['y'], b['x']))
    return bubbles


def organize_into_questions(bubbles, n_choices=5):
    if not bubbles:
        return []

    sorted_bubbles = sorted(bubbles, key=lambda b: b['y'])
    rows = []
    current_row = [sorted_bubbles[0]]
    avg_radius = np.mean([b['r'] for b in bubbles])
    y_threshold = int(avg_radius * 1.5)

    for bubble in sorted_bubbles[1:]:
        if abs(bubble['y'] - current_row[-1]['y']) <= y_threshold:
            current_row.append(bubble)
        else:
            if len(current_row) >= n_choices:
                sorted_row = sorted(current_row, key=lambda b: b['x'])[:n_choices]
                rows.append(sorted_row)
            elif len(current_row) == n_choices - 1:
                sorted_row = sorted(current_row, key=lambda b: b['x'])
                rows.append(sorted_row)
            current_row = [bubble]

    if len(current_row) >= n_choices - 1:
        sorted_row = sorted(current_row, key=lambda b: b['x'])
        if len(sorted_row) >= n_choices:
            sorted_row = sorted_row[:n_choices]
        rows.append(sorted_row)

    return rows


def classify_bubble(model, gray_img, bubble, size=40):
    x, y, r = bubble['x'], bubble['y'], bubble['r']
    pad = int(r * 1.3)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(gray_img.shape[1], x + pad), min(gray_img.shape[0], y + pad)
    crop = gray_img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    center_region = crop[crop.shape[0]//4:3*crop.shape[0]//4,
                         crop.shape[1]//4:3*crop.shape[1]//4]
    darkness = 255 - np.mean(center_region)
    crop_resized = cv2.resize(crop, (size, size))
    crop_norm = crop_resized.astype(np.float32) / 255.0
    crop_input = np.expand_dims(np.expand_dims(crop_norm, -1), 0)
    pred = model.predict(crop_input, verbose=0)[0]
    cnn_filled_score = pred[1]
    combined_score = cnn_filled_score * 0.6 + (darkness / 255.0) * 0.4
    return combined_score


# -------------------------
# Main CLI flow
# -------------------------
def main(args):
    # Load key
    try:
        answer_key = load_answer_key(args.key)
    except Exception as e:
        print(f"Error loading answer key: {e}", file=sys.stderr)
        sys.exit(1)
    total_questions = len(answer_key)
    print(f"Loaded answer key with {total_questions} questions.")

    # Load image
    file_bytes = np.frombuffer(open(args.image, "rb").read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("Could not read image.", file=sys.stderr)
        sys.exit(1)

    # Preprocess & detect
    img_processed, gray = preprocess_image(img_bgr)
    bubbles = detect_bubbles_hough(gray, debug=False)
    print(f"Detected {len(bubbles)} bubbles (after header cutoff).")

    if len(bubbles) < total_questions * 3:
        print(f"Warning: expected ~{total_questions * 5} bubbles but found {len(bubbles)}")

    questions = organize_into_questions(bubbles, n_choices=5)
    print(f"Organized into {len(questions)} question rows.")

    if len(questions) == 0:
        print("Could not organize bubbles into rows. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Train model
    model = train_model(epochs=args.epochs)

    # Grade
    results = []
    correct_count = 0
    FILL_THRESHOLD = 0.40

    for q_num, question_bubbles in enumerate(questions, start=1):
        if q_num > total_questions:
            break
        scores = []
        for bubble in question_bubbles:
            score = classify_bubble(model, gray, bubble)
            scores.append(score)
        max_score = max(scores) if scores else 0.0
        if max_score < FILL_THRESHOLD:
            selected_letter = "NA"
            selected_idx = None
            selected_conf = 0.0
            is_correct = False
        else:
            selected_idx = int(np.argmax(scores))
            selected_conf = scores[selected_idx]
            correct_idx = answer_key.get(q_num, None)
            is_correct = (selected_idx == correct_idx) if correct_idx is not None else False
            selected_letter = chr(65 + selected_idx)
        if is_correct:
            correct_count += 1
        correct_idx = answer_key.get(q_num, None)
        correct_letter = chr(65 + correct_idx) if correct_idx is not None else "N/A"
        results.append({
            'Question': q_num,
            'Selected': selected_letter,
            'Confidence': f"{max_score:.2f}",
            'Correct': correct_letter,
            'Status': '✅' if is_correct else ('⚠️' if selected_letter == "NA" else '❌')
        })

    # Save results CSV
    out_prefix = args.out_prefix
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_csv_path = f"{out_prefix}_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to: {results_csv_path}")

    # Final score
    score_pct = (correct_count / total_questions * 100) if total_questions > 0 else 0.0
    print(f"Final Score: {correct_count}/{total_questions} ({score_pct:.1f}%)")

    # Visualization: annotate detected bubbles and correct answers
    vis_img = img_processed.copy()
    for q_num, question_bubbles in enumerate(questions[:total_questions], start=1):
        for i, bubble in enumerate(question_bubbles):
            x, y, r = bubble['x'], bubble['y'], bubble['r']
            correct_idx_for_q = answer_key.get(q_num, -1)
            is_correct_ans = (i == correct_idx_for_q)
            color = (0, 255, 0) if is_correct_ans else (200, 200, 200)
            thickness = 3 if is_correct_ans else 1
            cv2.circle(vis_img, (x, y), r, color, thickness)
            cv2.putText(vis_img, chr(65 + i), (x - 8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    viz_path = f"{out_prefix}_viz.png"
    cv2.imwrite(viz_path, cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    print(f"Visualization saved to: {viz_path}")

    # Optionally: quick matplotlib show if asked
    if args.show:
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"OMR Detection - Score {correct_count}/{total_questions}")
        plt.show()


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL OMR Grader (CLI version)")
    parser.add_argument("--image", required=True, help="Path to OMR image (jpg/png)")
    parser.add_argument("--key", required=True, help="Path to answer key CSV")
    parser.add_argument("--out-prefix", default=f"omr_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Prefix for output files (results CSV and viz PNG)")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for synthetic-data training")
    parser.add_argument("--show", action="store_true", help="Show final visualization window (matplotlib)")
    args = parser.parse_args()
    main(args)