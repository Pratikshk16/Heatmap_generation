import cv2
import numpy as np
import os
import json

VIDEO_DIR = 'videos'
HEATMAP_DIR = 'outputs/heatmaps'
OVERLAY_DIR = 'outputs/frames'
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

def export_activity_stats(heatmap, filename):
    stats = {}
    h, w = heatmap.shape
    grid_size = 4  # 4x4 grid
    gh, gw = h // grid_size, w // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            block = heatmap[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            avg_activity = np.mean(block)
            stats[f"block_{i}_{j}"] = round(float(avg_activity), 2)

    json_path = os.path.join("outputs", f"{filename}_stats.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"📊 Saved activity stats JSON: {json_path}")

def overlay_heatmap_on_video(video_path, heatmap_img, output_path):
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    heatmap_img = cv2.resize(heatmap_img, (width, height))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        overlay = cv2.addWeighted(frame, 0.6, heatmap_img, 0.4, 0)
        out.write(overlay)

    cap.release()
    out.release()
    print(f"🎞️ Saved overlay video: {output_path}")

def create_animated_heatmap(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    heatmap = np.zeros((height, width), dtype=np.float32)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = bg_subtractor.apply(gray)
        _, fg = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)

        heatmap += fg.astype(np.float32)
        norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)
        out.write(overlay)

    cap.release()
    out.release()
    print(f"🎬 Saved animated heatmap video: {output_path}")

def process_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    accumulated_heatmap = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

        if accumulated_heatmap is None:
            accumulated_heatmap = np.zeros_like(fg_mask, dtype=np.float32)

        accumulated_heatmap += fg_mask.astype(np.float32)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"{filename}: Processed {frame_count} frames")

    cap.release()

    normalized = cv2.normalize(accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    colored_heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

    heatmap_path = os.path.join(HEATMAP_DIR, f"{filename}_heatmap.jpg")
    cv2.imwrite(heatmap_path, colored_heatmap)
    print(f"✅ Saved heatmap image: {heatmap_path}")

    # Overlay video
    overlay_path = os.path.join(OVERLAY_DIR, f"{filename}_overlay.mp4")
    overlay_heatmap_on_video(video_path, colored_heatmap, overlay_path)

    # Animated heatmap video
    animated_path = os.path.join(OVERLAY_DIR, f"{filename}_animated.mp4")
    create_animated_heatmap(video_path, animated_path)

    # Export activity JSON
    export_activity_stats(normalized, filename)

def main():
    for file in os.listdir(VIDEO_DIR):
        if file.endswith('.mpg'):
            video_path = os.path.join(VIDEO_DIR, file)
            filename = os.path.splitext(file)[0]
            print(f"\n🎬 Processing: {file}")
            process_video(video_path, filename)

if __name__ == "__main__":
    main()
