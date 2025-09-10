# Heatmap Generation for Visualizing Human Activity in Surveillance Video

### 🎓 Mini-Project Report (AIML7445E04 - Image and Video Analytics)  
**Department of Computer Science and Engineering**  
**School of Engineering and Technology, CHRIST (Deemed to be University), Bengaluru**  
**September 2025**  

**Authors:**  
- Pratik Suchak (2262162)  
- Pihoo Yagnik (2262131)  

---

## 📌 Project Introduction
In modern surveillance, massive amounts of video footage are generated daily, yet most of it is underutilized beyond security monitoring. Identifying **high-activity regions** in these videos can provide valuable insights for:

- 🛒 Retail analytics (footfall patterns)  
- 👥 Crowd management  
- 🛡️ Public safety monitoring  

This project focuses on generating **heatmaps** from surveillance videos to visualize regions of frequent human activity. By applying **background subtraction** and accumulating motion data across frames, we create **color-coded heatmaps** that highlight:

- 🔴 “Hot zones” → frequently visited areas  
- 🔵 “Cold zones” → rarely used spaces  

The heatmaps are overlaid on video frames and exported as both static images and animated videos, with JSON statistics for numerical analysis.

---

## ⚙️ Implementation

The project is implemented in **Python (3.x)** using **OpenCV** and **NumPy**.

### Pipeline
1. **Preprocessing**
   - Read input video (`.mpg` format).  
   - Convert frames to grayscale.  

2. **Motion Detection**
   - Background subtraction using `cv2.createBackgroundSubtractorMOG2()`.  
   - Thresholding to remove shadows/noise.  

3. **Heatmap Generation**
   - Accumulate binary motion masks across frames.  
   - Normalize values and apply `COLORMAP_JET`.  

4. **Visualization & Output**
   - Save static heatmap images.  
   - Overlay heatmap onto original video.  
   - Create animated heatmap video.  
   - Export activity stats in JSON format (4x4 grid).  

---

## 💻 Code

The full implementation is in `main.py`.  
Key functions:  

- `process_video(video_path, filename)` → generates heatmaps, overlays, JSON stats.  
- `overlay_heatmap_on_video(video_path, heatmap_img, output_path)` → blends heatmap with original video.  
- `create_animated_heatmap(video_path, output_path)` → shows activity accumulation.  
- `export_activity_stats(heatmap, filename)` → saves 4x4 grid activity JSON.  

---

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Heatmap-Analytics.git
   cd Heatmap-Analytics
2. Install dependencies
   ```bash
     pip install opencv-python numpy
3. Place .mpg surveillance videos inside the videos/ folder.
4. Run python main.py
