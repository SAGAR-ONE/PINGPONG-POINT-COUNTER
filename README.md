# 🏓 Table Tennis Tracker

An automated table tennis referee system that uses a camera and computer vision to watch a game and keep score in real time — no human referee needed.

---

## Overview

The camera is placed above the table at an angle and captures 30 frames per second. The software tracks the ball using computer vision, detects bounces and points, and updates the scoreboard automatically.

---

## Features

- **Ball Detection** — Color + shape + trajectory-based tracking using HSV color space
- **Ball Registration** — Calibrate the ball size before the game to avoid false positives
- **Table Detection** — Click 4 corners to define the table boundary, net, and player sides
- **Bounce Detection** — Tracks vertical velocity to detect when the ball bounces; two bounces on one side awards a point to the opponent
- **Trajectory Prediction** — Stores the last 5–10 positions to predict the ball's next location, smoothing out jitter and handling brief disappearances
- **Perspective Correction** — Uses OpenCV's `getPerspectiveTransform` (homography) to flatten the trapezoidal camera view for accurate measurements
- **Point Detection** — Awards points when the ball exits bounds, bounces twice on one side, or disappears for 25 frames; includes a 2-second cooldown to prevent double counting
- **Serve Detection** — Detects or manually marks who is serving
- **Rally Counter** — Counts how many times the ball crosses the net per rally
- **Multiple Game Sets** — Supports best-of-5 sets (first to 11 per set), following real table tennis rules
- **Sound Effects** — Plays audio on point scored, bounce, and game winner
- **Game History & Stats** — Shows total points, longest rally, most consecutive points, and average rally length after each game
- **Replay Last Point** — Stores the last 5 seconds of frames; replays in slow motion when a point is scored
- **Web Dashboard** — Live scoreboard viewable on any phone in the room via a browser
- **Scoreboard Overlay** — Drawn directly onto the video frame in real time, showing player names and scores
- **Automatic Camera Calibration** — Detects table edges automatically using color (blue/green table surfaces)
- **Mobile Phone as Camera** — Supports DroidCam or similar apps to stream a phone camera over WiFi

---

## How It Works

### Camera Input
The camera is placed above the table and captures 30 frames per second. Each frame is passed through the computer vision pipeline.

### Computer Vision Pipeline
```
Raw Frame
  → Convert to HSV color space
  → Apply color mask (filter ball color only)
  → Clean up mask (remove noise)
  → Find circular shapes (contours)
  → Match against registered ball size
  → Get ball position (x, y)
```

### Why HSV Instead of RGB?
RGB is sensitive to lighting — the same ball can look completely different in shadow vs. bright light. HSV separates color (Hue) from brightness (Value), making detection reliable across different lighting conditions.

### Ball Registration
Before the game, hold the ball in front of the camera for 30 frames. The system records the average, minimum, and maximum radius in pixels. During the game, any circle outside this size range is ignored.

### Table Setup
Click the 4 corners of the table. The system calculates:
- The full table boundary
- The net position (midpoint of left and right edges)
- Player 1 side (left half)
- Player 2 side (right half)

### Point Detection Logic
| Event | Result |
|---|---|
| Ball bounces twice on one side | Opponent scores |
| Ball exits table boundary | Opponent scores |
| Ball disappears for 25 frames | Opponent scores |
| 2-second cooldown | Prevents double counting |

### Trajectory Prediction
```
Store positions: [(x1,y1), (x2,y2), (x3,y3) ...]
Calculate velocity: dx = x2 - x1, dy = y2 - y1
Predict next:     x4 = x3 + dx, y4 = y3 + dy
```

---

## Setup & Usage

### Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- A webcam or USB camera mounted above the table
- Optional: DroidCam or similar app for using a phone as a camera

### Running the Tracker
```bash
python main.py
```

### Step-by-step
1. Launch the application
2. Hold the ball in front of the camera and press **Register Ball** — hold still for 30 frames
3. Click the **4 corners** of the table on the screen
4. Enter player names
5. Press **Start Game**
6. The scoreboard updates automatically as you play

---

## Stats & History

After each game, a stats screen displays:
- Total points played
- Longest rally
- Most consecutive points won
- Average rally length per set

---

## Web Dashboard

The live scoreboard is also available in your browser — anyone in the room can view it on their phone without needing the Java window open.

```
http://localhost:8080
```

---

## Camera Tips

- Mount the camera directly above or at a high angle over the table for best results
- Ensure consistent lighting — avoid strong backlight or shadows across the table
- A USB camera mounted on a stand works best; a phone on a tripod with DroidCam also works well
- The automatic table detection works best on blue or green tables

---

## Project Structure

```
table-tennis-tracker/
├── main.py                 # Entry point
├── vision/
│   ├── ball_detector.py    # HSV masking, contour detection, registration
│   ├── trajectory.py       # Position history and prediction
│   ├── bounce_detector.py  # Vertical velocity tracking
│   └── homography.py       # Perspective correction
├── game/
│   ├── scoring.py          # Point detection logic and cooldown
│   ├── sets.py             # Multi-set game management
│   └── stats.py            # Rally counter and match history
├── ui/
│   ├── scoreboard.py       # Overlay drawn on video frames
│   ├── dashboard.py        # Web scoreboard server
│   └── replay.py           # Last-point slow-motion replay
├── audio/
│   └── sounds.py           # Sound effects (javax.sound / pygame)
└── README.md
```

---

## Acknowledgements

Built with OpenCV for computer vision and Python for game logic. Inspired by the challenge of bringing automated officiating to recreational table tennis.
