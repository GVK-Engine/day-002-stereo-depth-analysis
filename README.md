# Day 2 - Stereo Camera Depth Analysis

**Series 1: Perception | Project 2 of 12**

Part of my 90-day robotics portfolio series.
MS Robotics and Autonomous Systems Engineering, Arizona State University, Dec 2026.

________________________________________

## The Problem

Every self-driving car that uses cameras needs to answer one question:
how accurate is camera depth really, and at what distance does it become unsafe?

Tesla chose cameras only for FSD. Mobileye builds their entire stack around cameras.
But stereo geometry has a fundamental physical limitation that no algorithm can fully overcome.

This project measures that limitation on real sensor data and finds the exact distance
where camera depth becomes unreliable for safe driving.

________________________________________

## What the Industry Does Today

Most stereo depth papers report accuracy on a single benchmark frame or a small curated set.
What they rarely publish is per-distance-band accuracy with standard deviation across many frames.

That variance matters. A consistent 2m error can be compensated for by downstream systems.
A 2m average error with 2m standard deviation cannot, because the system never knows
whether the error is 0.5m or 4m at any given moment.

This project measures both.

________________________________________

## How It Works

The pipeline runs in seven stages.

Load the left and right camera images captured simultaneously from two cameras
mounted 0.47 meters apart on the car roof.

Compute a disparity map using Semi-Global Matching. SGM finds matching pixels
between the left and right images and measures how far each pixel shifted.
Close objects shift more. Far objects shift less.

Convert disparity to metric depth using the stereo triangulation formula:
depth = (focal length x baseline) / disparity.
This converts pixel shift directly into real meters.

Load the LiDAR point cloud and transform it from the LiDAR coordinate frame
to the camera coordinate frame using the velo-to-cam rigid body transform.
Then project onto the image plane using the camera projection matrix P2.
This gives sparse but highly accurate ground truth depth at each projected pixel.

Dilate the sparse LiDAR depth map to increase pixel overlap with the dense
camera depth map. This is standard preprocessing in stereo evaluation papers.

Compare camera depth against LiDAR ground truth across four distance bands
and compute MAE and RMSE per band.

________________________________________

## Sensor Setup

    Dataset          KITTI Raw 2011_09_26_drive_0001
    Frames tested    9 frames - 1, 5, 10, 20, 30, 50, 70, 90, 100
    Left camera      image_02 - 1242 x 375 pixels
    Right camera     image_03
    Focal length     721.54 pixels
    Baseline         0.4706 meters
    LiDAR            Velodyne HDL-64E - accuracy +/- 2cm

________________________________________

## Results

4-panel visualization showing camera image, disparity map, depth map, and accuracy chart:
https://drive.google.com/file/d/1f_SQ3wEw5SeGKK5YpB0dLtt-PyOnST1z/view?usp=drive_link

Benchmark results averaged across 9 real KITTI frames:

    Distance Band    Avg MAE     Std Dev     Verdict
    0-10m            1.039m      0.014m      Just above 1.0m safety threshold
    10-30m           2.302m      0.121m      2.3x over safety limit
    30-50m           5.377m      0.662m      Dangerously inaccurate
    50m+             8.813m      2.136m      Completely unreliable

Safety threshold: 1.0m MAE
Beyond 1.0m average error, depth estimates cannot be trusted for driving decisions.

________________________________________

## Key Engineering Findings

Finding 1 - The safety threshold is crossed immediately.

Even at 0-10m the average error is 1.039m. There is no distance range where
classical SGM stereo depth is comfortably within the safe zone on its own.

Finding 2 - Long range errors are unpredictable, not just large.

At 0-10m the standard deviation is 0.014m. The error is consistent frame to frame.
At 50m the standard deviation grows to 2.136m. The error varies wildly between frames.
Consistent errors can be modeled and compensated for by downstream systems.
Random errors cannot. A system that produces 6m error on one frame and 11m error
on the next frame at highway range cannot be made safe through calibration alone.

Finding 3 - This is physics, not an algorithm limitation.

The stereo depth formula is depth = (f x b) / disparity.
At large distances disparity is very small, just a few pixels.
A one-pixel matching error at 50m causes a 6 meter depth error.
No algorithm reliably matches pixels to sub-pixel accuracy at long range in all conditions.
Neural network depth estimation partially overcomes this by learning scene priors,
but the fundamental geometric constraint does not disappear.

________________________________________

## Why This Matters to the Industry

Waymo uses LiDAR as its primary ranging sensor and cameras for classification.
They do not rely on stereo geometry for depth beyond close range.

Tesla uses cameras only but compensates with massive training data, optical flow
across video frames, and neural networks that learn depth from context rather than
from stereo geometry alone. Pure SGM stereo is not their final architecture.

This project benchmarks the geometric baseline that motivates those architectural decisions.
Project 8 in this series builds the LiDAR-camera fusion pipeline that addresses
the limitation measured here.

________________________________________

## Run It Yourself

    git clone https://github.com/GVK-Engine/day-002-stereo-depth-analysis
    cd day-002-stereo-depth-analysis
    pip install -r requirements.txt

Single frame analysis:

    py -3.11 stereo_depth.py

Full 9-frame benchmark:

    py -3.11 benchmark.py

KITTI data download (free after registration):
https://www.cvlibs.net/datasets/kitti/raw_data.php
Download: 2011_09_26_drive_0001 synced+rectified data and calibration files.

________________________________________

## Project Structure

    day-002-stereo-depth-analysis/
    ├── stereo_depth.py      full pipeline on one frame with visualization
    ├── benchmark.py         9-frame averaged accuracy evaluation
    ├── results/             saved output visualizations
    ├── requirements.txt     Python dependencies
    └── README.md

________________________________________

## Stack

Python 3.11   OpenCV 4.x   NumPy   SciPy   KITTI Dataset

________________________________________

## Series Progress

    P1.1    LiDAR Obstacle Detection Pipeline            Complete
    P1.2    Stereo Camera Depth Analysis                 Complete
   
