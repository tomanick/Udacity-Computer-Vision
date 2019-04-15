# Project2 Landmark Detection and Tracking (SLAM)

## Project overview
In this project, I'll implement SLAM (Simultaneous Localization and Mapping) for a 2 dimensional world! 
I’ll combine what I know about robot sensor measurements and movement to create a map of an environment from only sensor and motion data gathered by a robot, over time. 
SLAM gives me a way to track the location of a robot in the world in real-time and identify the locations of landmarks such as buildings, 
trees, rocks, and other world features. This is an active area of research in the fields of robotics and autonomous systems.

Below is an example of a 2D robot world with landmarks (purple x's) and the robot (a red 'o') located and found using only sensor and motion data collected by that robot. 
This is just one example for a 50x50 grid world; in your work you will likely generate a variety of these maps.
![image](https://github.com/tomanick/Udacity-Computer-Vision/blob/master/Project3_landmark_detection_and_tracking(slam)/images/robot_world.png)

## Project files
The project will be broken up into a few main parts in four Python notebooks:

Notebook 1 : Robot Moving and Sensing

Notebook 2 : Omega and Xi, Constraints

Notebook 3 : Landmark Detection and Tracking

robot_class.py: The CNN-RNN architecture
