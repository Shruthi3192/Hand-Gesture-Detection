# Feature-Recognition-and-AI-Bot-Interaction
This repo uses open cv and tensor flow to communicate with Furhatos lab developed AI BOT


Annotated_Videos.py 
This file takes landmarks from csv file and annotates them on the video.

the csv file format is as below:

Column Content:
1      Source Video File name
       A string indicating the file name of the video being referenced.
       
2      Frame count within the video
       An integer indicating which frame of the video is being referenced
       
3      Camera Facing Side
       One of [dorsal, palm]
       
4      Gesture
       One of [open, fist, three]
       
5 - 44  Landmark Positions


In pixel (pairs of (x,y)). First all the palm side landmarks starting with the
root and moving towards the fingertips. The order of the fingers is thumb,
index, middle, ring, pinky finger. The same order is repeated for all the
landmarks of the dorsal side. If a landmark is not present it’s value is (0,0).
