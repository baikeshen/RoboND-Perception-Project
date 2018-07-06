
# Project: Perception Pick & Place
### In this lesson, we practiced using several image process techniques to extract perception information and recognize objects from images. Then, with the object information, we can command the robot to perform pick and place motion accordingly.

#### This project is separated into two sections - the exercise and the project itself.

#### The main focus on the exercise part is to build the perception functions step by step. Here, we started from downsizing the image grid, reducing focus area, segmentation, RANASAC plane fitting, clustering, and then use a small machine learning technique to train the object features for object recognition.

#### In the project, we implemented the perception functions to a well-prepared pick & place testing cell with a PR2 robot to test the object recognition function. We also used the same machine learning technique to train the potential objects in advance and let the robot to recognize and pick up the objects.

[//]: # (Image References)

[image1]: ./misc_images/voxel_downsample.JPG
[image2]: ./misc_images/pass_through_filter.JPG
[image3]: ./misc_images/exercise_2_objects.JPG
[image4]: ./misc_images/exercise_2_cluster.JPG
[image5]: ./misc_images/exercise_3_capture.JPG
[image6]: ./misc_images/exercise_3_SVM_training.JPG
[image7]: ./misc_images/exercise_3_object_recognition.JPG
[image8]: ./misc_images/train_svm_graph.JPG
[image9]: ./misc_images/recognition_rviz_1.JPG
[image10]: ./misc_images/recognition_rviz_2.JPG
[image11]: ./misc_images/recognition_rviz_3.JPG
[image12]: ./misc_images/original_table.JPG
[image13]: ./misc_images/exercise_1_inliers.JPG
[image14]: ./misc_images/exercise_1_outliers.JPG

# [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
The rubric points will be considered individually and each point will be addressed in my implementation.  

---

## Exercise 1, 2 and 3 Pipeline Implemented:
In the RoboND-Perception-Exercise repository, a static set of image data are provided to practice implementing the perception pipeline. In the first two exercises, we implemented several perception filtering functions to segment RGB images. Then in the exercise 3, we use Support Vector Machine (SVM) to perform object recognition.


The original Table is showed as below:
![Original Table][image12]

These functions include:

### 1. VoxelGrid Downsamplying Filter
This function is used to reduce the sampling amount in the imaging process. In the picture below, the voxel grid size is set to 0.01 cubic meter.

```
vox = cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.01
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
```

Voxel_downsample.JPG 
![alt text][image1]

### 2. PassThrough Filter
This function is used to filter out and keep only image data from certain area in space. The following codes are for setting up a filter in Z-axis. In the project, three separated filters are used to limit the space in X, Y and Z directions.

```
# PassThrough filter
# Create a PassThrough filter object.
passthrough = cloud_filtered.make_passthrough_filter()

# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.2
passthrough.set_filter_limits(axis_min, axis_max)
```

the Pass Through Filter section to select a region of interest from my Voxel Downsample Filtered point cloud is showned as below.
![alt text][image2]


### 3. RANASAC Plane Fitting
To identify points belong to a particular model from the entire image dataset.
The table fits the rectangle plane and is extracted from the dataset from all the objects above or below it.

```
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.01
seg.set_distance_threshold(max_distance)
```
![alt text][image13]

### 4. Outlier Removal Filter
To remove noise effect due to external factors. Since the provided image has no noise, there is no need to implemented in the exercise. But this filter will be used in the Perception project.

### 5. Euclidean Clustering
Clustering is a process to find similarities among individual points.
Density-Based Spatial Clustering of Applications with Noise (DBSACN), also known as "Euclidean Clustering", is used in this exercise and the following project for clustering points based on their density or distance between point in a cluster.

```sf
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.05)
ec.set_MinClusterSize(10)
ec.set_MaxClusterSize(1000)
```
Here is the entire point cloud published in ROS in rviz.
![alt text][image3]

Here is the point cloud published to the ROS after clustering.
![alt text][image4]


### 6. Support Vector Machine (SVM)
SVM is a supervised machine learning algorithm to be used to trained for object recognition.

In this exercise, we first run the file `capture_feature.py` to capture the objects' features in rviz environment, and then run `train_svm.py ` to train SVM. The following pictures show the recognition result after training the SVM with 10 orientations for all of the seven objects and with using `hsv` instead of `xyz`. As the result shows, the success rate is around 70~80 %. Later in the project, I tried to capture 150 orientations for the 8 objects and could reach 95 % success rate. However, it took morethan one hour to finish the training.

## Pick and Place Setup

### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

For the project, I reset the values for the filtering, RANSAC, and segmentation, given that the values from the previous exercises were not on par with the new setup. After recalibrating those values, I proceeded on capturing the features (capture_features.py) and training the model (see train_svm.py results in **Figure**  **8**).

![alt text][image6]
###### **Figure**  **8** : Project Confusion Matrix Results

Due the project requiring to recognize more objects for pick_list_3, I decided to perform the capture_features.py job using a 50 iterations loop. This made the training_set.sav data to be more accurate and producing better model.sav data for object recognition efforts. 

The following images display the 3 object recognition tasks with the 3 different pick_list yaml files. Managed to obtain a 100% object recognition result for 2 out of 3 pick_list (miss one object for pick_list_3.yaml).

![alt text][image9]
###### **Figure**  **9** : Object Recognition with pick_list_1.yaml

![alt text][image10]
###### **Figure**  **10** : Object Recognition with pick_list_2.yaml

![alt text][image11]
###### **Figure**  **11** : Object Recognition with pick_list_3.yaml


### 2. Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.

Most of the effort was already performed for the Perception Exercises, therefore; aside from testing out new filter, RANSAC, and segmentation values; the output files were the main concern. To obtain the pick_pose I followed the instructions from the [Output Yaml Files Lesson] by obtaining the object cloud centroid. 

For the place_pose and arm_name, I used the following code which is based on the group (color) from the object_list and then deciding which arm to use based on the dropbox where each object was supposed to be placed.

    # Used the groups color information to select which arm to use and which location to place the object
    if object_group == 'green':
        arm_name.data = 'right'
        place_pose.position.x = dropbox_param[1]['position'][0]
        place_pose.position.y = dropbox_param[1]['position'][1]
        place_pose.position.z = dropbox_param[1]['position'][2]
    else:
        arm_name.data = 'left'
        place_pose.position.x = dropbox_param[0]['position'][0]
        place_pose.position.y = dropbox_param[0]['position'][1]
        place_pose.position.z = dropbox_param[0]['position'][2]

One of the troubles I came across, was that the model was mistakenly recognizing the colored edges of the boxes as objects, therefore, I decided to use another passthrough filter to narrow it's vision. 

    passthrough = cloud_filtered.make_passthrough_filter()

    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min,axis_max)
    cloud_filtered = passthrough.filter()
	
Given more time, I would pursue a full pick_place_routine by using techniques learned from the Kinematics Lessons. This is a very interesting topic and I'm hoping to look further into other possible techniques within computer vision and more complex object recognition.

