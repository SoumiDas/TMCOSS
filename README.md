# Thresholded Multi-Criteria Online Subset Selection

*Paper*: TMCOSS: Thresholded Multi-Criteria Online Subset Selection for Data-Efficient Autonomous Driving

*Authors*: Soumi Das, Harikrishna Patibandla, Suparna Bhattacharya, Kshounis Bera, Niloy Ganguly, Sourangshu Bhattacharya

# Prerequisites

* Python, NumPy, OpenCV, cvxpy, PyTorch, scikit-learn, Matplotlib
* Data and its corresponding model

# Usage

1. Divide your data into existing and incoming sets.
2. Train your target model on the current existing set and find the loss values on the entire training data.
3. Compute features of the current existing set and the next incoming set using any pretrained model features (here VGG). (```feature_compute.py```)
4. Find neighbours from the current existing set. (```neighbours.py , fix_nb.py```)
5. Similarity computation between the current existing and incoming set. (```sift_compute.py```)
6. Subset finding using similarity value pairs and loss values of the current existing and incoming set.(```subset_find.py```)
7. The obtained subset becomes the current existing set for the next round.
8. Steps 2-7 are to be repeated till the subset is found on the entire dataset.
9. Now, you can train the model using the subset and obtain the performance metrics.

# Sample run
One can run ```python sample_subset.py``` on the sample data and obtain the subset. 

Pre-given: 

1. Sample data
2. The loss values obtained using the driving model.
3. Neighbour files from existing set.
4. Similarity values for the sample data.
