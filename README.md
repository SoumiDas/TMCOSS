# Thresholded Multi-Criteria Online Subset Selection

*Paper*: TMCOSS: Thresholded Multi-Criteria Online Subset Selection for Data-Efficient Autonomous Driving

*Authors*: Soumi Das, Harikrishna Patibandla, Suparna Bhattacharya, Kshounis Bera, Niloy Ganguly, Sourangshu Bhattacharya

# Prerequisites

* Python, NumPy, cvxpy, PyTorch, scikit-learn, Matplotlib
* Data and its corresponding model

# Steps to execute the algorithm

1. Divide your data into existing and incoming sets.
2. Train your target model on the current existing set and find the loss values on the entire training data.
3. Compute features of the current existing set and the next incoming set using any pretrained model features.
4. Find neighbours from the current existing set.
5. Similarity computation between the current existing and incoming set.
6. Subset finding using similarity value pairs and loss values of the current existing and incoming set.
7. The obtained subset becomes the current existing set for the next round.
8. Steps 2-7 are to be repeated till the subset is found on the entire dataset.
