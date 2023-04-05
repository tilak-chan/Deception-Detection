# Deception-Detection
Building a Machine Learning Model to detect if a person is lying or not. Dataset used: Bag-Of-Lies

The mmodalities in the Datasets are Video, Audio, EEG and Gaze Data. 

EEG:

Using 'Annotations.csv', I was able to get the first 1000 datapoints from each user's 'EEG.csv' file. I concatenated all the values in a single .csv file with approximately 3 lakh data points and then started to pre process the data. 
- Removed the rows with 'Quality'
- Removed the timestamp
- Added a row called Truth with the truth values: 
    - '0' => False
    - '1' => True

Then tried out different ML models like RandomForest, Decision Tree Classifier, SVM etc.
Got an accuracy of 90% using the Random Forest Classifier.
