Explanation on how I used the give notebooks.

1) Audio_FeatureExtraction.ipynb:

    This is a Python script that extracts audio features using OpenSmile, which is a toolkit for audio analysis. It also converts .mp4 files to .wav files, annotates the .wav files, and stores the information in a CSV file.

    Here is an explanation of the different parts of the script:
      - The script imports necessary packages, such as os, sys, glob, csv, numpy, pandas, matplotlib, sklearn, and keras.
      - The next section of the script stores the path for videos from the "Bag of Lies" dataset. It uses the "walk" function to get all the files in a directory and its subdirectories that have a ".mp4" extension.
      - The script then converts all the .mp4 files from the above dataset into .wav files using FFmpeg.
      - After the .wav files are created, the script annotates them and stores the information in a CSV file. The CSV file contains the following columns:
        * Path_for_mp4_video: the path for the original .mp4 video file
        * Path_for_wav_file: the path for the .wav file
        * csv_file_name: the name of the CSV file that contains the annotated audio features
        * csv_file_name_path_fullvideo: the path for the CSV file that contains the annotated audio features for the entire video
        * csv_file_name_path_perframe: the path for the CSV file that contains the annotated audio features for each frame
        * label: the label for the video (either "deceptive" or "truthful")

    It then uses SMILExtract to extract all the audio features. To know more - https://audeering.github.io/opensmile/reference.html#smilextract-command-line-options
    Then all the .csv files are copied to a new folder.
    
2) Unimodal_Audio.ipynb:
  - This is a Python script for a deep learning project to detect deception using audio features.
  - Uses a sequential model in Keras. 
  - Model: It uses an LSTM layer as the input layer, followed by two fully connected layers, and finally an output layer with a sigmoid activation function.  The LSTM layer has 64 units and a recurrent dropout of 0.35, which helps prevent overfitting. The fully connected layers have 64 units each, with ReLU activation functions and a dropout rate of 0.2. Both layers also have a kernel regularizer with an L2 penalty of 0.001, which helps prevent overfitting by encouraging the model weights to be small. The optimizer used is stochastic gradient descent with a learning rate of 0.0001 and momentum of 0.9. The loss function is binary cross-entropy and the evaluation metric is accuracy.
  
