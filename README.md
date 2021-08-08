# face_recognition
This is a facial recognition project using the face_recognition package in python, and uses the following models to train and predict faces:

K nearest Neighbors

Support Vector Machines

Neural Network

The KNN model performed the best on average across the three models on the given dataasets (database_1, database_2, database_3), which are 
subsets of data from the pubfig dataset from kaggle https://www.kaggle.com/kaustubhchaudhari/pubfig-dataset-256x256-jpg which contains in 
total ~ 59,000 images of celebrity faces.

About the models:

I wanted to create an accurate model off little input, so each of the models were trained on only 4 images of 60 celebrities. the test set
contains the same 60 celebrities, but different, unseen, images of those celebrities. The test set also contains 15 **"unknown"** celebrities, which
are celebrities that **were not** included in the training set. For these celebrities, the model should ouput "unknown" because they were not trained on these 
celebrities. SVM can not output "unknown" while KNN and NN can. In the end, I found that the KNN model was very promising with an average accuracy around
96-97% for both "unknown" classification and "known" claasification (celebrities in the training set)
