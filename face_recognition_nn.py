import face_recognition
from sklearn.neural_network import MLPClassifier
import os
import pickle
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from face_recognition.face_recognition_cli import image_files_in_folder
import random
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Function to train the neural network
@ignore_warnings(category=ConvergenceWarning)
def train_nn(trainingDirectory, modelSavePath=None):
    # trainingDirectory is the path to the training directory
    # modelSavePath is the path to where you want to save the trained NN model
    # RETURNS** Trained NN Classifier

    encodings = []
    names = []
    train_dir = os.listdir(trainingDirectory)
    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir("trainCondensed/" + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file("trainCondensed/" + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " was skipped and can't be used for training")

    # Create, tune, and train the Neural network
    random.seed(1)
    nnModel = MLPClassifier()
    hidden_layers = [(150, 100, 50), (120, 80, 40), (100, 50, 30)]
    activation = ["identity", "logistic", "tanh", "relu"]
    solver = ["lbfgs", "sgd", "adam"]
    alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    learning_rate = ["constant", "invscaling", "adaptive"]
    params = {"hidden_layer_sizes": hidden_layers, "activation": activation, "solver": solver, "alpha": alpha,
              "learning_rate": learning_rate}
    nn_tuned = RandomizedSearchCV(nnModel, params, cv=3, random_state=1)
    nn_tuned.fit(encodings, names)

    if modelSavePath is not None:
        with open(modelSavePath, 'wb') as f:
            pickle.dump(nn_tuned, f)

    return nn_tuned


# Function to predict an image
def nn_predict(image_path, nn_clf=None, model_path=None, threshold=0.5):
    # image_path is the path to the image to be predicted
    # nn_clf: Neural network model to be used for the predcition, if not provided, model_path MUST be provided
    # model_path: path to the saved NN model. must be provided if nn_clf is not provided
    # threshold: probability threshold for which to predict "unknown"

    if nn_clf is None and model_path is None:
        raise Exception("Must supply NN classifier either through nn_clf or model_path")
    if nn_clf is None:
        with open(model_path, "rb") as f:
            nn_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(image_path)
    X_face_locations = face_recognition.face_locations(X_img)

    if len(X_face_locations) == 0:
        return []

    no = len(X_face_locations)
    # print("Number of faces detected:", no)

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    names = []

    for faces in range(no):
        test_image_enc = faces_encodings[faces]
        try:
            name = nn_clf.best_estimator_.predict_proba([test_image_enc])
        except:
            name = nn_clf.predict_proba([test_image_enc])

        # get index of the highest probability
        nameIdx = np.argmax(name)
        # get value of highest probability
        probability = name[0][nameIdx]
        # get name of person to be predicted
        name = nn_clf.classes_[nameIdx]

        # if probability is less than threshold, predict "unknown"
        if probability < threshold:
            name = "unknown"
        names.append(name)

    return names


# Function to calculate the accuracy of the predictions
def getScore(predictions, testDirectory, skip = None):
    # predictions: array of predictions
    # testDirectory: path to testDirectory where actual names are ( to be compared against predictions)
    # returns: tuple, with first position representing accuracy, and 2nd position contains an array of correct (actual) names
    actual = []
    for directory in os.listdir(testDirectory):
        if directory == "twoPeople":
            continue
        if skip is not None:
            if directory == skip:
                continue
        for image_file in image_files_in_folder(os.path.join(testDirectory, directory)):
            full_file_path = image_file
            X_img = face_recognition.load_image_file(full_file_path)
            X_face_locations = face_recognition.face_locations(X_img)
            if len(X_face_locations) == 0:
                continue
            actual.append(directory)

    correct = 0
    i = 0
    for name in predictions:
        if name == actual[i]:
            correct += 1
        i += 1
    return ((correct / len(actual), actual))


def get_unknown_score(predictions):
    correct = 0
    for name in predictions:
        if name == "unknown":
            correct += 1
    return correct / len(predictions)



    namesPredictedFinal = []
    testPath = "testCelebs/"
    modelPath = "trainedModels/trained_tuned_nn_model"
    thresholds = [0.75, 0.85, 0.95]
    accuracies = []
    unknownAccuracies = []
    knownAccuracies = []

    # STEP 1: Train NN classifier
    print("Training Neural Network...")
    #nn_model = train_nn("trainCondensed/", "trainedModels/trained_tuned_nn_model")
    print("Training Done!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    # Also find the most optimal threshold value
    for i in thresholds:
        namesPredicted = []
        unknownPredictions = []
        knownPredictions = []
        for directory in os.listdir(testPath):
            for image_file in image_files_in_folder(os.path.join(testPath, directory)):
                full_file_path = image_file

                # print("Looking for faces in {}".format(image_file))

                # Find all people in the image using a trained classifier model
                # Note: You can pass in either a classifier file name or a classifier model instance
                predictions = nn_predict(full_file_path, model_path=modelPath, threshold=i)

                # Print results on the console
                for name in predictions:
                    # print("Found :", name)
                    if directory == "twoPeople":
                        continue
                    if directory == "unknown":
                        unknownPredictions.append(name)
                    else:
                        knownPredictions.append(name)
                    namesPredicted.append(name)

        nn_accuracy = getScore(namesPredicted, testPath)[0]
        unknown_accuracy = get_unknown_score(unknownPredictions)
        known_accuracy = getScore(knownPredictions,testPath, skip="unknown")[0]
        print()
        print("THRESHOLD: ", i)
        print("Total accuracy: " + str(nn_accuracy))
        print("Unknown accuracy: " + str(unknown_accuracy))
        print("known accuracy: " + str(known_accuracy))
        accuracies.append(nn_accuracy)
        unknownAccuracies.append(unknown_accuracy)
        knownAccuracies.append(known_accuracy)


    #index = np.argmax(accuracies)
    #index2 = np.argmax(unknownAccuracies)
    #print("optimal threhsold: ", thresholds[index])
    #print("Accuracy: ", accuracies[index])
    #print("optimal threshold for UNKNOWN: ", thresholds[index2])
    #print("UNKNOWN ACCURACY: ", unknownAccuracies[index2])
