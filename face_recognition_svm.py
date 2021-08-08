# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
from sklearn import svm
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
def trainSVM(trainingDirectory,modelSavePath = None):
    # trainingDirectory is the path to the training directory
    # modelSavePath is the path to where you want to save the trained svm model
    # RETURNS** Trained SVM Classifier

    encodings =[]
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

            #If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " was skipped and can't be used for training")

    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings,names)
    #If save path given, save model
    if modelSavePath is not None:
        with open(modelSavePath, 'wb') as f:
            pickle.dump(clf, f)
    return clf

# Load the test image with unknown faces into a numpy array
# test_image = face_recognition.load_image_file('test_image.jpg')

def predictImage(X_img_path, svm_clf = None, model_path = None):
    # x_img_path is the path to the image to be predicted
    # svm_clf is the SVM classifier to be passed in (if not specified, model_path MUST be specified
    # model_path is the path to the saved svm model, must be specified if svm_clf is not specified

    # Raise exception if image path is not compatible or if svm classifier is not provided
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    if svm_clf is None and model_path is None:
        raise Exception("Must supply svm classifier either through svm_clf or model_path")

    # load svm classifier if not directly provided
    if svm_clf is None:
        with open(model_path, "rb") as f:
            svm_clf = pickle.load(f)

    # load image
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # if no images are found, return empty
    if len(X_face_locations) == 0:
        return []

    no = len(X_face_locations)
    print("number of faces detected: ", no)
    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # predict each face in image
    names = []
    # probabilities = []
    #decisionOutput = []
    for faces in range(no):
        test_image_enc = faces_encodings[faces]
        try:
            name = svm_clf.best_estimator_.predict([test_image_enc])
            print("Best Estimator used")
        except:
            name = svm_clf.predict([test_image_enc])

        out = svm_clf.decision_function([test_image_enc])
        names.append(*name)
        # probabilities.append(prob)
        #decisionOutput.append(out)
    return names

def train_tuned_svm(trainingDirectory,modelSavePath = None):
   # trainingDirectory is the path to the training directory
    # modelSavePath is the path to where you want to save the trained svm model
    # RETURNS** Trained SVM Classifier
    train_dir = os.listdir(trainingDirectory)

    encodings =[]
    names = []
    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(trainingDirectory + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(trainingDirectory + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            #If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " was skipped and can't be used for training")

    # Create and train the SVC classifier
    clf = svm.SVC(decision_function_shape='ovo')
    # create distribution of parameters to tune for
    c = [0.001,0.01,0.1,1,10,100]
    kernel = ["linear","poly","rbf","sigmoid"]
    gamma = ["scale","auto"]
    shrinking = [True, False]
    params = {"C":c, "kernel":kernel,"gamma":gamma,"shrinking":shrinking}
    svm_clf = RandomizedSearchCV(clf,params,cv=3)

    svm_model = svm_clf.fit(encodings,names)
    #If save path given, save model
    if modelSavePath is not None:
        with open(modelSavePath, 'wb') as f:
            pickle.dump(svm_model, f)
    return svm_model

def getScore(predictions, testDirectory):

    actual = []
    for directory in os.listdir(testDirectory):
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
    return correct / len(actual)


def get_unknown_score(predictions, target = "unknown"):
    correct = 0
    for name in predictions:
        if name == target:
            correct += 1
    return correct / len(predictions)

if __name__ == "__main__":

    #train tuned svm model
    print("Training SVM Model...")
    #svm_tuned = train_tuned_svm("trainCondensed/","trained_tuned_svm_model") #only need to train once, then just use model path
    print("Training Done!")

    namesPredicted = []
    unknownPredictions = []
    knownPredictions = []
    testPath = "testCelebs/"
    modelPath = "trained_tuned_svm_model"
    # STEP 2: Using the trained classifier, make predictions for unknown images
    for directory in os.listdir(testPath):
        for image_file in image_files_in_folder(os.path.join(testPath, directory)):
            full_file_path = image_file

            print("Looking for faces in {}".format(image_file))

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predictImage(full_file_path, model_path=modelPath)

            # Print results on the console
            for name in predictions:
                print("Found :", name)
                if directory == "twoPeople":
                    continue
                if directory == "unknown":
                    unknownPredictions.append(name)
                else:
                    knownPredictions.append(name)

                namesPredicted.append(name)

    svm_accuracy = getScore(namesPredicted, testPath)
    unknown_accuracy = get_unknown_score(unknownPredictions)
    print("TOTAL ACCURACY: ", svm_accuracy)
    print("UNKNOWN ACCURACY: ", unknown_accuracy)
    print("KNOWN ACCURACY: ", get_unknown_score(knownPredictions, target=""))
