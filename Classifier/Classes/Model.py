############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2020
# Project:       AllDS2020 Classifier
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Model Helper Class
# Description:   Model helper functions for data the ALL Detection System 2020 (AllDS2020)
#                ALL Classifier.
# License:       MIT License
# Last Modified: 2020-03-05
#
############################################################################################

import cv2
import json
import os
import random
import requests
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from numpy.random import seed
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

from Classes.Helpers import Helpers


class Model():
    """ Model Class
    
    Model helper class for the Paper 1 Evaluation.
    """

    def __init__(self, optimizer, do_augmentation = False):
        """ Initializes the Model class. """

        self.Helpers = Helpers("Model", False)
        self.optimizer = optimizer
        self.do_augmentation = do_augmentation
        self.testing_dir = self.Helpers.confs["cnn"]["data"]["test"]
        self.valid = self.Helpers.confs["cnn"]["data"]["valid_types"]
        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        if self.do_augmentation == False:
            self.seed = self.Helpers.confs["cnn"]["data"]["seed_" + self.optimizer]
            self.weights_file = self.Helpers.confs["cnn"]["model"]["weights"]
            self.model_json = self.Helpers.confs["cnn"]["model"]["model"]
        else:
            self.seed = self.Helpers.confs["cnn"]["data"]["seed_" + self.optimizer + "_augmentation"]
            self.weights_file = self.Helpers.confs["cnn"]["model"]["weights_aug"]
            self.model_json = self.Helpers.confs["cnn"]["model"]["model_aug"]
            
        random.seed(self.seed)
        seed(self.seed)
        tf.random.set_seed(self.seed)
            
        self.Helpers.logger.info("Model class initialization complete.")

    def build_network(self, X_train,  X_test, y_train, y_test):
        """ Builds the network. 
        
        Replicates the networked outlined in the  Acute Leukemia Classification 
        Using Convolution Neural Network In Clinical Decision Support System paper
        using Tensorflow 2.0.

        https://airccj.org/CSCP/vol7/csit77505.pdf
        """
        
        if self.do_augmentation == False:
            self.val_steps = self.Helpers.confs["cnn"]["train"]["val_steps"]
            self.batch_size = self.Helpers.confs["cnn"]["train"]["batch"] 
            self.epochs = self.Helpers.confs["cnn"]["train"]["epochs"]
        else:
            self.val_steps = self.Helpers.confs["cnn"]["train"]["val_steps_augmentation"]
            self.batch_size = self.Helpers.confs["cnn"]["train"]["batch_augmentation"] 
            self.epochs = self.Helpers.confs["cnn"]["train"]["epochs_augmentation"]
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.ZeroPadding2D(
                padding=(2, 2), input_shape=self.X_train.shape[1:]),
            tf.keras.layers.Conv2D(30, (5, 5), strides=1,
                                   padding="valid", activation='relu'),
            tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
            tf.keras.layers.Conv2D(30, (5, 5), strides=1,
                                   padding="valid", activation='relu'),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Activation('softmax')
        ], 
        "AllCnn")
        self.model.summary()
        self.Helpers.logger.info("Network built")

    def compile_and_train(self):
        """ Compiles the Paper 1 Evaluation model. """

        if self.optimizer == "adam":            
            self.Helpers.logger.info("Using Adam Optimizer.")
            optimizer =  tf.keras.optimizers.Adam(lr=self.Helpers.confs["cnn"]["train"]["learning_rate_adam"], 
                                                  decay = self.Helpers.confs["cnn"]["train"]["decay_adam"])
            #optimizer =  tf.keras.optimizers.Adam()
        else:
            self.Helpers.logger.info("Using RMSprop Optimizer.")
            optimizer = tf.keras.optimizers.RMSprop(lr = self.Helpers.confs["cnn"]["train"]["learning_rate_rmsprop"], 
                                                    decay = self.Helpers.confs["cnn"]["train"]["decay_rmsprop"])
        
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'),
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.AUC(name='auc') ])

        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), 
                                      validation_steps=self.val_steps, epochs=self.epochs)

        print(self.history)
        print("") 
    
    def predictions(self):
        """ Makes predictions on the test set. """
        
        self.train_preds = self.model.predict(self.X_train)
        self.test_preds = self.model.predict(self.X_test)
        
        self.Helpers.logger.info("Training predictions: " + str(self.train_preds))
        self.Helpers.logger.info("Testing predictions: " + str(self.test_preds))
        print("")

    def evaluate_model(self):
        """ Evaluates the Paper 1 Evaluation model. """
        
        metrics = self.model.evaluate(self.X_test, self.y_test, verbose=0)        
        for name, value in zip(self.model.metrics_names, metrics):
            self.Helpers.logger.info("Metrics: " + name + " " + str(value))
        print()
        
    def visualize_metrics(self):
        """ Visualize our metrics. """
        
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim((0, 1))
        plt.legend(['Train', 'Validate'], loc='upper left')
        plt.savefig('Model/Plots/Accuracy.png')
        plt.show()
        
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validate'], loc='upper left')
        plt.savefig('Model/Plots/Loss.png')
        plt.show()
        
        plt.plot(self.history.history['auc'])
        plt.plot(self.history.history['val_auc'])
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validate'], loc='upper left')
        plt.savefig('Model/Plots/AUC.png')
        plt.show()
        
        plt.plot(self.history.history['precision'])
        plt.plot(self.history.history['val_precision'])
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validate'], loc='upper left')
        plt.savefig('Model/Plots/Precision.png')
        plt.show()
        
        plt.plot(self.history.history['recall'])
        plt.plot(self.history.history['val_recall'])
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validate'], loc='upper left')
        plt.savefig('Model/Plots/Recall.png')
        plt.show()
        
    def confusion_matrix(self):
        """ Prints/displays the confusion matrix. """
        
        self.matrix = confusion_matrix(self.y_test.argmax(axis=1), 
                                       self.test_preds.argmax(axis=1))
        
        self.Helpers.logger.info("Confusion Matrix: " + str(self.matrix))
        print("")
        
        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Confusion matrix ')
        plt.colorbar()
        plt.savefig('Model/Plots/Confusion-Matrix.png')
        plt.show()
            
    def figures_of_merit(self):
        """ Calculates/prints the figures of merit. 
        
        https://homes.di.unimi.it/scotti/all/
        """
        
        test_len = len(self.X_test)
        
        TP = self.matrix[1][1]
        TN = self.matrix[0][0]
        FP = self.matrix[0][1]
        FN = self.matrix[1][0]
        
        TPP = (TP * 100)/test_len
        FPP = (FP * 100)/test_len
        FNP = (FN * 100)/test_len
        TNP = (TN * 100)/test_len
        
        specificity = TN/(TN+FP) 
        
        misc = FP + FN        
        miscp = (misc * 100)/test_len 
        
        self.Helpers.logger.info("True Positives: " + str(TP) + "(" + str(TPP) + "%)")
        self.Helpers.logger.info("False Positives: " + str(FP) + "(" + str(FPP) + "%)")
        self.Helpers.logger.info("True Negatives: " + str(TN) + "(" + str(TNP) + "%)")
        self.Helpers.logger.info("False Negatives: " + str(FN) + "(" + str(FNP) + "%)")
        
        self.Helpers.logger.info("Specificity: " + str(specificity))
        self.Helpers.logger.info("Misclassification: " + str(misc) + "(" + str(miscp) + "%)")        
        
    def save_weights(self):
        """ Saves the model weights. """
            
        self.model.save_weights(self.weights_file)  
        self.Helpers.logger.info("Weights saved " + self.weights_file)
        
    def save_model_as_json(self):
        """ Saves the model to JSON. """
        
        with open(self.model_json, "w") as file:
            file.write(self.model.to_json())
            
        self.Helpers.logger.info("Model JSON saved " + self.model_json)
        
    def load_model_and_weights(self):
        """ Loads the model and weights. """
        
        with open(self.model_json) as file:
            m_json = file.read()
        
        self.AllModel = tf.keras.models.model_from_json(m_json) 
        self.AllModel.load_weights(self.weights_file)
            
        self.Helpers.logger.info("Model loaded ")
        
        self.AllModel.summary() 
        
    def test_classifier(self):
        """ Tests the trained model. """
        
        files = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for testFile in os.listdir(self.testing_dir):
            if os.path.splitext(testFile)[1] in self.valid:

                files += 1
                fileName = self.testing_dir + "/" + testFile

                img = cv2.imread(fileName).astype(np.float32)
                self.Helpers.logger.info("Loaded test image " + fileName)

                dx, dy, dz = img.shape
                delta = float(abs(dy-dx))

                if dx > dy:
                    img = img[int(0.5*delta):dx-int(0.5*delta), 0:dy]
                else:
                    img = img[0:dx, int(0.5*delta):dy-int(0.5*delta)]
                    
                img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim_augmentation"], 
                                       self.Helpers.confs["cnn"]["data"]["dim_augmentation"]))
                dx, dy, dz = img.shape
                input_data = img.reshape((-1, dx, dy, dz))
                
                predictions = self.AllModel.predict_proba(input_data)
                prediction = predictions[0]
                prediction  = np.argmax(prediction)
                prediction  = self.Helpers.confs["cnn"]["data"]["labels"][prediction]
                
                msg = ""
                if prediction == 1 and "_1." in testFile:
                    tp += 1
                    msg = "ALL correctly detected (True Positive)"
                elif prediction == 1 and "_0." in testFile:
                    fp += 1
                    msg = "ALL incorrectly detected (False Positive)"
                elif prediction == 0 and "_0." in testFile:
                    tn += 1
                    msg = "ALL correctly not detected (True Negative)"
                elif prediction == 0 and "_1." in testFile:
                    fn += 1
                    msg = "ALL incorrectly not detected (False Negative)"
                self.Helpers.logger.info(msg)
                    
        self.Helpers.logger.info("Images Classifier: " + str(files))
        self.Helpers.logger.info("True Positives: " + str(tp))
        self.Helpers.logger.info("False Positives: " + str(fp))
        self.Helpers.logger.info("True Negatives: " + str(tn))
        self.Helpers.logger.info("False Negatives: " + str(fn))

    def send_request(self, img_path):
        """ Sends image to the inference API endpoint. """

        self.Helpers.logger.info("Sending request for: " + img_path)
        
        _, img_encoded = cv2.imencode('.png', cv2.imread(img_path))
        response = requests.post(
            self.addr, data=img_encoded.tostring(), headers=self.headers)
        response = json.loads(response.text)
        
        return response

    def test_http_classifier(self):
        """ Tests the trained model via HTTP. """
        
        msg = ""
        result = ""
        
        files = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        self.addr = "http://" + self.Helpers.confs["cnn"]["api"]["server"] + \
            ':'+str(self.Helpers.confs["cnn"]["api"]["port"]) + '/Inference'
        self.headers = {'content-type': 'image/jpeg'}

        for data in os.listdir(self.testing_dir):
            if os.path.splitext(data)[1] in self.valid:
                
                response = self.send_request(self.testing_dir + "/" + data)
                
                msg = ""
                if response["Classification"] == 1 and "_1." in data:
                    tp += 1
                    msg = "ALL correctly detected (True Positive)"
                elif response["Classification"] == 1 and "_0." in data:
                    fp += 1
                    msg = "ALL incorrectly detected (False Positive)"
                elif response["Classification"] == 0 and "_0." in data:
                    tn += 1
                    msg = "ALL correctly not detected (True Negative)"
                elif response["Classification"] == 0 and "_1." in data:
                    fn += 1
                    msg = "ALL incorrectly not detected (False Negative)"
                
                files += 1
                
                self.Helpers.logger.info(msg)
                print()
                time.sleep(7)
                    
        self.Helpers.logger.info("Images Classifier: " + str(files))
        self.Helpers.logger.info("True Positives: " + str(tp))
        self.Helpers.logger.info("False Positives: " + str(fp))
        self.Helpers.logger.info("True Negatives: " + str(tn))
        self.Helpers.logger.info("False Negatives: " + str(fn))

    def http_classify(self, req):
        """ Classifies an image sent via HTTP. """
            
        if len(req.files) != 0:
            print("image read")
            img = np.fromstring(req.files['file'].read(), np.uint8)
        else:
            print("image data")
            img = np.fromstring(req.data, np.uint8)
            
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

        dx, dy, dz = img.shape
        delta = float(abs(dy-dx))

        if dx > dy:
            img = img[int(0.5*delta):dx-int(0.5*delta), 0:dy]
        else:
            img = img[0:dx, int(0.5*delta):dy-int(0.5*delta)]
            
        img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim_augmentation"], 
                                self.Helpers.confs["cnn"]["data"]["dim_augmentation"]))
        dx, dy, dz = img.shape
        input_data = img.reshape((-1, dx, dy, dz))
        
        predictions = self.AllModel.predict_proba(input_data)
        prediction = predictions[0]
        prediction  = np.argmax(prediction)
        
        return self.Helpers.confs["cnn"]["data"]["labels"][prediction]

    def vr_http_classify(self, img):
        """ Classifies an image sent via from VR via HTTP. """

        dx, dy, dz = img.shape
        delta = float(abs(dy-dx))

        if dx > dy:
            img = img[int(0.5*delta):dx-int(0.5*delta), 0:dy]
        else:
            img = img[0:dx, int(0.5*delta):dy-int(0.5*delta)]
            
        img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim_augmentation"], 
                                self.Helpers.confs["cnn"]["data"]["dim_augmentation"]))
        dx, dy, dz = img.shape
        input_data = img.reshape((-1, dx, dy, dz))
        
        predictions = self.AllModel.predict_proba(input_data)
        prediction = predictions[0]
        prediction  = np.argmax(prediction)
        
        return self.Helpers.confs["cnn"]["data"]["labels"][prediction]