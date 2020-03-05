############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2020
# Project:       AllDS2020 Classifier
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         AllDS2020 Class
# Description:   Core wrapper class for the Tensorflow 2.0 AllDS2020 classifier.
# License:       MIT License
# Last Modified: 2020-03-05
#
############################################################################################

import sys

from Classes.Helpers import Helpers
from Classes.Data import Data as Data
from Classes.Model import Model as Model


class AllDS2020():
    """ AllDS2020 Wrapper Class

    Core wrapper class for the Tensorflow 2.0 AllDS2020 classifier.
    """

    def __init__(self):

        self.Helpers = Helpers("Core")
        self.optimizer = "Adam"
        self.mode = "Local"
        self.do_augmentation = True

    def do_data(self):
        """ Creates/sorts dataset. """
    
        self.Data = Data(self.optimizer, self.do_augmentation)
        self.Data.data_and_labels_sort()
        
        if self.do_augmentation == False:
            self.Data.data_and_labels_prepare()
        else:
            self.Data.data_and_labels_augmentation_prepare()
        
        self.Data.shuffle()
        self.Data.get_split()

    def do_model(self):
        """ Creates & trains the model. 
        
        Replicates the networked and data splits outlined in the  Acute Leukemia Classification 
        Using Convolution Neural Network In Clinical Decision Support System paper
        using Tensorflow 2.0.

        https://airccj.org/CSCP/vol7/csit77505.pdf
        """

        self.Model = Model(self.Data.X_train, self.Data.X_test, 
                           self.Data.y_train, self.Data.y_test, 
                           self.optimizer, self.do_augmentation)
        
        self.Model.build_network()
        self.Model.compile_and_train()
        
        self.Model.save_model_as_json()
        self.Model.save_weights()

    def do_evaluate(self):
        """ Predictions & Evaluation """

        self.Model.predictions()
        self.Model.evaluate_model()

    def do_metrics(self):
        """ Predictions & Evaluation """
        self.Model.plot_metrics()
        
        self.Model.confusion_matrix()
        self.Model.figures_of_merit()


AllDS2020 = AllDS2020()


def main():
    
    server = sys.argv[2].lower()
    
    if server == "adam" or server == "rmsprop":
        AllDS2020.optimizer = server
    else:
        print("Optimizer not supported")
        exit()
        
    if sys.argv[1] == "Server" or sys.argv[1] == "Train" or sys.argv[1] == "Classify":
        AllDS2020.mode = sys.argv[1]
    else:
        print("Mode not supported! Server, Train or Classify")
        exit()
        
    if sys.argv[3] == 'True':
        AllDS2020.do_augmentation = True
    else:
        AllDS2020.do_augmentation = False
        
    AllDS2020.do_data()
        
    if AllDS2020.mode == "Train":
        """ Creates and trains the classifier """
        
        AllDS2020.do_model()
        AllDS2020.do_evaluate()
        AllDS2020.do_metrics()
        
    if AllDS2020.mode == "Server":
        """ Runs the classifier in server mode
        
        Runs the classifier in server mode and provides 
        an endpoint, exposing the classifier."""
        
        print("Not implemented")
        exit()
        
    if AllDS2020.mode == "Classify":
        """ Runs the classifier locally
        
        Runs the classifier in local mode."""
        
        print("Not implemented")
        exit()


if __name__ == "__main__":
    main()
