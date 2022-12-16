import argparse
import numpy as np
import data
import network
import tqdm
import matplotlib.pyplot as plt

# import image

def binary_class_encoder(data_set, class_a, class_b):
    (X, Y) = data_set
    binary_class_index = np.append(np.where(Y == class_a),
                                   np.where(Y == class_b))  # Create an array containg index of where class = a or b
    bin_X = X[binary_class_index, :]
    bin_Y = Y[binary_class_index]
    bin_Y = bin_Y == class_a  # Encode class as either 0 or 1 from 0 to 9 class labelling
    return (bin_X, bin_Y)
def binary_classification(X_train,y_train,hyperparameter,a=0,b=5):
    batch_size = hyperparameter.batch_size
    epochs = hyperparameter.epochs
    k_folds = hyperparameter.k_folds

    training_accuracy_over_ephoc = np.zeros(epochs)
    training_loss_over_ephoc = np.zeros(epochs)
    validation_accuracy_over_ephoc = np.zeros(epochs)
    validation_loss_over_ephoc = np.zeros(epochs)
    (binary_x, binary_y) = binary_class_encoder((X_train,y_train),a,b)
    binary_x = data.append_bias(binary_x) # Adding bias term
    network_list = np.array([])

    for set in data.generate_k_fold_set((binary_x, binary_y),k_folds): # Cross Validation
        ((cv_train_x,cv_train_y),(cv_validation_x , cv_validation_y)) = set

        logistic = network.Network(hyperparameter,network.sigmoid,network.binary_cross_entropy,1) # Reinitialization of class, so weighting get reset for every fold
        """
        Save logistic object for picking the best network
        """
        training_loss = np.array([])
        training_accuracy = np.array([])
        validation_loss = np.array([])
        validation_accuracy = np.array([])
        for k in tqdm.trange(epochs,desc='Binary Classification'):
            (cv_train_x, cv_train_y) = data.shuffle((cv_train_x, cv_train_y))
            training_batch_accuracy = np.array([])
            training_batch_loss = np.array([])
            validation_batch_loss = np.array([])
            validation_batch_accuracy = np.array([])
            for X, t in data.generate_minibatches((cv_train_x,cv_train_y),batch_size = batch_size):
                single_train_loss , single_train_accuracy = logistic.train((X,np.reshape(t,(-1,1))))
                training_batch_loss = np.append(training_batch_loss,single_train_loss)
                training_batch_accuracy = np.append(training_batch_accuracy,single_train_accuracy)
            for X, t in data.generate_minibatches((cv_validation_x , cv_validation_y),batch_size = batch_size):
                # Doing validation in by mini-batches becuase the test function is written to take mini-batches
                single_validation_loss , single_validation_accuracy = logistic.test((X , np.reshape(t,(-1,1))))
                validation_batch_loss = np.append(validation_batch_loss,single_validation_loss)
                validation_batch_accuracy = np.append(validation_batch_accuracy,single_validation_accuracy)
            training_loss = np.append(training_loss,np.mean(training_batch_loss))
            training_accuracy = np.append(training_accuracy,np.mean(training_batch_accuracy))
            validation_loss = np.append(validation_loss,np.mean(validation_batch_loss))
            validation_accuracy = np.append(validation_accuracy,np.mean(validation_batch_accuracy))
        training_loss_over_ephoc = np.vstack((training_loss_over_ephoc,training_loss))
        training_accuracy_over_ephoc = np.vstack((training_accuracy_over_ephoc,training_accuracy))

        validation_loss_over_ephoc = np.vstack((validation_loss_over_ephoc,validation_loss))
        validation_accuracy_over_ephoc = np.vstack((validation_accuracy_over_ephoc,validation_accuracy))

        network_list = np.append(network_list, logistic)
    best_network_index = np.argmax(validation_accuracy_over_ephoc[:, -1])  # Getting the index of network w/ best accuracy
    best_network = network_list[best_network_index-1]

    training_loss_over_ephoc = np.mean(training_loss_over_ephoc[1:,:], axis = 0)
    training_accuracy_over_ephoc = np.mean(training_accuracy_over_ephoc[1:,:], axis = 0)

    validation_loss_over_ephoc = np.mean(validation_loss_over_ephoc[1:,:], axis =0)
    validation_accuracy_over_ephoc = np.mean(validation_accuracy_over_ephoc[1:,:], axis = 0)
    return (training_loss_over_ephoc, validation_loss_over_ephoc,
            training_accuracy_over_ephoc, validation_accuracy_over_ephoc,best_network)

def multi_class_classification(X_train,y_train,hyperparameter):
    batch_size = hyperparameter.batch_size
    epochs = hyperparameter.epochs
    k_folds = hyperparameter.k_folds
    training_accuracy_over_ephoc = np.zeros(epochs) # k fold average training accuracy over ephoc Shape (epochs , 0)
    training_loss_over_ephoc = np.zeros(epochs) # k fold average training accuracy over ephoc Shape (epochs , 0)
    validation_accuracy_over_ephoc = np.zeros(epochs)
    validation_loss_over_ephoc = np.zeros(epochs)
    network_list = np.array([])

    for set in data.generate_k_fold_set((X_train,y_train),k_folds):
        ((cv_train_x,cv_train_y),(cv_validation_x , cv_validation_y)) = set

        cv_train_x = data.append_bias(cv_train_x) # Adding bias term
        cv_validation_x = data.append_bias(cv_validation_x)
        cv_validation_y = data.onehot_encode(cv_validation_y)
        cv_train_y = data.onehot_encode(cv_train_y)
        # Reinitialization of class, so weighting get reset for every fold
        Soft_max = network.Network(hyperparameter,network.softmax,network.multiclass_cross_entropy,10)
        training_loss = np.array([])
        training_accuracy = np.array([])
        validation_loss = np.array([])
        validation_accuracy = np.array([])
        for k in tqdm.trange(epochs,desc='Multiclass Classification'):
            (cv_train_x, cv_train_y) = data.shuffle((cv_train_x, cv_train_y))
            training_batch_accuracy = np.array([])
            training_batch_loss = np.array([])
            validation_batch_loss = np.array([])
            validation_batch_accuracy = np.array([])
            for X, t in data.generate_minibatches((cv_train_x,cv_train_y),batch_size = batch_size):
                single_train_loss , single_train_accuracy = Soft_max.train((X,t))
                training_batch_loss = np.append(training_batch_loss,single_train_loss)
                training_batch_accuracy = np.append(training_batch_accuracy,single_train_accuracy)
            for X, t in data.generate_minibatches((cv_validation_x , cv_validation_y),batch_size = batch_size):
                single_validation_loss , single_validation_accuracy = Soft_max.test((X ,t))
                validation_batch_loss = np.append(validation_batch_loss,single_validation_loss)
                validation_batch_accuracy = np.append(validation_batch_accuracy,single_validation_accuracy)
            training_loss = np.append(training_loss,np.mean(training_batch_loss))
            training_accuracy = np.append(training_accuracy,np.mean(training_batch_accuracy))
            validation_loss = np.append(validation_loss,np.mean(validation_batch_loss))
            validation_accuracy = np.append(validation_accuracy,np.mean(validation_batch_accuracy))

        training_loss_over_ephoc = np.vstack((training_loss_over_ephoc,training_loss))
        training_accuracy_over_ephoc = np.vstack((training_accuracy_over_ephoc,training_accuracy))

        validation_loss_over_ephoc = np.vstack((validation_loss_over_ephoc,validation_loss))
        validation_accuracy_over_ephoc = np.vstack((validation_accuracy_over_ephoc,validation_accuracy))

        network_list = np.append(network_list, Soft_max)
    # We need to choose the best network across all folds:
    best_network_index = np.argmax(validation_accuracy_over_ephoc[:, -1])
    # Getting the index of network w/ best accuracy
    best_network = network_list[best_network_index-1]

    training_loss_over_ephoc = np.mean(training_loss_over_ephoc[1:,:], axis = 0)
    training_accuracy_over_ephoc = np.mean(training_accuracy_over_ephoc[1:,:], axis = 0)

    validation_loss_over_ephoc = np.mean(validation_loss_over_ephoc[1:,:], axis =0)
    validation_accuracy_over_ephoc = np.mean(validation_accuracy_over_ephoc[1:,:], axis = 0)
    return (training_loss_over_ephoc, validation_loss_over_ephoc,
            training_accuracy_over_ephoc, validation_accuracy_over_ephoc,best_network)

    """
    This function takes input:
    1. Training data set
    2. Hyperparameters
    
    Output:
    1. losses for all training folds & validation folds
    2. test accuracy for all training folds
    3. Network on fold with best accuracy
    """

def tune_hyperparameter(X_train, y_train):
    batch_size_set = np.array([64,128,512,1024])
    epochs_set = np.array([50,100,200,400])
    learning_rate_set = np.array([0.0001,0.001,0.01,0.1])
    normalizer_set = np.array(['z-score', 'min-max'])
    k_fold_set = np.array([3,5,10,20])

    results = np.zeros(8)

    with tqdm.tqdm(total=100,desc='Tuning:') as pbar:
        for bs in batch_size_set:
            for ep in epochs_set:
                for lr in learning_rate_set:
                    for norm in normalizer_set:
                        for kf in k_fold_set:
                            if norm == 'z-score':
                                (X_tune,norm_1,norm_2) = data.z_score_normalize(X_train)
                            else:
                                (X_tune, norm_1, norm_2) = data.min_max_normalize(X_train)
                            hyperparameters.batch_size = bs
                            hyperparameters.epochs = ep
                            hyperparameters.learning_rate = lr
                            hyperparameters.k_folds = kf
                            (plane_dog_training_loss, plane_dog_validation_loss,
                             plane_dog_training_accuracy, plane_dog_validation_accuracy, plane_dog_logistic) = binary_classification(
                                X_tune, y_train, hyperparameters, a=0, b=5)  # a ,b are class indices
                            (cat_dog_training_loss, cat_dog_validation_loss,
                             cat_dog_training_accuracy, cat_dog_validation_accuracy, cat_dog_logistic) = binary_classification(
                                X_tune, y_train, hyperparameters, a=3, b=5)
                            (multi_class_training_loss, multi_class_validation_loss,
                             multi_class_training_accuracy, multi_class_validation_accuracy,
                             Soft_max) = multi_class_classification(X_tune, y_train, hyperparameters)
                            results = np.vstack((results,np.array([bs,ep,lr,norm,kf,
                                                                   plane_dog_validation_accuracy[-1], # Getting last epoch accuracy
                                                                   cat_dog_validation_accuracy[-1],
                                                                   multi_class_validation_accuracy[-1],
                                                                   ])))
                            pbar.update(10) # Progress bar for every set of hyperpara
    return results
def main(hyperparameters=None):
    batch_size = hyperparameters.batch_size
    epochs = hyperparameters.epochs
    (X_train, y_train) = data.shuffle(data.load_data()) # Loading training data
    (X_test, y_test) = data.shuffle(data.load_data(train=False)) # Loading testing data
    import matplotlib.pyplot as plt


    # def unique_image(X,y):
    #     plt.ion()
    #     for i in range(10):
    #         index = np.where(y == i)[0][1] # retrieve index for each class
    #         plt.figure(i)
    #         plt.imshow(np.reshape(X[index,:],(32,32))) # show corresponding image
    # unique_image(X_train,y_train)

    if hyperparameters.tune_hyperparameter:
        best_hyperparameter = tune_hyperparameter(X_train, y_train)
        print(best_hyperparameter)

    (X_train,X_train_mu,X_train_sd) = data.z_score_normalize(X_train) # Normalization

    (X_test,X_test_max,X_test_min) = data.z_score_normalize(X_test) # Normalization
    X_test = data.append_bias(X_test) # Appending Bias
    #
    # (binary_training_loss,binary_validation_loss,
    #  binary_training_accuracy, binary_validation_accuracy,logistic) = binary_classification(X_train,y_train,hyperparameters,a=0,b=5) # a ,b are class indices
    #
    # (X_binary_test, y_binary_test) = binary_class_encoder((X_test, y_test),0,5) # Extract class 0,5 data and convert labels into 0 1
    # (binary_test_loss, binary_test_accuracy) = logistic.test((X_binary_test, np.reshape(y_binary_test,(-1,1))))
    # binary_test_loss = np.repeat(binary_test_loss, epochs)
    # print("Test_set Accuracy on Plane & Dogs = {}".format(binary_test_accuracy))
    # binary_test_accuracy = np.repeat(binary_test_accuracy, epochs)
    # plt.figure("Plane & Dog Classification Average Loss bs={} e={} lr={} kf={}".format(
    #     hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    # plt.plot(binary_training_loss,label = 'training_loss')
    # plt.plot(binary_validation_loss, label = 'validation_loss')
    # plt.plot(binary_test_loss, label = 'test_set_loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Binary Cross Entropy')
    # plt.legend()
    # plt.title("Plane & Dog Classification Average Loss bs={} e={} lr={} kf={}".format(
    #     hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    # plt.figure("Plane & Dog Classification Average Accuracy bs={} e={} lr={} kf={}".format(
    #     hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    # plt.plot(binary_training_accuracy, label = 'training_accuracy')
    # plt.plot(binary_validation_accuracy, label = 'validation_accuracy')
    # plt.plot(binary_test_accuracy, label='test_set_accuracy')
    # plt.title("Plane & Dog Classification Average Accuracy bs={} e={} lr={} kf={}".format(
    #     hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    # plt.xlabel('Epochs')
    # plt.ylabel('Average Classification Accuracy')
    # plt.legend()
    # plt.figure("Plane & Dog Network Weight Visualization")
    # plt.imshow(np.reshape(logistic.weights[1:],(32,32))) # Visualizing Weights

    # plt.figure('Hadamard product') # Meaningless
    # plt.imshow(np.reshape(logistic.weights[1:,0] * (X_train[np.where(y_train == 5)[0][1], :] * X_train_sd + X_train_mu),(32,32)))

    # (binary_training_loss,binary_validation_loss,
    #  binary_training_accuracy, binary_validation_accuracy,logistic) = binary_classification(X_train,y_train,hyperparameters,a=3,b=5)
    #
    # (X_binary_test, y_binary_test) = binary_class_encoder((X_test, y_test),3,5)
    # (binary_test_loss, binary_test_accuracy) = logistic.test((X_binary_test, np.reshape(y_binary_test,(-1,1))))
    # binary_test_loss = np.repeat(binary_test_loss, epochs)
    # print("Test_set Accuracy on Cat & Dogs = {}".format(binary_test_accuracy))
    # binary_test_accuracy = np.repeat(binary_test_accuracy, epochs)
    # plt.figure("Cat & Dog Classification Average Loss bs={} e={} lr={} kf={}".format(
    #     hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    # plt.plot(binary_training_loss,label = 'training_loss')
    # plt.plot(binary_validation_loss, label = 'validation_loss')
    # plt.plot(binary_test_loss, label = 'test_set_loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Binary Cross Entropy')
    # plt.legend()
    # plt.title("Cat & Dog Classification Average Loss bs={} e={} lr={} kf={}".format(
    #     hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    # plt.figure("Cat & Dog Classification Average Accuracy bs={} e={} lr={} kf={}".format(
    #     hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    # plt.plot(binary_training_accuracy, label = 'training_accuracy')
    # plt.plot(binary_validation_accuracy, label = 'validation_accuracy')
    # plt.plot(binary_test_accuracy, label='test_set_accuracy')
    # plt.title("Cat & Dog Classification Average Accuracy bs={} e={} lr={} kf={}".format(
    #     hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    # plt.xlabel('Epochs')
    # plt.ylabel('Average Classification Accuracy')
    # plt.legend()
    # plt.figure("Cat & Dog Network Weight Visualization")
    # plt.imshow(np.reshape(logistic.weights[1:],(32,32)))
    #
    (multi_class_training_loss, multi_class_validation_loss,
     multi_class_training_accuracy, multi_class_validation_accuracy,Soft_max) = multi_class_classification(X_train,y_train,hyperparameters)

    y_multi_class_test = data.onehot_encode(y_test)
    (multi_class_test_loss, multi_class_test_accuracy) = Soft_max.test((X_test,y_multi_class_test))
    print("Test_set Accuracy on Multiclass = {}".format(multi_class_test_accuracy))
    multi_class_test_loss = np.repeat(multi_class_test_loss, epochs)
    multi_class_test_accuracy = np.repeat(multi_class_test_accuracy, epochs)
    plt.figure("Multiclass Classification Average Loss bs={} e={} lr={} kf={}".format(
        hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    plt.plot(multi_class_training_loss,label = 'training_loss')
    plt.plot(multi_class_validation_loss, label = 'validation_loss')
    plt.plot(multi_class_test_loss,label = 'test_set_loss')
    plt.title("Multiclass Classification Average Loss bs={} e={} lr={} kf={}".format(
        hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    plt.xlabel('Epochs')
    plt.ylabel('Multiclass Cross Entropy')
    plt.legend()
    plt.figure("Multiclass Classification Average Accuracy bs={} e={} lr={} kf={}".format(
        hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    plt.plot(multi_class_training_accuracy, label = 'training_accuracy')
    plt.plot(multi_class_validation_accuracy, label = 'validation_accuracy')
    plt.plot(multi_class_test_accuracy, label = 'test_set_accuracy')
    plt.title("Multiclass Classification Average Accuracy bs={} e={} lr={} kf={}".format(
        hyperparameters.batch_size, hyperparameters.epochs, hyperparameters.learning_rate, hyperparameters.k_folds))
    plt.xlabel('Epochs')
    plt.ylabel('Multiclass Classification Accuracy')
    plt.legend()
    for i in range(10):
        plt.figure("Class {} Network Weight Visualization".format(i))
        plt.imshow(np.reshape(Soft_max.weights[1:,i], (32, 32)))
        # plt.figure('Hadamard product Class {}'.format(i)) # Meaningless
        # plt.imshow(np.reshape(Soft_max.weights[1:,i] * (X_train[np.where(y_train == i)[0][1], :] * X_train_sd + X_train_mu),(32,32)))

    plt.show(block=True)

parser = argparse.ArgumentParser(description = 'CSE151B PA1')
parser.add_argument('--batch-size', # Optinal Arg
                    type = int,
                    default = 64,
                    help = 'input batch size for training (default: 1)')
parser.add_argument('--epochs', # Optinal Arg
                    type = int,
                    default = 100,
                    help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning-rate', # Optinal Arg
                    type = float,
                    default = 0.001,
                    help = 'learning rate (default: 0.001)')
parser.add_argument('--z-score', # Optinal Arg
                    dest = 'normalization',
                    action='store_const',
                    default = data.min_max_normalize,
                    const = data.z_score_normalize,
                    help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--k-folds', # Optinal Arg
                    type = int,
                    default = 5,
                    help = 'number of folds for cross-validation')
parser.add_argument('--tune-hyperparameter', # Optinal Arg
                    type = bool,
                    default = False,
                    help = 'settrue if want to tune hyperparameter')


hyperparameters = parser.parse_args()
main(hyperparameters)