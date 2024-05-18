# Mini-Project Summary: Fashion Mnist Dataset Analysis

## Objective:
The objective of this mini-project is to perform a comprehensive analysis of the Fashion MNIST dataset using various deep learning techniques and models. The analysis includes developing a CNN model from scratch, hyperparameter tuning using grid search, applying data augmentation techniques, and implementing transfer learning with the VGG-16 model.

## Details:

### a. CNN Model from Scratch
- Developed a Convolutional Neural Network (CNN) model from scratch.
- The model architecture includes 4 convolutional layers, followed by MaxPooling layers and fully connected layers.
- Applied Batch Normalization after each convolutional layer and Dropout for regularization in the fully connected layers.
- Trained the model using the Fashion Mnist dataset and evaluated its performance, including reporting performance metrics and displaying the learning curve.

### b. Hyperparameter Tuning with Grid Search
- Conducted grid search to find the optimal set of hyperparameters for the CNN model.
- Explored various combinations of activation functions (ReLU), optimizers (Adam, Adagrad), mini-batch sizes (4, 8, 16, 32), and learning rates (0.001, 0.0001).
- Reported the optimal hyperparameters and corresponding test accuracy achieved after hyperparameter tuning.

### c. Data Augmentation
- Applied five different image augmentation techniques on the Fashion Mnist training data to augment the dataset.
- Evaluated the previously designed CNN model on the augmented data to analyze its robustness and performance under varied input conditions.

### d. Transfer Learning with VGG-16
- Implemented Transfer Learning using the VGG-16 pre-trained model.
- Modified the architecture by adding additional layers, including global average pooling and fully connected layers, while freezing the base VGG16 layers to leverage pre-trained features.
- Evaluated the performance of the Transfer Learning model on the Fashion Mnist dataset.

## Conclusion:
This mini-project aims to showcase the application of deep learning techniques and methodologies on the Fashion MNIST dataset, demonstrating the process of model development, hyperparameter tuning, data augmentation, and transfer learning for image classification tasks.
