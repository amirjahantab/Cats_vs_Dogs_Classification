

# Image Classification (Cats vs Dogs)

This project focuses on classifying images of cats and dogs using Convolutional Neural Networks (CNNs) with PyTorch. The dataset used for this project is from Kaggle's ["Dogs vs Cats Redux"](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) competition.

## Objectives

- **Image classification**
- **Four key components of any ML system (in PyTorch):**
  - Data (Images)
  - Model (CNN)
  - Loss (Cross Entropy)
  - Optimization (SGD, Adam, ..)
- **Convolutional Neural Networks (CNNs)**
- **Overfitting and how to handle it**
- **Data augmentation techniques**
- **Transfer learning**

## Dataset

The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).


## Libraries and Tools

The following libraries and tools are used in this project:

- **Python**: Programming language
- **PyTorch**: Main library for building and training the model
- **NumPy**: For numerical operations
- **Matplotlib**: For plotting and visualization
- **PIL (Python Imaging Library)**: For image processing
- **tqdm**: For progress bars
- **scikit-learn**: For additional metrics like confusion matrix

## Usage

1. **Training the Model:**
   - Open the Jupyter Notebook `cat_vs_dog.ipynb`.
   - Follow the steps in the notebook to preprocess the data, build the model, train it, and evaluate its performance.

2. **Evaluating the Model:**
   - After training, provided various metrics and visualizations to evaluate the model's performance.

## Detailed Explanation of Steps

### Data Loading and Preprocessing

- **Data Augmentation:** Techniques like random cropping, horizontal flipping, and normalization are applied to increase the diversity of the training data.
- **Data Loading:** PyTorch's `DataLoader` is used to load the dataset in batches.

### Model Building

- **Convolutional Neural Network (CNN):** A custom CNN is built using PyTorch's `nn.Module`.
- **Transfer Learning:** Pre-trained models (like ResNet, VGG) from `torchvision.models` are used for better performance.

### Training

- **Loss Function:** Cross-Entropy Loss is used for classification.
- **Optimization:** Stochastic Gradient Descent (SGD) and Adam optimizers are used.
- **Training Loop:** The training process involves forward pass, loss computation, backward pass, and optimizer step.

### Evaluation

- **Confusion Matrix:** To visualize the performance of the model on the test data.
- **Accuracy:** Accuracy on validation data with overfit is $61\%$.

### Handling Overfitting

- **Data Augmentation** 
- **Transfer Learning** 

## Results

- **Accuracy:** The final accuracy of the model on the test set is $97\%$.


## Conclusion

This project demonstrates the steps to build a robust image classification model using CNNs and transfer learning. The techniques of data augmentation and regularization play a crucial role in achieving good performance.


## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the dataset
- [PyTorch](https://pytorch.org/) for the deep learning library



