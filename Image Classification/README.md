# Image Classification Machine Learning Model

This project demonstrates the creation of an image classification model using a pre-trained Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The model can classify images into predefined categories, and we also provide an example of how to classify a sample image.

## About

Image classification is a fundamental task in computer vision, where the goal is to assign a label or category to an image based on its content. This project uses a pre-trained VGG16 model to classify images into one of the ten predefined categories in the CIFAR-10 dataset.

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. It is commonly used for benchmarking image classification algorithms.

## Features

- Classification of images into 10 predefined categories.
- Option to test the model with a sample image.

## Prerequisites

To run the code, you'll need the following libraries:

- TensorFlow
- NumPy
- OpenCV (for image loading and preprocessing)

You can install the required libraries using the following command:

```bash
pip install tensorflow numpy opencv-python
```

## Usage

1. Clone the repository or download the code to your local machine.

2. Ensure that you have installed the required libraries mentioned in the "Prerequisites" section.

3. Run the code by executing the `imageClassify.py` script:

   ```bash
   python imageClassify.py
   ```

4. The model will be trained on the CIFAR-10 dataset and evaluated for accuracy.

5. After training, you can test the model by providing the path to a sample image using the `sample_image_path` variable in the `imageClassify.py` script.

6. The complete training of the model is present in the `cnn_cifar10_image_classify.ipynb` file

## Sample Image Classification

To classify a sample image, you can follow these steps:

1. Replace the `airplaneImage.jpg` file in the project directory with your own image or use the provided sample image.

2. Set the `sample_image_path` variable in the `imageClassify.py` script to the path of your sample image.

3. Run the script as described in the "Usage" section.

4. The script will load the sample image and classify it into one of the ten CIFAR-10 categories.

5. The predicted category label will be displayed in the terminal.

## Dataset

The CIFAR-10 dataset is used for this project. You can learn more about the dataset at [CIFAR-10 - Object Recognition in Images](https://www.cs.toronto.edu/~kriz/cifar.html).

## Acknowledgments

- This project is for educational purposes and demonstrates image classification using pre-trained models.
- The CIFAR-10 dataset is a widely used benchmark dataset for image classification.

Feel free to use, modify, and expand upon this code for your image classification projects!