# Document Image Analysis - Font Classification

## Information

### Authors:
* Davide Morelli
* Mattéo Bonvin

### Project Overview
The goal of this project is to classify words from 15 (initially) different fonts.

In the following image, we can see the general workflow needed to be done in order to solve the task.
![Example Image](readme_images/project_steps.PNG)

### Workplan and Milestones
| Date     | What to do                                                                   |
|----------|------------------------------------------------------------------------------|
| March 12 | Presentation of challenges, initial discussion and team building             |
| March 19 | Definitive task assignments within groups and initial protocol specification |
| April 9  | First protocol with precise evaluation protocol                              |
| April 14 | Definitive protocol specification and milestones                             |
| April 16 | Delivery of first trivial end-to-end experiment                              |
| Mai 7    | delviery of final implementation with extensive results                      |
| Mai 12   | Deadline for report                                                          | 
| Mai 14   | Results improvements                                                         |
| Mai 21   | Oral presentations and final discussions                                     |
| Mai 28   | Conclusion, catch up if needed                                               |


## Initial Idea
Our first idea is to achieve the font classification using a ML model.

### First look at the dataset
The dataset (`fonts/`) is composed of 15 different folders, each one containing 1'000 images.

From an initial look at the different images, we can say that each font contains images of word of different colors, and
the words from a font to another are different in general (there could be some cases that some fonts contain the same word,
but we can't take that as an absolute truth).

### Principal difficulties
The principal difficulties that we can see are:
* Correctly loading the dataset
* Preprocessing the images
* Choosing the right model
* Evaluating the model

### Evaluation metrics
We can use a handful of metrics to evaluate the performance of our model. The ones we are choosing are:
* Accuracy: the ratio of correctly predicted observation to the total observations
* Precision: the ratio of correctly predicted positive observations to the total predicted positive observations
* Recall: the ratio of correctly predicted positive observations to the all observations in actual class
* F1-score: the weighted average of Precision and Recall
* Confusion matrix (15x15 matrix): a table used to describe the performance of a classification model
* ROC curve: a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied

### Workload estimation and task assignment
The main tasks that we can see are:
* Data loading and preprocessing
* Model selection and training
* Model evaluation

We can assign the tasks as follows:
* Person 1: Data loading and preprocessing
* Person 2: Model selection and training
* Both: Model evaluation

## Data Loading and Preprocessing
The first step is to load the data and preprocess it in order to be able to use it in the model.

### Data Loading
The data is loaded using the `tf.keras.preprocessing.image_dataset_from_directory` function, which allows us to load the data
directly from the directory, and to split it into training and validation sets.

### Data Preprocessing
The data is preprocessed using the following step:
* Resizing the images to 32x150 pixels

## Model Selection and Training
The following table shows the different models that we are going to use to solve the task, with some information about them.

| Model name            | # Parameters | GFLOPS |
|-----------------------|--------------|--------|
| SqueezeNet1_1         | 1.2M         | 0.35   |
| ShuffleNet_V2_X0_5_V1 | 1.4M         | 0.04   |
| MNASNet0_5            | 2.2M         | 0.1    |
| MobileNet_V3_Small    | 2.5M         | 0.06   |
| EfficientNet_B1_V2    | 7.8M         | 0.69   |
| ResNet18_V1           | 11.7M        | 1.81   |

More information about the models can be found here: [TorchVision Models](https://pytorch.org/vision/stable/models.html)

We can also work with different sizes of dataset:
* 50 images per font
* 100 images per font
* 250 images per font
* 500 images per font
* 1000 images per font (Original dataset)

Or even with a different number of fonts:
* 5 fonts
* 10 fonts
* 15 fonts (Original dataset)
* 25
* 40

## Results
For this, we evaluate each model with all the images of the dataset, and then we use the best model to evaluate the potential
differences between the different sizes of dataset.

### Evaluation of the original dataset
In the following table, we can see the results of the different models with the original dataset.
The time taker for the training is to be taken with a grain of salt, as it can vary a lot depending on the machine used. But it
can give a general idea of the time needed to train the model, and which model is faster to train.

| Model name            | # total images | # images per font | # epochs | Accuracy (%) | Precision | Recall | F1-score | Training time (s) | Medium training time per epoch (s) |
|-----------------------|----------------|-------------------|----------|--------------|-----------|--------|----------|-------------------|------------------------------------|
| SqueezeNet1_1         | 15'000         | 1'000             | 40       | 97.24        | 0.97      | 0.97   | 0.97     | 588.16            | 14.70                              |
| ShuffleNet_V2_X0_5_V1 | 15'000         | 1'000             | 40       | 97.42        | 0.97      | 0.97   | 0.97     | 568.72            | 14.22                              |
| MNASNet0_5            | 15'000         | 1'000             | 40       |              |           |        |          |                   |                                    |
| MobileNet_V3_Small    | 15'000         | 1'000             | 40       | 99.02        | 0.99      | 0.99   | 0.99     | 583.72            | 14.59                              |
| EfficientNet_B1_V2    | 15'000         | 1'000             | 40       | 99.38        | 0.99      | 0.99   | 0.99     | 2908.18           | 72.70                              |
| ResNet18_V1           | 15'000         | 1'000             | 40       | 99.16        | 0.99      | 0.99   | 0.99     | 1266.63           | 31.67                              |

The different models are saved as `<model_name>_font_classifier.h5` files in the `models/` folder, so we can use them later to evaluate them.

Based on the above table, all the models perform similarly, with the EfficientNet_B1_V2 model being the best one, with a 99.38% accuracy.
But the time taken to train the model is quite high, so we can use the MobileNet_V3_Small model, which has a 99.02% accuracy, and a smaller training time.

### Evaluation of the different sizes of dataset
In the following table, we can see the results of the MobileNet_V3_Small model with different sizes of dataset.

| Model name         | % of total images | # total images | # images per font | Accuracy | Precision | Recall | F1-score | Training time (s) | Medium training time per epoch (s) |
|--------------------|-------------------|----------------|-------------------|----------|-----------|--------|----------|-------------------|------------------------------------|
| MobileNet_V3_Small | 5%                | 750            | 50                |          |           |        |          |                   |                                    |
| MobileNet_V3_Small | 10%               | 1'500          | 100               |          |           |        |          |                   |                                    |
| MobileNet_V3_Small | 20%               | 3'000          | 200               |          |           |        |          |                   |                                    |
| MobileNet_V3_Small | 50%               | 7'500          | 500               |          |           |        |          |                   |                                    |
| MobileNet_V3_Small | 100%              | 15'000         | 1'000             |          |           |        |          |                   |                                    |

### Evaluation of the different number of fonts
In the following table, we can see the results of the MobileNet_V3_Small model with different number of fonts.

| Model name         | # total fonts | # images per font | Accuracy | Precision | Recall | F1-score | Training time (s) | Medium training time per epoch (s) |
|--------------------|---------------|-------------------|----------|-----------|--------|----------|-------------------|------------------------------------|
| MobileNet_V3_Small | 5             | 1000              |          |           |        |          |                   |                                    |
| MobileNet_V3_Small | 10            | 1000              |          |           |        |          |                   |                                    |
| MobileNet_V3_Small | 15            | 1000              |          |           |        |          |                   |                                    |
| MobileNet_V3_Small | 25            | 1000              |          |           |        |          |                   |                                    |
| MobileNet_V3_Small | 40            | 1000              |          |           |        |          |                   |                                    |

