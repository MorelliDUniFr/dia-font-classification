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

### Data Loading

### Data Preprocessing

## Model Selection and Training
The following table shows the different models that we are going to use to solve the task, with some information about them.

| Model name            | # Parameters |
|-----------------------|--------------|
| ShuffleNet_V2_X0_5_V1 | 1.4M         |
| MobileNet_V2_V2       | 3.5M         |
| EfficientNet_B1_V2    | 7.8M         |
| DenseNet121_V1        | 8.0M         |
| ResNet18_V1           | 11.7M        |

We can also work with different sizes of dataset:
* 50 images per font
* 100 images per font
* 250 images per font
* 500 images per font
* 1000 images per font (Original dataset)

## Results
For this, we evaluate each model with all the images of the dataset, and then we use the best model to evaluate the potential
differences between the different sizes of dataset.
In the following table, we can see the results of the different models with the original dataset.
The time taker for the training is to be taken with a grain of salt, as it can vary a lot depending on the machine used. But it
can give a general idea of the time needed to train the model, and which model is faster to train.

| Model name            | # total images | # images per font | # epochs | Accuracy (%) | Precision | Recall | F1-score | Training time (s) | Medium training time per epoch (s) |
|-----------------------|----------------|-------------------|----------|--------------|-----------|--------|----------|-------------------|------------------------------------|
| ShuffleNet_V2_X0_5_V1 | 15'000         | 1'000             | 40       | 96.40        | 0.96      | 0.96   | 0.96     | 598.31            | 14.96                              |
| MobileNet_V2_V2       | 15'000         | 1'000             | 40       |              |           |        |          |                   |                                    |
| EfficientNet_B1_V2    | 15'000         | 1'000             | 40       |              |           |        |          |                   |                                    |
| DenseNet121_V1        | 15'000         | 1'000             | 40       |              |           |        |          |                   |                                    |
| ResNet18_V1           | 15'000         | 1'000             | 40       | 99.07        | 0.99      | 0.99   | 0.99     | 1059.28           | 26.48                              |

| Model name | % of total images | # total images | # images per font | Accuracy | Precision | Recall | F1-score |
|------------|-------------------|----------------|-------------------|----------|-----------|--------|----------|
|            | 5%                | 750            | 50                |          |           |        |          |
|            | 10%               | 1'500          | 100               |          |           |        |          |
|            | 20%               | 3'000          | 200               |          |           |        |          |
|            | 50%               | 7'500          | 500               |          |           |        |          |
|            | 100%              | 15'000         | 1'000             |          |           |        |          |

