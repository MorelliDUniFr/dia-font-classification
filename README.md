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
The dataset (`font/`) is composed of 15 different folders, each one containing 1'000 images.

From an initial look at the different images, we can say that each font contains images of word of different colors, and
the words from a font to another are different in general (there could be some cases that some fonts contain the same word,
but we can't take that as an absolute truth).

