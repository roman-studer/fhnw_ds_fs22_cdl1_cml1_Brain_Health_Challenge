# Understanding Brain Health - International Challenge

## Information

This semester we will focus on brain health. We will work on medical image classification and visualization tasks, i.e., the challenge contains equal part machine learning and data visualization. We have an international partnerships for this challenge with the New York University School of Medicine / Neuroscience. We will work with an open source data set that includes brain scans and some 'tabular' data regarding patients' characteristics for Alzheimer’s disease.

Each of you will gain experience with machine learning for image classification in the domain of medical imaging, as well as how to integrate visualizations in the process to better understand and explain the process (explainable artificial intelligence (XAI)).

### Description of the disease
Description of the Disease
Human brain, just like any other part of human anatomy, loses some function as we age. Some of this is 'healthy aging', i.e., it happens to everybody. However, a relatively large percent of the aging population also experience cognitive decline that is more extreme than others. A milder version of this is called 'mild cognitive impairment' (MCI), where the person can mostly function. For some of the MCI patients, the disease is progressive, and leads to dementia (Alzheimer's disease is a form of dementia). In this project we have brain imaging, cognitive testing, and laboratory data from three groups of people: healthy-aging individuals ("healthy controls"), people with MCI and people with Alzheimer's disease (AD). AD is a crippling disease in its advanced stages that prevents normal interaction, retention of even the most prominent memories (names of close family members), and individuals with advanced AD must, in most cases, be institutionalized. It is one of the most common causes of death in modern times.

Understanding memory / cognition related dysfunction, being able to detect, predict, and monitor brain health is a socially and economically relevant applied science issue. Any new insight that data science methods might bring will help towards developing solutions that can detect, predict and monitor brain health.
### Dataset

The ADNI data set contains of two main "types" of data:
1) Tabular data that contains information about the patient (age, gender, profession, medical history etc.) and patients' responses to some tests that assess their cognitive abilities, and

2) Imaging data from MRI or other medical scanners.

In 2010, ADNI contained a total of 819 subjects (229 normal control subjects, 398 subjects with mild cognitive impairment (MCI), and 192 subjects with mild Alzheimer's disease (AD)) (see 2010 article). Since then, the database grew much larger, and these numbers grow as more and more institutions join the effort and upload their datasets into the collection too.

### Tasks
#### Task 1: Discovering Relationships in Tabular Data
Create an interactive visualization environment and conduct exploratory data analysis on the six tables provided on gitlab to get a sense of what these datasets contain. One contains demographic information and the other five are selected cognitive tests:
-- PRDEMOG (Subject Demographics [ADNI1,GO,2,3])
-- CDR (Clinical Dementia Rating scale)
-- GDSCALE (Geriatric Depression scale)
-- MMSE (Mini Mental State Examination)
-- MoCA (Montreal Cognitive Assessment)
-- NEUROBAT Neuropsychological Battery [ADNI1,GO,2,3]
You can connect these tables as well as the imaging data by using the RID column (not to be confused with "ID"). More information is provided under "Linking Images and Tabular Data".
Explore the relationships using unsupervised machine learning within the tabular data multiple observations e.g., based on clustering: Do MoCA / MMSE / CDR and NEUROBAT are similar in some cases? What are those cases? How do each test correlate with demographic variables? Which of the tests correlate strongest with the diagnosis outcomes?
Is there a relationship between GDSCALE (depression) and any of the measured variables (including demographics)?
Is there a relationship between any of the observed variables and an MCI/AD diagnosis? If yes what is the degree of this correlation (in terms of effect size) and is it statistically significant based on t-test and analysis of variance (ANOVA)?
Discuss your results with the international collaborator and your advisors, and document your findings
#### Task 2: Making Sense of the Imaging Data
Make an overview of the available image scans and check wether they were correctly assigned (class + gender) using the overview sheet.
Analyze and explore the provided image scans (#subjects, research groups, #visits per subject, age, gender, image axes, image formats, ...) and extract relevant 2D slices (axial, one or multiple per subject on interesting areas), see play notebook (needs permissions) in resources for data loading.
Make sure you create data-science-sound datasets for your deep learning experiments.
Classify the images using deep learning and binary classification (see DL notebook (needs permissions) and switchtube video in resources) into: healthy (CN) vs. unhealthy (MCI, AD)
Conduct 3-4 different deep learning experiments and analyze their differences:
Varying 2D slices or balance of the dataset (age, gender, …)
Varying DL models, layers, …
Be creative and propose your own ideas
See also options below
Visualize the differences of two or three classes by using explainable artificial intelligence (XAI), such as grad-CAM, SHAP or layerwise-relevance-propagation (LRP). Note LRP by innvestigate only works for tensorflow 1.x.
Discuss your results with the international collaborator and your advisors, and document your findings
Optionally: classify the selected images using deep learning and multi-class classification into three classes: CN, AD, MCI
Note: Venugopalan et al. 2021 reported that early MCI are similar to CN and late MCI are similar to AD; hence, you may also try splitting MCI subjects accordingly and  classify only AD vs. CN.
Optionally: propose your own data analysis and ideas

#### Task 3: Connecting the Findings from Imaging and Tabular Data
Filter the data in the tabular datasets to match the imaging dataset and the observations recorded in tables
Link tabular data and results of the image analysis (agnostic / correlation driven, exploratory approach): i.e., which of the measures correlate strongly with the AD diagnosis, what are the patterns and anomalies for the two patient groups?
Discuss your results with the international collaborator and document your findings

## Project Structure
```
├── application                     - holds the final application
├── data                            - contains raw, transformed and processed data
│   └── external               
│   └── interim
│   └── processed
│   └── raw
│     └── miniset                   - contains a small sample dataset for test purposes
├── models                          - contains serialized models
├── notebooks                       - contains jupyter notebooks 
│   ├── 0_eda
│   └── 1_preprocessing
│   └── 2_modelling
│   └── 3_evaluation
│   └── 4_visualizations
│   └── 5_miscellaneous
├── reports                         - contains figures and texts for the final report
│   └── figures
└── scripts                         - contains python scripts (including helper functions)
│   └── helpers
└── CONFIG.yml                      - config file to ajust parameters that are used over multiple scripts
└── README.md
└── requirements.txt

```

## Setup

### Flattening the data structure
The image data is extremely nested and thus inconvenient to handle. The information of the names of the subfolders
are contained in the filename itself. Which renders the subfolders unnecessary.

We have written some functions to flatten the data structure:
1. Make sure the raw data is in the project folder `data/raw/data`
2. Run the notebook `notebooks/5_miscellaneous/data_extraction.ipynb`. Make sure to uncomment the line
`# copy_to_flat()` in the last cell. 

Note that this creates a copy of the data. Additional space on disk is required. After that one can theoretically disperse of the raw, nested data.

### Get train- and testset annotations
We provide functionality to split the data into a train and into testset by creating two CSV-files:
`test_labels.csv` and `train_labels.csv` in the `data/annotations`-folder. 
To create the split (always stratified) with the proportions you desire run the `notebooks/1_preprocessing/Train_Test_Splitter.ipynb`-notebook.

You can adapt the split in the following way:
 - "proportion": This argument defines the proportion of data in the trainset (therefore has to be between 0 and 1)
 - "group_identifier": A column in the input dataframe. Allows us to split the data based on the values of a column
 - "save_files": If true, the above mentioned csv-files will be created. If false, the test- and trainset dataframe will be returned.
