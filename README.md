# DSAI2 - Aktiv Walter - Exercise Recommendation for individual pains
## 1) Introduction
The project suggests physical exercises for users who want to reduce a certain problem like back pain. The following chapters describe the project files.

## 2) Content
The Project includes the data folder with the raw and processed datasets used for the notebooks and the jupyter notebooks for the different project parts. This readme explains the structure of the datasets and the purpose of each notebook and its relation to the other components of the project.

## 3) Data
The project includes two datasets for fitness performance of large studies conducted in South Korea and the U.S. Also, there is a couple of merged datasets with IDs (foreign keys) that map the relationship between different datasets.

### 3.1) bodyPerformancy.csv
The bodyPerformance dataset has attributes such as age, gender, weight, and different performance indicators for various exercises to categorize the participants into different fitness levels. It is used later for training the classification model in the fitness_categories.ipynb notebook.

### 3.2) exercise_dataset.csv
This dataset includes average calories for activities and exercises for different weight groups and also provides information about the calories per kg bodyweight which makes the different activities more comparable in terms of how intense they are.

### 3.3) data.json
This file contains the raw json data retrieved from the python web scrapper that browses medical journals to retrieve exercises for patients with different problems such as diabetes, back pain or high blood pressure. The json object comes with the attributes Problem (e.g. disease), an exercise, the SourceHead (i.e. the title of the article/blog post) and the SourceLink for traceability. 

### 3.4) data.csv
To better process the scrapped data, this file includes the identical information as the data.json but in .csv format.

### 3.5) exercise_dataset_with_exercise_id.csv
This file builds upon the exercise_dataset.csv but adds two important attributes. First, the category of the exercise indicates again the difficulty/fitness level similar to the bodyPerformance.csv above. Second, the exercise_id groups together exercises from the same type. For instance, exercises "running 5 mph" and "running 9 mph" are both given the same exercise_id. The category attribute then differentiates between the two different levels.

### 3.6) sources_with_exercise_id.csv
Similar to the previous file. The data.csv is also extended with an exercise_id following the same principle as before. This allows the two files "exercise_dataset_with_exercise_id.csv" and "sources_with_exercise_id.csv" to be merged an processed for further steps and eventually for our exercise suggestions for our app users.

## 4) Business Logic
This next part describes all notebooks and their interrelation in our project. There are also references back to the datasets described before and how they were used and processed.

### 4.1) Create_Activity_Sources_dataset.ipynb
We used EdgeGPT (https://github.com/acheong08/EdgeGPT), a wrapper for the Bing Chat to gather a dataset including the columns "Problem", "Exercise", "SourceHead", "SourceLink". The data was received in json format and had to be cleaned to be used in a ML model. We cleaned the dataset manually and with the help of OpenAI's GPT3. The cleaned dataset can be found under **./data/data**.csv. The code for gathering the data is in **Create_Activity_Sources_dataset.ipynb**. 

1. The code defines a function called requestBingChat that uses the EdgeGPT library to interact with the chatbot and retrieve exercise recommendations for a given problem.
2. It creates an instance of the Chatbot class, sends a prompt to the chatbot, and retrieves a response.
3. The response is processed, removing unnecessary elements from the sources.
4. The response is converted to JSON format and added to a list called jsonList.
5. The function is closed.
6. The code iterates over the problems list, calling the requestBingChat function for each problem.
7. The resulting jsonList is saved to a JSON file called "data.json". If the file already exists, the code appends the jsonList to the existing data.

### 4.2) fitness_categories.ipynb

In this file we split the users of the bodyPerformance dataset into four performance groups. The groups are based on the quartiles of the "Calories per kg" column. The groups are stored in the "category" column. In a next step we test different classifiers to predict the category of a user based on the bodyPerformance dataset. The classifiers are: Random Forest Classifier (RFC), Decision Tree Classifier (DTC), XGBoost Classifier (XGBC), Logistic Regression (LR), and Support Vector Machine (SVM). The classifiers are trained and evaluated on the bodyPerformance dataset. Accuracy scores and confusion matrices are printed for each classifier. The best classifier is the Random Forest Classifier with an accuracy of 0.75. In the keyword matching part, the exercises of the exercise_dataset are matched with the exercises of the sources dataset to get relevant sources for each exercise. 

#### 4.2.1) Data Loading and Preprocessing:

Two datasets are loaded: 'exercise_dataset.csv' and 'bodyPerformance.csv'.
'exercise_dataset.csv' is assigned to the DataFrame variable 'df', and 'bodyPerformance.csv' is assigned to 'dfBody'.
'category' column is created in 'df' based on quartiles of 'Calories per kg'.
Gender and class columns in 'dfBody' are encoded using mapping dictionaries.

#### 4.2.2) Feature Importance:

ExtraTreesClassifier and XGBClassifier are used to determine feature importances in 'dfBody'.
The feature importances are visualized using bar plots.

#### 4.2.3) Classification Models:

Random Forest Classifier (RFC), Decision Tree Classifier (DTC), XGBoost Classifier (XGBC), Logistic Regression (LR), and Support Vector Machine (SVM) classifiers are trained and evaluated on 'dfBody'.
Accuracy scores and confusion matrices are printed for each classifier.

#### 4.2.4) Random Forest Classifier Fine-Tuning:

GridSearchCV is used to find the best hyperparameters for the RFC.
The best model is extracted and evaluated on the test set.
The best model is saved to 'best_model.pkl' using pickle.

#### 4.2.5) Keyword Extraction and Matching:

The 'Exercise' column in 'sources' DataFrame is given unique IDs and stored in the 'exercise_id' column.
The 'exercise_id' column is filled in 'df' by matching exercise names from 'sources' with activity names in 'df'.
Unique 'exercise_id' values in 'df' are outputted.
'df' and 'sources' are saved to new CSV files.

### 4.3) Create_exercise_for_user.ipynb

This notebook contains code to create a personalized exercise recommendation for a user. It uses the best model created in **fitness_categories.ipynb
** to predict the performance category of the user. Based on the pain defined by the user, an exercise is loaded from the exercise dataset (**exercise_dataset_with_exercise_id.csv**) in combination with a source from the sources dataset (**sources_with_exercise_id.csv**). The predicted exercise is matched with the sources dataset to get a source for the exercise. The source is printed to the user.

1. Data Preparation:
   - A dictionary `data` is created with information about an individual.
   - The dictionary is converted into a pandas DataFrame `x`.
   - The 'gender' column in the DataFrame is encoded using a mapping dictionary, converting 'M' to 0 and 'F' to 1.

2. Model Loading and Prediction:
   - A pre-trained model is loaded from the 'best_model.pkl' file using pickle.
   - The loaded model is used to predict the class labels for the input data `x`.
   - The predicted labels are stored in `y_pred`.

3. Mapping the Predicted Label:
   - A mapping dictionary is defined to map the numeric labels to corresponding classes.
   - The predicted label `y_pred` is mapped to a user-friendly class label `user_class`.

4. Loading Additional Data:
   - Two datasets are loaded from CSV files: 'sources_with_exercise_id.csv' and 'exercise_dataset_with_exercise_id.csv'.
   - These datasets contain information about sources and exercises, respectively.

5. Searching for Exercise IDs:
   - A variable `problem` is set to search for specific problems (e.g., "back pain").
   - The `sources` DataFrame is filtered to find rows where the 'Problem' column contains the search keyword.
   - If any exercise IDs are found, a random one is selected, and the associated source head and link are stored in `sourceHead` and `sourceLink` variables.
   - If no exercise IDs are found, `-1` is assigned to `ex_id`.

6. Retrieving Exercise Details:
   - The `exercises` DataFrame is filtered to find rows where the exercise ID matches `ex_id`.
   - If any exercises are found, a random one is selected from the filtered DataFrame.
   - If no exercises are found, a random exercise is selected from the entire `exercises` DataFrame.

7. Output Summary:
   - The variables `problem`, `exercise`, `user_class`, `sourceHead`, and `sourceLink` are printed.


### 4.4) Create_exercise_for_user_with_LLM.ipynb
This file contains the same code as **Create_exercise_for_user.ipynb** but with the addition of a language model. The language model is used to generate a personalized exercise recommendation for a user. This recommendation should be used as a push notification for a mobile app to motivate the user to do something against his/her pain.


### 4.5) falcon-7b.ipynb
This notebook contains code to download the Falcon-7b-Instruct model from Huggingface and some test prompts. It shows that the 8-bit quantized model can run inference with less than 4GB VRAM. It was executed on Kaggle cloud.
