## DISASTER RESPONSE PIPELINE

This project showcases data engineering skills gained as part of the Data Science Nanodegree designed by [Udacity](https://eu.udacity.com/course/data-scientist-nanodegree--nd025). This repo contains the code to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

Basis for the analysis are two datasets. One contains disaster messages, whereas the second accounts for their respective response labels (36 in total). A data preprocessing pipeline is build that stores cleaned data in a SQLite database. A machine learning pipeline takes the cleaned data as an input to train a model classifying the messages. A Flask web app uses the pre-trained classifier allowing to predict in which response category the message is likelier to belong and providing visualisations of the data. This will help organizations to reduce potential reaction time.

## Table of Contents

1. [File Descriptions](#file-descriptions)
2. [Code Descriptions](#code-descriptions)
3. [Licensing, Authors, and Acknowledgments](#licensing)

## File Descriptions

The project is structured as follows:

- **ETL pipeline - `/data` directory:**

	* Joins the two given datasets (disaster-messages.csv & categories-messages.csv).
	* Removes duplicates and identifies messages' labels utilising pandas.
	* Saves cleaned dataframe to pre-defined database.
	
		**Usage:**  
            When in 'data' directory specify the file paths of the two datasets you want to analyze as first and second argument. The third argument specifies the database name that will be stored in the respective folder. For further information execute `python process_data.py --help`. Exemplary execution:  

			python process_data.py -m disaster_messages.csv -c disaster_categories.csv -d DisasterResponse

- **Machine Learning pipeline for NLP - `/models` directory:**

	* Loads the data from database and creates training and test set.
    * Instantiates a pipeline for text processing and machine learning.
	* Trains and tunes hyperparameters of a given model using grid search.
	* Prints model performance on test set to terminal.
    * Stores the trained classifier in a pickle file.
	* Exports the final model as a pickle file  

		**Usage:** 
			From the `/models` directory provide the filepath and the name of the database as well as the table name storing the cleaned data. Moreover, provide the filepath of the pickle file to save the model. Exemplary execution:    

			 python train_classifier.py -db data/DisasterResponse -dt df_clean -m classifier.pkl


- **Flask app - `/app` directory.**

	Displays basic information on the data stored in the database. After a user inputs a message into the app, the app returns classification results for all 36 labels.

	**Usage:**   
	- From the `/app` directory Run the following command to run the web app:

	```  
	python run.py   
	```   

	- Go to http://0.0.0.0:3001/   

## Code Descriptions

- `app/`
  - `template/`
    - `master.html`  -  Master HTML file of app.
    - `go.html`  -  Result page after classification.
  - `run.py`  - Flask app's main script

- `data/`
  - `disaster_categories.csv`  - Disaster categories dataset.
  - `disaster_messages.csv`  - Disaster Messages dataset.
  - `process_data.py` - Data processing pipeline.
  - `DisasterResponse.db`   - Database containing the merged and cleand data.

 - `media/` 
 	- `app.gif` - A gif showcasing the app interface.
- `models/`
  - `train_classifier.py` - The NLP and ML pipeline.

## Further Improvements

Although the model in use already gives quite good results there is definitely room for improvement. According to literature LSTM (Long short-term memory) models give state-of-the-art results for NLP like the one given. Consequently, it seems promising to further look into this by e.g. using Keras (Tensorflow) solutions.

## Licensing

For some code examples, especially the Flask web app, credit goes to Udacity. Otherwise, feel free to use the code as you like.

