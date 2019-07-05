# This script covers the data processing part of the Disaster Response pipeline
# EXAMPLE: python process_data.py -m disaster_messages.csv -c disaster_categories.csv -d DisasterResponse

# Import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import argparse

def load_data(messages_path, categories_path):
    """ This function takes two csv columns as 
    input and merges them into one dataframe.

    Parameters
    ----------
    messages_path : string
        file location of messages file
    categories_path : string
        file location of labels file
    Returns
    -------
    pandas.DataFrame
        The merged dataframe
    """
    # load message data and remove duplicates
    messages = pd.read_csv(messages_path)
    messages.drop_duplicates(inplace=True)

    # load categories dataset and remove duplicates
    categories = pd.read_csv(categories_path)
    categories.drop_duplicates(inplace=True)

    # merge datasets on "id"
    df = messages.merge(categories, how="inner", on="id")

    return df

def clean_data(df):
    """ This function takes the preprocessed 
    dataframe as input and prepares it for analysis.

    Parameters
    ----------
    df: pd.Dataframe

    Returns
    -------
    pandas.DataFrame
        The cleaned dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    rows = categories.iloc[0]

    # rename the columns of `categories`
    category_colnames = []
    for row in rows:
        category_colnames.append(row[0:-2])
    categories.columns = category_colnames

    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))

    # drop the original categories column from `df`
    df.drop(labels="categories", axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    frames = [df, categories]
    df = pd.concat(frames, axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data_to_db(df, database_name):
    """ This function takes the preprocessed 
    dataframe as input as well as a database name 
    for final storage. If run multiple times each run
    overwrites the existing database.

    Parameters
    ----------
    df: pd.Dataframe
    database_name: string

    Returns
    -------
    None
    """
    
    engine = create_engine("sqlite:///{}.db".format(database_name))
    df.to_sql('df_clean', engine, if_exists='replace', index=False)

def main():
    """
    Run 'python process_data.py --help' for information
    on the expected inputs.
    """
    # parse command line inputs using argparse
    parser = argparse.ArgumentParser(description='Data processing pipeline that requires the following inputs:')
    parser.add_argument(
         "-m", "--messages", help="Valid file path for messages.csv.", required=True)
    parser.add_argument(
        "-c", "--categories", help="Valid file path for categories.csv.", required=True)
    parser.add_argument(
        "-d", "--database", help="Database name to store the clean data in.", required=True)
    args = vars(parser.parse_args())
    
    # save script inputs to global variables
    MESSAGES = args.get("messages", "")
    CATEGORIES = args.get("categories", "")
    DATABASE = args.get("database", "")

    # run data loading and cleaning
    df = load_data(MESSAGES, CATEGORIES)
    df = clean_data(df)
    save_data_to_db(df, DATABASE)
    print("Successfully loaded data from CSV files and saved clean dataframe to database.")


if __name__ == "__main__":
    main()