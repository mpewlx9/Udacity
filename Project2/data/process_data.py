import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function is to load the messages and categories files into two data frames as the first step for data processing
    Args:
        messages_filepath (string):     the file path for disaster_messages csv file
        categories_filepath (string):   the file path for disaster_categories csv file
    Returns:
        df (pandas dataframe): merged dataframe 
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head(2)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head(2)
    
    # merge datasets
    df = messages.merge(categories, how='inner', on = ['id'])
    df.head(2)
    
    return df


def clean_data(df):
    """
      Function:
      clean up the Dataframe
      
      Args:
      df (DataFrame): Input dataframe
      
      Return:
      df (DataFrame): dataframe after clean up
      
    """

        
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    categories.head(1)  
    
    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories. one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =  row.applymap(lambda x: x[:-2]).iloc[0, :]
       
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
     
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #remove data with value not in (0,1)
    categories = categories[categories["related"] < 2]
        
    # replace categories columns in df with new category columns    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)  
        
    # concatenate the original dataframe with the new `categories` dataframe   
    df = pd.concat([df, categories], axis = 1, join='inner')   
        
    # drop duplicates  
    df.drop_duplicates(inplace=True)   
        
    # make number of duplicates is zero
    print('# of duplicates is ', sum(df.duplicated()))  
        
    return df

        
def save_data(df, database_filename):
    """
    This function is to save the cleaned data into  SQLite database.
    
    Args:
    df: dataframe, dataframe with clean version of data
    database_filename: string, Filename for output SQLite database.
       
    returns:
    None
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)   


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()