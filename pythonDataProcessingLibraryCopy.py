import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datetime import datetime


class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()


class MissingValueHandler:
    """
    The MissingValueHandler class provides methods to handle missing values.

    Attributes:
        data (DataFrame): The dataset to be processed.

    Methods:
        fill_missing_values_with_mean(columns=None): Fills missing values with column means.
        fill_missing_values_with_median(columns=None): Fills missing values with column medians.
        fill_missing_values_with_constant(constant, columns=None): Fills missing values with the specified constant.
        delete_missing_values(columns=None): Deletes rows containing missing values from the dataset.
    """

    def __init__(self, data):
        """
        Constructor method for the MissingValueHandler class.

        Args:
            data (DataFrame): The dataset to be processed.
        """
        self.data = data

    def fillingWithMean(self, columns=None):
        """
        Fills missing values with column means.

        Args:
            columns (list, optional): The columns to operate on. By default, all columns are used.
        """
        try:
            self.data.fillna(self.data.mean(), inplace=True, subset=columns)
        except Exception as e:
            print(f"An error occurred while filling missing values with mean: {e}")

    def fillingWithMedian(self, columns=None):
        """
        Fills missing values with column medians.

        Args:
            columns (list, optional): The columns to operate on. By default, all columns are used.
        """
        try:
            self.data.fillna(self.data.median(), inplace=True, subset=columns)
        except Exception as e:
            print(f"An error occurred while filling missing values with median: {e}")

    def filingWithConstant(self, constant, columns=None):
        """
        Fills missing values with the specified constant.

        Args:
            constant: The constant value to fill missing values with.
            columns (list, optional): The columns to operate on. By default, all columns are used.
        """
        try:
            self.data.fillna(constant, inplace=True, subset=columns)
        except Exception as e:
            print(f"An error occurred while filling missing values with constant: {e}")

    def deletingMissingValues(self, columns=None):
        """
        Deletes rows containing missing values from the dataset.

        Args:
            columns (list, optional): The columns to operate on. By default, all columns are used.
        """
        try:
            self.data.dropna(inplace=True, subset=columns)
        except Exception as e:
            print(f"An error occurred while deleting missing values: {e}")




class OutlierHandler:
    """
    A class to handle outliers in a dataset using the Interquartile Range (IQR) method.

    Attributes:
        data (DataFrame): The input DataFrame containing the dataset.
        
    Methods:
        handle_outliers_iqr(column, threshold=1.5):
            Handle outliers in the specified column using the IQR method.
    
    """

    def __init__(self, data):
        """
        Initializes the OutlierHandler with the input dataset.

        Parameters:
            data (DataFrame): The input DataFrame containing the dataset.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data

    def handlingOutliersIqr(self, column, threshold=1.5):
        """
        Handles outliers in the specified column using the Interquartile Range (IQR) method.

        Outliers are identified based on the IQR score of the column.
        Data points lying beyond the threshold times the IQR above the third quartile
        or below the first quartile are considered outliers and removed from the dataset.

        Parameters:
            column (str): The name of the column to handle outliers.
            threshold (float, optional): The threshold multiplier for defining outliers.
                Default is 1.5.

        Returns:
            None. Modifies the data attribute in-place by removing outliers from the specified column.
        """
        try:
            q1 = self.data[column].quantile(0.25)
            q3 = self.data[column].quantile(0.75)
        except KeyError:
            raise KeyError(f"Column '{column}' does not exist in the dataset.")

        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        try:
            self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
        except KeyError:
            raise KeyError(f"Column '{column}' does not exist in the dataset.")


class Scaler:
    """
    A class for scaling numerical data using Min-Max scaling and Standard scaling methods.
    
    Attributes:
        data (DataFrame): The input DataFrame containing numerical data to be scaled.
    
    Methods:
        min_max_scaling(columns):
            Scale the specified numerical columns using Min-Max scaling method.
        
        standard_scaling(columns):
            Scale the specified numerical columns using Standard scaling method.
    """

    def __init__(self, data):
        """
        Initialize the Scaler object with the input DataFrame.

        Args:
            data (DataFrame): The input DataFrame containing numerical data.
        """
        self.data = data

    def minMaxScaling(self, columns):
        """
        Scale the specified numerical columns using Min-Max scaling method.

        Args:
            columns (list): A list of column names to be scaled.

        Returns:
            None. Modifies the DataFrame in-place.

        Raises:
            ValueError: If any column specified in `columns` does not exist in the DataFrame.
            ZeroDivisionError: If attempting to perform Min-Max scaling and the range of a column is zero.
        """
        for column in columns:
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
            min_val = self.data[column].min()
            max_val = self.data[column].max()
            if min_val == max_val:
                raise ZeroDivisionError(f"Cannot perform Min-Max scaling on column '{column}' because the range is zero.")
            self.data[column] = (self.data[column] - min_val) / (max_val - min_val)

    def standardScaling(self, columns):
        """
        Scale the specified numerical columns using Standard scaling method.

        Args:
            columns (list): A list of column names to be scaled.

        Returns:
            None. Modifies the DataFrame in-place.

        Raises:
            ValueError: If any column specified in `columns` does not exist in the DataFrame.
            ZeroDivisionError: If attempting to perform Standard scaling and the standard deviation of a column is zero.
        """
        for column in columns:
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
            mean_val = self.data[column].mean()
            std_val = self.data[column].std()
            if std_val == 0:
                raise ZeroDivisionError(f"Cannot perform Standard scaling on column '{column}' because the standard deviation is zero.")
            self.data[column] = (self.data[column] - mean_val) / std_val



class TextCleaner:
    """
    A class to perform text cleaning operations on a pandas DataFrame column.

    Attributes:
        data (pandas.DataFrame): The DataFrame containing the text data.

    Methods:
        remove_stopwords(column):
            Removes stopwords from the specified column using NLTK's English stopwords list.
            Args:
                column (str): The name of the column containing text data.

        lowercase_text(column):
            Converts text in the specified column to lowercase.
            Args:
                column (str): The name of the column containing text data.

        remove_punctuation(column):
            Removes punctuation from the specified column.
            Args:
                column (str): The name of the column containing text data.

        lemmatize_text(column):
            Lemmatizes the text in the specified column using NLTK's WordNetLemmatizer.
            Args:
                column (str): The name of the column containing text data.
    """

    def __init__(self, data):
        """
        Initializes the TextCleaner object with the provided DataFrame.

        Args:
            data (pandas.DataFrame): The DataFrame containing the text data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data

    def removeStopwords(self, column):
        """
        Removes stopwords from the specified column using NLTK's English stopwords list.

        Args:
            column (str): The name of the column containing text data.
        """
        try:
            stop_words = set(stopwords.words('english'))
            self.data[column] = self.data[column].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            self.data[column] = self.data[column].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

    def lowercaseText(self, column):
        """
        Converts text in the specified column to lowercase.

        Args:
            column (str): The name of the column containing text data.
        """
        try:
            self.data[column] = self.data[column].str.lower()
        except AttributeError:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    def removePunctuation(self, column):
        """
        Removes punctuation from the specified column.

        Args:
            column (str): The name of the column containing text data.
        """
        try:
            self.data[column] = self.data[column].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        except AttributeError:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    def lemmatizeText(self, column):
        """
        Lemmatizes the text in the specified column using NLTK's WordNetLemmatizer.

        Args:
            column (str): The name of the column containing text data.
        """
        try:
            lemmatizer = WordNetLemmatizer()
            self.data[column] = self.data[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
        except LookupError:
            nltk.download('wordnet')
            lemmatizer = WordNetLemmatizer()
            self.data[column] = self.data[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
        except AttributeError:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")


class FeatureEngineer:
    """
    A class for feature engineering tasks on a given dataset.

    Attributes:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing the data to be engineered.

    Methods:
    --------
    create_new_features():
        Creates new features based on existing ones. 
        Generates 'total_sales' by multiplying 'product_price' and 'quantity', 
        and 'age_category' by categorizing 'age' into predefined bins.

    calculate_percentage():
        Calculates the percentage of individual sales with respect to total sales.
        Requires 'individual_sales' and 'total_sales' columns to be present.

    normalize_features(columns):
        Normalizes specified numerical features in the dataset.
        The normalization is done by scaling the values between 0 and 1.

    create_interaction_terms(feature1, feature2):
        Creates interaction terms between two specified features.
        Generates a new feature representing the product of the given features.

    detect_weekend(date_column):
        Detects whether each date in the specified column falls on a weekend.
        Adds a binary column 'is_weekend' indicating if the date is a weekend day (1) or not (0).
    """
    def __init__(self, data):
        """
        Initializes the FeatureEngineer object with the provided data.

        Parameters:
        -----------
        data : pandas.DataFrame
            The input DataFrame containing the data to be engineered.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data

    def createNewFeatures(self):
        """
        Creates new features based on existing ones.
        Generates 'total_sales' by multiplying 'product_price' and 'quantity', 
        and 'age_category' by categorizing 'age' into predefined bins.
        """
        if {'product_price', 'quantity', 'age'}.issubset(self.data.columns):
            self.data['total_sales'] = self.data['product_price'] * self.data['quantity']
            self.data['age_category'] = pd.cut(self.data['age'], bins=[0, 18, 30, 50, np.inf], labels=['0-18', '19-30', '31-50', '51+'])
        else:
            raise ValueError("Columns 'product_price', 'quantity', and 'age' are required for feature creation.")

    def calculatePercentage(self):
        """
        Calculates the percentage of individual sales with respect to total sales.
        Requires 'individual_sales' and 'total_sales' columns to be present.
        """
        if {'individual_sales', 'total_sales'}.issubset(self.data.columns):
            self.data['percentage_of_total'] = (self.data['individual_sales'] / self.data['total_sales']) * 100
        else:
            raise ValueError("Columns 'individual_sales' and 'total_sales' are required for percentage calculation.")

    def normalizeFeatures(self, columns):
        """
        Normalizes specified numerical features in the dataset.
        The normalization is done by scaling the values between 0 and 1.

        Parameters:
        -----------
        columns : list
            A list of column names to be normalized.
        """
        for column in columns:
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' not found in the dataset.")
            self.data[f'{column}_normalized'] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())

    def createInteractionTerms(self, feature1, feature2):
        """
        Creates interaction terms between two specified features.
        Generates a new feature representing the product of the given features.

        Parameters:
        -----------
        feature1 : str
            Name of the first feature.
        feature2 : str
            Name of the second feature.
        """
        if {feature1, feature2}.issubset(self.data.columns):
            self.data[f'{feature1}_{feature2}_interaction'] = self.data[feature1] * self.data[feature2]
        else:
            raise ValueError(f"Columns '{feature1}' and '{feature2}' are required for interaction term creation.")

    def detectWeekend(self, date_column):
        """
        Detects whether each date in the specified column falls on a weekend.
        Adds a binary column 'is_weekend' indicating if the date is a weekend day (1) or not (0).

        Parameters:
        -----------
        date_column : str
            Name of the column containing date values.
        """
        if date_column not in self.data.columns:
            raise ValueError(f"Column '{date_column}' not found in the dataset.")
        self.data['is_weekend'] = self.data[date_column].dt.dayofweek // 5


class DataTypeConverter:
    """
    A class for converting data types of columns in a pandas DataFrame.

    Attributes:
        data (DataFrame): The input DataFrame containing the data to be converted.

    Methods:
        convert_to_numeric: Converts specified columns to numeric data type.
        convert_to_categorical: Converts specified columns to categorical data type.
    """

    def __init__(self, data):
        """
        Initializes the DataTypeConverter with the input DataFrame.

        Args:
            data (DataFrame): The input DataFrame containing the data to be converted.
        """
        self.data = data

    def convertNumeric(self, columns):
        """
        Converts specified columns to numeric data type.

        Args:
            columns (list): A list of column names to be converted to numeric data type.
        """
        self.data[columns] = self.data[columns].apply(pd.to_numeric, errors='coerce')

    def convertToCategorical(self, columns):
        """
        Converts specified columns to categorical data type.

        Args:
            columns (list): A list of column names to be converted to categorical data type.
        """
        self.data[columns] = self.data[columns].astype('category')




class CategoricalEncoder:
    """
    A class for encoding categorical variables in a pandas DataFrame.

    Parameters:
    -----------
    data : pandas DataFrame
        The DataFrame containing the categorical variables to be encoded.

    Methods:
    --------
    one_hot_encode(columns):
        Encode categorical variables using one-hot encoding.

        Parameters:
        -----------
        columns : list
            A list of column names containing categorical variables to be one-hot encoded.

        Returns:
        --------
        None. Updates the DataFrame in-place with one-hot encoded columns.

    label_encode(columns):
        Encode categorical variables using label encoding.

        Parameters:
        -----------
        columns : list
            A list of column names containing categorical variables to be label encoded.

        Returns:
        --------
        None. Updates the DataFrame in-place with label encoded columns.
    """
    def __init__(self, data):
        """
        Initialize the CategoricalEncoder object with a pandas DataFrame.

        Parameters:
        -----------
        data : pandas DataFrame
            The DataFrame containing the categorical variables to be encoded.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data

    def oneHotEncode(self, columns):
        """
        Encode categorical variables using one-hot encoding.

        Parameters:
        -----------
        columns : list
            A list of column names containing categorical variables to be one-hot encoded.

        Returns:
        --------
        None. Updates the DataFrame in-place with one-hot encoded columns.
        """
        try:
            self.data = pd.get_dummies(self.data, columns=columns)
        except KeyError as e:
            print(f"Error: One or more specified columns {e} not found in the DataFrame.")

    def labelEncode(self, columns):
        """
        Encode categorical variables using label encoding.

        Parameters:
        -----------
        columns : list
            A list of column names containing categorical variables to be label encoded.

        Returns:
        --------
        None. Updates the DataFrame in-place with label encoded columns.
        """
        label_encoder = LabelEncoder()
        try:
            for column in columns:
                self.data[column] = label_encoder.fit_transform(self.data[column])
        except KeyError as e:
            print(f"Error: One or more specified columns {e} not found in the DataFrame.")
        except ValueError as e:
            print(f"Error: Unable to perform label encoding. {e}")






class DateTimeHandler:
    """
    A class to handle date and time features extraction from a DataFrame.

    Attributes:
        data (DataFrame): The DataFrame containing the date and time column.

    Methods:
        extract_date_features(column): Extracts various date and time features from the specified column 
                                        and adds them as new columns to the DataFrame.
    """

    def __init__(self, data):
        """
        Initializes the DateTimeHandler object with the provided DataFrame.

        Args:
            data (DataFrame): The DataFrame containing the date and time column.
        """
        self.data = data

    def extractDateFeatures(self, column):
        """
        Extracts various date and time features from the specified column and adds them as new columns to the DataFrame.

        Args:
            column (str): The name of the column containing the date and time information.
        
        Raises:
            ValueError: If the specified column does not exist in the DataFrame.
            TypeError: If the data type of the specified column is not compatible with datetime conversion.
        """
        try:
            # Convert the specified column to datetime
            self.data[column] = pd.to_datetime(self.data[column])

            # Extract year, month, day, day of week, hour, and minute features
            self.data['year'] = self.data[column].dt.year
            self.data['month'] = self.data[column].dt.month
            self.data['day'] = self.data[column].dt.day
            self.data['day_of_week'] = self.data[column].dt.dayofweek  # Weekday (0: Monday, 1: Tuesday, ..., 6: Sunday)
            self.data['hour'] = self.data[column].dt.hour  # Hour
            self.data['minute'] = self.data[column].dt.minute  # Minute

        except KeyError:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        except TypeError:
            raise TypeError("The data type of the specified column is not compatible with datetime conversion.")



















