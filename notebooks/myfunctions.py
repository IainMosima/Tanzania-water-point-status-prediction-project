from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import pandas as pd
from category_encoders import OrdinalEncoder

# function to help with importing sets from thr analysis-df
def set_importer(path, y=False):
    import pandas as pd
    set = pd.read_csv(path)
    set.drop('Unnamed: 0', axis=1, inplace=True)

    if y == True:
        return set.squeeze()

    return set

#  function to help with printing score
def scores(y_true, y_preds):
    p_s = precision_score(y_true, y_preds)
    r_s = recall_score(y_true, y_preds)
    a_s = accuracy_score(y_true, y_preds)
    f_1 = f1_score(y_true, y_preds)


    print('The precision score is:\t', p_s)
    print('The recall score is:\t', r_s)
    print('The accuracy score is:\t', a_s)
    print('The f1 score is:\t', f_1)

# function for returning the number of outliers column.
def outliers(set, column, iqr=True):
    # using IQR to determine outliers.
    if (iqr):
        q1 = set[column].quantile(.25)
        q3 = set[column].quantile(.75)

        iqr = q3 - q1
        outliers = list( set[ (set[column]<(q1-1.5*iqr)) | (set[column]>(q1+1.5*iqr))][column] )

    else:
        temp_df = set.copy()
        temp_df['temp_col'] = temp_df[column]
        temp_df['temp_col'] = temp_df['temp_col'].apply(lambda x: (x-temp_df['temp_col'].mean())/temp_df['temp_col'].std() )
        outliers = list( temp_df[ (temp_df['temp_col'] <= -3) & (temp_df['temp_col'] >= 3) ].drop('temp_col', axis=1) )

    return len(outliers), outliers

# function for removing outliers given a certain set and column
def outliers_remover(set, column, iqr=True):
    # using IQR to determine outliers.

    if (iqr):
        q1 = set[column].quantile(.25)
        q3 = set[column].quantile(.75)

        iqr = q3 - q1
        result = set[ (set[column]>(q1 - 1.5*iqr)) & (set[column]<(q3 + 1.5*iqr)) ]


    
    # using z-score to remove outliers
    else:
        temp_df = set.copy()
        temp_df['temp_column'] = temp_df[column]
        temp_df['temp_column'] = temp_df['temp_column'].apply(lambda x: (x-temp_df['temp_column'].mean())/temp_df['temp_column'].std() )
        result = temp_df[ (temp_df['temp_column'] >= -3) & (temp_df['temp_column'] <= 3) ]
        result.drop('temp_column', axis=1, inplace=True) # dropping the term_column.
    
    
    return result

# function summarizing the whole data cleaning process
def data_cleaner(temp_df, target):
    # combining X and y
    temp_df = pd.concat([temp_df, target], axis=1)
    # converting to lower case and removing any trailing spaces
    temp_df_cleaned = temp_df.applymap(lambda x: x.lower().strip() if type(x) == str else  x)

    # dealing with null values
    # 1. scheme_name column
    temp_df_cleaned = temp_df_cleaned.drop('scheme_name', axis=1)
    # 2. scheme_management
    temp_df_cleaned = temp_df_cleaned.drop('scheme_management', axis=1)
    # 3. installer
    temp_df_cleaned.installer = temp_df_cleaned.installer.fillna('unknown')
    #  4. funder
    temp_df_cleaned.funder = temp_df_cleaned.funder.fillna('notknown')
    # 5. public_meeting
    temp_df_cleaned.drop('public_meeting', axis=1, inplace=True)
    # 6. permit
    permit_mode = temp_df_cleaned.permit.mode()[0]
    temp_df_cleaned.permit = temp_df_cleaned.permit.fillna(permit_mode)
    # 7. subvillage
    temp_df_cleaned.drop('subvillage', axis=1, inplace=True)

    # dealing with outliers
    # 1. longitude
    longitude_outliers_no, longitude_outliers = outliers(temp_df_cleaned, 'longitude')
    longitudes_median = temp_df_cleaned.longitude.median()
    temp_df_cleaned.longitude = temp_df_cleaned.longitude.apply(lambda x: longitudes_median if x in longitude_outliers else x)

    # 2.population
    median_pop = temp_df_cleaned.population.median()
    temp_df_cleaned.population = temp_df_cleaned.population.apply(lambda x: median_pop if x<median_pop else x)
    temp_df_cleaned = outliers_remover(temp_df_cleaned, 'population')

    # dropping irrelevant columns
    columns_to_drop = ['id', 'amount_tsh', 'date_recorded', 'installer', 'funder', 'num_private', 'region', 'lga', 'ward', 'recorded_by', 'wpt_name', 'management', 'quantity', 'extraction_type', 'extraction_type_group', 'payment_type', 'water_quality', 'source', 'source_type', 'waterpoint_type']

    # dropping these columns
    temp_df_cleaned.drop(columns_to_drop, axis=1, inplace=True)

    # abnormality
    # district_code
    temp_df_cleaned = temp_df_cleaned.loc[temp_df_cleaned.district_code != 0]

    # splitting back to normal
    X_temp_df = temp_df_cleaned.drop('status_group', axis=1)
    y_temp_df = temp_df_cleaned['status_group']
    return X_temp_df, y_temp_df


# function for encoding remaining categorical columns
def myScaler(set):
    numerics = ['gps_height', 'longitude', 'latitude', 'region_code', 'district_code', 'population', 'permit', 'construction_year'] # selecting columns to scale
    columns_to_scale = set.drop(numerics, axis=1)
    
    # Using standardscaler I will set all numerical values to be on the same scale.
    sc = StandardScaler()
    numericals_scaled = sc.fit_transform(columns_to_scale)

    numericals_scaled_df = pd.DataFrame(numericals_scaled, columns=columns_to_scale.columns, index=columns_to_scale.index)

    # dropping the numerical columns and then adding the new scaled columns
    result = pd.concat([set[numerics], numericals_scaled_df], axis = 1)

    return result

# data preparation function
def data_preparation(set, target):
    # Data conversion 
    # 1. Label encoding permit column
    le = LabelEncoder()
    le.fit(set.permit)
    set.permit = le.transform(set.permit)
  

    # 2. Label encoding the target column
    # Conveting target variable to a binary foramt
    le = LabelEncoder()
    target_transformed = target.to_frame().status_group.apply(lambda x: 1 if x == 'non functional' else 0)


    # Data Scaling
    numerics = ['gps_height', 'longitude', 'latitude', 'region_code', 'district_code', 'population', 'permit', 'construction_year'] # selecting columns to scale
    numericals = set[numerics]
    # selecting non-numerical dtypes.
    not_numericals = set.drop(numericals, axis=1)
    sc = StandardScaler()
    numericals_scaled = sc.fit_transform(numericals)
    numericals_scaled_df = pd.DataFrame(numericals_scaled, columns=numericals.columns, index=numericals.index)
    # dropping the numerical columns and then adding the new scaled columns
    set_scaled = pd.concat([numericals_scaled_df, not_numericals], axis = 1)

    # Ordinal encoding
    columns = ['basin', 'extraction_type_class', 'management_group', 'payment', 'quality_group', 'quantity_group', 'source_class', 'waterpoint_type_group']
    oe = OrdinalEncoder(cols=columns)
    set_ordinal_encoded = oe.fit_transform(set_scaled)
    set_ordinal_encoded = myScaler(set_ordinal_encoded)

    # 3. Onehot encoding the set
    set_onehotencoded = pd.get_dummies(set, drop_first=True)
    set_onehotencoded = myScaler(set_onehotencoded)


   

   

    return set_ordinal_encoded, set_onehotencoded, target_transformed
