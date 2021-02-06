# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)


def preprocessing(df, return_df = False):
    categorical = df.select_dtypes("object")
    numerical = df.select_dtypes(exclude=["object"])
    print(f"Rows Number: {df.shape[0]}")
    print(f"Columns Number: {df.shape[1]} \n")
    print(f"Variables List: \n{list(df.columns)}\n")
    print(f"--Categorical--: \n{list(categorical.columns)} \n")
    print(f"--Numerical--: \n{list(numerical.columns)} \n")
    print(f"Number of NaN: \n{df.isnull().sum()}\n")
    print(f"Data types: \n{df.dtypes}")
    if return_df == True:
        return categorical, numerical

    
        
def missing(df):
        import seaborn as sns
        #frac_missing = df.isnull().sum()/len(df)
        percent_missing = df.isnull().sum()*100/len(df)
        return percent_missing.sort_values(ascending=False), sns.heatmap(df.isnull(), cbar=False)
    
    
def mean_imputer(df):
        '''This func takes dataset with missing values, 
            and impute them with mean values'''
        from sklearn.impute import SimpleImputer
        mean_imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
        imputed_df = mean_imputer.fit_transform(df)
        return imputed_df    

def regression_imputer(df):
    '''This func takes dataset with missing values, 
        and impute them with regression values'''
    import pandas as pd
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    column = df.columns
    reg_imputer = IterativeImputer()
    imputed_df = reg_imputer.fit_transform(df)
    return pd.DataFrame(imputed_df, columns=column)

def make_standard_normal(df_train, df_test):
    """
    In order to make the data closer to a normal distribution, take log
    transforms to reduce the skew.
    Then standardize the distribution with a mean of zero and standard deviation of 1. 
  
    Args:
      df_train (dataframe): unnormalized training data.
      df_test (dataframe): unnormalized test data.
  
    Returns:
      df_train_normalized (dateframe): normalized training data.
      df_test_normalized (dataframe): normalized test data.
    """
    # Remove skew by applying the log function to the train set, and to the test set
    df_train_unskewed = np.log(df_train)
    df_test_unskewed = np.log(df_test)
    
    #calculate the mean and standard deviation of the training set
    mean = np.mean(df_train_unskewed)
    stdev = np.std(df_train_unskewed, ddof=1)
    
    # standardize the training set
    df_train_standardized = (df_train_unskewed - mean)/stdev
    
    # standardize the test set (see instructions and hints above)
    df_test_standardized = (df_test_unskewed - mean)/stdev  
    return df_train_standardized, df_test_standardized

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def cindex(y_true, scores):
    '''

    Input:
    y_true (np.array): a 1-D array of true binary outcomes (values of zero or one)
        0: patient does not get the disease
        1: patient does get the disease
    scores (np.array): a 1-D array of corresponding risk scores output by the model

    Output:
    c_index (float): (concordant pairs + 0.5*ties) / number of permissible pairs
    '''
    n = len(y_true)
    assert len(scores) == n

    concordant = 0
    permissible = 0
    ties = 0
    
    # use two nested for loops to go through all unique pairs of patients
    for i in range(n):
        for j in range(i+1,n): #choose the range of j so that j>i
            
            # Check if the pair is permissible (the patient outcomes are different)
            if y_true[i] != y_true[j]:
                # Count the pair if it's permissible
                permissible +=1

                # For permissible pairs, check if they are concordant or are ties

                # check for ties in the score
                if scores[i] == scores[j]:
                    # count the tie
                    ties +=1
                    # if it's a tie, we don't need to check patient outcomes, continue to the top of the for loop.
                    continue

                # case 1: patient i doesn't get the disease, patient j does
                if y_true[i] == 0 and y_true[j] == 1:
                    # Check if patient i has a lower risk score than patient j
                    if scores[i] < scores[j]:
                        # count the concordant pair
                        concordant +=1
                    # Otherwise if patient i has a higher risk score, it's not a concordant pair.
                    else:
                        continue
                    # Already checked for ties earlier
                

                # case 2: patient i gets the disease, patient j does not
                if y_true[i] == 1 and y_true[j] == 0:
                    # Check if patient i has a higher risk score than patient j
                    if scores[i] > scores[j]:
                        #count the concordant pair
                        concordant +=1
                    # Otherwise if patient i has a lower risk score, it's not a concordant pair.
                    else:
                        continue
                    # We already checked for ties earlier
            
    # calculate the c-index using the count of permissible pairs, concordant pairs, and tied pairs.
    c_index = (concordant+0.5*ties)/permissible
    ### END CODE HERE ###
    
    return c_index

# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def add_interactions(X):
    """
    Add interaction terms between columns to dataframe.

    Args:
    X (dataframe): Original data

    Returns:
    X_int (dataframe): Original data with interaction terms appended. 
    """
    features = X.columns
    m = len(features)
    X_int = X.copy(deep=True)

    # 'i' loops through all features in the original dataframe X
    for i in range(m):
        
        # get the name of feature 'i'
        feature_i_name = features[i]
        
        # get the data for feature 'i'
        feature_i_data = X_int[feature_i_name]
        # choose the index of column 'j' to be greater than column i
        for j in range(i+1, m):
            
            # get the name of feature 'j'
            feature_j_name = features[j]
            
            # get the data for feature j'
            feature_j_data = X_int[feature_j_name]
            
            # create the name of the interaction feature by combining both names
            # example: "apple" and "orange" are combined to be "apple_x_orange"
            feature_i_j_name = f"{features[i]}_x_{features[j]}"
            
            # Multiply the data for feature 'i' and feature 'j'
            # store the result as a column in dataframe X_int
            X_int[feature_i_j_name] = feature_i_data*feature_j_data
    return X_int


def holdout_grid_search(clf, X_train, y_train, X_val, y_val, hyperparams, fixed_hyperparams={}):
    '''
    Conduct hyperparameter grid search on hold out validation set. Use holdout validation.
    Hyperparameters are input as a dictionary mapping each hyperparameter name to the
    range of values they should iterate over. Use the cindex function as your evaluation
    function.

    Input:
        clf: sklearn classifier
        X_train (dataframe): dataframe for training set input variables
        y_train (dataframe): dataframe for training set targets
        X_val_hp (dataframe): dataframe for validation set input variables
        y_val_hp (dataframe): dataframe for validation set targets
        hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                            names to range of values for grid search
        fixed_hyperparams (dict): dictionary of fixed hyperparameters that
                                  are not included in the grid search

    Output:
        best_estimator (sklearn classifier): fitted sklearn classifier with best performance on
                                             validation set
        best_hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                                 names to values in best_estimator
    '''
    best_estimator = None
    best_hyperparams = {}
    
    # hold best running score
    best_score = 0.0

    # get list of param values
    lists = hyperparams.values()
    
    # get all param combinations
    param_combinations = list(itertools.product(*lists))
    total_param_combinations = len(param_combinations)

    # iterate through param combinations
    for i, params in enumerate(param_combinations, 1):
        # fill param dict with params
        param_dict = {}
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
            
        # create estimator with specified params
        estimator = clf(**param_dict, **fixed_hyperparams)

        # fit estimator
        estimator.fit(X_train, y_train)
        
        # get predictions on validation set
        preds = estimator.predict_proba(X_val)
        
        # compute cindex for predictions
        estimator_score = cindex(y_val, preds[:,1])

        print(f'[{i}/{total_param_combinations}] {param_dict}')
        print(f'Val C-Index: {estimator_score}\n')

        # if new high score, update high score, best estimator
        # and best params 
        if estimator_score >= best_score:
                best_score = estimator_score
                best_estimator = estimator
                best_hyperparams = param_dict

    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    return best_estimator, best_hyperparams


def grid_search_temp():

    # Define ranges for the chosen random forest hyperparameters 
    hyperparams = { \
        'n_estimators': [10, 20, 30,],
        'max_depth': [ 50, 100],
        'min_samples_leaf': [8, 10, 15, 20, 25],
    }    
    fixed_hyperparams = {
        'random_state': 10,}   
    rf = RandomForestClassifier
    best_rf, best_hyperparams = holdout_grid_search(rf, X_train_dropped, y_train_dropped,
                                                    X_val_dropped, y_val_dropped, hyperparams,
                                                    fixed_hyperparams)

    print(f"Best hyperparameters:\n{best_hyperparams}")
    y_train_best = best_rf.predict_proba(X_train_dropped)[:, 1]
    print(f"Train C-Index: {cindex(y_train_dropped, y_train_best)}")
    y_val_best = best_rf.predict_proba(X_val_dropped)[:, 1]
    print(f"Val C-Index: {cindex(y_val_dropped, y_val_best)}")
    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    return best_rf, best_hyperparams