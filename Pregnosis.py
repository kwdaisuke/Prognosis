class Pregnosis:
    def __init__(self):
        pass

    def preprocessing(self, df, return_df = False):
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
        
    def missing(self, df):
        import seaborn as sns
        #frac_missing = df.isnull().sum()/len(df)
        percent_missing = df.isnull().sum()*100/len(df)
        return percent_missing.sort_values(ascending=False), sns.heatmap(df.isnull(), cbar=False)
    
    
    def mean_imputer(self, df):
        '''This func takes dataset with missing values, 
            and impute them with mean values'''
        from sklearn.impute import SimpleImputer
        mean_imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
        imputed_df = mean_imputer.fit_transform(df)
        return imputed_df    

    def regression_imputer(self, df):
        '''This func takes dataset with missing values, 
            and impute them with regression values'''
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        reg_imputer = IterativeImputer()
        imputed_df = reg_imputer.fit_transform(df)
        return imputed_df
    
    def logistic_classification(self, df, targe_variable):
        '''This func split data into train and test, then train into train and valid
            to cross-validate with grid-search and return the algorithm?'''
        from sklean.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        
        y = df.pop(target_variable).values
        X = df.values
        
        classifier = LogisticRegression(max_iter=10000, tol=0.1)
        params_grid = {'logistic_C': np.logspace(-4,4,4)}
        search = GridSearchCV(classifier, params_grid, njobs=-1)
        search.fit(X, y)
        print("Best parameter (CV score=%0.3f):"%search.best_score_)
        print(search.best_params_)
        