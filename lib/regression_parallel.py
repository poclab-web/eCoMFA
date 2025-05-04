from itertools import product
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LassoLars
from sklearn.model_selection import KFold
from multiprocessing import Pool, cpu_count

def regression(X_train,X_test,y_train,y,method):
    """
    Performs regression analysis using ElasticNet or Partial Least Squares (PLS) methods.

    This function applies one of two regression techniques, based on the specified method string:
    - ElasticNet: A linear regression model with both L1 (Lasso) and L2 (Ridge) regularization.
    - PLS (Partial Least Squares): A dimensionality reduction and regression method for data with
      high collinearity or when predictors outnumber observations.

    Args:
        X_train (numpy.ndarray): The training data matrix of shape (n_samples, n_features).
        X_test (numpy.ndarray): The test data matrix of shape (m_samples, n_features).
        y_train (numpy.ndarray): The target values for training, of shape (n_samples,).
        y (numpy.ndarray): The full range of target values, used to clip predictions.
        method (str): A string specifying the regression method and its parameters:
                      - `"ElasticNet alpha l1_ratio"`: Use ElasticNet with specified `alpha` and `l1_ratio`.
                      - `"PLS n_components"`: Use PLS with the specified number of components.

    Returns:
        tuple:
            - coef (numpy.ndarray): The coefficients of the regression model.
            - predict (numpy.ndarray): The predicted values for `X_test`, clipped to the range of `y`.

    Raises:
        ValueError: If the `method` string does not match the expected format or contains unsupported options.

    Notes:
        - ElasticNet: Uses the `alpha` parameter for regularization strength and `l1_ratio` for the balance
          between L1 and L2 regularization.
        - PLS: Uses `n_components` to specify the number of latent variables in the regression.
        - Predictions are clipped to the range of `y` to prevent extreme outliers in the output.

    Example:
        # ElasticNet regression
        coef, pred = regression(X_train, X_test, y_train, y, method="ElasticNet 0.1 0.5")

        # PLS regression
        coef, pred = regression(X_train, X_test, y_train, y, method="PLS 3")
    """
    if "Ridge" in method:
        alpha=float(method.split()[1])
        model=Ridge(alpha=alpha, fit_intercept=False,max_iter=10000)
        model.fit(X_train, y_train)
        coef=model.coef_
        predict=model.predict(X_test)
    elif "Lasso" in method:
        alpha=float(method.split()[1])
        model=Lasso(alpha=alpha, fit_intercept=False,max_iter=10000)
        model.fit(X_train, y_train)
        coef=model.coef_
        predict=model.predict(X_test)
    elif "ElasticNet" in method:
        alpha,l1ratio=map(float, method.split()[1:3])
        model=ElasticNet(alpha=alpha,l1_ratio=l1ratio, fit_intercept=False,max_iter=10000,warm_start=True)
        model.fit(X_train, y_train)
        coef=model.coef_
        predict=model.predict(X_test)
    elif "PLS" in method:
        n_components=int(method.split()[1])
        model = PLSRegression(n_components=n_components,scale=False)
        model.fit(X_train, y_train)
        coef=model.coef_[0]
        predict=model.predict(X_test)[:,0]
    predict=np.clip(predict, np.min(y), np.max(y))
    return coef,predict

def regression_parallel(input):
    X_train,X,y_train,y,method=input
    coef,predict=regression(X_train,X,y_train,y,method)
    cvs=[]
    sort_index=[]
    kf = KFold(n_splits=len(y_train), shuffle=False)
    # kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(y_train):
        _,cv=regression(X_train[train_index],X_train[test_index],y_train[train_index],y,method)
        cvs.extend(cv)
        sort_index.extend(test_index)
    
    original_array = np.empty_like(cvs)
    original_array[sort_index] = cvs
    return method,coef,predict,original_array
def nan_rmse(x,y):
    """
    Calculates the Root Mean Square Error (RMSE) while ignoring NaN values.

    This function computes the RMSE between two arrays, where NaN values in the
    first array (`x`) are ignored in the calculation.

    Args:
        x (numpy.ndarray or pandas.Series): Predicted values, which may contain NaN values.
        y (numpy.ndarray or pandas.Series): Actual values, corresponding to `x`.

    Returns:
        float: The RMSE value, calculated as:
               \[
               \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - x_i)^2}
               \]
               where \( N \) is the number of non-NaN values in `x`.
    """
    return np.sqrt(np.nanmean((y-x)**2))

def nan_r2(x,y):
    """
    Calculates the coefficient of determination (R²) while ignoring NaN values.

    This function computes the R² score between two arrays, where NaN values in
    the first array (`x`) are ignored. The R² score indicates the proportion of
    variance in `y` that is predictable from `x`.

    Args:
        x (numpy.ndarray or pandas.Series): Predicted values, which may contain NaN values.
        y (numpy.ndarray or pandas.Series): Actual values, corresponding to `x`.

    Returns:
        float: The R² value, calculated as:
               \[
               R^2 = 1 - \frac{\sum (y_i - x_i)^2}{\sum (y_i - \bar{y})^2}
               \]
               where:
               - \( \bar{y} \) is the mean of the non-NaN `y` values.
               - The summations ignore NaN values in `x`.
    """
    x,y=x[~np.isnan(x)],y[~np.isnan(x)]
    return 1-np.sum((y-x)**2)/np.sum((y-np.mean(y))**2)
def regression_(path,names):
    """
    Performs regression analysis on molecular grid data using ElasticNet and PLS methods.

    This function reads a preprocessed dataset from a pickle file, normalizes electronic and electrostatic grid 
    data, and applies regression techniques to predict experimental free energy differences (`ΔΔG.expt.`).
    Results include regression coefficients, predictions for training and test sets, and cross-validation scores.

    Args:
        path (str): Path to the input pickle file containing preprocessed molecular data. 
                    The file must include:
                    - electronic and electrostatic grid data columns (e.g., "electronic_fold ...").
                    - "ΔΔG.expt.": Experimental free energy difference values.
                    - "test": Indicator column (0 for training data, 1 for test data).

    Returns:
        None: Saves the regression results and coefficients to new files:
              - A pickle file with predictions and regression results.
              - A CSV file with regression coefficients for grid points.

    Workflow:
        1. Load the dataset from the pickle file and separate training and test sets.
        2. Normalize electronic and electrostatic grid data by their respective norms.
        3. Combine electronic and electrostatic data into feature matrices (`X_train`, `X`) for regression.
        4. Define regression methods:
            - ElasticNet: Using various combinations of alpha (regularization strength) and l1_ratio.
            - PLS (Partial Least Squares): Using varying numbers of components.
        5. Apply regression for each method:
            - Train the model on the training set.
            - Predict values for training and test sets.
            - Perform 5-fold cross-validation on the training set.
        6. Save:
            - Regression coefficients to a CSV file.
            - Predictions, cross-validation results, and regression outputs to a pickle file.

    Example:
        regression_("/path/to/preprocessed_data.pkl")

    Notes:
        - ElasticNet: Combines L1 (Lasso) and L2 (Ridge) regularization.
        - PLS: Performs dimensionality reduction and regression, suitable for collinear data.
        - Coefficients are scaled back to the original grid data scale before saving.

    Raises:
        - FileNotFoundError: If the input file does not exist.
        - KeyError: If required columns are missing in the dataset.
    """
    print(path)
    # print()
    df=pd.read_pickle(path)#.sort_values(by="test")
    # df["test"]=0
    df_train=df[df["test"]==0]
    # names=["electronic","electrostatic"]
    trains=[]
    train_tests=[]
    stds=[]
    for name in names:
        train = df_train.filter(like=f'{name}_fold').to_numpy()
        std=np.linalg.norm(train)
        train_test = df.filter(like=f'{name}_fold').to_numpy()
        train/=std
        train_test/=std
        train_tests.append(train_test)
        trains.append(train)
        stds.append(std)

    y_train,y=df_train["ΔΔG.expt."].values,df["ΔΔG.expt."].values
    methods=[]
    for alpha in np.logspace(-20,-11,10,base=2):
        methods.append(f'Lasso {alpha}')
    for alpha in np.logspace(-20,-11,10,base=2):
        methods.append(f'Ridge {alpha}')
    for alpha,l1ratio in product(np.logspace(-20,-11,10,base=2),[0.5]):#np.round(np.linspace(0.1, 0.9, 9),decimals=10)
        methods.append(f'ElasticNet {alpha} {l1ratio}')
    for n_components in range(1,10):
        methods.append(f'PLS {n_components}')

    #electronic_electrostatic
    grid=pd.DataFrame(index=[col.replace("electronic_fold ","") for col in df.filter(like='electronic_fold ').columns])
    print(np.concatenate(trains,axis=1).shape)
    with Pool(24) as pool:
        # results = pool.map(regression_parallel, [(np.concatenate(trains,axis=1),
        #                                           np.concatenate(train_tests,axis=1),
        #                                           y_train,y,method) for method in methods])
        results = list(pool.imap_unordered(regression_parallel, [(np.concatenate(trains,axis=1),
                                                  np.concatenate(train_tests,axis=1),
                                                  y_train,y,method) for method in methods]))
    for result in results:
        method,coef,predict,original_array=result
        print(method)
        for name,std,coef_ in zip(names,stds,np.split(coef,len(names))):
            grid[f"{method} {name}_coef"]=coef_/std
        df[f'{method} regression'] = np.where(df["test"] == 0, predict, np.nan)
        df[f'{method} prediction'] = np.where(df["test"] == 1, predict, np.nan)
        df.loc[df["test"]==0,f'{method} cv']=original_array
    feature_names = "_".join(names)
    df.to_pickle(path.replace(".pkl", f"_{feature_names}_regression.pkl"))
    grid.to_csv(path.replace(".pkl",f"_{feature_names}_regression.csv"))
    # nan_rmse(df[regression].values,df["ΔΔG.expt."].values)
    


from itertools import combinations

def generate_combinations(elements):
    result = []
    for r in range(1, len(elements) + 1):
        result.extend([list(c) for c in combinations(elements, r)])
    return result

if __name__ == '__main__':
    for feat,path in product(generate_combinations(["electronic","electrostatic"]),[
        "/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/cbs.pkl",
                 "/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/DIP.pkl",
                 "/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/alpine_borane.pkl"
                 ]):#,"binary","lumo"

        regression_(path,feat)
