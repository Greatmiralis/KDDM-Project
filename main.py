import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from Preprocess import preprocessor, target_transform, CorrelationFilter, categorical_features, role_feature, numerical_features
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from missingpy import MissForest
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
model = Pipeline( steps=[
    ("preprocessor", preprocessor), # preprocessing
    ("rfe", RFE(Ridge(alpha=1), n_features_to_select=100)),
    ("ridge", Ridge(alpha=1))
    ]    
)
    
def compareArrays(arr1, arr2):
    
    arr2 = arr2.to_numpy().flatten()
    mask = arr2 <= 1
    
    arr1_filtered = arr1[mask]
    arr2_filtered = arr2[mask]
    
    max_diff = np.max(np.abs(arr1_filtered - arr2_filtered))
    mean_diff = np.mean(np.abs(arr1_filtered - arr2_filtered))
    print("Max Diff:" + str(max_diff))
    print("Mean Diff:" + str(mean_diff))

    plt.scatter(arr1_filtered, arr2_filtered)
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()])   # max of both axes
    ]

    plt.plot(lims, lims, 'r--', label='Ideal: y = x')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.title('Scatter plot of arrays')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data_input.csv")
    df_filtered = df[df["win_prob"] <= 1].copy()
    df_filtered = df_filtered[df_filtered["win_prob"].notna()]
    df_filtered = df_filtered.drop("intelligence", axis=1)
    
    y = df_filtered[["win_prob"]]
    df_filtered = df_filtered.drop("win_prob", axis=1)    
        
    name_col = df_filtered['name']
    df_to_impute = df_filtered.drop(columns=['name'])
    role_col = df_to_impute['role']
    df_to_impute = df_to_impute.drop(columns=['role'])
    
    original_categories = {}
    for col in categorical_features:
        df_to_impute[col] = df_to_impute[col].astype('category')
        original_categories[col] = df_to_impute[col].cat.categories
        # Convert to codes, replace -1 with np.nan for missing
        df_to_impute[col] = df_to_impute[col].cat.codes.replace(-1, np.nan)
        
    imputer = MissForest(max_iter=40, random_state=43, max_features='sqrt')
    df_imputed_np = imputer.fit_transform(df_to_impute)
    df_imputed = pd.DataFrame(df_imputed_np, columns=df_to_impute.columns)
    df_imputed['name'] = name_col.values
    df_imputed['role'] = role_col.values

    cols = df_filtered.columns.tolist()
    df_filtered = df_imputed[cols]
        
    for col in categorical_features:
        # Round and convert to int before using as codes
        codes = df_filtered[col].round().astype(int)
        df_filtered[col] = pd.Categorical.from_codes(codes, categories=original_categories[col])
    
    df_filtered.to_csv("Outputs/ImputedAndDroppedData.csv")
    x = df_filtered

            
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

    y_transformed = target_transform.fit_transform(y_train)

    param_grid = {
        'rfe__n_features_to_select': [50, 75, 100, 106],  # try different numbers of features
        'ridge__alpha': [0.01, 0.1, 0.5, 1, 5, 10, 100],          # Ridge regularization strength
        'preprocessor__num__poly__degree': [1, 2],        # polynomial degree
        'preprocessor__num__poly__interaction_only': [False, True],  # interaction terms only or full polynomial
    }
    
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_transformed.ravel())
    print("Best parameters found:")
    print(grid_search.best_params_)
    model = grid_search.best_estimator_
    pred = model.predict(X_test)
    
    bias = np.mean(y_test.to_numpy().flatten() - pred)
    print("bias pre: ", bias)
    compareArrays(pred, y_test)
    
    r2 = r2_score(y_test, pred)
    print("R²:", r2)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("RMSE:", rmse)
    
    rmse = mean_squared_error(y_test, pred)
    print("MSE:", rmse)
    
    pred_df = pd.DataFrame(pred)
    df_combined = pd.concat([y_test.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    df_combined.columns = ["Target values", "Prediction"]
    df_combined["residuals"] = df_combined["Target values"] - df_combined["Prediction"]
    df_combined.to_csv("Outputs/Target_and_predictions", index=False)
    
    # Task 2
    
    x = pd.read_csv("Data-20250331/Task2-superheroes-villains-edited.csv")
    name = x["name"]
    pred = model.predict(x)
    print("Prediction: ", pred)
    pred_df = pd.DataFrame(pred)
    pred_df.columns = ["pred win_prob"]
    df_combined = pd.concat([x.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    df_combined.to_csv("Outputs/Task2 win probs", index=False)
    
    preprocessor = model.named_steps['preprocessor']
    cat_pipe = preprocessor.named_transformers_["cat"]
    onehot_encoder = cat_pipe.named_steps["encoder"]
    onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    
    role_pipe = preprocessor.named_transformers_["role"]
    role_encode = role_pipe.named_steps["encoder"]
    role_feat_names = role_encode.get_feature_names_out(role_feature)
    
    num_pipe = preprocessor.named_transformers_["num"]
    num_encode = num_pipe.named_steps["poly"]
    num_feat_names = num_encode.get_feature_names_out(numerical_features)
    
    rfe = model.named_steps["rfe"]
    selected_mask = rfe.support_  # Boolean mask for selected features
    

    all_feature_names = np.array(list(num_feat_names) + list(onehot_feature_names) + list(role_feat_names))
    selected_features = all_feature_names[selected_mask]
    for i in range(x.shape[0]):
        row = x.iloc[[i]]
        
        X_transformed = preprocessor.transform(row)
        X_transformed = model.named_steps["rfe"].transform(X_transformed)
        coefficients = model.named_steps['ridge'].coef_ 
        intercept = model.named_steps['ridge'].intercept_

        # Compute contribution per feature
        contributions = X_transformed.flatten() * coefficients
        prediction = contributions.sum() + intercept
        
        explanation = pd.DataFrame({
            'feature': selected_features,
            'value': X_transformed.flatten(),
            'coefficient': coefficients,
            'contribution': contributions
        }).sort_values(by='contribution', key=abs, ascending=False)
        explanation.to_csv("Outputs/Task2 feature contribution" + row["name"].values[0], index=False)
    
    
    # Task3
    
    x = pd.read_csv("Data-20250331/Task3_villain.csv")
    pred = model.predict(x.drop(" win_prob", axis=1))
    print("Prediction: ", pred)
    pred_df = pd.DataFrame(pred)
    pred_df.columns = ["pred win_prob"]
    df_combined = pd.concat([x.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    df_combined.to_csv("Outputs/Task3 win probs", index=False)
    
    row = x.iloc[[0]]

    X_transformed = preprocessor.transform(row)
    X_transformed = model.named_steps["rfe"].transform(X_transformed)
    coefficients = model.named_steps['ridge'].coef_ 
    intercept = model.named_steps['ridge'].intercept_

    # Compute contribution per feature
    contributions = X_transformed.flatten() * coefficients
    prediction = contributions.sum() + intercept

    explanation = pd.DataFrame({
        'feature': selected_features,
        'value': X_transformed.flatten(),
        'coefficient': coefficients,
        'contribution': contributions
    }).sort_values(by='contribution', key=abs, ascending=False)
    explanation.to_csv("Outputs/Task3 feature contribution", index=False)
    
    