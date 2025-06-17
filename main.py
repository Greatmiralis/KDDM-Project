import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from Preprocess import preprocessor, target_transform, CorrelationFilter, categorical_features, role_feature, numerical_features
from Dataloader import load_characters, split_eval
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor


model = Pipeline( steps=[
    ("preprocessor", preprocessor), # preprocessing
    #("corr_filter", CorrelationFilter(threshold = 0.9)), # filter all columns with correlation over threshold
    ("ridge", Ridge(alpha=1))
    ]    
)

def outputTransformedDFs(x, y, y_transformed):
    x_transformed = preprocessor.fit_transform(x)
    cat_pipe = preprocessor.named_transformers_["cat"]
    onehot_encoder = cat_pipe.named_steps["encoder"]
    onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    
    role_pipe = preprocessor.named_transformers_["role"]
    role_encode = role_pipe.named_steps["encoder"]
    role_feat_names = role_encode.get_feature_names_out(role_feature)
    
    num_pipe = preprocessor.named_transformers_["num"]
    num_encode = num_pipe.named_steps["poly"]
    num_feat_names = num_encode.get_feature_names_out(numerical_features)

    all_feature_names = list(num_feat_names) + list(onehot_feature_names) + list(role_feat_names)
    
    x_df = pd.DataFrame(x_transformed, columns=all_feature_names)
    
    y_df = pd.DataFrame(y_transformed, index=y.index, columns=y.columns if hasattr(y, 'columns') else ['win_prob'])
    df_combined = pd.concat([x_df, y_df], axis=1)   # add both togheter
    df_combined.to_csv("transformed_data", index=False)
    
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
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.title('Scatter plot of arrays')
    plt.show()

if __name__ == "__main__":
    # load and filter data
    df = pd.read_csv("Data-20250331/data.csv")
    df_filtered = df[df["win_prob"] <= 1].copy()

    # train test split
    x = df_filtered.drop("win_prob", axis=1)
    y = df_filtered[["win_prob"]]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

    # target transformation
    y_transformed = target_transform.fit_transform(y_train)
    outputTransformedDFs(X_train, y_train, y_transformed)
    
    # train model
    model.fit(X_train, y_transformed.ravel())
    pred = model.predict(X_test)
    bias = np.mean(y_test.to_numpy().flatten() - pred)
    print("bias pre: ", bias)
    pred = pred + bias
    bias = np.mean(y_test.to_numpy().flatten() - pred)
    print("bias post: ", bias)
    compareArrays(pred, y_test)

    # evaluation
    r2 = r2_score(y_test, pred)
    print("R²:", r2)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("RMSE:", rmse)
    
    pred_df = pd.DataFrame(pred)
    df_combined = pd.concat([y_test.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    df_combined.columns = ["Target values", "Prediction"]
    df_combined["residuals"] = df_combined["Target values"] - df_combined["Prediction"]
    df_combined.to_csv("Target_and_predictions", index=False)
    
    ######### Task 2 Predict win probabilities #########
    # load data
    x = pd.read_csv("Data-20250331/Task2-superheroes-villains-edited.csv")
    name = x["name"]

    # predict win probabilities
    pred = model.predict(x)
    print("Prediction: ", pred)
    pred_df = pd.DataFrame(pred)
    pred_df.columns = ["pred win_prob"]
    df_combined = pd.concat([x.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    df_combined.to_csv("Task2 win probs", index=False)

    # get feature names
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

    # combined feature names
    all_feature_names = list(num_feat_names) + list(onehot_feature_names) + list(role_feat_names)

    # loop over all characters and explain predictions
    for i in range(x.shape[0]):
        row = x.iloc[[i]]
        
        X_transformed = preprocessor.transform(row)
        coefficients = model.named_steps['ridge'].coef_ 
        intercept = model.named_steps['ridge'].intercept_

        # Compute contribution per feature
        contributions = X_transformed.flatten() * coefficients
        prediction = contributions.sum() + intercept
        
        explanation = pd.DataFrame({
            'feature': all_feature_names,
            'value': X_transformed.flatten(),
            'coefficient': coefficients,
            'contribution': contributions
        }).sort_values(by='contribution', key=abs, ascending=False)
        explanation.to_csv("Task2 feature contribution" + row["name"].values[0], index=False)
    
    
    ########## Task3 Villain Analysis #########

    # load data
    x = pd.read_csv("Data-20250331/Task3_villain.csv")

    # prediction
    pred = model.predict(x.drop(" win_prob", axis=1))
    print("Prediction: ", pred)
    pred_df = pd.DataFrame(pred)
    pred_df.columns = ["pred win_prob"]
    df_combined = pd.concat([x.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    df_combined.to_csv("Task3 win probs", index=False)
    
    row = x.iloc[[0]]

    # transform data and retreive coefficients and intercept
    X_transformed = preprocessor.transform(row)
    coefficients = model.named_steps['ridge'].coef_ 
    intercept = model.named_steps['ridge'].intercept_

    # Compute contribution per feature
    contributions = X_transformed.flatten() * coefficients
    prediction = contributions.sum() + intercept

    explanation = pd.DataFrame({
        'feature': all_feature_names,
        'value': X_transformed.flatten(),
        'coefficient': coefficients,
        'contribution': contributions
    }).sort_values(by='contribution', key=abs, ascending=False)
    explanation.to_csv("Task3 feature contribution", index=False)
    
    