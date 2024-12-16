import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from tqdm.auto import tqdm
import constants


# Models to evaluate
PAPER_MODELS = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "ANN": MLPClassifier(),
}

# Hyperparameters for GridSearch
PAPER_PARAM_GRIDS = {
    "KNN": {"classifier__n_neighbors": np.arange(3, 10), "classifier__weights": ["uniform", "distance"]},
    "SVM": {"classifier__C": np.logspace(-2, 0, 3),
            "classifier__kernel": ["linear", "rbf"],
            "classifier__class_weight": ["balanced", None],
            "classifier__gamma": ["scale", "auto"],
            },
    "DecisionTree": {"classifier__max_depth": np.arange(3, 10),
                     "classifier__criterion": ["gini", "entropy"],
                     "classifier__class_weight": ["balanced", None],
                     "classifier__max_features": ["sqrt", "log2", None], # no 'auto' option for DT in Sklearn
                     },

    "RandomForest": {"classifier__n_estimators": [50, 100],
                     "classifier__max_depth": [3, 5, 10],
                     "classifier__criterion": ["gini", "entropy"],
                     "classifier__class_weight": ["balanced", None],
                     "classifier__max_features": ["sqrt", "log2", None], # no 'auto' option for DT in Sklearn
                     },

    "ANN": {"classifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            # "classifier__activation": ["relu", "tanh"],
            # "classifier__dropout": np.linspace(0, 1, 10)},
            # "classifier__solver": ["adam", "sgd"],
            # "classifier__learning_rate_init": np.logspace(-4, 0, 5),
            # "classifier__max_iter": np.logspace(1, 3, 3).astype(int),
            }
}


NEW_MODELS = {
    "LR": LogisticRegression(),
    "AdaBoost": AdaBoostClassifier(),
}

NEW_PARAM_GRIDS = {
    "LR": {"classifier__class_weight": ["balanced", None]},
    "AdaBoost": {"classifier__n_estimators": [50, 100, 150, 200]}, #, "classifier__max_depth": [1, 2, 3],},
    "LDA": {"classifier__shrinkage": [None, "auto"]},
}


# ----------------------------------------------------------------------------# 
# --------------------           Preprocessing            --------------------# 
# ----------------------------------------------------------------------------# 


def make_preprocessor(X):
    """ """
    
    categorical_features = [col for col in X.columns if col != "age"]
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])
    
    transformers = [("cat", categorical_transformer, categorical_features)]

    if "age" in X.columns:
        numeric_features = ["age"]
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        transformers.append(("num", numeric_transformer, numeric_features))

    preprocessor = ColumnTransformer(transformers)
    return preprocessor


# ----------------------------------------------------------------------------# 
# --------------------           Running Models           --------------------# 
# ----------------------------------------------------------------------------# 


def run_models(X, y, models, param_grids, preprocessor, seed=42):
    """ """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=seed)

    # Results storage
    results, best_hyperparameters = {}, {}
    pbar = tqdm(total=len(models))
    for name, model in models.items():
        pbar.set_description(f"Training '{name} - Model'")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=2, scoring="roc_auc")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        hp_values = best_model["classifier"].get_params()

        hp_list = [hp.replace("classifier__", "") for hp in param_grids[name].keys()]
        best_hyperparameters[name] = {hp: hp_values[hp] for hp in hp_list}
        best_hp_str = "\n".join(f"{hp}: {hp_values[hp]}".title() for hp in hp_list)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results[name] = {
            "Best Hyperparameters": best_hp_str,
            "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp),
            "PPV": tp / (tp + fp),
            "NPV": tn / (tn + fn),
            "AUC": auc,
            "Accuracy": acc,
            "Classification Report": classification_report(y_test, y_pred)
        }
        pbar.update(1)

    return results, best_hyperparameters


# \section run_models

def run_model_and_feature_sets(X_df, y, feature_sets, models, param_grids):
    """ """
    pdf_set, hp_set = [], []
    feature_set_results = {}
    for feature_set_name, feature_set_list in feature_sets.items():
        print(f"Running models for feature set: {feature_set_name}")
        
        X = X_df[feature_set_list]
        feature_set_preprocessor = make_preprocessor(X)
        results_i, hp_i = run_models(X, y, models, param_grids, feature_set_preprocessor)
        feature_set_results[feature_set_name] = results_i
        pdf = pd.DataFrame(feature_set_results[feature_set_name]).T
        pdf["Feature Set"] = feature_set_name
        pdf_set.append(pdf)
        hp_set.append(hp_i)

    pdf = pd.concat(pdf_set).drop(columns=["Classification Report", "Best Hyperparameters"])
    pdf.set_index('Feature Set', append=True, inplace=True)
    pdf.sort_index(inplace=True)
    pdf = (pdf * 100).astype(str).applymap(lambda x: x[:5]) + "%"
    
    return pdf, hp_set


# ----------------------------------------------------------------------------# 
# --------------------                Main                --------------------# 
# ----------------------------------------------------------------------------# 


def main():
    """ """

    raw_thyroid_df = pd.read_csv(constants.THRYOID_CANCER_DATA_PATH)
    raw_thyroid_df.rename(columns={'Hx Radiothreapy': 'Hx Radiotherapy'}, inplace=True)
    raw_thyroid_df.columns = [c.lower().strip().replace(" ", "_") for c in raw_thyroid_df.columns]

    y = raw_thyroid_df["recurred"]
    X_df = raw_thyroid_df

    feature_sets = {
        "ATA Risk": ["risk"],
        "ATA Risk Excluded": [col for col in raw_thyroid_df.columns if col not in ["risk", "recurred"]],
        "Full": [col for col in raw_thyroid_df.columns if col != "recurred"]
    }


    paper_model_results, paper_hp_set = run_model_and_feature_sets(X_df, y, feature_sets, PAPER_MODELS, PAPER_PARAM_GRIDS)
    dfi.export(paper_model_results, "paper_model_results_table.png")

    new_model_results, new_hp_set = run_model_and_feature_sets(X_df, y, feature_sets, NEW_MODELS, NEW_PARAM_GRIDS)
    dfi.export(new_model_results, "new_model_results_table.png")

    print(new_model_results)
    print(new_hp_set[0])

# # Visualization of feature importance for Random Forest
# best_rf = models["RandomForest"]
# best_rf.fit(preprocessor.fit_transform(X_train), y_train)

# feature_names = (
#     numeric_features + list(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out())
# )
# importances = best_rf.feature_importances_
# indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(10, 6))
# sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
# plt.title("Feature Importances - Random Forest")
# plt.show()


if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
