import os
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
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
            "classifier__activation": ["relu", "tanh"],
            "classifier__dropout": np.linspace(0, 1, 10)},
            "classifier__solver": ["adam", "sgd"],
            "classifier__learning_rate_init": np.logspace(-4, 0, 5),
            "classifier__max_iter": np.logspace(1, 3, 3).astype(int),
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


#\section paper plots

def figure_one(raw_thyroid_df, save_path=None):
    """ """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.tight_layout(h_pad=4, w_pad=4)
    sns.histplot(data=raw_thyroid_df, x="age", hue="gender", kde=True, stat="percent", common_norm=True, ax=axes[0, 0])
    axes[0, 0].set(title="Patient Age Distributions")

    raw_thyroid_df.groupby('pathology')['risk'].value_counts(normalize=True).unstack('risk').plot.bar(stacked=True, ax=axes[0, 1], width=0.9)
    # .legend(loc="center right")
    axes[0, 1].legend(bbox_to_anchor=(.95, 0.5), loc='center left', borderaxespad=0.)
    axes[0, 1].set(title="Risk by Pathology Type", xlabel="Pathology", ylabel="Fraction")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)

    sns.boxplot(data=raw_thyroid_df, x="stage", y="age", hue="stage", ax=axes[1, 0])
    axes[1, 0].set(title="Age Distributions by Age", xlabel="Age")

    sns.boxplot(data=raw_thyroid_df, x="recurred", y="age", hue="recurred", ax=axes[1, 1])
    axes[1, 1].set(title="Age Distributions by Recurrence", xlabel="Recurred")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')


def feature_table_plot(X_df, y, save_path):
    """ """
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=100, random_state=42)

    set_labels = ["Train", "Validation", "Total"]
    sets = [X_train, X_test, X_df]

    items = []
    features, feature_values = [], []
    set_arrays = dict(zip(set_labels, [[], [], []]))

    for feature in X_df.columns:
        if feature == "age":
            continue

        for feature_item in X_df[feature].unique():
            features.append(feature.replace("_", " ").replace("hx", "History of ").title())
            feature_values.append(feature_item)

            for i, (label, arr) in enumerate(set_arrays.items()):
                val = sets[i][feature] == feature_item
                counts, percent = int(np.sum(val)), np.mean(val) * 100
                arr.append(f"{counts} ({percent:0.0f}%)")


    df = pd.DataFrame(set_arrays)
    df["Features"] = features
    df[""] = feature_values
    df.set_index(['Features', ""], inplace=True)

    # Render each half to an image
    dfi.export(df.iloc[:28], "df1.png")
    dfi.export(df.iloc[28:], "df2.png")

    # Combine the two images into a single figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    fig.tight_layout(w_pad=-9.1)
    for ax, img_path in zip(axes, ["df1.png", "df2.png"]):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    os.remove("df1.png")
    os.remove("df2.png")


def decision_tree_plot(tree, features, X_df, save_path):
    """ """
    flattened_features = []
    for feature in features:
        if feature == "age":
            flattened_features += ["age"]
        else:
            flattened_features += [feature +"-"+ val for val in X_df[feature].unique()]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_tree(tree, feature_names=flattened_features, class_names=["No Recurrence", "Recurred"],
              filled=True, node_ids=False, impurity=True, label='all', rounded=True, ax=ax)
    
    fig.savefig("figures/tree_nodes_example.png", bbox_inches="tight", dpi=600)


def ROC_plot(models, X_df, y, save_path):
    """ """
    X_train, X_test, y_train, y_test = train_test_split(X_df, y == "Yes", test_size=100, random_state=42)

    print(len(y_test))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred = y_pred == "Yes"
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
    
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    fig.savefig(save_path)


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
    best_models = {}
    for name, model in models.items():
        pbar.set_description(f"Training '{name} - Model'")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=2, scoring="roc_auc")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
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

    return results, best_hyperparameters, best_models


# \section run_models

def run_model_and_feature_sets(X_df, y, feature_sets, models, param_grids):
    """ """
    pdf_set, hp_set = [], []
    feature_set_results = {}
    best_models_set = []
    for feature_set_name, feature_set_list in feature_sets.items():
        print(f"Running models for feature set: {feature_set_name}")
        
        X = X_df[feature_set_list]
        feature_set_preprocessor = make_preprocessor(X)
        results_i, hp_i, best_models = run_models(X, y, models, param_grids, feature_set_preprocessor)
        feature_set_results[feature_set_name] = results_i
        pdf = pd.DataFrame(feature_set_results[feature_set_name]).T
        pdf["Feature Set"] = feature_set_name
        pdf_set.append(pdf)
        hp_set.append(hp_i)
        best_models_set.append(best_models)

    pdf = pd.concat(pdf_set).drop(columns=["Classification Report", "Best Hyperparameters"])
    pdf.set_index('Feature Set', append=True, inplace=True)
    pdf.sort_index(inplace=True)
    pdf = (pdf * 100).astype(str).applymap(lambda x: x[:5]) + "%"
    
    return pdf, hp_set, best_models_set


# ----------------------------------------------------------------------------# 
# --------------------                Main                --------------------# 
# ----------------------------------------------------------------------------# 


def main():
    """ """

    raw_thyroid_df = pd.read_csv(constants.THRYOID_CANCER_DATA_PATH)
    raw_thyroid_df.rename(columns={'Hx Radiothreapy': 'Hx Radiotherapy'}, inplace=True)
    raw_thyroid_df.columns = [c.lower().strip().replace(" ", "_") for c in raw_thyroid_df.columns]

    figure_one(raw_thyroid_df, save_path="figures/figure_1.png")

    y = raw_thyroid_df["recurred"]
    X_df = raw_thyroid_df

    feature_sets = {
        "ATA Risk": ["risk"],
        "ATA Risk Excluded": [col for col in raw_thyroid_df.columns if col not in ["risk", "recurred"]],
        "Full": [col for col in raw_thyroid_df.columns if col != "recurred"]
    }


    feature_table_plot(X_df, y, "figures/feature_table.png")


    paper_model_results, paper_hp_set, best_models_set = run_model_and_feature_sets(X_df, y, feature_sets, PAPER_MODELS, PAPER_PARAM_GRIDS)
    dfi.export(paper_model_results, "figures/paper_model_results_table.png")

    new_model_results, new_hp_set, new_models_set = run_model_and_feature_sets(X_df, y, feature_sets, NEW_MODELS, NEW_PARAM_GRIDS)
    dfi.export(new_model_results, "figures/new_model_results_table.png")

    # print(new_model_results)
    # print(new_hp_set[0])

    tree = best_models_set[-1]["DecisionTree"][-1]
    decision_tree_plot(tree, feature_sets["Full"], X_df, "figures/tree_nodes_example.png")

    fi = 1
    all_best_models = {}
    all_best_models.update(best_models_set[fi])
    all_best_models.update(new_models_set[fi])
    
    ROC_plot(all_best_models, X_df, y, "figures/ROC.png")

    # best_rf.fit(preprocessor.fit_transform(X_train), y_train)


    # # Visualization of feature importance for Random Forest
    # best_rf = best_models_set[-1]["RandomForest"]
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
