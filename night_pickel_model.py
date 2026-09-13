import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score



# feature columns


feature_cols = [
    "h3.3_gene_body_ct12",
    "h3.3_promoters_ct12",
    "h3k27ac_ct12",
    "h3k9ac_ct12",
    "h3k4me3_ct12",
    "h3k4me1_ct12",
    "h3k36me3_gene_body_ct12",
    "h3k79me2_gene_body_ct12",
    "per1_promoters_ct12",
    "per2_promoters_ct12",
    "rnapol2_promoter_ct12",
    "per1_gene_body_ct12",
    "per2_gene_body_ct12",

    "h3.3_gene_body_ct16",
    "h3.3_promoters_ct16",
    "h3k27ac_ct16",
    "h3k9ac_ct16",
    "h3k4me3_ct16",
    "h3k4me1_ct16",
    "h3k36me3_gene_body_ct16",
    "h3k79me2_gene_body_ct16",
    "per1_promoters_ct16",
    "per2_promoters_ct16",
    "rnapol2_promoter_ct16",
    "per1_gene_body_ct16",
    "per2_gene_body_ct16",

    "h3.3_gene_body_ct20",
    "h3.3_promoters_ct20",
    "h3k27ac_ct20",
    "h3k9ac_ct20",
    "h3k4me3_ct20",
    "h3k4me1_ct20",
    "h3k36me3_gene_body_ct20",
    "h3k79me2_gene_body_ct20",
    "per1_promoters_ct20",
    "per2_promoters_ct20",
    "rnapol2_promoter_ct20",
    "per1_gene_body_ct20",
    "per2_gene_body_ct20"

]


feature_names = [
    "h33_gb",
    "h33_pm",
    "k27ac",
    "k9ac",
    "k4me3",
    "k4me1",
    "k36me3_gb",
    "k79me2_gb",
    "per1_promoters",
    "per2_promoters",
    "rnapol2_promoter",
    "per1_gb",
    "per2_gb"
]


if __name__ == "__main__":


    # Load complete 45k non-clock genes dataset


    data_all = pd.read_csv(
        "/home/ibab/Downloads/mrop/codes/cross_testing_clock_vs_non_clock/data_sets/"
        "clock_genes_1500.csv",
        na_values=["NA", "null", "?", " "],
        engine="python"
    )



    # Load the test set


    data_ct0 = pd.read_csv(
        "/home/ibab/Downloads/mrop/codes/cross_testing_clock_vs_non_clock/data_sets/"
        "clock_genes_150_test_set.csv",
        na_values=["NA", "null", "?", " "],
        engine="python"
    )


    print("Total genes in complete dataset:", len(data_all))
    print("test set genes in selected file:", len(data_ct0))



    # Identify gene-ID columns


    gene_col_all = data_all.columns[0]
    gene_col_ct0 = data_ct0.columns[0]


    # Remove spaces from gene IDs
    data_all[gene_col_all] = (
        data_all[gene_col_all]
        .astype(str)
        .str.strip()
    )

    data_ct0[gene_col_ct0] = (
        data_ct0[gene_col_ct0]
        .astype(str)
        .str.strip()
    )



    # Remove all test set genes from training


    ct0_gene_ids = set(
        data_ct0[gene_col_ct0]
    )

    training_data = data_all[
        ~data_all[gene_col_all].isin(ct0_gene_ids)
    ].copy()


    removed_gene_count = (
        len(data_all) - len(training_data)
    )

    print("test set genes removed from training:", removed_gene_count)
    print("Genes remaining for training:", len(training_data))



    # Prepare training features and target


    X_train = training_data[
        feature_cols
    ].values

    y_train = training_data[[
        "ct12_rpkm_cm_avg",
        "ct16_rpkm_cm_avg",
        "ct20_rpkm_cm_avg"]
    ].values


    # Scale training features


    scalar_X = RobustScaler()

    X_train_scaled = scalar_X.fit_transform(
        X_train
    )



    # Log-transform training RNA-expression target

    y_train_scaled = np.log1p(
        y_train
    )



    # Define Random Forest model


    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_split=2,
        max_depth=None,
        n_jobs=-1,
        random_state=999
    )



    # 10-fold cross-validation on the training data

    #
    # cv = KFold(
    #     n_splits=10,
    #     shuffle=True,
    #     random_state=999
    # )
    #
    # cv_r2_scores = cross_val_score(
    #     model,
    #     X_train_scaled,
    #     y_train_scaled.ravel(),
    #     cv=cv,
    #     scoring="r2"
    # )
    #
    # print(
    #     "\n10-fold CV R2 scores:",
    #     np.round(cv_r2_scores, 4)
    # )
    #
    # print(
    #     "Mean CV R2: {:.4f} (+/- {:.4f})".format(
    #         cv_r2_scores.mean(),
    #         cv_r2_scores.std()
    #     )
    # )



    # Train final Random Forest on full training data


    model.fit(
        X_train_scaled,
        y_train_scaled
    )


    print("\nFinal Random Forest model trained successfully on full training set.")



    # Create pickle object


    pickle_object = {
        "model": model,
        "scaler_X": scalar_X,
        "feature_names": feature_names,
        "feature_columns": feature_cols,
        "target_column": ["ct12_rpkm_cm_avg", "ct16_rpkm_cm_avg", "ct20_rpkm_cm_avg"],
        "target_transformation": "log1p",
        "training_group": (
            "All genes excluding test set genes"
        ),
        # "cv_r2_scores": cv_r2_scores,
        # "cv_r2_mean": cv_r2_scores.mean(),
        # "cv_r2_std": cv_r2_scores.std()
    }



    # Save pickle object


    pickle_file = (
        "/home/ibab/Downloads/mrop/codes/cross_testing_clock_vs_non_clock/day_vs_night/"
        "clock_night_model.pkl"
    )

    with open(pickle_file, "wb") as file:
        pickle.dump(
            pickle_object,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )


    print("\nPickle object saved as:")
    print(pickle_file)