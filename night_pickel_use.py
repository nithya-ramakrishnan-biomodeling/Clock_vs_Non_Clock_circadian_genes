import pickle
import numpy as np
import pandas as pd


from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)



# features used while training the saved clock night model


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



# Load the saved Clock night model pickle object


pickle_file = ("/home/ibab/Downloads/mrop/codes/cross_testing_clock_vs_non_clock/day_vs_night/clock_night_model.pkl")

with open(pickle_file, "rb") as file:
    pickle_object = pickle.load(file)

model = pickle_object["model"]
scalar_X = pickle_object["scaler_X"]
feature_names = pickle_object["feature_names"]

print("clock night model loaded successfully")

# print("\nTraining 10-fold CV Results")
# print("---------------------------")
# print("CV R2 scores :", pickle_object["cv_r2_scores"])
# print("Mean CV R2   :", pickle_object["cv_r2_mean"])
# print("Std CV R2    :", pickle_object["cv_r2_std"])

# Load test genes


ct0_data = pd.read_csv(
    "/home/ibab/Downloads/mrop/codes/cross_testing_clock_vs_non_clock/data_sets/non_clock_genes_1500.csv",
    na_values=["NA", "null", "?", " "],
    engine="python"
)

print("Number of clock test genes:", len(ct0_data))


# Prepare features of test genes


X_ct0_genes = ct0_data[
    feature_cols
].values


# Use the scaler fitted during CT0 model training
X_ct0_genes_scaled = scalar_X.transform(
    X_ct0_genes
)



# Predict RNA expression


y_pred_log = model.predict(
    X_ct0_genes_scaled
)


# Convert predictions back to original expression scale
y_pred_original = np.expm1(
    y_pred_log
)



# Actual RNA expression

y_actual_original = ct0_data[[
    "ct12_rpkm_cm_avg", "ct16_rpkm_cm_avg", "ct20_rpkm_cm_avg"]
].values

y_actual_log = np.log1p(
    y_actual_original
)



# Metrics on log1p scale


r2_log = r2_score(
    y_actual_log,
    y_pred_log
)
#
mae_log = mean_absolute_error(
    y_actual_log,
    y_pred_log
)
#
# mse_log = mean_squared_error(
#     y_actual_log,
#     y_pred_log
# )
#
#
# print("\nNON-CLOCK test GENES: CT0 EXPRESSION PREDICTION")
# print("R2:", r2_log)
# print("MAE:", mae_log)
# print("MSE:", mse_log)


r2_ct12 = r2_score(
    y_actual_log[:, 0],
    y_pred_log[:, 0]
)

r2_ct16 = r2_score(
    y_actual_log[:, 1],
    y_pred_log[:, 1]
)

r2_ct20 = r2_score(
    y_actual_log[:, 2],
    y_pred_log[:, 2]
)

print("\nCross-testing on clock genes")
print("CT12 R²:", r2_ct12)
print("CT16 R²:", r2_ct16)
print("CT20 R²:", r2_ct20)
print("Overall R²:", r2_log)



mae_ct12 = mean_absolute_error(
    y_actual_log[:, 0],
    y_pred_log[:, 0]
)

mae_ct16 = mean_absolute_error(
    y_actual_log[:, 1],
    y_pred_log[:, 1]
)

mae_ct20 = mean_absolute_error(
    y_actual_log[:, 2],
    y_pred_log[:, 2]
)

print("CT12 MAE:", mae_ct12)
print("CT16 MAE:", mae_ct16)
print("CT20 MAE:", mae_ct20)
print("Overall MAE²:", mae_log)




