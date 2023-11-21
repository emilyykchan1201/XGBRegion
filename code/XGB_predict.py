import xgboost as xgb
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from feature_extraction import feature
from natsort import natsorted
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for plot tables
from plottable import Table, ColumnDefinition
from plottable.plots import percentile_bars, progress_donut

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# define data directory
Dir_model = '/Users/emilyykchan/Library/CloudStorage/OneDrive-ImperialCollegeLondon/test-code/XGBRegion/model'
Dir_data_X = '/Users/emilyykchan/Library/CloudStorage/OneDrive-ImperialCollegeLondon/test-code/XGBRegion/X/SlapFight' # MODIFY if change sport
Dir_data_Y = '/Users/emilyykchan/Library/CloudStorage/OneDrive-ImperialCollegeLondon/test-code/XGBRegion/Y'
Dir_results = '/Users/emilyykchan/Library/CloudStorage/OneDrive-ImperialCollegeLondon/test-code/XGBRegion/result'
# define data files
File_Y = 'SlapFight-test.csv'  # MODIFY if change sport

# Load kinematics data
os.chdir(Dir_data_X)
X_file_list = natsorted([f for f in os.listdir('.') if os.path.isfile(f)])
print(f'Testing cases: {X_file_list}')

# Extract feature for X
X_features = feature(X_file_list, dof=range(4, 12))
X_features = np.reshape(X_features, (X_features.shape[0], -1))

# Load MPS of the test data for validation (if the MPS is available)
os.chdir(Dir_data_Y)
if File_Y != '':
    Y = pd.read_csv(File_Y).drop(columns='subject')

# Load BrainStem model
os.chdir(Dir_model)
model = xgb.XGBRegressor()
model.load_model("model_xgb.json")  # MODIFY if change model
y_pred = model.predict(X_features)
print(y_pred)

# Load multiregion model to predict
os.chdir(Dir_model)

# Output prediction
os.chdir(Dir_results)
pd.DataFrame(y_pred, columns=['BrainStemPred']).to_excel('Predicted_MPS_Slap.xlsx', index=False)  # MODIFY if change sport


# Evaluate model on the test set
if File_Y != '':
    print('R2: ', round(r2_score(Y, y_pred), ndigits=3))
    print('MAE: ', round(mean_absolute_error(Y, y_pred), ndigits=3))
    print('RMSE: ', round(np.sqrt(mean_squared_error(Y, y_pred)), ndigits=3))


# visualise prediction results - Distribution plot
# ref_strain = pd.read_excel(Dir_model+'/Rugby_simulation_1046.xlsx')['90PercStrain']
# plt.figure(figsize=(8, 6))
# plt.title('Where does the predicted strain sits in the 1046 Rugby training cases? ')
# sns.displot(ref_strain, kde=True, bins=20, edgecolor='white', legend=False)
# sns.scatterplot(y_pred)
# plt.show()

# visualise prediction results - position at bar
# scale_max = 0.58  # the max strain in whole population
# res_df = pd.DataFrame([['%.4f' % n for n in y_pred], y_pred/scale_max]).transpose()
# res_df.columns = ['predicted strain', 'percentile of max']
# # Init a figure
# fig, ax = plt.subplots(figsize=(6, len(res_df)*0.3))
# # draw table
# col_defs = [ColumnDefinition("percentile of max", plot_fn=percentile_bars, plot_kw={"is_pct": True,}),]
# tab = Table(res_df, cell_kw={"linewidth": 0,  "edgecolor": "k"},
#             textprops={"ha": "center"},
#             column_definitions=col_defs,
#            )
# plt.show()
