# %% [markdown]
# ## Regresie Logistica - Exemplu Bank Customer Data

# %%
import numpy as np
import pandas as pd

from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# This bank marketing dataset is open-sourced and available for download at the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#).
# 
# It was originally created by: [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

# %%
bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
bank_full.head()

# %%
bank_full.info()

# %%
X = bank_full.iloc[:,18:37].values
y = bank_full.iloc[:,17].values

# %%
LogReg = LogisticRegression()
LogReg.fit(X, y)

# %%
james = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1]).reshape(1, -1)
y_pred = LogReg.predict(james)
y_pred_proba = LogReg.predict_proba(james)
# get name of variable james

print('Utilizatorul James', 'nu accepta oferta' if y_pred[0] == 1 else 'nu accepta oferta')
print("Probabilitatea ca el sa accepte: ", y_pred_proba[0][1])

# %%
bill = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1]).reshape(1, -1)
y_pred = LogReg.predict(bill)
y_pred_proba = LogReg.predict_proba(bill)
# get name of variable james

print('Utilizatorul Bill', 'accepta oferta' if y_pred[0] == 1 else 'nu accepta oferta')
print("Probabilitatea ca el sa accepte: ", y_pred_proba[0][1])

# %%



