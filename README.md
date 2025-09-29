## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
 
<img width="373" height="487" alt="image" src="https://github.com/user-attachments/assets/fe1c712d-c6ab-4638-8d2c-188ede32cbc4" />
 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
 
<img width="161" height="264" alt="image" src="https://github.com/user-attachments/assets/77666097-cd43-4963-a6c9-4843d3b7dfd8" />
 
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
 
<img width="413" height="488" alt="image" src="https://github.com/user-attachments/assets/3d854986-d21d-4dc3-a940-fcd62c8bff3e" />

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

<img width="428" height="493" alt="image" src="https://github.com/user-attachments/assets/2eab026b-aefa-4dff-b778-6646d5e1c503" />

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

<img width="882" height="493" alt="image" src="https://github.com/user-attachments/assets/1a4c22fc-1b56-4204-8fe0-3d0ff4f40801" />

pd.get_dummies(df2,columns=["nom_0"])

<img width="1182" height="490" alt="image" src="https://github.com/user-attachments/assets/eef4e066-1964-4438-9fd8-fd26417d72fb" />

pip install --upgrade category_encoders

<img width="1665" height="484" alt="image" src="https://github.com/user-attachments/assets/ac7a74af-cc36-4ad1-a777-f146faf976dd" />

from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df

<img width="622" height="499" alt="image" src="https://github.com/user-attachments/assets/2dd5be54-a86e-4c22-ab32-1b34b9a58162" />

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df

<img width="630" height="504" alt="image" src="https://github.com/user-attachments/assets/87a08ba2-b6f8-47c6-b878-093e813ff6d9" />

dfb=pd.concat([df,nd],axis=1)
dfb

<img width="911" height="498" alt="image" src="https://github.com/user-attachments/assets/88463595-41da-4428-aa17-23ca6ce76556" />

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

<img width="721" height="496" alt="image" src="https://github.com/user-attachments/assets/97382fbd-06b8-4782-a9c1-cd93c039dfe8" />

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

<img width="1036" height="568" alt="image" src="https://github.com/user-attachments/assets/2b90cead-044a-47e6-a859-35aa39942220" />

df.skew()

<img width="349" height="274" alt="image" src="https://github.com/user-attachments/assets/d59c8f38-3b2f-42f8-a284-2ae607ba3ee4" />

np.log(df["Highly Positive Skew"])

<img width="325" height="625" alt="image" src="https://github.com/user-attachments/assets/c23a8d43-c6e9-4bff-991a-e29d30c40b30" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="404" height="633" alt="image" src="https://github.com/user-attachments/assets/aa16656e-a8dc-4587-a7c4-29fbb72fbf11" />

np.sqrt(df["Highly Positive Skew"])

<img width="335" height="637" alt="image" src="https://github.com/user-attachments/assets/30664e97-afcb-4547-b635-adc7d900aa1c" />

np.square(df["Highly Positive Skew"])

<img width="323" height="629" alt="image" src="https://github.com/user-attachments/assets/4798ccf9-2c32-4583-9d55-b8986e262e50" />

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

<img width="1335" height="585" alt="image" src="https://github.com/user-attachments/assets/ee0dddd3-e1f0-4f11-b37d-65f591da4a1b" />

df.skew()

<img width="445" height="326" alt="image" src="https://github.com/user-attachments/assets/4c2c2484-381e-4dea-9e67-38196f20a5ff" />

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

<img width="451" height="367" alt="image" src="https://github.com/user-attachments/assets/c370f7d1-47fd-49f3-8ec2-e8ba068d48c8" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

<img width="1662" height="616" alt="image" src="https://github.com/user-attachments/assets/77dde33f-09d7-4177-bf8c-e176bc70fb7c" />

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="811" height="608" alt="image" src="https://github.com/user-attachments/assets/9e980aa3-3425-4660-90ec-0033fb775a0e" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

<img width="798" height="607" alt="image" src="https://github.com/user-attachments/assets/394e3385-1f6c-4915-b364-19bf1d4537ec" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="787" height="605" alt="image" src="https://github.com/user-attachments/assets/df2e0116-bc7c-400b-ba02-5c02a5a8dac8" />

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

<img width="787" height="610" alt="image" src="https://github.com/user-attachments/assets/c48fe39d-ce00-46b0-8ff5-7a7907d897f3" />

dt=pd.read_csv("titanic_dataset.csv")
dt

<img width="1569" height="579" alt="image" src="https://github.com/user-attachments/assets/f0e4ec35-0357-4944-81d3-084070c545d5" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

<img width="781" height="605" alt="image" src="https://github.com/user-attachments/assets/95210449-4394-4a64-b27b-68a1de694561" />

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

<img width="790" height="607" alt="image" src="https://github.com/user-attachments/assets/132a9243-12bd-4cfc-b3e1-886c4c36b158" />

# RESULT:
The given data is read and performed Feature Encoding and Transformation process and the data is saved to a file       

       
