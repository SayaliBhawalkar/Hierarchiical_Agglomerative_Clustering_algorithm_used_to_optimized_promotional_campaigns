import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# dataset creation
np.random.seed(42)
data = pd.DataFrame({'Age':np.random.randint(18,65,100),
                     'Income': np.random.randint(30000,100000,100),
                      'SpendingScore': np.random.randint(1,100,100)})

scaler = StandardScaler() # for standardization
scaled_data = scaler.fit_transform(data)

clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
data['Cluster']=clustering.fit_predict(scaled_data)

# visualization
plt.scatter(data['Income'],data['SpendingScore'],c=data['Cluster'],cmap='viridis')
plt.xlabel('Income')
plt.ylabel('SpendingScore')
plt.title('Customer Segmentation')
plt.show()

# user input
new_customer = pd.DataFrame({'Age': [30],
                             'Income':[50000],
                             'SpendingScore':[70]
                            })
scaled_new_customer = scaler.transform(new_customer)
predicted_cluster = clustering.fit_predict(scaled_new_customer)
print(f"The new customer predicted cluster is : {predicted_cluster[0]}")

