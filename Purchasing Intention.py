#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[15]:


shoppers_df = pd.read_csv(r"D:\Marketing analytics_course\marketing analytics\online_shoppers_intention.csv")


# In[16]:


# Transform Boolean and String Values into Numbers
shoppers_df[["Weekend", "Revenue"]] = shoppers_df[["Weekend", "Revenue"]].values.astype(int)


# In[17]:


# First the dichotomous “VisitorType”, next the slightly more complex one for “Month

shoppers_df["VisitorType"] = np.asarray([1 if val == "Returning_Visitor" else 0 for val in shoppers_df["VisitorType"].values])


# In[18]:


shoppers_df[:2]


# In[19]:


# convert month
months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for i, val in enumerate(shoppers_df["Month"]):
    shoppers_df["Month"][i] = months.index(val)


# In[ ]:


#  a .map function which does all this much more easily
# mapping_dict = {False : 0, True : 1}
# shoppers_df["Weekend", "Revenue"].map(mapping_dict)


# In[20]:


# normalization could be finished with just one more function
def normalize(column):
    shoppers_df[column] = np.asfarray((shoppers_df[column])/
    float(max(shoppers_df[column]) * 0.99) + 0.01)


# In[21]:


# Rearranged the dataframe so that the output variable “Revenue” is the first instead of the last column
column_list = shoppers_df.columns.tolist()
column_list.insert(0, column_list[-1])
column_list.pop()
shoppers_df = shoppers_df[column_list]


# In[ ]:


# Create Train and Test Data


# In[22]:


from sklearn.model_selection import train_test_split
shoppers_train, shoppers_test = train_test_split(shoppers_df, test_size=0.15)
shoppers_train.to_csv(r"D:\Marketing analytics_course\marketing analytics\shoppers_train.csv",index = None, header = True)
shoppers_test.to_csv(r"D:\Marketing analytics_course\marketing analytics\shoppers_test.csv",index = None, header = True)


# In[25]:


# input_nodes = 17
# hidden_nodes = 8
# output_nodes = 1
# learning_rate = 0.2
# n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# In[72]:


# from scipy import special
# class NeuralNetwork:
    
#     def __init__(self, inputnodes, hiddennodes, outputnodes,      learningrate):
#         self.inodes = inputnodes
#         self.hnodes = hiddennodes
#         self.onodes = outputnodes
 
#         self.lr = learningrate
        
#         self.activation_function = lambda x: scipy.special.expit(x)
        
#         self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
#         self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
    
#     def train(self, inputs_list, targets_list):
        
#         inputs = np.array(inputs_list, ndmin=2).T
#         targets = np.array(targets_list, ndmin=2).T
        
#         hidden_inputs = np.dot(self.wih, inputs)
#         hidden_outputs = self.activation_function(hidden_inputs)
        
#         final_inputs = np.dot(self.who, hidden_outputs)
#         final_outputs = self.activation_function(final_inputs)
        
#         output_errors = targets - final_outputs
#         hidden_errors = np.dot(self.who.T, output_errors) 
        
#         self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
#         #print("Weights Hidden-Output :", self.who)
#         #print("inputs :", inputs)
        
#         self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


# In[73]:


# input_nodes = 17
# hidden_nodes = 8
# output_nodes = 1
# learning_rate = 0.2
# n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# In[75]:


in_li = shoppers_train.iloc[:,1:]
ta_li =shoppers_train.iloc[:,:1]


# In[90]:


from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[98]:


# regressor3 = RandomForestClassifier(bootstrap=True,
#                                    class_weight=None,
#                                    criterion='gini',
#                                     max_depth=9,
#                                     max_features='auto',
#                                     max_leaf_nodes=None,
#                                     min_samples_leaf=1,
#                                     min_impurity_split=2,
#                                     min_weight_fraction_leaf=0.0,
#                                     n_estimators=300,
#                                     n_jobs=None                            
#                                    )

# regressor3.fit(in_li,ta_li)


# In[97]:


x = shoppers_df.iloc[:,1:]
y = shoppers_df.iloc[:,:1]
X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=12)


# In[99]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=24, random_state=0)
random = regressor.fit(X_train, y_train)


# In[100]:


y_pred = random.predict(X_test)
random.score(X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




