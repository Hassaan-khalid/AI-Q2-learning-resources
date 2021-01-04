#!/usr/bin/env python
# coding: utf-8

# # Question# 1

# In[1]:


import numpy as np
arr=np.array([0,1,2,3,4,5,6,7,8,9])


# In[2]:


print("Numpy 1D Arrray:")
print(arr)


# In[3]:



arr=np.array([0,1,2,3,4,5,6,7,8,9])
arr_2d=np.reshape(arr, (2,5))
print(arr_2d)


# # Question # 02

# In[4]:


import numpy as np


# In[5]:


arr = np.array([[0,1,2,3,4],[5,6,7,8,9],[1,1,1,1,1],[1,1,1,1,1]])


# In[6]:


out_arr = np.vstack((arr))


# In[7]:


out_arr


# # Question # 03

# In[8]:


import numpy as np


# In[9]:


a = np.array((0,1,2,3,4,1,1,1,1,1))
b = np.array((5,6,7,8,9,1,1,1,1,1))
np.hstack((a,b))


# In[10]:


a = np.array([[0],[1],[2],[3],[4],[1],[1],[1],[1],[1]])
b = np.array([[5],[6],[7],[8],[9],[1],[1],[1],[1],[1]])


# In[11]:


arr=np.hstack((a,b))
arr


# # Question # 04

# In[12]:


import numpy as np


# In[13]:


array2D = np.array([[0,1,2], [3,4,5], [6,7,8]])
# printing initial arrays 
print("Given array:\n",array2D)
# Using flatten()
res = array2D.flatten()


# In[14]:


print("Flattened array:\n ", res)


# # Question # 05

# In[15]:


import numpy as np
arr2D = np.array([[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]])
# printing initial arrays 
print("Given array:\n",arr2D)
# Using flatten() to change dimension
res = arr2D.flatten()


# In[16]:


print("Flattened array:\n ", res)


# # Question# 06

# In[17]:


import numpy as np


# In[19]:


arr = np.arange(15).reshape(-1,3)
arr


# # Question 7

# In[ ]:


import numpy as np


# In[ ]:


x = np.random.random((5,5))
print("Original Array:")
print(x) 
arr_2d_square = np.square(x)
print(f'square array :\n{arr_2d_square}')


# # Question # 08

# In[ ]:


import numpy as np


# In[ ]:


np.random.seed(456)


# In[20]:


arr = np.random.randint(30,size=(5,6))
print(arr)
arr.mean()


# # Question no 09

# In[ ]:


import numpy as np


# In[ ]:


np.random.seed(456)
arr = np.random.randint(30,size =(5,6))
print(arr)


# In[21]:


np.std(arr)


# # Question no 10

# In[ ]:


import numpy as np


# In[22]:


np.random.seed(456)
arr = np.random.randint(30,size =(5,6))
np.median(arr)


# # Question no 11

# In[23]:


import numpy as np


# In[24]:


np.random.seed(456)
arr = np.random.randint(30,size=(5,6))
arr


# In[25]:


arr.T


# # Question no 12

# In[26]:


import numpy as np


# In[27]:


arr = np.arange(16).reshape(4,4)
print(arr)
np.diagonal(arr)


# # Question no 13

# In[28]:


import numpy as np


# In[31]:


arr = np.arange(16).reshape(4,4)
print(arr)
np.linalg.det(arr)


# # Question no 14

# In[32]:


import numpy as np


# In[33]:


arr = np.arange(10)
print(arr)
print(np.percentile(arr,5))
print(np.percentile(arr,95))


# # Question no 15

# In[34]:


import numpy as np


# In[38]:


arr = np.array([[0,0,0],[2,4,6],[9,0,8],[8,7,6]])
arr


# In[39]:


arr_with_nan = np.isnan(arr)


# In[42]:


(arr_with_nan)


# #                     OR

# In[43]:


import numpy as np
     
b = [[1,2,3],[np.nan,np.nan,2]] 
arr = np.array(b)

print(arr)
print(np.isnan(arr))

x = np.isnan(arr)

#replacing NaN values with 0
arr[x] = 0
print("After replacing NaN values:")
arr


# In[ ]:




