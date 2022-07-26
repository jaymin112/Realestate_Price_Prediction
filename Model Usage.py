#!/usr/bin/env python
# coding: utf-8

# In[2]:


from joblib import dump, load
import numpy as np
model = load ('Realstate.joblib')


# In[6]:


features= np.array([[-0.3283834 , -0.48074454, -0.46974328, -0.27361709, -0.16439335,
       -1.16044209, -1.14593679,  0.01453841, -0.64922514, -0.62529642,
        1.17963243, -0.70762584, -0.16487235]])
model.predict(features)


# In[ ]:





# In[ ]:




