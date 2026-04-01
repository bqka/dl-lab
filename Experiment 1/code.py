#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import torch


# Creating Tensors/Arrays

# In[20]:


a1 = np.array([1, 2, 3, 4])
a2 = np.array([[1, 2], [3, 4]])
a3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])


# In[21]:


t1 = torch.tensor([1, 2, 3, 4])
t2 = torch.tensor([[1, 2], [3, 4]])
t3 = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

t1


# Basic Operations

# In[22]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b
a * b
a ** 2


# In[23]:


x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

print(x + y)
print(x * y)
print(x ** 2)


# Slicing and Boolean Masking

# In[27]:


a = np.array([[1, 2, 3], [4, 5, 6]])
t = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(t[0, 2])
print(t[:, 2])
print(t[1, :])

print(a[a > 2])
print(t[t > 2])


# Sub Tensors and Reshaping

# In[ ]:


print(t.view(3, 2))
print(t.reshape(3, 2))

x = torch.tensor([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
])

sub = x[0:2, 1:3]
print(sub)


# Squeezing and Broadcasting

# In[31]:


x = torch.tensor([1, 2, 3])

y = x.unsqueeze(0)
z = x.unsqueeze(1)

print(y)
print(z)

x = torch.tensor([[1, 2, 3]])

y = x.squeeze()
print(y)

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([10, 20, 30])

c = a + b
print(c)


# In Place vs Out of Place Operations

# In[38]:


# Out of Place
x = torch.tensor([1, 2, 3])
y = x.add(5)

print("original doesn't change", x)
print(y)

x = torch.tensor([1, 2, 3])
x.add_(5)   # underscore = in-place

print("original changes", x)


# Time Comparison b/w Tensors, Lists and Arrays

# In[26]:


import time

N = 6_700_000

l1 = list(range(N))
l2 = list(range(N))

a1 = np.arange(N)
a2 = np.arange(N)

t1 = torch.arange(N)
t2 = torch.arange(N)

start = time.perf_counter()
l1 = [1, 2, 3, 4]
l2 = [1, 2, 3, 4]

for i in range(len(l1)):
    l2[i] *= l1[i]

end = time.perf_counter()

print("Python Lists:", end - start)

start = time.perf_counter()
a3 = np.multiply(a1, a1)
end = time.perf_counter()
print("Numpy Arrays:", end - start)

start = time.perf_counter()
mult = torch.multiply(t1, t1)
end = time.perf_counter()
print("Tensors:", end - start)


# In[ ]:




