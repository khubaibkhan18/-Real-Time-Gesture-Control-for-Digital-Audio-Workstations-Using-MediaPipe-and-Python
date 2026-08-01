import numpy as np
# creating a list
lst = np.arange(0, 20, 1)
print(type(lst))
# list slicing and methods
lst = []
for idx in range(20):
    lst.append(idx)
print(lst[:5])
print(lst[5:])
print(lst)
print(lst[-1])
print(len(lst))
lst.append('Banana')
print(lst)
print(lst.pop(5)) # this removes this part from the list
lst.remove('Banana')
print(sorted(lst, reverse= True))
print(max(lst))
print(min(lst))

# Nested lists
new_lst = ['Tiger', 'Unicorn', 'Hyena', 'Dog']
lst.append(new_lst)
print(lst)