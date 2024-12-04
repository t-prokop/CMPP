#%%
import os
import numpy as np
import matplotlib.pyplot as plt
with open('./PanTadeusz.txt', 'r') as f:
    txtstring = f.read()
characterlist = list(txtstring)
characters,counts = np.unique(characterlist, return_counts=True)
#%%
#filter for ascii characters from [34,126]
range_chars = range(34,127)
counts_filtered = [count for i,count in enumerate(counts) if ord(characters[i]) in range_chars]
characters_filtered = [repr(char) for char in characters if ord(char) in range_chars]

#all characters
# counts_filtered = counts
# characters_filtered = [repr(char) for char in characters]

#calculate probabilities
num_all_chars = np.sum(counts_filtered)
prob_char = counts_filtered/num_all_chars
#%%
#make histogram
fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot()
ax.bar(characters_filtered,prob_char, width = 0.8, align = "center")
#%%
#calc entropy
entropy = -np.sum([prob*np.log2(prob) for prob in prob_char if prob > 0])
print(f'Entropy: {entropy}')
len_encoded = int(np.ceil(entropy*num_all_chars))
print(f"Expected length of encoded: {len_encoded} bits = {len_encoded/8/1024:.1f} kB ")
print(f"Size of ZIP: {os.stat('./PanTadeusz.zip').st_size/1024:.1f}kB")
print("Różnica wynika z tego, że znaki w Panu Tadeuszu nie są przypadkowymi ciągami i wykorzystując korelacje między ich ciągami można bardziej zwiększyć kompresję.")
# %%
#huffman coding
#from n bits -> n+1 symbols can be encoded
