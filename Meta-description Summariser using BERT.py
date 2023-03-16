#!/usr/bin/env python
# coding: utf-8

# # Bert Meta Description Model

# In[ ]:


#importing the required libraries
#pip install bert-extractive-summarizer
import pandas as pd
from summarizer import Summarizer


# In[ ]:


#Reading the Meta description file in CSV format
df = pd.read_csv("your file path goes here", header=0, encoding = "ISO-8859-1") 


# In[ ]:


#Dropping null values if any
df.dropna()
#df= df.iloc[:2 , :]


# In[ ]:


#Printing the shape of the data frame
df.shape


# In[ ]:


#importing the required libraries for pre processing 
import nltk
nltk.download('punkt')
import string
import re


# In[ ]:


#defining a function for cleaning the text 
def cleantext(x):
    text=re.sub("[^a-zA-Z]"," ",x)
    text=text.lower()
    text=text.split()
    text=" ".join(text)
    return(text)


# In[ ]:


#Applying the cleantext function for cleaning the textual data.
df["Scraped"]= df["Scraped"].apply(lambda x : cleantext(x))
test_text["Utterance"]=test_text["Utterance"].apply(lambda x : cleantext(x))
val_text["Utterance"]=val_text["Utterance"].apply(lambda x : cleantext(x))


# In[ ]:


# Create a empty list to store the MDs
metadesc = []


# In[ ]:


# For each URL in the input CSV run the analysis and store the results in the list 
for i in range(len(df)):
    # Here is the bodytext TBA
    body = str(df.iloc[i][1])
    model = Summarizer()
    result = model(body)
    full = ''.join(result)
    metadesc.append(full)


# In[ ]:


#save stored values in a in a column
df['Meta_Des'] = metadesc
columnnames=["Meta_Des"]
df = pd.DataFrame(columns = columnnames)


# In[ ]:


#save output
output = df.to_csv('meta_output.csv')

