#!/usr/bin/env python

import pandas as pd
import numpy  as np

#For 
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('movie_data.csv', encoding='utf-8')

print(df.head(3))
