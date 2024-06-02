import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import pickle
import os



current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'models', 'svr_model_5.26.24.sav')


#load model
savename = model_path
with open(savename, 'rb') as f:
    pipeline = pickle.load(f)

st.set_page_config(layout="wide")

st.title("Inflation estimator")

gas = st.number_input("How much more expensive will gasoline compared to 12 months prior? Input a value as a percent without the percent sign?")
milk = st.number_input("How much more expensive will milk compared to 12 months prior? Input a value as a percent without the percent sign?")
rent = st.number_input("How much more expensive will rent compared to 12 months prior? Input a value as a percent without the percent sign?")
prev_month = st.number_input("What was the previous month's inflation? Input a value as a percent without the percent sign?")


gasinflation = gas
milkinflation = milk #Needs updated
rentinflation = rent
previous_month_inflation = prev_month
postpan = 1

X = [[gasinflation,milkinflation,rentinflation,previous_month_inflation,postpan,gasinflation,milkinflation,rentinflation]]

normalized_x = pipeline.named_steps['standardscaler'].transform(X)
predictions = pipeline.named_steps['svr'].predict(normalized_x)

print(predictions)

st.text(f'\nInflation will likely be {predictions[0]}%, compared to {previous_month_inflation} year-over-year inflation last month.')