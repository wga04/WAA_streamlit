import numpy as np
import pandas as pd
import streamlit as st
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()
import plotly.express as px
from PIL import Image
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import os
import squarify
import plotly.graph_objects as go
from statsmodels import api as sm
import pylab as py
import matplotlib.dates as dates
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import kstest,norm
from scipy.stats import norm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import plot_importance
from sklearn.utils import resample
from scipy.stats import chi2_contingency
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit.components.v1 as components


# Setting page icon, title and layout
st.set_page_config(
    page_title="Streamlit: Online Consumer Behavior",
    page_icon="📊",
    layout="wide",
)
st.markdown(f'<h1 style="color:#494949;font-size:50px;">{"📊   Online Consumer Behavior"}</h1>', unsafe_allow_html=True)

# naming pages to shuffle between
page = st.selectbox('Category ⬇️',('Home Page','Data & Report','Visualizations with Insights', 'Cart to Purchase Prediction'))


# Upload CSV data
with st.sidebar.subheader('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Please Upload Here", type=["csv"])


# Home page where I included an image a brief introduction and an image
if page == "Home Page":
    image = "https://chameleoncollective.com/wp-content/uploads/2018/04/e-commerce-blog-post-scaled.jpg"
    st.image(image, use_column_width = True)
    st.text_area("Overview",'''E-Consumer is a tool that people are relying on more and more each day, where it is rarely to find firms and mid to big size businesses that don't offer eCommerce on their platforms. This is why you can use this tool to visualize you data, reveal business insights, understand your RFM analysis and finally utilize machine learning to predict if your customers will purchase products after adding them to cart, this prediction is based on factors that you will input such as brand, price, weekday, category and activity on the website upon shopping.''')


# Spaces so that upon scrolling down nothing shows
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")



# Dataset is showna along automatic report generated to give statistics on every column in the dataset with final visuals and insights

if page == "Data & Report":
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')
            return csv
        df = load_csv()
        pr = ProfileReport(df, explorative=True, dark_mode=True)
        st.header('*E-commerce Dataset*')
        st.write(df)
        st.write('---')
        st.header('*Automatic Pandas Profiling Report*')
        st_profile_report(pr)
    else:
        st.info('Awaiting for CSV file to be uploaded.')

csv = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')


# Visual shared from tableau public where I included 4 visuals having some insights about the brand ranking, sales and category
if page == "Visualizations with Insights":

    def main():
        html_temp = """<div class='tableauPlaceholder' id='viz1653054254937' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;2S&#47;2SQ4MBZQN&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;2SQ4MBZQN' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;2S&#47;2SQ4MBZQN&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1653054254937');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
        components.html(html_temp, width=100000, height=800)
    if __name__ == "__main__":
        main()

# ML PART to predict if the client will move from in cart to purchase or will cancel the purchase
if page == "Cart to Purchase Prediction":

    # setting the label and features to run the model upon
    target = csv.iloc[:,9]
    features = csv[['brand','price','event_weekday','category_code_level1','category_code_level2','activity_count']]

    #checking for data balance
    is_purchase_set = csv[csv['is_purchased'] == 1]
    not_purchase_set = csv[csv['is_purchased'] == 0]


    # encoding categorical variables
    features.loc[:,'brand'] =LabelEncoder().fit_transform(csv.loc[:,'brand'].copy())
    features.loc[:,'event_weekday'] = LabelEncoder().fit_transform(csv.loc[:,'event_weekday'].copy())
    features.loc[:,'category_code_level1'] = LabelEncoder().fit_transform(csv.loc[:,'category_code_level1'].copy())
    features.loc[:,'category_code_level2'] = LabelEncoder().fit_transform(csv.loc[:,'category_code_level2'].copy())

    # Train set 80% and test set 20%
    X_train ,X_test ,y_train,y_test = train_test_split(features,target,test_size = 0.2,random_state = 0)


    col1,col2 = st.columns(2)
    with col1:


        # For each feature we are creating a list of unique values then iterating on them to create a dictionary from the list and iterations starting from zero
        list = csv["brand"].unique()
        brand = st.selectbox("Brand", list)
        dict = { i:list[i] for i in range(0,len(list))}

        list1 = csv["category_code_level1"].unique()
        category_code_level1 = st.selectbox("Category", list1)
        dict1 = { i:list1[i] for i in range(0,len(list1))}

        list2 = csv["category_code_level2"].unique()
        category_code_level2 = st.selectbox("Subcategory", list2)
        dict2 = { i:list2[i] for i in range(0,len(list2))}

        # function that takes the users input and dictionary and will give the value which is passed to the model
        def get_value(val,my_dict):
            	for key,value in my_dict.items():
            		if val == key:
            			return value

        brand_in = get_value(brand,dict)
        category_code_level1_in = get_value(category_code_level1,dict1)
        category_code_level2_in = get_value(category_code_level2,dict2)
    with col2:

        event_weekday = st.selectbox("Event Weekday", ("Monday","Thursday"))
        dict3 = {"Monday":0,"Thursday":4}
        event_weekday_in = get_value(event_weekday,dict3)
        price = st.number_input("Price",csv["price"].min(),csv["price"].max())
        activity_count = st.number_input("Customer Activity",csv["activity_count"].min(),csv["activity_count"].max())

    test = np.array([[brand_in, price, event_weekday_in, category_code_level1_in, category_code_level2_in,activity_count]])

    # Aftering running the data on jupyter we tested on logistic regression, xgboost and classification tree and we got the highest accuracy using the xgboost where we did hyper tunning and increased the accuracy
    #choosing the xgboost hyperparameters that we want to tune
    xgb_parameters = [
    { # XGBoost
        'n_estimators': [10,20,30],
        'max_depth': [3,4,5],
        'subsample': [0.2,0.3,0.4]
    }]
    xg_model = XGBClassifier(learning_rate = 0.1)

    #optimizing the performance of xgboost by choosing the best hyperparameters combination
    grid_xgb= GridSearchCV(estimator=xg_model, param_grid=xgb_parameters)
    grid_xgb.fit(X_train,y_train)

    y_pred = grid_xgb.predict(test)

    # if result is 1 then the client will purchase and if 0 will not.
    if  y_pred[0]==1:
         st.header("We are 66% sure that the client will purchase the product in cart")
    elif y_pred[0]==0:
            st.header("We are 66% sure that the client will not purchase the product in cart")
