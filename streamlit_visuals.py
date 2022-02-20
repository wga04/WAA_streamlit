import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
from PIL import Image

# Interactive features to the visualization: 1) page selection. 2) in last page called interactive chosing the rank of the university where the graph changes accordingly
st.set_page_config(layout="wide")
st.balloons()
logo = "https://raw.githubusercontent.com/wga04/WAA_streamlit/main/LOGO.jpg"
st.image(logo, width=150)

st.title('W.A ANALYTICS')

data = pd.read_csv('https://raw.githubusercontent.com/wga04/WAA_streamlit/main/World%20Universities%20-%20Top%20Business%20Schools%20-%20Ranking%20Location%20%20Salary.csv')
df1 = data


page = st.sidebar.selectbox('Select Topic For Data Visuals',
  ['Dataset','Top 10 Universities','Universities Distribution By Countries',
  'Salary Batches For Countries','MAP: Top Universities Location','DATA Summary: 3D Plot',
  'Salary Rank Relationship','Interactive'])


if page == 'Dataset':
    st.header("World Universities: Top 100 Business Schools in 2020")
    st.write(df1)


if page == 'Top 10 Universities':
    st.header("Top 10 Universities world wide")

# Plotting Inline
# visualizing the top 10 universities in the world
# since the order of the universities starts from rank 1 and are incremented by 1 we take the top 10 using head function
    df2 = df1.head(10)
    data = [Bar(x=df2['School name'],y=df2['Rank in 2020'])]
    layout1 = Layout(xaxis_title="University Name", yaxis_title="University Rank")
    fig = dict(data=data, layout=layout1)
    st.write(fig)


if page == 'Universities Distribution By Countries':
    # I used a piechart because it is better than the graph shown above to reflect the frequency of Top 100 universities sorted by Country
    st.header("Ditribution of top 100 universities on different countries")
    st.markdown('The Pie chart shows the distribution of the 100 top ranked universities along the countries.')
    df3 = df1['Country'].value_counts()
    df4 = pd.DataFrame(df3)
    df5 = px.data.tips()
    fig = px.pie(df4, values='Country', names=df4.index)
    st.write(fig)

    st.header("Frequency of top universities per country")
    # Universities are sorted according to the country that they are existed in
    # We are interested in the frequency of universities in the top 100 list sorted by country
    df3 = df1['Country'].value_counts()
    df4 = pd.DataFrame(df3)
    st.write(df4)

    # Plotting Inline
    # After sorting the data in a new dataframe we can now visualize it showing the frequency of country occupying top universities
    st.markdown('The below barchart reflects the values indicated in the table showing the frequency of top universities in different country locations.')
    data = [Bar(x=df4.index,y=df4['Country'])]
    layout1=Layout(xaxis_title="Country", yaxis_title="Frequency")
    fig = dict(data=data, layout=layout1)
    st.write(fig)


if page == 'Salary Batches For Countries':
    st.header("Salary Batches for Top Universities per Country per Rank")
    df2 = df1.head(10)
    st.markdown("The below diagram is an informative visualization of the top 10 ranked universities in the world. The categories are set according to country where every country is represented in various color and shows the weighted salary paid in addition to its rank accordingly")
    # visualizing the top 10 universities in the world in 2020 in terms of rank, salaries and country
    fig = px.scatter(df2, x="Rank in 2020", y="Weighted salary (US$)", size="Rank in 2020", color="Country", log_x=True, size_max=50)
    st.write(fig)


if page == 'MAP: Top Universities Location':
    st.header("Location of Universities Represented in Different Colors According to University Ranks")
    st.markdown('This is an interactive maps that shows the location of the top 100 universities and are spotted in colors according to their rank.')
    # Plotting Interactive Maps
    fig = px.choropleth(df1, locations="Country",
                       locationmode="country names",
                       color="Rank in 2020",
                       hover_name="Country",
                       color_continuous_scale=px.colors.sequential.Plasma)
    st.write(fig)


if page == 'DATA Summary: 3D Plot':
    st.header("Variation of salaries according to university rank in different countries")
    st.markdown('This 3D plot is interactive where once you press on any dot it will provide with information like university name, country of location, university rank and salary.')
    # 3D Plot showing the relation between the university rank, salaries and country of the universities are found in
    fig = px.scatter_3d(df1,x="Country", y="Rank in 2020", z="Weighted salary (US$)", color="School name")
    st.write(fig)


if page == 'Salary Rank Relationship':
    st.header("Animation visualization showing the variation of weighted salary for the top 100 universities")
    st.markdown('This animation shows the variation of salaries as the university rank goes from rank 1 to 100 along with the country of the selected university and its name.')
    # Animated Plots: in different datasets we may use this scatter in order to show informative and meaningful observation as for our dataset here I just added this visualization in order to practice all kinds of charts included in the tutirial
    fig = px.scatter(df1, x="Rank in 2020", y="Weighted salary (US$)", animation_frame="Rank in 2020", size="Weighted salary (US$)", color="Weighted salary (US$)", hover_name="Country", log_x=True, size_max=50, range_x=[1,100], range_y=[50000,250000], labels=dict(Length="Length (in seconds)"))
    st.write(fig)


if page == 'Interactive':
    st.header("University's Weighted Salary and Country of Location")
    df1['Rank in 2020'] = df1['Rank in 2020'].apply(str)
    clist = df1['Rank in 2020'].unique()
    rank = st.sidebar.selectbox("Select University's Rank",clist)
    fig = px.scatter(df1[df1['Rank in 2020'] == rank],
        x = "Country", y = "Weighted salary (US$)")
    st.plotly_chart(fig)
