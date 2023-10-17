# import library
import plotly.express as px
 
# load in iris data
df = px.data.iris()
 
# create 3D scatter plot
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')
fig.show()