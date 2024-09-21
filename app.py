import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Alcohol Dashboard", page_icon="🍾", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,.1);
    } 
</style>
""", unsafe_allow_html=True)


import ast

# Create sample data
@st.cache_data
def create_sample_data(df = None, n_records=1000):

    def extract_category_filter(row):
        filter_str = row['raw.ec_category_filter']
        try:
            # Safely evaluate the string as a Python literal
            filter_list = ast.literal_eval(filter_str)
            if filter_list and isinstance(filter_list[0], str):
                categories = filter_list[0].split("|")
                # Take the last three categories, or pad with None if fewer than 3
                return (categories[-3:] + [None, None, None])[:3]
        except (ValueError, SyntaxError, IndexError):
            # Return None values if there's any error in parsing
            return [None, None, None]

    # Apply the function to create new columns
    df[['cat_filter2', 'cat_filter3', 'cat_filter4']] = df.apply(extract_category_filter, axis=1, result_type='expand')
    
    # np.random.seed(42)
    # # country
    # regions = ['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Nova Scotia']
    # # brand
    # brands = ['Smirnoff', 'Bacardi', 'Absolut', 'Grey Goose', 'Jack Daniels', 'Johnnie Walker', 'Hennessy', 'Moet & Chandon']
    # # category
    # types = ['Vodka', 'Rum', 'Whiskey', 'Gin', 'Tequila', 'Champagne', 'Cognac']

    # start_date = datetime(2023, 1, 1)
    # end_date = datetime(2023, 12, 31)
    # date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    # data = []
    # for _ in range(n_records):
    #     date = np.random.choice(date_range)
    #     region = np.random.choice(regions)
    #     brand = np.random.choice(brands)
    #     type_ = np.random.choice(types)
    #     price = np.random.uniform(10, 100)
    #     rating = np.random.uniform(1, 5)
    #     inventory = np.random.randint(0, 1000)
        
    #     data.append({
    #         'date': date,
    #         'region': region,
    #         'brand': brand,
    #         'category': type_,
    #         'price': price,
    #         'rating': rating,
    #         'inventory': inventory
    #     })

    # df = pd.DataFrame(data)
    
    # Convert the 'Date' column to datetime, assuming the current year is 2023
    df_final = pd.DataFrame()
    df_final['store'] = ['Store 1']*df.shape[0]
    df_final['date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df_final['brand'] = df['raw.ec_brand'].replace("undefined", np.nan)
    df_final['category'] = df['cat_filter2'].replace("undefined", np.nan)
    df_final['price'] = df['raw.ec_price'].replace("undefined", np.nan).astype('float')
    df_final['region'] = df['raw.lcbo_region_name'].replace("undefined", np.nan)

    df_final['rating'] = df['raw.ec_rating'].replace("undefined", np.nan).astype('float')
    df_final['inventory'] = df['raw.stores_inventory'].replace("undefined", np.nan).astype('float')
    df_final['title'] = df['title']

    # Replace 'undefined' with NaN
    # df_final['price'] = df[column_name].replace('undefined', pd.NA)
    # df_final.dropna(inplace=True)
    return df_final

df = pd.read_excel('Alcohol data.xlsx')
df = create_sample_data(df)
# df.to_csv("clean_data.csv", index = False)


# Sidebar
st.sidebar.title("Filters")
min_date = df['date'].min().date()
max_date = df['date'].max().date()

selected_store = st.sidebar.multiselect("Stores", options=df['store'].unique(), default=df['store'].unique())

date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

selected_brands = st.sidebar.multiselect("Brands", options=df['brand'].unique(), default=df['brand'].unique())

selected_types = st.sidebar.multiselect("Types", options=df['category'].unique(), default=df['category'].unique())

min_price, max_price = st.sidebar.slider("Price Range", float(df['price'].min()), float(df['price'].max()), (float(df['price'].min()), float(df['price'].max())))

selected_regions = st.sidebar.multiselect("Regions", options=df['region'].unique(), default=df['region'].unique())

# Filter data
filtered_df = df[
    (df['store'].isin(selected_store)) & 
    (df['date'].dt.date >= date_range[0]) & 
    (df['date'].dt.date <= date_range[1]) &
    (df['region'].isin(selected_regions)) &
    (df['brand'].isin(selected_brands)) & 
    (df['category'].isin(selected_types)) &
    (df['price'] >= min_price) &
    (df['price'] <= max_price)
]

# Main content
st.title("🍾 Alcohol Sales Dashboard")

# Top rated and priced products
col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 3 Rated Products")
    top_rated = filtered_df.nlargest(3, 'rating')
    for i, row in top_rated.iterrows():
        st.write(f"{row['title']} - Rating: {row['rating']}⭐")

with col2:
    st.subheader("Top 3 Priced Products")
    top_priced = filtered_df.nlargest(3, 'price')
    for i, row in top_priced.iterrows():
        st.write(f"{row['title']} - Price: 💲{row['price']:.2f}")


# Key Metrics
# col1, col2, col3, col4 = st.columns(4)
# col1, col2 = st.columns(2)

# with col1:
#     # st.metric("Total Products", f"{filtered_df['brand'].nunique():,}")
#     # st.markdown('<div class="title-container">', unsafe_allow_html=True)
#     top3_rated_products = df.sort_values(by = ['rating'], ascending=False)[:3][['title', 'rating']].values
#     st.write("#### Top 3 rated products")
#     st.write(f"{top3_rated_products[0][0]}, {top3_rated_products[0][1]}⭐")
#     st.write(f"{top3_rated_products[1][0]}, {top3_rated_products[1][1]}⭐")
#     st.write(f"{top3_rated_products[2][0]}, {top3_rated_products[2][1]}⭐")
#     # st.markdown('</div>', unsafe_allow_html=True)

# with col2:
#     top3_priced_products = df.sort_values(by = ['price'], ascending=False)[:3][['title', 'price']].values
#     st.write("#### Top 3 price products")
#     st.write(f"{top3_priced_products[0][0]}, 💲{top3_priced_products[0][1]}")
#     st.write(f"{top3_priced_products[1][0]}, 💲{top3_priced_products[1][1]}")
#     st.write(f"{top3_priced_products[2][0]}, 💲{top3_priced_products[2][1]}")
#     # st.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")
# # with col3:
# #     st.metric("Avg Price", f"${filtered_df['price'].mean():.2f}")
# # with col4:
# #     st.metric("Total Inventory", f"{filtered_df['inventory'].sum():,}")


# Product count comparison
st.subheader("Product Count Comparison")
brand_category_counts = filtered_df.groupby(['brand', 'category']).size().unstack(fill_value=0)
brand_category_counts_pct = brand_category_counts.div(brand_category_counts.sum(axis=1), axis=0) * 100

fig = px.bar(brand_category_counts_pct.reset_index().melt(id_vars='brand', var_name='category', value_name='percentage'),
             x='brand', y='percentage', color='category', title="Product Category Distribution by Brand",
             labels={'percentage': 'Percentage', 'brand': 'Brand', 'category': 'Category'},
             height=400)
fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig, use_container_width=True)

# Sales Distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Distribution by Category")
    sales_by_category = filtered_df.groupby('category')['price'].sum()
    fig = px.pie(values=sales_by_category.values, names=sales_by_category.index, title="Sales Distribution by Category")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Selling Products")
    top_selling = filtered_df.groupby('title')['price'].sum().nlargest(5)
    fig = px.bar(x=top_selling.index, y=top_selling.values, title="Top 5 Selling Products",
                 labels={'x': 'Product', 'y': 'Total Sales'})
    st.plotly_chart(fig, use_container_width=True)

# Pricing architecture
st.subheader("Pricing Architecture")
fig = px.box(filtered_df, x='brand', y='price', color='category', title="Price Distribution by Brand and Category",
             height=500)
fig.update_layout(xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig, use_container_width=True)

# Inventory levels
st.subheader("Inventory Levels")
inventory_df = filtered_df.groupby('brand')['inventory'].mean().sort_values(ascending=False)
fig = px.bar(inventory_df, x=inventory_df.index, y='inventory', title="Average Inventory by Brand",
             labels={'inventory': 'Average Inventory', 'index': 'Brand'},
             color=inventory_df.values, color_continuous_scale='RdYlGn')
st.plotly_chart(fig, use_container_width=True)

# ================== NEW PAGE ======================= #
# Rating Dashboard
st.title("Alcohol Rating Dashboard")

# Average Rating by Brand
st.subheader("Average Rating by Brand")
avg_rating_by_brand = filtered_df.groupby(['brand', 'category'])['rating'].mean().unstack()
fig = px.bar(avg_rating_by_brand.reset_index().melt(id_vars='brand', var_name='category', value_name='avg_rating'),
             x='brand', y='avg_rating', color='category', title="Average Rating by Brand and Category",
             labels={'avg_rating': 'Average Rating', 'brand': 'Brand', 'category': 'Category'},
             height=400)
fig.update_layout(xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig, use_container_width=True)

# Top Rated Products
st.subheader("Top Rated Products")
top_rated = filtered_df.nlargest(10, 'rating')
fig = px.bar(top_rated, x='title', y='rating', color='category', title="Top 10 Rated Products",
             labels={'rating': 'Rating', 'title': 'Product'},
             height=400)
fig.update_layout(xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig, use_container_width=True)

# Rating Trends
st.subheader("Rating Trends")
rating_trends = filtered_df.groupby(['date', 'brand'])['rating'].mean().unstack()
fig = px.line(rating_trends, x=rating_trends.index, y=rating_trends.columns, title="Rating Trends by Brand",
              labels={'value': 'Average Rating', 'variable': 'Brand', 'date': 'Date'},
              height=400)
st.plotly_chart(fig, use_container_width=True)

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")


st.title("Should I Remove Below Prev Dashboard")
# Price Distribution
st.subheader("Price Distribution")
fig = px.histogram(filtered_df, x='price', nbins=50, title="Price Distribution")
st.plotly_chart(fig, use_container_width=True)

# Rating vs Price Scatter Plot
st.subheader("Rating vs Price")
fig = px.scatter(filtered_df, x='price', y='rating', color='region', hover_data=['brand'], title="Rating vs Price by Region")
st.plotly_chart(fig, use_container_width=True)

# Top Rated Products
st.subheader("Top Rated Products")
top_rated = filtered_df.nlargest(10, 'rating')
fig = px.bar(top_rated, x='brand', y='rating', color='region', title="Top 10 Rated Products")
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, use_container_width=True)


# Region Comparison
st.subheader("Region Comparison")
region_stats = filtered_df.groupby('region').agg({
    'price': 'mean',
    'rating': 'mean',
    'brand': 'count'
}).reset_index()

fig = px.scatter(region_stats, x='price', y='rating', size='brand', color='region', hover_name='region', labels={'price': 'Average Price', 'rating': 'Average Rating', 'brand': 'Number of Products'}, title="Region Comparison: Price vs Rating")
st.plotly_chart(fig, use_container_width=True)

# Inventory Heatmap
st.subheader("Inventory Levels")
pivot_df = filtered_df.pivot_table(values='inventory', index='brand', columns='category', aggfunc='mean')
fig = px.imshow(pivot_df, title="Average Inventory Levels")
st.plotly_chart(fig, use_container_width=True)

# Product Table
st.subheader("Product Details")
product_df = filtered_df.groupby(['brand', 'category']).agg({
    'price': 'mean',
    'rating': 'mean',
    'inventory': 'mean'
}).reset_index()

product_df = product_df.sort_values('rating', ascending=False)
st.dataframe(product_df.style.format({
    'price': '${:.2f}',
    'rating': '{:,.2f}',
    'inventory': '{:,.0f}'
}), use_container_width=True)