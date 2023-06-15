import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pandas import Series, DataFrame
from plotly.subplots import make_subplots
from scipy import stats
import math
from itertools import permutations,combinations
import streamlit as st
from streamlit_option_menu import option_menu
st.set_page_config(layout="wide")

# In[31]:

mypath=""
sales=pd.read_csv(mypath+"\SalesForCourse_quizz_table.csv")
sales=sales.drop(['index','Column1'],axis=1)
sales=sales.drop(34866,axis=0)
sales_co=sales.copy()
sales['Profit']=sales['Revenue']-sales['Cost']
sales['Profit_indicator']=np.where(sales['Profit']>0,'Positive','Negative')
category=sales['Product Category'].value_counts(normalize=True)
sales['Year']=sales['Year'].astype(int)
sales['Date']=pd.to_datetime(sales['Date'])
sales['DayofWeek']=sales['Date'].dt.weekday
sales['DayofWeek']=sales['DayofWeek'].apply(lambda x: 'Weekend'if x>5 else 'Weekday')
n=sales['Profit_indicator']=='Negative'
p=sales['Profit_indicator']=='Positive'
num_name={'Customer Age':'Customer Age','Quantity':'Quantity','Unit Cost':'Unit Cost','Unit Price':'Unit Price','Cost':'Total Cost','Revenue':'Total Revenue','Profit':'Profit'}
cat_name={k:k for k in sales.select_dtypes(include='object').columns}
target='Profit_indicator'
cat_name.update({'DayofWeek':'Day of Week','Year':'Year','Profit_indicator':'Profit Indicator'})
all_name=num_name.copy()
all_name.update(cat_name)
chi_dict={v:stats.chi2_contingency(pd.crosstab(sales[target],sales[k]))[0] for k,v  in cat_name.items()}
chi_dict=sorted(chi_dict.items(),key=lambda x:x[1],reverse=True)










def get_key(val):
    for k,v in num_name.items():
        if val==v:
            return k
    for k,v in cat_name.items():
        if val==v:
            return k
def fit_distribution(data):
    mu = np.mean(data)
    sigma = np.std(data)
    dist = stats.norm(mu, sigma)
    return dist
fit_distribution(sales['Cost'])
def profit_predict(df,tv='Profit_indicator'):
    positive=df[df[tv]=='Positive'].copy()
    negative=df[df[tv]=='Negative'].copy()
    p_prior=df[tv].value_counts(normalize=True)['Positive']
    n_prior=df[tv].value_counts(normalize=True)['Negative']
    p_p={}
    p_n={}
    for i in df.columns:
        if df[i].dtype=='object':
            p_con=dict(positive[i].value_counts(normalize=True))
            n_con=dict(negative[i].value_counts(normalize=True))
            p_p[i]=df[i].map(p_con)
            p_n[i]=df[i].map(n_con)
        elif df[i].dtype=='float64':
            p_p[i]=fit_distribution(positive[i]).pdf(df[i])
            p_n[i]=fit_distribution(negative[i]).pdf(df[i])          
    final_p=pd.DataFrame(p_p).prod(axis=1)*p_prior
    final_n=pd.DataFrame(p_n).prod(axis=1)*n_prior
    predicted=(final_p>final_n).map({True:'Positive',False:'Negative'})
    final=pd.concat([predicted,df[tv]],axis=1)
    final.columns=['Predicted','Actual']
    acc=final['Predicted']==final['Actual']
    #print(f"The final accuracy is {np.round(acc.mean(),4)}")
    return final,acc.mean()
def chi_square(col1):
    table=pd.crosstab(sales[col1],sales['Profit_indicator'])
    chi=stats.chi2_contingency(table)
    return f"Chi-Square: {np.round(chi[0],2)}\np_value: {np.round(chi[1],4)}"

melted_num=pd.melt(sales,id_vars=['Profit_indicator'],value_vars=[get_key(i) for i in list(num_name.values())])
melted_num=melted_num[melted_num['variable']!='Profit']
corr_df=sales[list(num_name.keys())].corr().round(4)






with st.sidebar: 
	selected = option_menu(
		menu_title = 'Navigation Pane',
		options = ['Abstract', 'Background Information', 'Data Cleaning','Exploratory Analysis','Testing Naive Bayes Predictions','Data Analysis', 'Conclusion', 'Bibliography'],
		menu_icon = 'arrow-down-right-circle-fill',
		icons = ['bookmark-check', 'book', 'box', 'map', 'boxes', 'bar-chart', 
		'check2-circle'],
		default_index = 0,
		)
if selected=='Abstract':
    st.title("SuperMarket Sales Abstract")
    st.markdown("In this case study, we will create a prediction model based on the sales dataset, focusing on different regions and product category and how consumers influence profits. I made a custom function that applies the Naive Bayes Algorithm to any dataset. I then used this function to predict profits in a sports gear sales dataset.")
if selected=='Background Information':
    st.title("Background Information")
    st.markdown("The global sportswear market is currently expanding noticeablely with an expected annual growth rate of 6.4% <sup>1</sup> from 2022 to 2030. This trend is mainly cased by 'Continuous innovations and rapid technological advancements to keep pace with dynamic consumer preferences <sup>1</sup>' and the growing awareness of the importance of healthy lifestyles arised from Covid-19. Covid-19 did adversely affect the industry as various sports events are cancelled or postponed. Therefore, 'stakeholders in the market are trying to assess the downstream impact arising from disrupted cash flows, insecurities, and the potential declines in long-term attendance and engagement.'" ,unsafe_allow_html=True)
    st.markdown("Our dataset contains information focsuing on the sales of sportswere in stores across one-year period from 2015 to 2016. ")
if selected=='Data Cleaning':
    st.title("Feature Engineering")
    st.markdown("The provided data is mostly clean, but I would like to add a few more features by utilizing existing columns and create a target variable called 'Profit Indicator' that is more suitable for applying the Naive Bayes algorithm. Additionally, I created the 'Day of Week' column from the 'Date' column to show which day of the week it is, which may be helpful in finding patterns.")
    st.markdown("This is the original dataframe")
    st.dataframe(sales_co)
    st.subheader("Making 'Profit Indicator' and'Day of Week'")
    
    code_1='''sales['Profit']=sales['Revenue']-sales['Cost']
sales['Profit_indicator']=np.where(sales['Profit']>0,'Positive','Negative')'''
    st.code(code_1,language='python')
    st.markdown("The new column created")
    st.dataframe(sales[['Profit','Profit_indicator']])
    st.markdown("First the 'Date' column must be converted into a datetime object. After this, I used a Pandas datetime function in order to know which day is it in a week. When all the days are known, I sorted them into 'Weekend' and 'Weekdays'.")
    code_2='''sales['Date']=pd.to_datetime(sales['Date'])
sales['DayofWeek']=sales['Date'].dt.weekday
sales['DayofWeek']=sales['DayofWeek'].apply(lambda x: 'Weekend'if x>5 else 'Weekday')'''
    st.code(code_2,language='python')
    st.dataframe(sales['DayofWeek'])
    st.markdown("Here's the completely cleaned dataframe")
    st.dataframe(sales)
if selected=='Exploratory Analysis':
    st.title('Exploratory Analysis')
    st.markdown("In this section the aim is to explore each variable and see how much they affect profit. Also, we will examine the numeric and categorical columns seperately as they influence our target variable in different ways. Through this section we will find factors that contribute the most or the least to sports gear profit. The following interactive graphs will allow you to perform feature selection for use in the Naive Bayes Algorithm function.")
            
    col3,col4=st.columns([3,5])      
    col3.header("Box Plot for Categorical Values")
    with st.form("Submit"):
        var_option2_2=col3.selectbox("Select Numeric variables",num_name.values(),key=2)
        submitted2=st.form_submit_button("Submit to compare one or multiple columns")
        if submitted2:
            fig2=px.box(sales,x='Profit_indicator',y=get_key(var_option2_2),labels=cat_name)
            fig2.update_xaxes(title_text='<b> Profit Indicator</b>')
            fig2.update_yaxes(title_text=f'<b>{var_option2_2}</b>')
            col4.plotly_chart(fig2)
            
    
    col5,col6=st.columns([3,5])
    col5.header("Histograms Showing Numeric Variable's Distrubution")
    with st.form("Submit numeric variables"):
        var_option3=col5.selectbox("Select Numeric Variable",num_name.values(),key=3)
        key=get_key(var_option3)
        check1=col5.checkbox("Control bin size",key=4)
        check2=col5.checkbox("Normalized Histogram",key=10)
        percent_hist=None
        number=None
        if check1:
            number=col5.number_input('Insert a number',min_value=10,max_value=300,step=10)
        if check2:
            percent_hist='percent'
        submitted3=st.form_submit_button("Submit Numeric Variables")
        if submitted3:
            fig3=px.histogram(sales,x=get_key(var_option3),nbins=number,title=f"<b>Distrubution of {var_option3}",color='Profit_indicator',barmode='group',histnorm=percent_hist,labels=cat_name)
            fig3.update_xaxes(title_text=f'<b>{var_option3}</b>')
            fig3.update_yaxes(title_text='<b>Frequency</b>')
            col6.plotly_chart(fig3)
    
    col8,col9=st.columns([3,5])
    col8.header("KDE Distribution Based on Different Category")
    col8.subheader("What is a KDE Graph")
    with st.form("Submit Catgorical Variables"):
        var_option4=col8.selectbox("Select Numeric Variable",num_name.values(),key=6)
        var_option5=col8.selectbox("Select Categorical Variable",cat_name.values(),key=5)
        var_option6=col8.selectbox("Select Different Category",sales[get_key(var_option5)].unique(),key=7)
        submitted4=st.form_submit_button("Submit Variables")
        if submitted4:
            key=get_key(var_option4)
            key1=get_key(var_option5)
            data=sales[sales[key1]==var_option6][key]
            kde=stats.gaussian_kde(data)
            fig4=px.line(sales,x=np.arange(data.min(),data.max()+1),
                         y=kde.evaluate(np.arange(data.min(),data.max()+1)),width=800,height=500,title=f"<b>Kernel Density of {var_option3} in {var_option5} {var_option6}</b>")
            fig4.add_vline(data.mean())
            fig4.add_vline(data.mean()+np.std(data,ddof=0),line_color='red')
            fig4.add_annotation(x=data.mean()+0.1,y=0.95,text="mean",align="center",xref='x',yref='paper',showarrow=False)
            fig4.add_annotation(x=data.mean()+np.std(data,ddof=0),y=0,text="1 standard deviation",align="center",xref='x',yref='paper')
            fig4.update_yaxes(title_text='<b>Density</b>')
            fig4.update_xaxes(title_text=f'<b>{var_option3}')
            col9.plotly_chart(fig4)
    col8.markdown("KDE stands for Kernel Density Estimation and is used to visualize the probability density of a particular event. And probability density is the likelihood of a particular value occurring in a given dataset. KDE uses this concept to estimate the underlying probability density function of a dataset. Unlike histograms, the reason why KDE graph is a good demonstration of probability is that it generates a smooth and continuous curve. First we would determine the bandwidth which is the length of each kernel and is analagous to choosing the bin size in a histogram. Then for each kernel we would assign a weight according to a particular kernel function like Gaussian. We then sum up all the kernels to get the curve.")
            
    
    
    
    
    
    chi_col1,chi_col2=st.columns([3,5])
    chi_col1.header("Chi-Square")
    chi_col1.markdown("A Chi-Square test of independence can be used to determine whether there is a statistical relationship between two categorical variables. A larger Chi-Square statistic implies that a feature is more strongly related than a feature with a lower test statistic. Therefore we can use this for feature selection by comparing the categorical predictors to the target variable 'Profit Indicator'.")
    chi_explore={}
    with st.form("Submit Mutiple Variables"):
        var_option7=chi_col1.multiselect("Select Predicator Variables",cat_name.values(),key=8)
        var_option8=chi_col1.selectbox("Select Target Variable",np.setdiff1d(list(cat_name.values()),var_option7),key=9)
        key_target=get_key(var_option8)
        check_log=chi_col1.checkbox("Display log scale")
        logy=False
        if check_log:
            logy=True
        submitted5=st.form_submit_button("Submit Variables to see")
        if submitted5:
            key_chi=[get_key(i) for i in var_option7]
            chi_explore=[[i]+list(stats.chi2_contingency(pd.crosstab(sales[i],sales[key_target]))[0:2]) for i in key_chi]
            
            chi_data=pd.DataFrame(chi_explore,columns=['Predictor','Chi-Square Stat','P-Value'])
            fig5=px.bar(chi_data,x='Predictor',y='Chi-Square Stat',hover_name='Predictor',hover_data=['P-Value'],color='Predictor',title=f'<b> Chi-Square Value of Predictor Variables and {var_option8}</b>',log_y=logy)
            fig5.update_traces(hovertemplate="<b>%{x}</b> <br><br> Chi-Square Stat: %{y:.2f}<br> P-Value: %{customdata:.2e} <extra></extra>")
            fig5.update_xaxes(title_text='<b>Predictor</b>')
            fig5.update_yaxes(title_text='<b>Chi-Square Stat</b>')
            chi_col2.plotly_chart(fig5)
            
            
if selected=="Testing Naive Bayes Predictions":
    st.title("Interactive Naive Bayes Predictions")
    st.header("Testing accuracy")
    st.markdown("Select the feature you found in the exploratory analysis section and see how well they predict the 'Profit Indicator'.")
    with st.form("Accuracy"):
        predictors=st.multiselect("Select the features you wish to predict profit",np.setdiff1d(list(all_name.values()), ["Profit Indicator"]),key=20,default=None)
        key2=[get_key(i) for i in predictors]
        submitted10=st.form_submit_button("Submit view predictions and accuracy")
        if submitted10:
            predicted,acc=profit_predict(sales[key2+['Profit_indicator']])
            st.markdown(f"The final accuracy is {np.round(acc,4)}")
            st.dataframe(predicted)
    st.markdown("If you are interested in understanding how the function works, click to see my implementation of the Naive Bayes Classifier")
    with st.expander("View the function"):
        st.code('''def fit_distribution(data):
            mu = np.mean(data)
            sigma = np.std(data)
            dist = stats.norm(mu, sigma)
            return dist
        fit_distribution(sales['Cost'])
def profit_predict(df,tv='Profit_indicator'):
    positive=df[df[tv]=='Positive'].copy()
    negative=df[df[tv]=='Negative'].copy()
    p_prior=df[tv].value_counts(normalize=True)['Positive']
    n_prior=df[tv].value_counts(normalize=True)['Negative']
    p_p={}
    p_n={}
    for i in df.columns:
        if df[i].dtype=='object':
            p_con=dict(positive[i].value_counts(normalize=True))
            n_con=dict(negative[i].value_counts(normalize=True))
            p_p[i]=df[i].map(p_con)
            p_n[i]=df[i].map(n_con)
        elif df[i].dtype=='float64':
            p_p[i]=fit_distribution(positive[i]).pdf(df[i])
            p_n[i]=fit_distribution(negative[i]).pdf(df[i])          
    final_p=pd.DataFrame(p_p).prod(axis=1)*p_prior
    final_n=pd.DataFrame(p_n).prod(axis=1)*n_prior
    predicted=(final_p>final_n).map({True:'Positive',False:'Negative'})
    final=pd.concat([predicted,df[tv]],axis=1)
    final.columns=['Predicted','Actual']
    acc=final['Predicted']==final['Actual']
    #print(f"The final accuracy is {np.round(acc.mean(),4)}")
    return final,acc.mean()''',language='python')
            
            
            
            

if selected=='Data Analysis':
    st.title("Data Analysis")
    st.markdown("In this section, we will focus on a specific variable, the 'Profit Indicator', which indicates whether a sale generates profits or not. Our goal is to test a simple Naive Bayes model by providing it with other variables, enabling it to generate predictions on the 'Profit Indicator' accordingly. Additionally, we will identify five predictors and use them as input for the model to achieve the highest possible accuracy")
    st.header("Explaining Naive Bayes Algorithm")
    st.markdown("Naive Bayes is an algorithm majorly used in classification and prediction models, as it takes in different types of variables such as binary or numeric and uses different probability density functions respectively to compute the conditional probability of that variable. Once all the probabilities are calculated, we will be able to know the overall probability and make assumptions. This module has several premises. The first one is that we suppose each variable to be independent from each other,including the class variable; the second one states that each variable we choose should follow a certain distribution like Normal or Gaussian distribution; the third one is the most important, which is the Bayes Formula used to calculate conditional probability. ")
    st.markdown("The construction of a Naive Bayes classifier typically involves three steps. First, we choose predictor and target variables and calculate their conditional probabilities using different probability density functions. Then, we split our dataset into a training set and a testing set. The training set is used to find the model that provides the most accurate predictions, while the testing set is used to evaluate the actual performance of the model")
    st.header("Accuracy")
    st.markdown("Accuracy is a crucial component when evaluating the performance of a Naive Bayes model. A baseline model refers to the accuracy we would expect to achieve by predicting the most common class for all observations in the dataset without any calculation. To calculate the baseline accuracy for our dataset, we can use the 'value_counts(normalize=True)' method from the Pandas library to find the percentage of the most frequent category of 'Profit Indicator'. This means that if we blindly predict positive profit, we would have an 84.87% accuracy. Therefore a good model must be higher than that.")
    st.write(sales['Profit_indicator'].value_counts(normalize=True))
    st.markdown("Above is the value count table and evidently the most common class is 'Positive' with 84%")
    
    
    
    
    
    col10,col11=st.columns([5,5])
    col10.header("Using Chi-Square to show Relevancy")
    st.markdown("Since we have chosen the 'Profit Indicator' variable as our target variable for prediction, we need to calculate the relevance of the other variables to it. Below is a dataframe that stores all the categorical variables along with their respective chi-square and p-values with respect to the 'Profit Indicator'.")
    keys=[get_key(i) for i in np.setdiff1d(list(cat_name.values()),["Profit Indicator"])]
    chi_data_analysis=pd.DataFrame([[i]+list(stats.chi2_contingency(pd.crosstab(sales[i],sales['Profit_indicator']))[0:2]) for i in keys],columns=['Predictor','Chi-Square Stat','P-Value'])
    
    st.dataframe(chi_data_analysis.sort_values("Chi-Square Stat",ascending=False).reset_index(drop=True))
    st.markdown("The dataframe clearly displays the size of the Chi-Square and P-values, which are essential for our subsequent analysis. We can use this information to filter out unnecessary variables and identify the most accurate model.")
    st.markdown("The chi-square value of a variable indicates its closeness of relation to the target variable. The bigger the chi-square value, the more closely related it is to the target. Conversely, a bigger p-value suggests less relevance. From the dataframe above, it is apparent that the 'Customer Gender' and 'Day of Week' columns are not very significant to profit. Therefore, we can consider them last. On the other hand, 'Sub Category' seems to be the most relevant variable to profit, so let's choose it first.")
    st.subheader("Testing Function")
    st.markdown("Let's first put 'Sub Category' into our function and see the result")
    code1='''profit_predict(sales[['Profit_indicator','Sub Category']])'''
    st.code(code1,language='python')
    st.markdown("The result shows an accuracy of 84.87% using only one predictor. While this may seem high initially, it is identical to the base-line accuracy, so we need to consider adding other features to improve the model. Although 'Product Category' also has a high value of Chi-Square, it is highly correlated with 'Sub Category', so we will not consider it. We will instead choose the third-highest value 'Year' and then 'State' as additional predictors. Next, we will consider two more predictors based on their difference in means and their correlation with each other.")
    st.subheader("Facet plots for selecting numeric predictors")
    fig6=px.box(melted_num,x='Profit_indicator',y='value',color='Profit_indicator',facet_col='variable',facet_col_wrap=3,facet_col_spacing=0.06,width=1000,height=900)
    fig6.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=14))
    fig6.update_yaxes(matches=None,title='',showticklabels=True)
    fig6.update_xaxes(showticklabels=False,title_text='')
    fig6.add_annotation(x=-0.05,y=0.4,text="<b>Value of Each Variable</b>", textangle=-90, xref="paper", yref="paper",font_size=14
)
    st.markdown("This facet plot is very important as it allows us to directly see the means of each variable with respect to positive and negative profits. Therefore, the greater the difference in means between the two categories, the more influence the variable has on the target variable, which we take into account when deciding on our predictors.")    
    
    st.plotly_chart(fig6)
    st.markdown("")
    
    st.header("Heatmap for correaltion between numeric variables")
    col_heat1,col_heat2=st.columns([3,5])
    fig7=px.imshow(corr_df,color_continuous_scale=px.colors.diverging.balance,text_auto=True,width=900,height=900,title="<b>Correlation of Each Numeric Variable")
    fig7.update_traces()
    col_heat2.plotly_chart(fig7)
    col_heat1.markdown("This heatmap displays the correlation between each numeric variable in our dataset by representing the strength of the correlation with color depth. The color red indicates a positive relationship, while blue indicates a negative one. Keep in mind that the Naive Bayes algorithm assumes that all predictor variables are independent of each other, but some of our numeric variables, such as 'Cost' and 'Unit Cost', are closely related, as shown by their dark red color on the heatmap. Therefore, these variables should not be used together as predictors. By analyzing the heatmap and the Chi-Square chart, we can identify variables that are highly significant to the target variable 'Profit Indicator' and independent of each other by selecting the ones with the highest Chi-Square and the largest difference in means. We should also ensure that these variables have lighter colors with each other to indicate that they are not highly correlated.")
    st.header("Final Choice")
    st.markdown("Finally I chose 'Sub Category','Country','Month','Customer Age', and 'Customer Gender'. ")
    st.markdown("It turns out that selecting variables based solely on their Chi-Square values and mean differences cannot necessarily generate the most accurate model. This is because these tests only consider the effect of each predictor on its own and not the combined effect of all predictors. Additionally, through experimentation, it has been found that even if some predictors do not have a high Chi-Square value, the accuracy can still be high as long as these predictors are not highly correlated with other predictors.")
    st.markdown("This model that I have found provides an accuracy of 88.36%, which can be considered successful as it surpasses the accuracy of the base-line model.")
    c1,c2=st.columns([5,5])
    c1.markdown("This is the baseline prediction for our dataset, with an accuracy of about 84.87%")
    c1.write(sales['Profit_indicator'].value_counts(normalize=True))
    data_result,acc_result=profit_predict(sales[['Profit_indicator','Sub Category','Country','Month','Customer Gender','Customer Age']])
    c2.write(np.round(acc_result,4))
    c2.dataframe(data_result)


if selected=='Conclusion':
    st.title("Conculsion")
    st.markdown("To summarize, these five factors produce the highest accuracy, mainly because they are each independent of each other. As we have seen, 'Sub Category' and 'Month' contribute a lot to profit, thus we could focus on these two variables and see which category and month have the best sales.")
    col_1,col_2,col_3=st.columns([10,10,10])
    col_1.subheader("Most Profitable Category")
    col_1.dataframe(sales.groupby(['Sub Category']).mean()['Profit'].sort_values(ascending=False))
    col_2.subheader("Most Profitable Month")
    col_2.dataframe(sales.groupby(['Month']).mean()['Profit'].sort_values(ascending=False))
    col_3.subheader("Most Profitable Month-Category Combined")
    col_3.dataframe(sales.groupby(['Month','Sub Category']).mean()['Profit'].sort_values(ascending=False))
    st.markdown("Bike Racks turn out to be the most profitable product category, and May is the most profitable month. However, when considering both factors, it appears that it is best to sell Bike Racks in July.")



if selected=='Bibliography':
    st.title("Bibliography")
    st.markdown("The dataset is downloaded from https://www.kaggle.com/datasets/thedevastator/analyzing-customer-spending-habits-to-improve-sa?resource=download")
    st.markdown("[1] https://www.grandviewresearch.com/industry-analysis/sports-equipment-market#:~:text=The%20global%20sports%20equipment%20market%20size%20was%20estimated%20at%20USD,largest%20sports%20equipment%20market%20share%3F")
    
