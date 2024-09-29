# imperial-capstone-project
Imperial College ML/IA Certificate - Capstone project

The aim of this project is to be able to forecast the price of different industry sector based on a long historical time series of industry aggregated prices using either a Convolutional Network or a Long Short-Term Memory neural network.    
Through predicted prices we can compute optimal weights giving the best return of the portfolio composed of all industry sectors.  

### Business goal

The business objective is to be able to predict weights for a portfolio composed of industry components.  
Having this prediction will enable to allocate on which industry an investor should bet for beating the market. 

### Data

Data is composed of financial market performance across 49 US industries on monthly periods from May 1925 until may 2024.  

Data was compiled by Professor Kenneth R. French and available on his website.  
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html  
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html  
The original data comes from CRSP financial database (https://www.crsp.org/) (affiliate of the University of Chicago)  
Dataset is composed of two CSV files.  

The first one is the market monthly including dividends returns of 49 industries and the second the number of firms in each industry.    

For our analysis we will only use data from January 2020.  
This represents a set of 293 months for 49 industries.  

I've used ex-dividend average value weighted (so a company in the industry return is represented by its value / (sum value of all companies of the industry sector)) return series.


### Diversification factor - Weights allocation computation

For searching optimal number of industries we should use on our model, I've use a dendrogram and compares to K-Means.  
This boundary will be reflected on the optimizer while bounding the maximum weight allowed.  

Dendrogram of industry returns covariance:

![Alt text](dendrogram.png?raw=true "Dendrogram")

Elbow chart of industry returns : 

![Alt text](elbow.png?raw=true "Title")

For computing weight, we will use the Sequential Least Squares Programming (SLSQP) optimization.  
Constraints will be added for having:
- all weights equal to 1 (no leverage) 
- weights between 0% (no short position) and 15% (as expected before)

# Predictive models : CNN and LSTM

Now we have set the maximum weight of each component our model will be use we have to know which component to select.  
We will try to forecast the performance of each industry.  
So, the output of the model is scalar.  
For the two models I have used the standard scaler for scaling our series.  
I have also tester MinMax scaler but gave me no so good results.
Once the forecasting part is done, we select the industries which we want to select on our universe for building our portfolio.  
For this I have tested two different models: a Convolutional Neural Network and a Long Short Term Memory recurrent neural network.  
The portfolio weight will be made through Sequential Least Squares Programming optimization technique knowing forecasted returns.  

### Convolutional Neural Network Model

For the convolutional layer I have used 64 kernels with kernel of 2.  
It is followed by three simple neural network layers.  
The optimizer uses the Adam algorithm (best performance during my trials).  
The loss function use is the mean squared error.  
The output our CNN is a scalar.

Example LSTM industry forecast : 

![Alt text](CNN_Fin_forecast.png?raw=true "Title")

CNN portfolio performance : 

![Alt text](CNN_portfolio.png?raw=true "Title")

CNN weights : 

![Alt text](CNN_weights.png?raw=true "Title")

### Long Short Term Memory model 

For the LSTM I have used 50 hidden states and a dropout of 0.1 following by a simple neural network layer.   
I've kept the same optimizer and loss function as the CNN model.  
The optimizer uses the Adam algorithm (best performance during my trials).  
The loss function use is the mean squared error.  
The output of the model is still a scalar.  

Example LSTM industry forecast : 

![Alt text](LSTM_Smoke_forecast.png?raw=true "Title")

LSTM portfolio performance : 

![Alt text](LSTM_portfolio.png?raw=true "Title")

LSTM weights : 

![Alt text](LSTM_weights.png?raw=true "Title")