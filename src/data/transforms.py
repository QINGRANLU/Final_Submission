from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DropCol(BaseEstimator, TransformerMixin):
    """
    Transformation class to drop single or multiple
    columns from a dataset.

    Methods:
    - fit():       Required by the sklearn preprocessing 
                   classes. Can ignore this.

    - transform(): Is called to transform the data and drop
                   columns.
    """
    def __init__(self, cols):
        self.cols_ = list(cols)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.cols_, axis=1)

class SelectCountry(BaseEstimator, TransformerMixin):
    """
    Transformation class to select a single country
    market from the customer dataset.

    Methods:
    - fit():       Required by the sklearn preprocessing 
                   classes. Can ignore this.

    - transform(): Is called to transform the data and select
                   the customers from a country.
    """
    def __init__(self, country):
        self.country = country
    
    def fit(self, X, y=None):
        return self

    def transform(self,X, y=None):
        return X[X[f"country_{self.country}"] == 1.0]
    
class SelectBrand(BaseEstimator, TransformerMixin):
    """
    Transformation class to select a single brand
    market from the product dataset.

    Methods:
    - fit():       Required by the sklearn preprocessing 
                   classes. Can ignore this.

    - transform(): Is called to transform the data and select
                   the products belonging to a brand.
    """
    def __init__(self, brand):
        self.brand = brand
    
    def fit(self, X, y=None):
        return self

    def transform(self,X, y=None):
        return X[X[f"brand_{self.brand}"] == 1.0]

class RemoveCustomerOutliers(BaseEstimator, TransformerMixin):
    """
    Transformation class to remove the outliers in the customer
    dataset as described in the report for Phase 1.

    Methods:
    - returnRateCut(): Applies the cutoff to the returnRate column.

    - returnsCut(): Applies the cutoff to the returns column.

    - salesCut(): Applies the cutoff to the sales column.

    - maleCut(): Applies the cutoff to the is_male column.

    - fit():       Required by the sklearn preprocessing 
                   classes. Can ignore this.

    - transform(): Is called to transform the data and 
                   apply the above methods.
    """
    def __init__(self):
        self.returnRateCol = "customerReturnRate"
        self.customerReturnsCol = "returnsPerCustomer"
        self.customerSalesCol = "salesPerCustomer"
        self.isMaleCol = "isMale"
        
        self.returnRateCutoff = 1.0  # Remove customers with invalid return rate
        self.customerReturnsCutoff = 400  # Remove customers with more than 400 returns
        self.customerSalesCutoffLow = 3  # Remove customers with less than 3 purchases
        self.customerSalesCutoffHigh = 500  # Remove customers with more than 500 purchases
        self.isMaleCutoff = 1

    def returnRateCut(self, X):
        return X[X[self.returnRateCol] < self.returnRateCutoff]
        
    def returnsCut(self, X):
        return X[X[self.customerReturnsCol] < self.customerReturnsCutoff]
    
    def salesCut(self, X):
        return X[(X[self.customerSalesCol] < self.customerSalesCutoffHigh)
                 & (X[self.customerSalesCol] > self.customerSalesCutoffLow)]
    
    def maleCut(self, X):
        return X[X[self.isMaleCol] <= self.isMaleCutoff]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = self.returnRateCut(X)
        X = self.returnsCut(X)
        X = self.salesCut(X)
        X = self.maleCut(X)
        return X.reset_index(drop=True)


class DropYearOfBirth(BaseEstimator, TransformerMixin):
    """
    Transformation class to remove all customers with an 
    invalid DOB.

    Methods:
    - fit():       Required by the sklearn preprocessing 
                   classes. Can ignore this.

    - transform(): Is called to transform the data and drop
                   customers with invalid DOBs.
    """
    def __init__(self):
        self.col_ = "yearOfBirth"
        self.cutoff_ = 1920
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[X[self.col_] > self.cutoff_].reset_index(drop=True)

class SelectLowAndHighReturningProducts(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.returnRateCol = "productReturnRate"
        self.returnRateCutoffHigh = 0.7
        self.returnRateCutoffLow = 0.3

    def returnRateCut(self, X):
        return X[(X[self.returnRateCol] > self.returnRateCutoffHigh) 
                    | (X[self.returnRateCol] < self.returnRateCutoffLow)]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.returnRateCut(X)
        return X.reset_index(drop=True)

class SelectLowAndHighReturningCustomers(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.returnRateCol = "customerReturnRate"
        self.returnRateCutoffHigh = 0.7
        self.returnRateCutoffLow = 0.3

    def returnRateCut(self, X):
        return X[(X[self.returnRateCol] > self.returnRateCutoffHigh) 
                    | (X[self.returnRateCol] < self.returnRateCutoffLow)]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.returnRateCut(X)
        return X.reset_index(drop=True)

class RemoveProductOutliers(BaseEstimator, TransformerMixin):
    """
    Transformation class to remove the outliers in the product
    dataset as described in the report for Phase 1.

    Methods:
    - returnRateCut(): Applies the cutoff to the returnRate column.

    - returnsCut(): Applies the cutoff to the returns column.

    - salesCut(): Applies the cutoff to the sales column.

    - priceCut(): Applies the cutoff to the avg_price column.

    - fit():       Required by the sklearn preprocessing 
                   classes. Can ignore this.

    - transform(): Is called to transform the data and 
                   apply the above methods.
    """
    def __init__(self):
        self.returnRateCol = "productReturnRate"
        self.productReturnsCol = "returnsPerProduct"
        self.productSalesCol = "salesPerProduct"
        self.productAvgPrice = "avgGbpPrice"
        
        self.returnRateCutoff = 1.0  # Remove products with invalid return rate
        self.productReturnsCutoff = 1000  # Remove products with more than 1000 returns
        self.productSalesCutoffLow = 3  # Remove products with less than 3 purchases
        self.productSalesCutoffHigh = 5000  # Remove products with more than 5000 purchases
        self.productAvgPriceCutoff = 500  # Remove products with a price higher than 500 GBP

    def returnRateCut(self, X):
        return X[X[self.returnRateCol] < self.returnRateCutoff]
        
    def returnsCut(self, X):
        return X[X[self.productReturnsCol] < self.productReturnsCutoff]
    
    def salesCut(self, X):
        return X[(X[self.productSalesCol] < self.productSalesCutoffHigh) 
                 & (X[self.productSalesCol] > self.productSalesCutoffLow)]
    
    def priceCut(self, X):
        return X[X[self.productAvgPrice] <= self.productAvgPriceCutoff]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = self.returnRateCut(X)
        X = self.returnsCut(X)
        X = self.salesCut(X)
        X = self.priceCut(X)
        return X.reset_index(drop=True)

class MinMaxScale(BaseEstimator, TransformerMixin):
    """
    This is a wrapper function for the MinMaxScaler from sklearn.
    This allows us to choose only certain columns to scale, i.e.
    we can avoid the one-hot-encoded columns.

    This was helpful in regularising the ML models when training.

    Methods:
    - fit():       Fits the min-max scaler to selected columns 
                   from the data.

    - transform(): Is called to transform the data and output a 
                   scaled version of the data
    """
    def __init__(self, cols):
        self.scaler = MinMaxScaler()
        self.cols = list(cols)

    def fit(self, X, y=None):
        self.scaler.fit(X[self.cols])
        return self

    def transform(self, X, y=None):

        X[self.cols] = self.scaler.transform(X[self.cols])

        return X
