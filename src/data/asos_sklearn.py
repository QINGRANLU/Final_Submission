import os
import shutil
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from numpy.random import default_rng

from .transforms import DropCol, RemoveCustomerOutliers, DropYearOfBirth, SelectCountry
from .transforms import RemoveProductOutliers, SelectBrand

CUSTOMER_TRANSFORMS = {
    "dropcol": DropCol,
    "removeoutliers": RemoveCustomerOutliers,
    "dropyob": DropYearOfBirth,
    "selectCountry": SelectCountry
}

PRODUCT_TRANSFORMS = {
    "dropcol": DropCol,
    "removeoutliers": RemoveProductOutliers,
    "selectBrand": SelectBrand
}

class ASOSData_sklearn():
    def __init__(self, load_path, reduced=False, validation_size=False, 
                        customer_transforms=None, product_transforms=None,
                        test=False, include_product_nodes=False,
                        validation=False):
        self.name = "ASOS Data - sklearn"
        self.load_path = load_path
        self.test = test
        self.validation = validation

        if self.test or self.validation:
            reduced = False
    
        if isinstance(customer_transforms, dict):
            self.customer_transforms = []
            for name, args in customer_transforms.items():
                if isinstance(args, dict):
                    self.customer_transforms.append((name, CUSTOMER_TRANSFORMS[name](**args)))
                else:
                    self.customer_transforms.append((name, CUSTOMER_TRANSFORMS[name]()))

            self.customer_transforms = Pipeline(self.customer_transforms)

        elif isinstance(customer_transforms, Pipeline):
            self.customer_transforms = customer_transforms

        else:
            self.customer_transforms = False

        if isinstance(product_transforms, dict):
            self.product_transforms = []
            for name, args in product_transforms.items():
                if isinstance(args, dict):
                    self.product_transforms.append((name, PRODUCT_TRANSFORMS[name](**args)))
                else:
                    self.product_transforms.append((name, PRODUCT_TRANSFORMS[name]()))

            self.product_transforms = Pipeline(self.product_transforms)
        elif isinstance(product_transforms, Pipeline):
            self.product_transforms = product_transforms

        else:
            self.product_transforms = False

        if not isinstance(reduced, bool):
            self.data_path = "gnn_reduced_data"
            self.is_reduced = True

            path = os.path.join(self.load_path, "raw", self.data_path)
            if os.path.exists(path):
                shutil.rmtree(path)

            os.mkdir(path, mode=0o777)
            
            self.create_reduced_data(reduced)

        else:
            self.is_reduced = False
            self.data_path = "gnn_data"

        if self.test:
            self.file_path = "testing.csv"
        elif self.validation:
            self.file_path = "validation.csv"
        else:
            self.file_path = "training.csv"

        self.process()

    def describe(self):
        return self.__str__

    def create_reduced_data(self, reduced):
        full_path = os.path.join(self.load_path, "raw", "gnn_data")
        # Load in the customer node features - training set
        df_customers = pd.read_csv(os.path.join(full_path, "customer_nodes_training.csv"))
        df_products = pd.read_csv(os.path.join(full_path, "product_nodes_training.csv"))
        df_events = pd.read_csv(os.path.join(full_path, "event_table_training.csv"))

        if self.customer_transforms:
            df_customers = self.customer_transforms.fit_transform(df_customers)
            df_events = pd.merge(df_events, df_customers["hash(customerId)"], 
                                        on="hash(customerId)", how="inner")

        if self.product_transforms:
            df_products = self.product_transforms.fit_transform(df_products)
            df_events = pd.merge(df_events, df_products["variantID"],
                                            on="variantID", how="inner")

        # Load in the customer node features - test set
        df_customers_test = pd.read_csv(os.path.join(full_path, "customer_nodes_testing.csv"))
        df_products_test = pd.read_csv(os.path.join(full_path, "product_nodes_testing.csv"))
        df_events_test = pd.read_csv(os.path.join(full_path, "event_table_testing.csv"))

        if self.customer_transforms:
            df_customers_test = self.customer_transforms.fit_transform(df_customers_test)
            df_events_test = pd.merge(df_events_test, df_customers["hash(customerId)"], 
                                        on="hash(customerId)", how="inner")

        if self.product_transforms:
            df_products_test = self.product_transforms.fit_transform(df_products_test)
            df_events_test = pd.merge(df_events_test, df_products["variantID"],
                                            on="variantID", how="inner")

        rng = default_rng(seed=np.random.get_state()[1][0])

        events_rnd = rng.integers(len(df_events), size=reduced)

        # df_events_reduced = df_events.iloc[events_rnd].drop_duplicates()
        if len(df_events) > reduced:
            df_events_reduced = df_events.sample(n=reduced, random_state=np.random.get_state()[1][0])
        else:
            df_events_reduced = df_events

        df_products_reduced = pd.merge(df_events_reduced["variantID"], df_products, 
                                            on="variantID", how="inner").drop_duplicates()
        df_customers_reduced = pd.merge(df_events_reduced["hash(customerId)"], df_customers, 
                                            on="hash(customerId)", how="inner").drop_duplicates()

        df_events_test_reduced = pd.merge(df_events_test, df_products_reduced["variantID"], on="variantID", how="inner")
        df_events_test_reduced = pd.merge(df_events_test_reduced, df_customers_reduced["hash(customerId)"], 
                                        on="hash(customerId)", how="inner")

        full_path_reduced = os.path.join(self.load_path, "raw", self.data_path)

        df_customers_reduced.to_csv(os.path.join(full_path_reduced, "customer_nodes_training.csv"), index=False)
        df_products_reduced.to_csv(os.path.join(full_path_reduced, "product_nodes_training.csv"), index=False)
        
        df_customers_reduced.to_csv(os.path.join(full_path_reduced, "customer_nodes_testing.csv"), index=False)
        df_products_reduced.to_csv(os.path.join(full_path_reduced, "product_nodes_testing.csv"), index=False)

        df_events_reduced.to_csv(os.path.join(full_path_reduced, "event_table_training.csv"), index=False)
        df_events_test_reduced.to_csv(os.path.join(full_path_reduced, "event_table_testing.csv"), index=False)

    @property
    def customer_data(self):
        df = pd.read_csv(os.path.join(self.load_path, "raw", self.data_path, f"customer_nodes_{self.file_path}"))
        return df

    @property
    def product_data(self):
        df = pd.read_csv(os.path.join(self.load_path, "raw", self.data_path, f"product_nodes_{self.file_path}"))
        return df

    @property
    def train_events_data(self):
        df = pd.read_csv(os.path.join(self.load_path, "raw", self.data_path, f"event_table_{self.file_path}"))
        df_product_inner = pd.merge(df, self.product_data["variantID"], on="variantID", how="inner")
        return pd.merge(df_product_inner, self.customer_data["hash(customerId)"], on="hash(customerId)", how="inner")

    def merge_datasets(self, df_customers, df_products, df_events):
        df_product_events = pd.merge(df_products, df_events, on="variantID", how="inner")
        df_full = pd.merge(df_customers, df_product_events, on="hash(customerId)", how="inner")
        return df_full

    def process(self):
        label_col = "isReturned"

        customers = self.customer_data
        products = self.product_data
        
        events = self.train_events_data

        if self.customer_transforms and not self.is_reduced:
            customers = self.customer_transforms.fit_transform(customers)

        if self.product_transforms and not self.is_reduced:
            products = self.product_transforms.fit_transform(products)

        data = self.merge_datasets(customers, products, events)

        data.drop(["hash(customerId)", "productID", "variantID"], axis=1, inplace=True)

        self.X, self.y = data.loc[:, data.columns != label_col], data[label_col]

    def remove_data(self):
        if self.is_reduced:
            shutil.rmtree(os.path.join(self.load_path, "raw", self.data_path))
        else:
            pass

if __name__ == "__main__":
    data_args = {
        "load_path": "../../raw/gnn_reduced_data"
    }

    data = ASOSData_sklearn(**data_args)
    print(data)
    print(data.X_train, data.y_train)