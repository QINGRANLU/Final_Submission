import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from sklearn.pipeline import Pipeline
from torch_geometric.data import HeteroData, InMemoryDataset, download_url, extract_zip

from .transforms import (
    DropCol,
    DropYearOfBirth,
    MinMaxScale,
    RemoveCustomerOutliers,
    RemoveProductOutliers,
    SelectBrand,
    SelectCountry,
    SelectLowAndHighReturningCustomers,
    SelectLowAndHighReturningProducts,
)

CUSTOMER_TRANSFORMS = {
    "dropcol": DropCol,
    "removeoutliers": RemoveCustomerOutliers,
    "dropyob": DropYearOfBirth,
    "selectCountry": SelectCountry,
    "minMaxScaler": MinMaxScale,
    "lowAndHighReturns": SelectLowAndHighReturningCustomers,
}

PRODUCT_TRANSFORMS = {
    "dropcol": DropCol,
    "removeoutliers": RemoveProductOutliers,
    "selectBrand": SelectBrand,
    "minMaxScaler": MinMaxScale,
    "lowAndHighReturns": SelectLowAndHighReturningProducts,
}

country_mapping = {
    "Australia": 0,
    "Austria": 1,
    "Denmark": 2,
    "France": 3,
    "Germany": 4,
    "Netherlands": 5,
    "Sweden": 6,
    "UK": 7,
    "United States": 8,
}

brand_mapping = {
    "ASOS DESIGN": 0,
    "ASOS Petite": 1,
    "Topshop": 2,
    "Stradivarius": 3,
    "Bershka": 4,
    "ASOS Curve": 5,
    "New Look": 6,
    "Collusion": 7,
    "Nike": 8,
    "other": 9,
    "Pull&Bear": 9,
    "ASOS Tall": 9,
}

type_mapping = {
    "Shirts": 0,
    "Dresses": 1,
    "Shorts": 2,
    "Trainers": 3,
    "T-shirts": 4,
    "Trousers": 5,
    "Jeans": 6,
    "Jackets": 7,
    "Jumpers": 8,
    "Crop Tops": 9,
    "other": 10,
}

cust_rr_mapping = {
    "customerId_level_(Looks different to image on site)_ratio": 0,
    "customerId_level_(More_than_one_size_ordered)_ratio": 1,
    "customerId_level_(Late_Delivery)_ratio": 2,
    "customerId_level_(Poor Quality)_ratio": 3,
    "customerId_level_(Doesn't suit)_ratio": 4,
    "customerId_level_(Incorrect item)_ratio": 5,
    "customerId_level_(Parcel Damaged)_ratio": 6,
    "customerId_level_(Faulty/Broken)_ratio": 7,
    "customerId_level_(Too Big/Long)_ratio": 8,
    "customerId_level_(Too Small/Short)_ratio": 9,
    "customerId_level_(Changed my Mind)_ratio": 10,
    "customerId_level_(Missing item from multipack)_ratio": 11,
    "customerId_level_(Doesn't fit)_ratio": 12,
}

var_rr_mapping = {
    "variantID_level_(Looks different to image on site)_ratio": 0,
    "variantID_level_(More_than_one_size_ordered)_ratio": 1,
    "variantID_level_(Late_Delivery)_ratio": 2,
    "variantID_level_(Poor Quality)_ratio": 3,
    "variantID_level_(Doesn't suit)_ratio": 4,
    "variantID_level_(Incorrect item)_ratio": 5,
    "variantID_level_(Parcel Damaged)_ratio": 6,
    "variantID_level_(Faulty/Broken)_ratio": 7,
    "variantID_level_(Too Big/Long)_ratio": 8,
    "variantID_level_(Too Small/Short)_ratio": 9,
    "variantID_level_(Changed my Mind)_ratio": 10,
    "variantID_level_(Missing item from multipack)_ratio": 11,
    "variantID_level_(Doesn't fit)_ratio": 12,
}

product_rr_mapping = {
    "variantID_level_(Looks different to image on site)_ratio": 0,
    "variantID_level_(More_than_one_size_ordered)_ratio": 1,
    "variantID_level_(Late_Delivery)_ratio": 2,
    "variantID_level_(Poor Quality)_ratio": 3,
    "variantID_level_(Doesn't suit)_ratio": 4,
    "variantID_level_(Incorrect item)_ratio": 5,
    "variantID_level_(Parcel Damaged)_ratio": 6,
    "variantID_level_(Faulty/Broken)_ratio": 7,
    "variantID_level_(Too Big/Long)_ratio": 8,
    "variantID_level_(Too Small/Short)_ratio": 9,
    "variantID_level_(Changed my Mind)_ratio": 10,
    "variantID_level_(Missing item from multipack)_ratio": 11,
    "variantID_level_(Doesn't fit)_ratio": 12,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


class ASOSData_pyg(InMemoryDataset):
    r"""A heterogeneous rating dataset, assembled by ASOS AI Research from
    an internal dataset of purchases made on ASOS.com, consisting of nodes of
    type :obj:`"customer"` and :obj:`"product"`.
    Customer returns for products are available as ground truth labels for the edges
    between the customers and the products :obj:`("customer", "returns", "product")`.

    Args:
        load_path (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        model_name (str, optional):
        reduced (bool, optional):
        test (bool, optional):
        include_product_nodes (bool, optional):
        include_country_nodes (bool, optional):

    """

    def __init__(
        self,
        load_path,
        customer_transforms: Optional[Callable] = None,
        product_transforms: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        model_name: Optional[str] = "asos-returns-model",
        reduced: Optional[bool or int] = False,
        test: Optional[bool] = False,
        validation: Optional[bool] = False,
        product_links: Optional[str or bool] = False,
        country_links: Optional[str or bool] = False,
        brand_links: Optional[str or bool] = False,
        type_links: Optional[str or bool] = False,
        cust_reason_links: Optional[str or bool] = False,
        var_reason_links: Optional[str or bool] = False,
        product_reason_links: Optional[str or bool] = False,
    ):

        # Load in variables
        self.name = "ASOS Data - pyg"
        self.load_path = load_path
        self.model_name = model_name
        self.test = test
        self.validation = validation
        self.product_links = (
            product_links
            if product_links == "virtual" or product_links == "direct"
            else False
        )
        self.country_links = (
            country_links
            if country_links == "virtual" or country_links == "direct"
            else False
        )
        self.brand_links = (
            brand_links
            if brand_links == "virtual" or brand_links == "direct"
            else False
        )
        self.type_links = (
            type_links if type_links == "virtual" or type_links == "direct" else False
        )
        self.cust_reason_links = (
            cust_reason_links
            if cust_reason_links == "virtual" or cust_reason_links == "direct"
            else False
        )
        self.var_reason_links = (
            var_reason_links
            if var_reason_links == "virtual" or var_reason_links == "direct"
            else False
        )
        self.product_reason_links = (
            product_reason_links
            if product_reason_links == "virtual" or product_reason_links == "direct"
            else False
        )

        # Require the full test set so no reduced data allowed
        if self.test or self.validation:
            reduced = False

        # Define and create file paths for reduced or full datasets
        if isinstance(customer_transforms, dict):
            self.customer_transforms = []
            for name, args in customer_transforms.items():
                if isinstance(args, dict):
                    self.customer_transforms.append(
                        (name, CUSTOMER_TRANSFORMS[name](**args))
                    )
                else:
                    if args and args != "test":
                        self.customer_transforms.append(
                            (name, CUSTOMER_TRANSFORMS[name]())
                        )

            self.customer_transforms = Pipeline(self.customer_transforms)

        elif isinstance(customer_transforms, Pipeline):
            self.customer_transforms = customer_transforms

        else:
            self.customer_transforms = False

        if isinstance(product_transforms, dict):
            self.product_transforms = []
            for name, args in product_transforms.items():
                if isinstance(args, dict):
                    self.product_transforms.append(
                        (name, PRODUCT_TRANSFORMS[name](**args))
                    )
                else:
                    if args and args != "test":
                        self.product_transforms.append(
                            (name, PRODUCT_TRANSFORMS[name]())
                        )

            self.product_transforms = Pipeline(self.product_transforms)

        elif isinstance(product_transforms, Pipeline):
            self.product_transforms = product_transforms

        else:
            self.product_transforms = False

        if not isinstance(reduced, bool):
            self.data_path = f"gnn_reduced_data"
            self.data_path_processed = (
                f"data_reduced_{self.model_name}_test.pt"
                if test
                else f"data_reduced_{self.model_name}.pt"
            )
            self.is_reduced = True

            path = os.path.join(self.load_path, "raw", self.data_path)

            if os.path.exists(path):
                shutil.rmtree(path)
                if os.path.exists(
                    os.path.join(self.load_path, "processed", self.processed_file_names)
                ):
                    os.remove(
                        os.path.join(
                            self.load_path, "processed", self.processed_file_names
                        )
                    )

            os.mkdir(path, mode=0o777)
            self.create_reduced_data(reduced)

        else:
            self.data_path = "gnn_data"
            if self.test:
                self.data_path_processed = f"data_{self.model_name}_test.pt"
            elif self.validation:
                self.data_path_processed = f"data_{self.model_name}_validation.pt"
            else:
                self.data_path_processed = f"data_{self.model_name}.pt"

            self.is_reduced = False

            if os.path.exists(
                os.path.join(self.load_path, "processed", self.processed_file_names)
            ):
                os.remove(
                    os.path.join(self.load_path, "processed", self.processed_file_names)
                )

        transform = None
        super().__init__(load_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """
        Required by PyG
        """
        if self.test:
            return [
                osp.join(self.data_path, "customer_nodes_testing.csv"),
                osp.join(self.data_path, "product_nodes_testing.csv"),
                osp.join(self.data_path, "event_table_testing.csv"),
            ]
        elif self.validation:
            return [
                osp.join(self.data_path, "customer_nodes_validation.csv"),
                osp.join(self.data_path, "product_nodes_validation.csv"),
                osp.join(self.data_path, "event_table_validation.csv"),
            ]
        else:
            return [
                osp.join(self.data_path, "customer_nodes_training.csv"),
                osp.join(self.data_path, "product_nodes_training.csv"),
                osp.join(self.data_path, "event_table_train.csv"),
            ]

    @property
    def processed_file_names(self) -> str:
        """
        Required by PyG
        """
        return self.data_path_processed

    def download(self):
        """
        Required by PyG
        """
        pass

    def describe(self):
        """
        Required by PyG
        """
        return self.data

    def create_reduced_data(self, reduced):
        """
        Method to sample and create the reduced dataset.
        """
        # Define the path to the full dataset
        full_path = os.path.join(self.load_path, "raw", "gnn_data")
        # Load in the customer node features - training set
        df_customers = pd.read_csv(
            os.path.join(full_path, "customer_nodes_training.csv")
        )
        df_products = pd.read_csv(os.path.join(full_path, "product_nodes_training.csv"))
        df_events = pd.read_csv(os.path.join(full_path, "event_table_train.csv"))

        if self.customer_transforms:
            df_customers = self.customer_transforms.fit_transform(df_customers)
            df_events = pd.merge(
                df_events,
                df_customers["hash(customerId)"],
                on="hash(customerId)",
                how="inner",
            )

        if self.product_transforms:
            df_products = self.product_transforms.fit_transform(df_products)
            df_events = pd.merge(
                df_events, df_products["variantID"], on="variantID", how="inner"
            )

        # Load in the customer node features - test set
        df_customers_test = pd.read_csv(
            os.path.join(full_path, "customer_nodes_testing.csv")
        )
        df_products_test = pd.read_csv(
            os.path.join(full_path, "product_nodes_testing.csv")
        )
        df_events_test = pd.read_csv(os.path.join(full_path, "event_table_testing.csv"))

        if self.customer_transforms:
            df_customers_test = self.customer_transforms.transform(df_customers_test)
            df_events_test = pd.merge(
                df_events_test,
                df_customers_test["hash(customerId)"],
                on="hash(customerId)",
                how="inner",
            )

        if self.product_transforms:
            df_products_test = self.product_transforms.transform(df_products_test)
            df_events_test = pd.merge(
                df_events_test,
                df_products_test["variantID"],
                on="variantID",
                how="inner",
            )

        # Check we have a large enough dataset to create a reduced set
        if len(df_events) > reduced:
            # Sample n events from full events (purchase) data
            df_events_reduced = df_events.sample(
                n=reduced, random_state=np.random.get_state()[1][0]
            )
        else:
            df_events_reduced = df_events

        # Only keep the customers and products we have events for in the customer and product data
        df_products_reduced = pd.merge(
            df_events_reduced["variantID"], df_products, on="variantID", how="inner"
        ).drop_duplicates()
        df_customers_reduced = pd.merge(
            df_events_reduced["hash(customerId)"],
            df_customers,
            on="hash(customerId)",
            how="inner",
        ).drop_duplicates()

        # df_events_test_reduced = pd.merge(df_events_test, df_products_reduced["variantID"], on="variantID", how="inner")
        # df_events_test_reduced = pd.merge(df_events_test_reduced, df_customers_reduced["hash(customerId)"],
        #                                 on="hash(customerId)", how="inner")

        # Create file path for reduced data.
        full_path_reduced = os.path.join(self.load_path, "raw", self.data_path)

        # Save all new reduced sets to folder
        df_customers_reduced.to_csv(
            os.path.join(full_path_reduced, "customer_nodes_training.csv"), index=False
        )
        df_products_reduced.to_csv(
            os.path.join(full_path_reduced, "product_nodes_training.csv"), index=False
        )

        df_customers_test.to_csv(
            os.path.join(full_path_reduced, "customer_nodes_testing.csv"), index=False
        )
        df_products_test.to_csv(
            os.path.join(full_path_reduced, "product_nodes_testing.csv"), index=False
        )

        df_events_reduced.to_csv(
            os.path.join(full_path_reduced, "event_table_train.csv"), index=False
        )
        df_events_test.to_csv(
            os.path.join(full_path_reduced, "event_table_testing.csv"), index=False
        )

    def add_variant_product_links(self, data, products, link_type):
        """
        Method for finding the variant-product links and incorporating these into the graph structure.
        Either by using virtual nodes or by adding direct links.
        """
        # Load in the variant-product links.
        file_path = (
            "variant_table_testing.csv" if self.test else "variant_table_training.csv"
        )
        variant_product_links = pd.read_csv(
            os.path.join(self.load_path, "raw", "gnn_data", file_path)
        )

        # Find all unique products and label the products dataset with indices from 0 to N (number of products)
        # This is necessary as productID is not a linear variable.
        product_mapping = (
            pd.DataFrame(products["productID"].unique())
            .reset_index()
            .set_index(0)
            .to_dict()["index"]
        )
        products["product_id"] = products["productID"].map(product_mapping)

        # Create a table of variants labelled with their product IDs.
        valid_variant_product_links = variant_product_links.merge(
            products[["variantID", "variant_id", "productID", "product_id"]],
            on=["variantID", "productID"],
            how="inner",
        )

        if link_type == "virtual":
            # Find the source (variant) and destination (product) nodes
            variant_src = torch.tensor(valid_variant_product_links["variant_id"].values)
            product_dst = torch.tensor(valid_variant_product_links["product_id"].values)

            # Create the edges.
            var_prod_edge = torch.stack([variant_src, product_dst])

            # Get virtual node information (this is currently just averages but could be adapted)
            avg_product_vals = (
                products.drop(["variantID", "productID", "variant_id"], axis=1)
                .groupby("product_id")
                .mean()
            )

            # Add node specific information to the graph product nodes
            data["product"].x = torch.from_numpy(avg_product_vals.to_numpy()).to(
                torch.float
            )

            # Get a list of indicies for product nodes
            product_nodes = list(range(0, len(avg_product_vals.index.unique())))

            data["product"].num_nodes = int(len(product_nodes))
            data["product"].node_index = torch.tensor(product_nodes)

            # Add the edges (and reverse edges) to the graph structure
            data["variant", "belongs_to", "product"].edge_index = var_prod_edge.to(
                torch.long
            )

        elif link_type == "direct":
            var_prod_links = valid_variant_product_links[["variant_id", "product_id"]]
            direct_var_prod_links = var_prod_links.merge(
                var_prod_links, how="left", on="product_id"
            )

            mask = (
                direct_var_prod_links["variant_id_x"]
                - direct_var_prod_links["variant_id_y"]
                == 0
            )
            direct_var_prod_links.drop(direct_var_prod_links[mask].index, inplace=True)

            variant_src = torch.tensor(direct_var_prod_links["variant_id_x"].values)
            variant_dst = torch.tensor(direct_var_prod_links["variant_id_y"].values)

            edge_index = torch.stack([variant_src, variant_dst])

            data["variant", "is_same_product_as", "variant"].edge_index = edge_index.to(
                torch.long
            )

        else:
            pass

    def add_variant_rr_links(self, data, products, link_type):
        """
        Method for finding the variant-product links and incorporating these into the graph structure.
        Either by using virtual nodes or by adding direct links.
        """
        # Load in the variant-product links.
        file_path = (
            "var_rrs_table_testing.csv" if self.test else "var_rrs_table_training.csv"
        )
        variant_rr_links = pd.read_csv(
            os.path.join(self.load_path, "raw", "gnn_data", file_path)
        )

        # Create a table of variants labelled with their product IDs.
        valid_variant_rr_links = variant_rr_links.merge(
            products[["variantID", "variant_id"]],
            on=["variantID"],
            how="inner",
        )

        if link_type == "virtual":
            # Find the source (variant) and destination (product) nodes
            variant_src = torch.tensor(valid_variant_rr_links["variant_id"].values)
            rr_dst = torch.tensor(valid_variant_rr_links["rr_id"].values)

            # Create the edges.
            var_rr_edge = torch.stack([variant_src, rr_dst])

            products_new = products.merge(
                variant_rr_links, on=["variantID"], how="inner"
            )

            # Get virtual node information (this is currently just averages but could be adapted)
            avg_rr_vals = (
                products_new.drop(["variantID", "variant_id"], axis=1)
                .groupby("rr_id")
                .mean()
            )

            # Add node specific information to the graph product nodes
            data["reason_var"].x = torch.from_numpy(avg_rr_vals.to_numpy()).to(
                torch.float
            )

            # Get a list of indicies for product nodes
            rr_nodes = list(range(0, len(avg_rr_vals.index.unique())))

            data["reason_var"].num_nodes = int(len(rr_nodes))
            data["reason_var"].node_index = torch.tensor(rr_nodes)

            # Add the edges (and reverse edges) to the graph structure
            data["variant", "top_reason_is", "reason_var"].edge_index = var_rr_edge.to(
                torch.long
            )

        elif link_type == "direct":
            var_rr_links = valid_variant_rr_links[["variant_id", "rr_id"]]
            var_rr_links.to_csv("var_reason_ID_for_sampling.csv")
            #build original table
            df_original_varreason = pd.read_csv("var_reason_ID_for_sampling.csv")
            df_copybase_varreason = pd.read_csv("var_reason_ID_for_sampling.csv")
            df_copybase_varreason.insert(loc=2, column='appearence_times', value=2)
            df_new_sampled_varreason = pd.DataFrame(columns=['variant_id_x', 'rr_id', 'variant_id_y'])
            for index, row in df_original_varreason.iterrows():

                # find the product id and rid
                variant_x = row['variant_id']
                variant_x_id = row['rr_id']

                if variant_x in df_copybase_varreason['variant_id'].values:

                    sample_seed = \
                        df_copybase_varreason.loc[
                            df_copybase_varreason['variant_id'] == variant_x, ['appearence_times']].values[0][0]

                    same_id_pool = df_copybase_varreason[df_copybase_varreason['variant_id'] != variant_x]
                    same_id_pool = same_id_pool[same_id_pool['rr_id'] == variant_x_id]

                    if len(same_id_pool) >= sample_seed:

                        temp_selection = same_id_pool.sample(sample_seed)

                        for index_temp, row_temp in temp_selection.iterrows():

                            variant_y = row_temp['variant_id']

                            df_copybase_varreason.loc[df_copybase_varreason['variant_id'] == variant_y, ['appearence_times']] = \
                                df_copybase_varreason.loc[
                                    df_copybase_varreason['variant_id'] == variant_y, ['appearence_times']] - 1

                            if df_copybase_varreason.loc[
                                df_copybase_varreason['variant_id'] == variant_y, ['appearence_times']].values <= 0:
                                df_copybase_varreason = df_copybase_varreason.drop(index_temp)

                            add_row = [variant_x, variant_x_id, variant_y]
                            row = pd.Series(add_row, index=df_new_sampled_varreason.columns)
                            df_new_sampled_varreason = df_new_sampled_varreason.append(row, ignore_index=True)

                    df_copybase_varreason = df_copybase_varreason.drop(index)
            direct_var_rr_links = df_new_sampled_varreason.astype('Int64')
            print(direct_var_rr_links)
            # direct_var_rr_links = var_rr_links.merge(
            #     var_rr_links, how="left", on="rr_id"
            # )
            #
            # mask = (
            #     direct_var_rr_links["variant_id_x"]
            #     - direct_var_rr_links["variant_id_y"]
            #     == 0
            # )
            # direct_var_rr_links.drop(direct_var_rr_links[mask].index, inplace=True)

            variant_src = torch.tensor(direct_var_rr_links["variant_id_x"].values)
            variant_dst = torch.tensor(direct_var_rr_links["variant_id_y"].values)

            edge_index = torch.stack([variant_src, variant_dst])

            data["variant", "same_top_reason_as", "variant"].edge_index = edge_index.to(
                torch.long
            )

        else:
            pass

    def add_product_rr_links(self, data, products, link_type):
        """
        Method for finding the product links based on different return reasons.
        Either by using virtual nodes or by adding direct links.
        """
        # Load in the variant-product links.
        file_path = (
            "sampled_data_testing.csv" if self.test else "sampled_data_training.csv"
        )
        product_rr_links = pd.read_csv(
            os.path.join(self.load_path, "raw", "gnn_data", file_path)
        )
        valid_product_rr_links = product_rr_links

        if link_type == "virtual":
            # Find the source (variant) and destination (product) nodes
            product_src = torch.tensor(valid_product_rr_links["product_id"].values)
            rr_dst = torch.tensor(valid_product_rr_links["reason_id"].values)

            # Create the edges.
            product_rr_edge = torch.stack([product_src, rr_dst])

            products_new = products.merge(
                product_rr_links, on=["variantID"], how="inner"
            )

            # Get virtual node information (this is currently just averages but could be adapted)
            avg_rr_vals = (
                products_new.drop(["productID", "product_id"], axis=1)
                    .groupby("reason_id")
                    .mean()
            )

            # Add node specific information to the graph product nodes
            data["reason_var"].x = torch.from_numpy(avg_rr_vals.to_numpy()).to(
                torch.float
            )

            # Get a list of indicies for product nodes
            rr_nodes = list(range(0, len(avg_rr_vals.index.unique())))

            data["reason_var"].num_nodes = int(len(rr_nodes))
            data["reason_var"].node_index = torch.tensor(rr_nodes)

            # Add the edges (and reverse edges) to the graph structure
            data["variant", "top_reason_is", "reason_var"].edge_index = product_rr_edge.to(
                torch.long
            )
            data["reason_var", "is_top_for", "variant"].edge_index = torch.flip(
                product_rr_edge.to(torch.long), [0]
            )

        elif link_type == "direct":
            #product_rr_links = valid_product_rr_links[["product_id", "reason_id"]]





            # direct_product_rr_links = product_rr_links.merge(
            #     product_rr_links, how="left", on="reason_id"
            # )
            ######################
            file_path_sample = (
                "new_product_reason_testing.csv" if self.test else "new_product_reason_training.csv"
            )
            ##############
            #file_path_sample = ("sampled_data.csv")
            direct_product_rr_links = pd.read_csv(
                os.path.join(self.load_path, "raw", "gnn_data", file_path_sample)
            )
            #direct_product_rr_links = pd.read_csv("sampled_data.csv")
            direct_product_rr_links = direct_product_rr_links.astype('Int64')
            print(direct_product_rr_links)
            print(type(direct_product_rr_links["product_id_x"].values))


            ##############################################
            # mask = (
            #     direct_product_rr_links["product_id_x"]
            #     - direct_product_rr_links["product_id_y"]
            #     == 0
            # )
            # direct_product_rr_links.drop(direct_product_rr_links[mask].index, inplace=True)

            product_src = torch.tensor(direct_product_rr_links["product_id_x"].values)
            product_dst = torch.tensor(direct_product_rr_links["product_id_y"].values)

            edge_index = torch.stack([product_src, product_dst])

            data["product", "same_top_reason_as", "product"].edge_index = edge_index.to(
                torch.long
            )
            print("test for runtime error")
        else:
            pass

    def add_customer_country_links(self, data, customers, link_type):
        """
        Method for finding the customer-country links and incorporating these into the graph structure.
        Either by using virtual nodes or by adding direct links.
        """
        # Load in the customer-country links.
        file_path = (
            "country_table_testing.csv" if self.test else "country_table_training.csv"
        )
        customer_country_links = pd.read_csv(
            os.path.join(self.load_path, "raw", "gnn_data", file_path)
        )

        # Create a table of customers labelled with their country IDs.
        valid_customer_country_links = customer_country_links.merge(
            customers[["customer_id", "hash(customerId)"]],
            on=["hash(customerId)"],
            how="inner",
        )

        if link_type == "virtual":
            # Find the source (customer) and destination (country) nodes
            customer_src = torch.tensor(
                valid_customer_country_links["customer_id"].values
            )
            country_dst = torch.tensor(valid_customer_country_links["countryID"].values)

            # Create the edges.
            cus_country_edge = torch.stack([customer_src, country_dst])

            # Delete country information as this is now encoded into the graph
            filter_col = [
                col for col in customers.columns if col.startswith("country_")
            ]
            customers.drop(filter_col, axis=1, inplace=True)

            # Get virtual node information (this is currently just averages but could be adapted)
            avg_country_vals = (
                customers.drop(["hash(customerId)", "customer_id"], axis=1)
                .groupby("shippingCountry")
                .mean()
            )

            # Add node specific information to the graph country nodes
            data["country"].x = torch.from_numpy(avg_country_vals.to_numpy()).to(
                torch.float
            )

            # Get a list of indicies for country nodes
            country_nodes = list(range(0, len(avg_country_vals.index.unique())))

            data["country"].num_nodes = int(len(country_nodes))
            data["country"].node_index = torch.tensor(country_nodes)

            # Add the edges (and reverse edges) to the graph structure
            data["customer", "is_from", "country"].edge_index = cus_country_edge.to(
                torch.long
            )

        elif link_type == "direct":
            cus_country_links = valid_customer_country_links[
                ["customer_id", "countryID"]
            ]
            cus_country_links.to_csv("cus_country_for_sampling.csv")
            df_original_cus= pd.read_csv("cus_country_for_sampling.csv")
            df_copybase_cus = pd.read_csv("cus_country_for_sampling.csv")
            # insert a column to appearence time which is the number of direct links
            df_copybase_cus.insert(loc=2, column='appearence_times', value=2)
            df_new_sampled_cus = pd.DataFrame(columns=['customer_id_x', 'countryID', 'customer_id_y'])
            for index, row in df_original_cus.iterrows():

                # find the pid and rid
                variant_x = row['customer_id']
                variant_x_id = row['countryID']

                if variant_x in df_copybase_cus['customer_id'].values:
                    #define sample_seed and sample pool
                    sample_seed = \
                        df_copybase_cus.loc[
                            df_copybase_cus['customer_id'] == variant_x, ['appearence_times']].values[0][0]

                    same_id_pool = df_copybase_cus[df_copybase_cus['customer_id'] != variant_x]
                    same_id_pool = same_id_pool[same_id_pool['countryID'] == variant_x_id]

                    if len(same_id_pool) >= sample_seed:

                        temp_selection = same_id_pool.sample(sample_seed)

                        for index_temp, row_temp in temp_selection.iterrows():

                            variant_y = row_temp['customer_id']

                            df_copybase_cus.loc[df_copybase_cus['customer_id'] == variant_y, ['appearence_times']] = \
                                df_copybase_cus.loc[
                                    df_copybase_cus['customer_id'] == variant_y, ['appearence_times']] - 1

                            if df_copybase_cus.loc[
                                df_copybase_cus['customer_id'] == variant_y, ['appearence_times']].values <= 0:
                                df_copybase_cus = df_copybase_cus.drop(index_temp)
                            #export direct links in new dataframe
                            add_row = [variant_x, variant_x_id, variant_y]
                            row = pd.Series(add_row, index=df_new_sampled_cus.columns)
                            df_new_sampled_cus = df_new_sampled_cus.append(row, ignore_index=True)

                    df_copybase_cus = df_copybase_cus.drop(index)
            direct_cus_country_links = df_new_sampled_cus.astype('Int64')
            print(direct_cus_country_links)


            # direct_cus_country_links = cus_country_links.merge(
            #     cus_country_links, how="left", on="countryID"
            # )
            #
            # mask = (
            #     direct_cus_country_links["customer_id_x"]
            #     - direct_cus_country_links["customer_id_y"]
            #     == 0
            # )
            # direct_cus_country_links.drop(
            #     direct_cus_country_links[mask].index, inplace=True
            # )

            # file_path_sample3 = (
            #     "mini_cus_country_testing.csv" if self.test else "mini_cus_country_training.csv"
            # )
            # ##############
            # # file_path_sample = ("sampled_data.csv")
            # direct_cus_country_links = pd.read_csv(
            #     os.path.join(self.load_path, "raw", "gnn_data", file_path_sample3)
            # )
            # # direct_product_rr_links = pd.read_csv("sampled_data.csv")
            # direct_cus_country_links = direct_cus_country_links.astype('Int64')


            customer_src = torch.tensor(
                direct_cus_country_links["customer_id_x"].values
            )
            customer_dst = torch.tensor(
                direct_cus_country_links["customer_id_y"].values
            )

            edge_index = torch.stack([customer_src, customer_dst])

            data[
                "customer", "from_same_country_as", "customer"
            ].edge_index = edge_index.to(torch.long)

        else:
            pass

    def add_variant_brand_links(self, data, products, link_type):
        """
        Method for finding the variant-brand links and incorporating these into the graph structure.
        Either by using virtual nodes or by adding direct links.
        """
        # Load in the variant-brand links.
        file_path = (
            "brand_table_testing.csv" if self.test else "brand_table_training.csv"
        )
        variant_brand_links = pd.read_csv(
            os.path.join(self.load_path, "raw", "gnn_data", file_path)
        )

        # Create a table of variants labelled with their variant IDs.
        valid_variant_brand_links = variant_brand_links.merge(
            products[["variant_id", "productID"]], on=["productID"], how="inner"
        )

        if link_type == "virtual":
            # Find the source (variant) and destination (brand) nodes
            variant_src = torch.tensor(valid_variant_brand_links["variant_id"].values)
            brand_dst = torch.tensor(valid_variant_brand_links["brandID"].values)

            # Create the edges
            var_brand_edge = torch.stack([variant_src, brand_dst])

            # Delete brand information as this is now encoded into the graph
            filter_col = [col for col in products.columns if col.startswith("brand_")]
            products.drop(filter_col, axis=1, inplace=True)

            # Get virtual node information (this is currently just averages but could be adapted)
            avg_brand_vals = (
                products.drop(["variantID", "productID", "variant_id"], axis=1)
                .groupby("brandDesc")
                .mean()
            )

            # Add node specific information to the graph brand nodes
            data["brand"].x = torch.from_numpy(avg_brand_vals.to_numpy()).to(
                torch.float
            )

            # Get a list of indicies for brand nodes
            brand_nodes = list(range(0, len(avg_brand_vals.index.unique())))

            data["brand"].num_nodes = int(len(brand_nodes))
            data["brand"].node_index = torch.tensor(brand_nodes)

            # Add the edges (and reverse edges) to the graph structure
            data["variant", "is_from", "brand"].edge_index = var_brand_edge.to(
                torch.long
            )

        elif link_type == "direct":
            var_brand_links = valid_variant_brand_links[["variant_id", "brandID"]]
            var_brand_links.to_csv("var_brand_ID_for_sampling.csv")
            df_original_brand = pd.read_csv("var_brand_ID_for_sampling.csv")
            df_copybase_brand = pd.read_csv("var_brand_ID_for_sampling.csv")
            df_copybase_brand.insert(loc=2, column='appearence_times', value=2)
            df_new_sampled_brand = pd.DataFrame(columns=['variant_id_x', 'brandID', 'variant_id_y'])
            for index, row in df_original_brand.iterrows():

                # find the pid and rid
                variant_x = row['variant_id']
                variant_x_id = row['brandID']

                if variant_x in df_copybase_brand['variant_id'].values:
                    #define the sample seed and sample pool
                    sample_seed = \
                    df_copybase_brand.loc[df_copybase_brand['variant_id'] == variant_x, ['appearence_times']].values[0][0]

                    same_id_pool = df_copybase_brand[df_copybase_brand['variant_id'] != variant_x]
                    same_id_pool = same_id_pool[same_id_pool['brandID'] == variant_x_id]

                    if len(same_id_pool) >= sample_seed:

                        temp_selection = same_id_pool.sample(sample_seed)

                        for index_temp, row_temp in temp_selection.iterrows():

                            variant_y = row_temp['variant_id']
                            # temp selection and calculate the number
                            df_copybase_brand.loc[df_copybase_brand['variant_id'] == variant_y, ['appearence_times']] = \
                            df_copybase_brand.loc[df_copybase_brand['variant_id'] == variant_y, ['appearence_times']] - 1

                            if df_copybase_brand.loc[df_copybase_brand['variant_id'] == variant_y, ['appearence_times']].values <= 0:
                                df_copybase_brand = df_copybase_brand.drop(index_temp)
                            # export into new dataframe
                            add_row = [variant_x, variant_x_id, variant_y]
                            row = pd.Series(add_row, index=df_new_sampled_brand.columns)
                            df_new_sampled_brand = df_new_sampled_brand.append(row, ignore_index=True)

                    df_copybase_brand= df_copybase_brand.drop(index)
            direct_var_brand_links = df_new_sampled_brand.astype('Int64')
            print(direct_var_brand_links)
            # direct_var_brand_links = var_brand_links.merge(
            #     var_brand_links, how="left", on="brandID"
            # )
            #
            # mask = (
            #     direct_var_brand_links["variant_id_x"]
            #     - direct_var_brand_links["variant_id_y"]
            #     == 0
            # )
            # direct_var_brand_links.drop(
            #     direct_var_brand_links[mask].index, inplace=True
            # )

            variant_src = torch.tensor(direct_var_brand_links["variant_id_x"].values)
            variant_dst = torch.tensor(direct_var_brand_links["variant_id_y"].values)

            edge_index = torch.stack([variant_src, variant_dst])

            data["variant", "from_same_brand_as", "variant"].edge_index = edge_index.to(
                torch.long
            )

        else:
            pass

    def add_variant_type_links(self, data, products, link_type):
        """
        Method for finding the variant-product type links and incorporating these into the graph structure.
        Either by using virtual nodes or by adding direct links.
        """
        # Load in the variant-product type links
        file_path = (
            "productType_table_testing.csv"
            if self.test
            else "productType_table_training.csv"
        )
        variant_type_links = pd.read_csv(
            os.path.join(self.load_path, "raw", "gnn_data", file_path)
        )

        # Create a table of vairants labelled with their product type IDs
        valid_variant_type_links = variant_type_links.merge(
            products[["variant_id", "productID"]], on=["productID"], how="inner"
        )

        if link_type == "virtual":
            # Find the source (variant) and destination (product type) nodes
            variant_src = torch.tensor(valid_variant_type_links["variant_id"].values)
            type_dst = torch.tensor(valid_variant_type_links["typeID"].values)

            # Create the edges
            var_type_edge = torch.stack([variant_src, type_dst])

            # Delete product type information as this is now encoded into the graph
            filter_col = [
                col for col in products.columns if col.startswith("productType_")
            ]
            products.drop(filter_col, axis=1, inplace=True)

            # Get virtual node information (this is currently just averages but could be adapted)
            avg_type_vals = (
                products.drop(["variantID", "productID", "variant_id"], axis=1)
                .groupby("productType")
                .mean()
            )

            # Add node specific information to the graph product type nodes
            data["product_type"].x = torch.from_numpy(avg_type_vals.to_numpy()).to(
                torch.float
            )

            # Get a list of indicies for product type nodes
            type_nodes = list(range(0, len(avg_type_vals.index.unique())))

            data["product_type"].num_nodes = int(len(type_nodes))
            data["product_type"].node_index = torch.tensor(type_nodes)

            # Add the edges (and reverse edges) to the graph structure
            data["variant", "is", "product_type"].edge_index = var_type_edge.to(
                torch.long
            )

        elif link_type == "direct":
            var_type_links = valid_variant_type_links[["variant_id", "typeID"]]
            ### add direct links
            var_type_links.to_csv("variant_type_ID_for_sampling.csv")
            df_original_type = pd.read_csv("variant_type_ID_for_sampling.csv")
            df_copybase_type = pd.read_csv("variant_type_ID_for_sampling.csv")
            df_copybase_type.insert(loc=2, column='appearence_times', value=2)
            df_new_sampled_type = pd.DataFrame(columns=['variant_id_x', 'typeID', 'variant_id_y'])
            for index, row in df_original_type.iterrows():

                # find pid and rid
                variant_x = row['variant_id']
                variant_x_id = row['typeID']

                if variant_x in df_copybase_type['variant_id'].values:
                    #define sample seed and sample pool
                    sample_seed = \
                        df_copybase_type.loc[
                            df_copybase_type['variant_id'] == variant_x, ['appearence_times']].values[0][0]

                    same_id_pool = df_copybase_type[df_copybase_type['variant_id'] != variant_x]
                    same_id_pool = same_id_pool[same_id_pool['typeID'] == variant_x_id]

                    if len(same_id_pool) >= sample_seed:

                        temp_selection = same_id_pool.sample(sample_seed)

                        for index_temp, row_temp in temp_selection.iterrows():

                            variant_y = row_temp['variant_id']

                            df_copybase_type.loc[df_copybase_type['variant_id'] == variant_y, ['appearence_times']] = \
                                df_copybase_type.loc[
                                    df_copybase_type['variant_id'] == variant_y, ['appearence_times']] - 1

                            if df_copybase_type.loc[
                                df_copybase_type['variant_id'] == variant_y, ['appearence_times']].values <= 0:
                                df_copybase_type = df_copybase_type.drop(index_temp)
                            #export into new dataframe
                            add_row = [variant_x, variant_x_id, variant_y]
                            row = pd.Series(add_row, index= df_new_sampled_type.columns)
                            df_new_sampled_type = df_new_sampled_type.append(row, ignore_index=True)

                    df_copybase_type = df_copybase_type.drop(index)
            direct_var_type_links = df_new_sampled_type.astype('Int64')
            print(direct_var_type_links)


            # direct_var_type_links = var_type_links.merge(
            #     var_type_links, how="left", on="typeID"
            # )
            #
            # mask = (
            #     direct_var_type_links["variant_id_x"]
            #     - direct_var_type_links["variant_id_y"]
            #     == 0
            # )
            # direct_var_type_links.drop(direct_var_type_links[mask].index, inplace=True)

            variant_src = torch.tensor(direct_var_type_links["variant_id_x"].values)
            variant_dst = torch.tensor(direct_var_type_links["variant_id_y"].values)

            edge_index = torch.stack([variant_src, variant_dst])

            data["variant", "is_same_type_as", "variant"].edge_index = edge_index.to(
                torch.long
            )

        else:
            pass


    def process(self):
        """
        Process method to construct the graph dataset.
        """
        # Define data variable
        data = HeteroData()

        # Read in customer information and perform transformations on this
        df_customers = pd.read_csv(self.raw_paths[0]).dropna()

        if (
            self.customer_transforms
            and not self.is_reduced
            and not (self.test or self.validation)
        ):
            df_customers = self.customer_transforms.fit_transform(df_customers)
        elif (
            self.customer_transforms
            and not self.is_reduced
            and (self.test or self.validation)
        ):
            df_customers = self.customer_transforms.transform(df_customers)
        else:
            pass

        # Insert a linear customer ID variable (required for labelling nodes)
        df_customers.insert(0, "customer_id", range(0, len(df_customers)))

        # Read in and transform product information
        df_products = pd.read_csv(self.raw_paths[1]).dropna()

        if (
            self.product_transforms
            and not self.is_reduced
            and not (self.test or self.validation)
        ):
            df_products = self.product_transforms.fit_transform(df_products)
        elif (
            self.product_transforms
            and not self.is_reduced
            and (self.test or self.validation)
        ):
            df_products = self.product_transforms.transform(df_products)
        else:
            pass

        # Insert a linear product ID variable (required for labelling nodes)
        df_products.insert(0, "variant_id", range(0, len(df_products)))

        # Read in the purchase links and merge with above datasets to ensure
        # node IDs are included (customer_id -> variant_id)
        df_events = pd.read_csv(self.raw_paths[2]).dropna()

        df_events = df_events.merge(
            df_customers[["hash(customerId)", "customer_id"]],
            on="hash(customerId)",
            how="inner",
        )

        df_valid_events = df_events.merge(
            df_products[["variantID", "variant_id"]], on="variantID", how="inner"
        )

        # Construct the purchase links
        customer_src = torch.tensor(df_valid_events["customer_id"])
        product_dst = torch.tensor(df_valid_events["variant_id"])
        edge_index = torch.stack([customer_src, product_dst])

        # Get the labels for these purchase links
        returned = torch.from_numpy(df_valid_events["isReturned"].values).to(torch.bool)

        # Get the indicies for return edges
        return_edge_index = edge_index[:, returned]

        # Add extra links
        if self.product_links:
            self.add_variant_product_links(
                data, df_products, link_type=self.product_links
            )
            df_products.drop(["product_id"], axis=1, inplace=True)

        if self.country_links:
            self.add_customer_country_links(
                data, df_customers, link_type=self.country_links
            )

        if self.brand_links:
            self.add_variant_brand_links(data, df_products, link_type=self.brand_links)

        if self.type_links:
            self.add_variant_type_links(data, df_products, link_type=self.type_links)

        if self.var_reason_links:
            self.add_variant_rr_links(
                data, df_products, link_type=self.var_reason_links
            )

        if self.product_reason_links:
            self.add_product_rr_links(
                data, df_products, link_type=self.product_reason_links
            )
        print("print self.product_reason_links")

        # Removes ids from node information, add it to index instead
        df_customers = df_customers.set_index("customer_id")
        df_products = df_products.set_index("variant_id")

        data["customer"].originalID = torch.from_numpy(
            df_customers["hash(customerId)"].to_numpy()
        ).to(torch.int)
        data["variant"].originalID = torch.from_numpy(
            df_products["variantID"].to_numpy()
        ).to(torch.int)

        # Drop non-useful information
        df_customers.drop(["hash(customerId)", "shippingCountry"], axis=1, inplace=True)
        df_products.drop(
            ["variantID", "productID", "brandDesc", "productType"], axis=1, inplace=True
        )

        # Add node features for customers and variants
        data["customer"].x = torch.from_numpy(df_customers.to_numpy()).to(torch.float)
        data["variant"].x = torch.from_numpy(df_products.to_numpy()).to(torch.float)

        # Add purchase links and labels for these
        data["customer", "purchases", "variant"].edge_index = edge_index.to(torch.long)
        data["customer", "purchases", "variant"].edge_label = returned.to(torch.long)

        # Add node information for graph
        customer_nodes = int(len(df_customers))
        product_nodes = int(len(df_products))
        data["customer"].num_nodes = customer_nodes
        data["variant"].num_nodes = product_nodes

        data["customer"].node_index = torch.tensor(df_customers.index)
        data["variant"].node_index = torch.tensor(df_products.index)
        print("before self.pre_transform")
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        print("AddSelfLoops")

        torch.save(self.collate([data]), self.processed_paths[0])
        print("torch.save")
    def remove_data(self):
        if self.is_reduced:
            shutil.rmtree(os.path.join(self.raw_dir, self.data_path))
            os.remove(os.path.join(self.processed_dir, self.processed_file_names))
        else:
            os.remove(os.path.join(self.processed_dir, self.processed_file_names))
        print("end remove_data")

if __name__ == "__main__":
    data = ASOSData(root="../..", reduced=True, test=True)
    data.process()
    dataset = data.data
    print(dataset)
