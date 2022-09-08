import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_file(file_name):
    result = pd.read_csv(file_name)
    print(f"Loaded {file_name}")
    return result


with ThreadPoolExecutor(max_workers=6) as e:
    cust_train = e.submit(read_file, "customer_nodes_training.csv")
    prod_train = e.submit(read_file, "product_nodes_training.csv")
    events_train = e.submit(read_file, "event_table_training.csv")

    cust_test = e.submit(read_file, "customer_nodes_testing.csv")
    prod_test = e.submit(read_file, "product_nodes_testing.csv")
    events_test = e.submit(read_file, "event_table_testing.csv")

    df_customers = cust_train.result()
    df_products = prod_train.result()
    df_events_full = events_train.result()

    df_events, df_events_val = train_test_split(
        df_events_full, test_size=10000, random_state=12, shuffle=True
    )

    df_customers_test = cust_test.result()
    df_products_test = prod_test.result()
    df_events_test = events_test.result()

    df_products_val = pd.merge(
        df_events_val["variantID"], df_products, on="variantID", how="inner"
    ).drop_duplicates()
    df_customers_val = pd.merge(
        df_events_val["hash(customerId)"],
        df_customers,
        on="hash(customerId)",
        how="inner",
    ).drop_duplicates()

    df_events.to_csv("event_table_train.csv", index=False)

    df_products_val.to_csv("product_nodes_validation.csv", index=False)
    df_customers_val.to_csv("customer_nodes_validation.csv", index=False)
    df_events_val.to_csv("event_table_validation.csv", index=False)


def get_top_return_reason_links():
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

    cust_rrs = df_customers[list(cust_rr_mapping.keys())]
    cust_top_rr = cust_rrs.idxmax(axis="columns")
    cust_top_rr = cust_top_rr.map(cust_rr_mapping).rename("rr_id")
    links_cust_rr = pd.concat([df_customers["hash(customerId)"], cust_top_rr], axis=1)

    var_rrs = df_products[list(var_rr_mapping.keys())]
    var_top_rr = var_rrs.idxmax(axis="columns")
    var_top_rr = var_top_rr.map(var_rr_mapping).rename("rr_id")
    links_var_rr = pd.concat([df_products["variantID"], var_top_rr], axis=1)

    links_cust_rr.to_csv("cust_rrs_table_training.csv", index=False)
    links_var_rr.to_csv("var_rrs_table_training.csv", index=False)

    cust_rrs_test = df_customers_test[list(cust_rr_mapping.keys())]
    cust_top_rr_test = cust_rrs_test.idxmax(axis="columns")
    cust_top_rr_test = cust_top_rr_test.map(cust_rr_mapping).rename("rr_id")
    links_cust_rr_test = pd.concat(
        [df_customers_test["hash(customerId)"], cust_top_rr_test], axis=1
    )

    var_rrs_test = df_products_test[list(var_rr_mapping.keys())]
    var_top_rr_test = var_rrs_test.idxmax(axis="columns")
    var_top_rr_test = var_top_rr_test.map(var_rr_mapping).rename("rr_id")
    links_var_rr_test = pd.concat(
        [df_products_test["variantID"], var_top_rr_test], axis=1
    )

    links_cust_rr_test.to_csv("cust_rrs_table_testing.csv", index=False)
    links_var_rr_test.to_csv("var_rrs_table_testing.csv", index=False)

    print("Finished constructing rr links.")


def get_variant_product_links():
    variant_product_links = (
        df_products[["variantID", "productID"]].drop_duplicates().reset_index(drop=True)
    )
    variant_product_links_test = (
        df_products_test[["variantID", "productID"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    variant_product_links.to_csv("variant_table_training.csv", index=False)
    variant_product_links_test.to_csv("variant_table_testing.csv", index=False)

    print("Finished constructing variant links!")


def get_country_links():
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

    customer_country_links = (
        df_customers[["hash(customerId)", "shippingCountry"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    customer_country_links["countryID"] = customer_country_links["shippingCountry"].map(
        country_mapping
    )

    customer_country_links_test = (
        df_customers_test[["hash(customerId)", "shippingCountry"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    customer_country_links_test["countryID"] = customer_country_links_test[
        "shippingCountry"
    ].map(country_mapping)

    customer_country_links.to_csv("country_table_training.csv", index=False)
    customer_country_links_test.to_csv("country_table_testing.csv", index=False)

    print("Finished constructing country links!")


def get_brand_links():
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
        "Pull&Bear": 9,
        "ASOS Tall": 9,
        "other": 9,
    }

    product_brand_links = (
        df_products[["productID", "brandDesc"]]
        .drop_duplicates("productID")
        .reset_index(drop=True)
    )
    product_brand_links["brandID"] = product_brand_links["brandDesc"].map(brand_mapping)

    product_brand_links_test = (
        df_products_test[["productID", "brandDesc"]]
        .drop_duplicates("productID")
        .reset_index(drop=True)
    )
    product_brand_links_test["brandID"] = product_brand_links_test["brandDesc"].map(
        brand_mapping
    )

    product_brand_links.to_csv("brand_table_training.csv", index=False)
    product_brand_links_test.to_csv("brand_table_testing.csv", index=False)

    print("Finished constructing brand links!")


def get_product_type_links():
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

    product_type_links = (
        df_products[["productID", "productType"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    product_type_links["typeID"] = product_type_links["productType"].map(type_mapping)

    product_type_links_test = (
        df_products_test[["productID", "productType"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    product_type_links_test["typeID"] = product_type_links_test["productType"].map(
        type_mapping
    )

    product_type_links.to_csv("productType_table_training.csv", index=False)
    product_type_links_test.to_csv("productType_table_testing.csv", index=False)

    print("Finished constructing product type links!")


with ThreadPoolExecutor(max_workers=5) as e:
    var_links = e.submit(get_variant_product_links)
    cou_links = e.submit(get_country_links)
    brand_links = e.submit(get_brand_links)
    prodtype_links = e.submit(get_product_type_links)
    rr_links = e.submit(get_top_return_reason_links)
