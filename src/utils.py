import os
from typing import Tuple

import pandas as pd

CUSTOMERS = 2000
PRODUCTS = 100
WEEKS = 50


def skeleton() -> pd.DataFrame:
    """Creates an skeleton dataframe containing customer_id, product_id & week.
    The dataframe will be used for later merges.

    Returns
    -------
    pd.DataFrame
        e.g.:
                customer_id	 product_id	week
            0   	      0	          0	   1
            1   	      0	          1	   1
            2   	      0	          2	   1
    """
    skeleton_df = pd.DataFrame({"customer_id": [], "product_id": [], "week": []})

    # Build customer product pairs
    customer_product_pair_gen = (
        (c, p) for c in range(CUSTOMERS) for p in range(PRODUCTS)
    )
    skeleton_cp = pd.DataFrame(
        customer_product_pair_gen, columns=["customer_id", "product_id"]
    )

    # Fill skeleton
    for w in range(1, WEEKS + 1):
        tmp = skeleton_cp.copy()
        tmp["week"] = [w] * len(skeleton_cp)
        skeleton_df = pd.concat([skeleton_df, tmp], ignore_index=True)

    assert len(skeleton_df) == CUSTOMERS * PRODUCTS * WEEKS

    return skeleton_df.astype(
        {"customer_id": "int64", "product_id": "int64", "week": "int64"}
    )


def merge_with_skeleton(
    skeleton_df: pd.DataFrame, train_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges Skeleton DataFrame with incomplete Training Data.
    By merging we're creating a full set of data: completed & incompleted transactions.

    Parameters
    ----------
    skeleton_df : pd.DataFrame
        Created with skeleton()
    train_df : pd.DataFrame
        Given Train Data with completed transactions only

    Returns
    -------
    pd.DataFrame
        A full dataset which can be used for training later.
    """
    train_df = train_df.copy()
    train_df["label"] = [1] * len(train_df)

    merged = skeleton_df.merge(
        right=train_df,
        how="left",
        on=["customer_id", "product_id", "week"],
        validate="1:m",
    )
    assert len(merged) == CUSTOMERS * PRODUCTS * WEEKS

    return merged


def features_customer(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates static and dynamic customer features for later merges into a training data set.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Complete Transaction Data created via merge_with_skeleton()

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames representing static & dynamic features
    """

    # 1. Dynamic Features
    customer_weekly = merged_df.groupby(by=["customer_id", "week"], as_index=False)[
        ["price", "label"]
    ].sum()
    customer_weekly.rename(
        columns={"label": "n_transactions_customer", "price": "weekly_spending"},
        inplace=True,
    )
    customer_weekly["sum_spending_last_week"] = customer_weekly[
        "weekly_spending"
    ].shift(periods=1)
    customer_weekly["n_transactions_last_week_customer"] = customer_weekly[
        "n_transactions_customer"
    ].shift(
        periods=1
    )  # .astype("int64") # NOTE: you may rm NaN before already

    customer_dynamic = customer_weekly[
        [
            "customer_id",
            "week",
            "sum_spending_last_week",
            "n_transactions_last_week_customer",
        ]
    ]

    assert len(customer_dynamic) == CUSTOMERS * WEEKS

    # 2. Static Features
    customer_static_features = {
        "customer_id": range(CUSTOMERS),
        "avg_n_transactions_weekly_customer": (
            customer_weekly[
                (customer_weekly.customer_id == c) & (customer_weekly.week < 50)
            ].n_transactions_customer.mean()
            for c in range(CUSTOMERS)
        ),
        "avg_spending_weekly": (
            customer_weekly[
                (customer_weekly.customer_id == c) & (customer_weekly.week < 50)
            ].weekly_spending.mean()
            for c in range(CUSTOMERS)
        ),
        # NOTE: Round here to avoid floating point errors.
        # However manipulating decimals could help improve feature quality
        "highest_price_paid": (
            round(
                merged_df[merged_df.customer_id == c].price.max(),
                2,
            )
            for c in range(CUSTOMERS)
        ),
        "lowest_price_paid": (
            round(
                merged_df[merged_df.customer_id == c].price.min(),
                2,
            )
            for c in range(CUSTOMERS)
        ),
    }
    customer_static = pd.DataFrame(customer_static_features)

    customer_sum_spending = (
        merged_df[["customer_id", "price"]]
        .groupby(by="customer_id", as_index=False)["price"]
        .sum()
    )
    customer_sum_spending["pays_above_avg"] = (
        customer_sum_spending.price > customer_sum_spending.price.mean()
    ).astype(int)

    customer_static = customer_static.merge(
        right=customer_sum_spending[["customer_id", "pays_above_avg"]],
        on="customer_id",
        how="inner",
    )

    assert len(customer_static) == CUSTOMERS

    return customer_static, customer_dynamic


def features_product(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # 1. Dynamic Features
    product_weekly = merged_df.groupby(by=["product_id", "week"], as_index=False)[
        ["price", "label"]
    ].sum()
    product_weekly.rename(
        columns={"label": "n_transactions_product", "price": "weekly_returns"},
        inplace=True,
    )

    product_weekly["sum_returns_last_week"] = product_weekly["weekly_returns"].shift(
        periods=1
    )
    product_weekly["n_transactions_last_week_product"] = product_weekly[
        "n_transactions_product"
    ].shift(periods=1)

    product_dynamic = product_weekly[
        [
            "product_id",
            "week",
            "sum_returns_last_week",
            "n_transactions_last_week_product",
        ]
    ]

    assert len(product_dynamic) == PRODUCTS * WEEKS

    # 2. Static Features
    product_static_features = {
        "product_id": range(PRODUCTS),
        "current_price": (  # NOTE: prices don't change in this example
            merged_df[merged_df.product_id == p].price.max() for p in range(PRODUCTS)
        ),
        "avg_n_transactions_weekly_product": (
            product_weekly[
                (product_weekly.product_id == p) & (product_weekly.week < 50)
            ].n_transactions_product.mean()
            for p in range(PRODUCTS)
        ),
        "avg_returns_weekly": (
            product_weekly[
                (product_weekly.product_id == p) & (product_weekly.week < 50)
            ].weekly_returns.mean()
            for p in range(PRODUCTS)
        ),
    }

    product_static = pd.DataFrame(product_static_features)

    product_prices = (
        merged_df[["product_id", "price"]]
        .groupby(by="product_id", as_index=False)["price"]
        .mean()
    )
    product_prices["costs_above_avg"] = (
        product_prices.price > product_prices.price.mean()
    ).astype(int)

    product_static = product_static.merge(
        right=product_prices[["product_id", "costs_above_avg"]],
        on="product_id",
        how="inner",
    )

    product_static.sort_values(by="product_id", inplace=True)

    assert len(product_static) == PRODUCTS

    return product_static, product_dynamic


def features_week(train_df: pd.DataFrame) -> pd.DataFrame:

    # 1. Calculate weekly returns
    weekly_returns = (
        train_df[["week", "price"]]
        .groupby(by="week", as_index=False)["price"]
        .sum()
        .rename(columns={"price": "sum_returns"})
    )
    # NOTE: This creates 4 weeks of NaNs! If you drop them later, you may loose information.
    # If you fill NaNs, the mapping between X and Y becomes less clear.
    weekly_returns["moving_avg_returns_lm"] = weekly_returns.sum_returns.rolling(
        4
    ).mean()

    # removing NaNs also removes quite a fraction of our target, so we better backfill missing values
    weekly_returns["moving_avg_returns_lm"].bfill(inplace=True)

    # Use last weeks data in current week
    weekly_returns.week = weekly_returns.week + 1

    # 2. Calculate weekly transactions
    weekly_transactions = pd.DataFrame(
        {
            "week": list(train_df["week"].value_counts().index),
            "sum_transactions": train_df["week"].value_counts().values,
        }
    )
    weekly_transactions.sort_values(by="week", inplace=True)

    weekly_transactions[
        "moving_avg_transactions_lm"
    ] = weekly_transactions.sum_transactions.rolling(4).mean()

    weekly_transactions["moving_avg_transactions_lm"].bfill(inplace=True)

    # Use last weeks data in current week
    weekly_transactions.week = weekly_transactions.week + 1

    weekly_aggregation = weekly_returns.merge(
        weekly_transactions, how="left", on="week", validate="1:1"
    )

    weekly_aggregation.rename(
        columns={
            "sum_returns": "sum_returns_last_week_general",
            "sum_transactions": "sum_transactions_last_week_general",
        },
        inplace=True,
    )

    return weekly_aggregation


def merge_all_data_sets(
    merged_df: pd.DataFrame,
    customer_static_df: pd.DataFrame,
    customer_dynamic_df: pd.DataFrame,
    product_static_df: pd.DataFrame,
    product_dynamic_df: pd.DataFrame,
    weekly_aggregation_df: pd.DataFrame,
):

    # 1. Static Customer Data to Skeleton
    merged_combo = merged_df.merge(
        right=customer_static_df, how="left", on="customer_id", validate="m:1"
    )

    # 2. Dynamic Customer Data to Skeleton
    merged_combo = merged_combo.merge(
        right=customer_dynamic_df,
        how="left",
        on=["customer_id", "week"],
        validate="m:1",
    )

    # 3. Static Product Data to Skeleton
    merged_combo = merged_combo.merge(
        right=product_static_df, how="left", on=["product_id"], validate="m:1"
    )

    # 4. Dynamic Product Data to Skeleton
    merged_combo = merged_combo.merge(
        right=product_dynamic_df, how="left", on=["product_id", "week"], validate="m:1"
    )

    # 5. Dynamic Aggregations to Skeleton
    merged_combo = merged_combo.merge(
        right=weekly_aggregation_df, how="left", on="week", validate="m:1"
    )

    # 6. Set Label
    merged_combo.label.fillna(0, inplace=True)  # NaN ~ no transaction

    # 7. Additional Features
    # Budget & Prices used to
    within_budget = (merged_combo.highest_price_paid >= merged_combo.current_price) & (
        merged_combo.current_price >= merged_combo.lowest_price_paid
    )

    merged_combo["within_budget"] = within_budget.astype(int)

    return merged_combo


def make_raw_data():

    # 1. Create Skeleton
    skeleton_data = skeleton()

    # 2. Read (incomplete) Train Data
    train_data = pd.read_csv("data/train.csv")

    # 3. Merge Skeleton & (incomplete) Train Data
    merged_data = merge_with_skeleton(skeleton_df=skeleton_data, train_df=train_data)

    # 4. Create Features based on Customer related Data
    customer_static_data, customer_dynamic_data = features_customer(
        merged_df=merged_data
    )

    # 5. Create Features based on Product related data
    product_static_data, product_dynamic_data = features_product(merged_df=merged_data)

    # 6. Create global aggregation features
    weekly_aggregation_data = features_week(train_df=train_data)

    # 7. Bringing all Data together
    all_data = merge_all_data_sets(
        merged_df=merged_data,
        customer_static_df=customer_static_data,
        customer_dynamic_df=customer_dynamic_data,
        product_static_df=product_static_data,
        product_dynamic_df=product_dynamic_data,
        weekly_aggregation_df=weekly_aggregation_data,
    )

    # Remove week 1 due to many NaNs
    all_data = all_data[all_data.week > 1].astype(
        {
            "label": "int64",
            "n_transactions_last_week_product": "int64",
            "moving_avg_transactions_lm": "int64",
            "n_transactions_last_week_customer": "int64",
            "sum_transactions_last_week_general": "int64",
        }
    )

    assert (
        all_data[(c for c in all_data.columns if c != "price")].isna().sum().sum() == 0
    )
    return all_data


def create_training_data_file(parquet_file: str = "training_01.parquet"):
    # Run in optimized mode when creating file
    os.environ["PYTHONOPTIMIZE"] = "1"

    data = make_raw_data()
    training = data.columns[4:]

    data[training][data.week < 50].to_parquet(parquet_file)


def main():
    create_training_data_file("training_02.parquet")


if __name__ == "__main__":
    main()
