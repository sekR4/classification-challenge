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
    customer_weekly["n_transactions_last_week_customer"] = (
        customer_weekly["n_transactions_customer"].shift(periods=1).astype("int64")
    )

    customer_dynamic = customer_weekly[
        [
            "customer_id",
            "week",
            "sum_spending_last_week",
            "n_transactions_last_week_customer",
        ]
    ]

    assert len(customer_dynamic) == CUSTOMERS * WEEKS

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
        # NOTE: Round here to avoid floating point errors. However manipulating decimals could help improve feature quality
        "highest_price_paid": (
            round(
                merged_df[merged_df.customer_id == c]
                .groupby(by="customer_id", as_index=False)["price"]
                .max(),
                2,
            )
            for c in range(CUSTOMERS)
        ),
        "lowest_price_paid": (
            round(
                merged_df[merged_df.customer_id == c]
                .groupby(by="customer_id", as_index=False)["price"]
                .max(),
                2,
            )
            for c in range(CUSTOMERS)
        ),
    }
    customer_static = pd.DataFrame(customer_static_features)

    assert len(customer_static) == CUSTOMERS

    return customer_static, customer_dynamic


def main():

    # 1. Create Skeleton
    skeleton_data = skeleton()

    # 2. Read (incomplete) Train Data
    train_data = pd.read_csv("train.csv")

    # 3. Merge Skeleton & (incomplete) Train Data
    merged_data = merge_with_skeleton(skeleton_df=skeleton_data, train_df=train_data)

    # 4. Create Features based on Customer related Data
    customer_static_data, customer_dynamic_data = features_customer(
        merged_df=merged_data
    )


if __name__ == "__main__":
    main()
