#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning", group="eda")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Dropping outliear with Minprice: {args.min_price}, Maximum price: {args.max_price}")
    # Drop outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Note how we did not impute missing values.
    # We will do that in the inference pipeline, so we will be able to handle missing values also in production.

    # drop rows in the dataset that are not in the proper geolocation. 
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # save the output artifact locally with index=False to not have an extra index column
    df.to_csv(args.output_artifact, index=False)

    # upload to wandb
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="the name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="the name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="the type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="a description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="the minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="the maximum price to consider",
        required=True
    )

    args = parser.parse_args()

    go(args)
