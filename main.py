import logging
from pathlib import Path
from typing import Any

import polars as pl
from pyscbwrapper import SCB

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
TABLES = {
    "month_tab": ("en", "AM", "AM0401", "AM0401I", "NAKUSysselYrke2012M"),
}
DEFAULT_TAB_ID = "month_tab"
OUTPUT_FILENAME = "scb_months.parquet"


def get_scb_client(tab_id: str = DEFAULT_TAB_ID) -> SCB:
    """Initialize and return the SCB client for a given table ID."""
    if tab_id not in TABLES:
        msg = f"Table ID '{tab_id}' not found in TABLES configuration."
        raise ValueError(msg)

    logger.info("Initializing SCB client for table ID: %s", tab_id)
    return SCB(*TABLES[tab_id])


def extract_metadata_keys(scb: SCB) -> dict[str, Any]:
    """
    Extract necessary metadata keys and values from the SCB client.

    Returns a dictionary containing keys and selected values for the query.
    """
    logger.info("Extracting metadata keys from SCB...")
    var_ = scb.get_variables()

    # Helper to find key by substring
    def find_key(substring: str) -> str:
        try:
            return next(k for k in var_ if substring.lower() in k.lower())
        except StopIteration as err:
            msg = f"Could not find metadata key containing: '{substring}'"
            raise KeyError(msg) from err

    dg_key = find_key("degree")
    occ_key = find_key("occupation")
    obs_key = find_key("observations")
    month_key = find_key("month")
    sex_key = find_key("sex")

    return {
        "attachment_key": "".join(dg_key.split()),
        "degree": var_[dg_key],
        "occupations_key": occ_key,
        "occupations": var_[occ_key],
        "observations_key": obs_key,
        "observations": var_[obs_key][0],
        "months_key": month_key,
        "months": var_[month_key],
        "sex_key": sex_key,
        "sex": var_[sex_key][:2],  # Assuming first two are Total/Men/Women or similar
    }


def fetch_scb_data(scb: SCB, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Configure query and fetch data from SCB API."""
    logger.info("Configuring query and fetching data...")

    scb.set_query(
        **{
            metadata["attachment_key"]: metadata["degree"],
            metadata["occupations_key"]: metadata["occupations"],
            metadata["months_key"]: metadata["months"],
            metadata["observations_key"]: metadata["observations"],
            metadata["sex_key"]: metadata["sex"],
        },
    )

    data = scb.get_data()
    return data.get("data", [])


def transform_data(
    raw_data: list[dict[str, Any]], scb: SCB, metadata: dict[str, Any],
) -> pl.DataFrame:
    """
    Transform raw SCB data into a clean Polars DataFrame.

    Applies mappings, filters, and formatting.
    """
    logger.info("Transforming data with Polars...")

    # Get code mappings from the query configuration
    query_info = scb.get_query()["query"]

    # Re-extracting codes from query selection
    occ_codes = next(
        q["selection"]["values"] for q in query_info if "Yrke" in q["code"]
    )
    occ_dict = dict(zip(occ_codes, metadata["occupations"], strict=True))

    sex_codes = next(
        q["selection"]["values"] for q in query_info if "Kon" in q["code"]
    )
    sex_dict = dict(zip(sex_codes, metadata["sex"], strict=True))

    df = (
        pl.DataFrame(raw_data)
        .with_columns(
            [
                pl.col("key").list.get(1).alias("code_1"),
                pl.col("key").list.get(2).alias("sex_code"),
                pl.col("key").list.get(3).alias("month_raw"),
                pl.col("values").list.get(0).alias("value"),
            ],
        )
        .drop(["key", "values"])
        .with_columns(
            [
                pl.col("code_1").replace(occ_dict).alias("occupation"),
                pl.col("sex_code").replace(sex_dict).alias("sex"),
            ],
        )
        .filter(~pl.col("code_1").is_in(["0002", "0000"]))
        .with_columns(
            [
                pl.col("code_1").cast(pl.Utf8),
                pl.col("occupation").cast(pl.Utf8),
                pl.col("sex").cast(pl.Utf8),
                pl.col("month_raw")
                .str.replace("M", "-")
                .str.strptime(pl.Date, "%Y-%m")
                .dt.strftime("%Y-%b")
                .alias("month"),
                pl.col("value").cast(pl.Utf8, strict=False),
            ],
        )
        .select(["code_1", "sex", "month", "value", "occupation"])
    )

    return df


def main():
    """Main execution flow."""
    try:
        # Setup paths
        root = Path.cwd().resolve()
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        scb_client = get_scb_client()

        # Metadata extraction
        metadata = extract_metadata_keys(scb_client)

        # Fetch
        raw_data = fetch_scb_data(scb_client, metadata)

        if not raw_data:
            logger.error("No data fetched from SCB.")
            return

        # Transform
        df = transform_data(raw_data, scb_client, metadata)

        # Save
        output_path = data_dir / OUTPUT_FILENAME
        df.write_parquet(output_path)
        logger.info("Successfully saved processed data to %s", output_path)
        logger.info("DataFrame shape: %s", df.shape)
        print(df.head(10))

    except Exception:
        logger.exception("An error occurred during execution")


if __name__ == "__main__":
    main()
