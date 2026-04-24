from __future__ import annotations

from pathlib import Path

import polars as pl

DAIOE_SOURCE = (
    "https://raw.githubusercontent.com/joseph-data/AI_Econ_daioe_years/"
    "development/data/daioe_scb_years_all_levels.parquet"
)
SCB_SOURCE = (
    "https://raw.githubusercontent.com/joseph-data/AI_Econ_daioe_months_v2/"
    "daioe_pull/data/scb_months.parquet"
)
OUTPUT_NAME = "scb_months_lvl1.parquet"


def pct_change(current: pl.Expr, shifted: pl.Expr) -> pl.Expr:
    """
    Calculate the percentage change between a current and a shifted expression.

    Returns None if the shifted value is null or zero to avoid division errors.
    """
    return (
        pl.when(shifted.is_not_null() & shifted.ne(0))
        .then((current / shifted - 1) * 100)
        .otherwise(None)
    )


def load_sources(daioe_source: str, scb_source: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Scan remote Parquet sources and return them as Polars LazyFrames."""
    return pl.scan_parquet(daioe_source), pl.scan_parquet(scb_source)


def build_scb_monthly_changes(scb_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean SCB monthly data and calculate 1m, 3m, and 6m absolute and percentage changes.

    This function:
    1. Filters out invalid codes.
    2. Extracts years and parses month dates.
    3. Aggregates values by category and month.
    4. Computes rolling changes using window functions.
    5. Sorts the results by occupation and time.
    """
    month_date_expr = pl.col("month").str.strptime(pl.Date, "%Y-%b", strict=False)

    scb_lf_clean = scb_lf.filter(pl.col("code_1").str.starts_with("0").not_()).with_columns(
        pl.col("month").str.extract(r"^(\d{4})", 1).cast(pl.Int64).alias("year"),
    )

    change_keys = [col for col in scb_lf_clean.collect_schema().names() if col != "value"]
    group_keys = [col for col in change_keys if col != "month"]

    emp = pl.col("emp_count")

    return (
        scb_lf_clean
        .with_columns(pl.col("value").cast(pl.Float64, strict=False))
        .group_by(change_keys)
        .agg(pl.col("value").sum().alias("emp_count"))
        .with_columns(month_date_expr.alias("_month_date"))
        .with_columns(
            emp.shift(i).over(group_keys, order_by="_month_date").alias(f"_emp_{i}m")
            for i in [1, 3, 6]
        )
        .with_columns(
            (emp - pl.col(f"_emp_{i}m")).alias(f"chg_{i}m") for i in [1, 3, 6]
        )
        .with_columns(
            pct_change(emp, pl.col(f"_emp_{i}m")).alias(f"pct_chg_{i}m") for i in [1, 3, 6]
        )
        .drop("_emp_1m", "_emp_3m", "_emp_6m")
        .sort(by=["code_1", "sex", "occupation", "_month_date"])
        .drop("_month_date")
    )


def build_weighted_daioe(daioe_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Filter DAIOE data for SSYK level 1 and calculate mean values for metrics."""
    return (
        daioe_lf
        .filter(pl.col("level") == "SSYK1")
        .select(
            pl.col(["level", "ssyk_code", "year", "weight_sum"]),
            pl.col("^daioe_.*$"),
            pl.col("^pctl_daioe_.*$"),
        )
        .group_by(["level", "ssyk_code", "year"])
        .agg(
            pl.col("weight_sum").mean().cast(pl.Int64),
            pl.col("^daioe_.*$").mean(),
            pl.col("^pctl_daioe_.*$").mean(),
        )
    )


def extend_daioe_years(
    base: pl.LazyFrame,
    scb_lazy_lf_changes: pl.LazyFrame,
) -> tuple[pl.LazyFrame, int, int, list[int]]:
    """
    Extend the DAIOE dataset to cover the years available in the SCB dataset.

    This uses a forward-fill strategy by repeating the data from the most recent
    available year in the DAIOE dataset for all missing future years.
    """
    daioe_max_res, scb_max_res = pl.collect_all(
        [
            base.select(pl.max("year")),
            scb_lazy_lf_changes.select(pl.max("year")),
        ]
    )
    daioe_max_year = daioe_max_res.item()
    scb_max_year = scb_max_res.item()
    missing_years = list(range(daioe_max_year + 1, scb_max_year + 1))

    extended = (
        base
        if not missing_years
        else pl.concat(
            [
                base,
                base
                .filter(pl.col("year") == daioe_max_year)
                .drop("year")
                .join(pl.LazyFrame({"year": missing_years}), how="cross")
                .select(base.collect_schema().names()),
            ],
            how="vertical",
        )
    )

    return extended, daioe_max_year, scb_max_year, missing_years


def build_monthly_panel(
    scb_lazy_lf_changes: pl.LazyFrame,
    daioe_lazy_lf_extended: pl.LazyFrame,
) -> pl.LazyFrame:
    """Join the cleaned SCB monthly data with the extended DAIOE data."""
    return (
        scb_lazy_lf_changes
        .join(
            daioe_lazy_lf_extended,
            left_on=["code_1", "year"],
            right_on=["ssyk_code", "year"],
            how="left",
        )
        .drop("level")
    )


def main() -> None:
    """Execute the main data processing pipeline."""
    root = Path.cwd().resolve()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / OUTPUT_NAME

    daioe_lf, scb_lf = load_sources(DAIOE_SOURCE, SCB_SOURCE)
    scb_lazy_lf_changes = build_scb_monthly_changes(scb_lf)
    weighted_daioe = build_weighted_daioe(daioe_lf)
    daioe_lazy_lf_extended, daioe_max_year, scb_max_year, missing_years = extend_daioe_years(
        weighted_daioe,
        scb_lazy_lf_changes,
    )
    scb_months_lf = build_monthly_panel(scb_lazy_lf_changes, daioe_lazy_lf_extended)

    scb_months_lf.sink_parquet(output_path)

    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    print(f"Wrote monthly panel to {output_path}")
    print(f"DAIOE max year: {daioe_max_year}")
    print(f"SCB max year: {scb_max_year}")
    print(f"Extended years: {missing_years or 'none'}")
    print(f"Output rows: {row_count}")


if __name__ == "__main__":
    main()
