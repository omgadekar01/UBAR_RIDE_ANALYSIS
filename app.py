"""Interactive Uber ride analytics dashboard for a college mini-project."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Keep the page layout wide so charts and tables have enough room.
st.set_page_config(
    page_title="Uber Rides | Predictive Analytics",
    page_icon="🚕",
    layout="wide",
)

# Use the CSV supplied with this project as the starting dataset.
PROJECT_FOLDER = Path(__file__).parent
DEFAULT_CSV = PROJECT_FOLDER / "uber_trips_dataset_50k.csv"
TIMESTAMP_ALIASES = [
    "pickup_datetime",
    "pickup_time",
    "pickup_date",
    "tpep_pickup_datetime",
    "lpep_pickup_datetime",
    "datetime",
    "date_time",
    "date/time",
    "timestamp",
]


def normalize_name(column_name):
    """Make different spellings of a column name easier to compare."""
    return re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower()).strip("_")


def find_timestamp_column(data):
    """Find a pickup date/time column without requiring one exact spelling."""
    normalized_columns = {
        normalize_name(column): column for column in data.columns
    }

    # Prefer familiar pickup timestamp names, including common taxi-data names.
    for alias in TIMESTAMP_ALIASES:
        normalized_alias = normalize_name(alias)
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    # Otherwise, use a column whose name clearly describes a pickup time.
    for normalized_name, original_name in normalized_columns.items():
        has_pickup = "pickup" in normalized_name
        has_time_part = any(
            word in normalized_name for word in ("time", "date", "timestamp")
        )
        if has_pickup and has_time_part:
            return original_name

    return None


def clean_data(raw_data, timestamp_column):
    """Remove exact duplicates and rows without a valid pickup timestamp."""
    original_rows = len(raw_data)

    # Count duplicates before removing them so the quality report can show the change.
    duplicate_rows = int(raw_data.duplicated().sum())
    cleaned_data = raw_data.drop_duplicates().copy()
    rows_after_duplicates = len(cleaned_data)

    # Convert the selected field to datetimes; invalid or blank values become NaT.
    cleaned_data[timestamp_column] = pd.to_datetime(
        cleaned_data[timestamp_column],
        errors="coerce",
        format="mixed",
    )
    invalid_timestamp_rows = int(cleaned_data[timestamp_column].isna().sum())

    # Rows without a usable pickup time cannot be used for time-based analysis.
    cleaned_data = cleaned_data.dropna(subset=[timestamp_column]).copy()
    return cleaned_data, {
        "original_rows": original_rows,
        "duplicate_rows": duplicate_rows,
        "rows_after_duplicates": rows_after_duplicates,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "cleaned_rows": len(cleaned_data),
        "original_columns": len(raw_data.columns),
    }


def find_coordinate_columns(data):
    """Return pickup latitude and longitude columns when both are available."""
    normalized_columns = {
        normalize_name(column): column for column in data.columns
    }
    latitude_names = (
        "pickup_lat",
        "pickup_latitude",
        "lat",
        "latitude",
    )
    longitude_names = (
        "pickup_lng",
        "pickup_lon",
        "pickup_longitude",
        "pickup_long",
        "lng",
        "lon",
        "longitude",
        "long",
    )

    latitude_column = next(
        (normalized_columns[name] for name in latitude_names if name in normalized_columns),
        None,
    )
    longitude_column = next(
        (
            normalized_columns[name]
            for name in longitude_names
            if name in normalized_columns
        ),
        None,
    )
    return latitude_column, longitude_column


def get_analysis_columns(data):
    """Select numeric measurements and leave identifier-like fields out."""
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    analysis_columns = []

    for column in numeric_columns:
        normalized_name = normalize_name(column)
        name_parts = normalized_name.split("_")
        looks_like_id = (
            normalized_name == "id"
            or normalized_name.endswith("_id")
            or name_parts[-1:] == ["id"]
        )
        has_values_to_compare = data[column].nunique(dropna=True) > 1

        # IDs identify records; their numeric ordering is not a useful measurement.
        if not looks_like_id and has_values_to_compare:
            analysis_columns.append(column)

    return analysis_columns


def show_correlation_analysis(data):
    """Display a correlation heatmap when there are enough useful measurements."""
    analysis_columns = get_analysis_columns(data)

    if len(analysis_columns) < 2:
        st.info(
            "Correlation analysis needs at least two varying numeric measurement "
            "columns. Identifier columns are intentionally excluded."
        )
        return

    correlation_matrix = data[analysis_columns].corr()
    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation between numeric measurements",
        labels={"x": "Column", "y": "Column", "color": "Correlation"},
    )
    figure.update_layout(height=520)
    st.plotly_chart(figure)
    st.caption(
        "Correlation describes how two measurements vary together. "
        "It does not prove that one causes the other."
    )


def make_hourly_demand(data, timestamp_column):
    """Turn individual rides into a continuous series of hourly ride counts."""
    hourly_counts = (
        data.set_index(timestamp_column)
        .sort_index()
        .resample("h")
        .size()
        .asfreq("h", fill_value=0)
    )
    return hourly_counts


def create_time_features(hourly_counts):
    """Create simple calendar and sequence features for the model."""
    timestamps = hourly_counts.index
    features = pd.DataFrame(index=timestamps)
    features["hour"] = timestamps.hour
    features["day_of_week"] = timestamps.day_of_week
    features["month"] = timestamps.month
    features["is_weekend"] = (timestamps.day_of_week >= 5).astype(int)
    features["time_index"] = np.arange(len(hourly_counts))
    return features


def show_predictive_model(data, timestamp_column):
    """Train and evaluate a Random Forest using an earlier-to-later split."""
    hourly_counts = make_hourly_demand(data, timestamp_column)

    # A short series cannot provide a useful chronological test set.
    if len(hourly_counts) < 10:
        st.info(
            "At least 10 hourly time buckets are needed to demonstrate a "
            "chronological train/test evaluation. Upload data spanning more hours."
        )
        return

    features = create_time_features(hourly_counts)
    target = hourly_counts.astype(float)
    split_index = int(len(hourly_counts) * 0.8)
    split_index = max(1, min(split_index, len(hourly_counts) - 1))

    # The first 80% of hours train the model; the latest 20% stay unseen for testing.
    train_features = features.iloc[:split_index]
    test_features = features.iloc[split_index:]
    train_target = target.iloc[:split_index]
    test_target = target.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)

    # A simple reference predicts the average ride count from the training period.
    baseline_predictions = np.full(len(test_target), train_target.mean())
    model_mae = mean_absolute_error(test_target, predictions)
    model_rmse = np.sqrt(mean_squared_error(test_target, predictions))
    baseline_mae = mean_absolute_error(test_target, baseline_predictions)
    baseline_rmse = np.sqrt(mean_squared_error(test_target, baseline_predictions))

    # R-squared is undefined for a one-row or constant-valued test set.
    r_squared = None
    baseline_r_squared = None
    if len(test_target) >= 2 and test_target.nunique() > 1:
        r_squared = r2_score(test_target, predictions)
        baseline_r_squared = r2_score(test_target, baseline_predictions)

    st.write(
        f"**Hourly buckets:** {len(hourly_counts):,} · "
        f"**Training buckets:** {len(train_target):,} · "
        f"**Test buckets:** {len(test_target):,}"
    )

    comparison = pd.DataFrame(
        {
            "Model": ["Random Forest", "Training-mean baseline"],
            "MAE (rides)": [model_mae, baseline_mae],
            "RMSE (rides)": [model_rmse, baseline_rmse],
            "R-squared": [r_squared, baseline_r_squared],
        }
    )
    st.subheader("Test-set evaluation")
    st.dataframe(
        comparison.style.format(
            {
                "MAE (rides)": "{:.2f}",
                "RMSE (rides)": "{:.2f}",
                "R-squared": lambda value: "Not available" if pd.isna(value) else f"{value:.3f}",
            }
        ),
        hide_index=True,
    )
    st.caption(
        "MAE is the average absolute prediction error in rides; lower is better. "
        "RMSE also measures error in rides and penalizes larger misses more. "
        "R-squared compares predictions with a mean-only reference; values closer "
        "to 1 are generally better, while a negative value can mean worse-than-mean "
        "predictions. Results describe this held-out period only."
    )

    chart_data = pd.DataFrame(
        {
            "Hour": test_target.index,
            "Actual rides": test_target.to_numpy(),
            "Predicted rides": predictions,
        }
    ).melt(id_vars="Hour", var_name="Series", value_name="Ride count")
    prediction_figure = px.line(
        chart_data,
        x="Hour",
        y="Ride count",
        color="Series",
        markers=True,
        title="Actual and predicted hourly rides (chronological test period)",
        labels={"Hour": "Pickup hour", "Ride count": "Number of rides"},
    )
    st.plotly_chart(prediction_figure)

    importance = pd.DataFrame(
        {
            "Feature": features.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=True)
    importance_figure = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest feature importance",
        labels={"Importance": "Relative importance", "Feature": "Input feature"},
    )
    st.plotly_chart(importance_figure)
    st.caption(
        "Feature importance shows which inputs the fitted forest used most. "
        "It does not show that a feature causes demand to change."
    )


def show_overview(data, timestamp_column, latitude_column, longitude_column):
    """Show filtered ride totals and the main pickup-time visualizations."""
    minimum_date = data[timestamp_column].min().date()
    maximum_date = data[timestamp_column].max().date()
    date_range = st.date_input(
        "Filter pickup dates",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
        key="overview_date_filter",
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    start_time = pd.Timestamp(start_date)
    end_time = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    filtered_data = data[
        (data[timestamp_column] >= start_time)
        & (data[timestamp_column] < end_time)
    ].copy()

    total_rides = len(filtered_data)
    unique_days = filtered_data[timestamp_column].dt.date.nunique()
    average_daily_rides = total_rides / unique_days if unique_days else 0

    # These cards summarize only rides inside the selected date range.
    kpi_one, kpi_two, kpi_three = st.columns(3)
    kpi_one.metric("Pickup records", f"{total_rides:,}")
    kpi_two.metric("Days with pickups", f"{unique_days:,}")
    kpi_three.metric("Average rides / active day", f"{average_daily_rides:,.1f}")

    if filtered_data.empty:
        st.warning("There are no pickup records in the selected date range.")
        return filtered_data

    daily_counts = (
        filtered_data.set_index(timestamp_column)
        .resample("D")
        .size()
        .rename("Pickups")
        .reset_index()
    )
    daily_figure = px.line(
        daily_counts,
        x=timestamp_column,
        y="Pickups",
        markers=True,
        title="Daily pickups",
        labels={timestamp_column: "Pickup date", "Pickups": "Number of rides"},
    )
    st.plotly_chart(daily_figure)

    hour_counts = (
        filtered_data[timestamp_column].dt.hour.value_counts().reindex(range(24), fill_value=0)
    )
    hour_data = pd.DataFrame(
        {"Hour of day": range(24), "Pickups": hour_counts.to_numpy()}
    )
    hour_figure = px.bar(
        hour_data,
        x="Hour of day",
        y="Pickups",
        title="Pickups by hour",
        labels={"Hour of day": "Hour (0–23)", "Pickups": "Number of rides"},
    )
    st.plotly_chart(hour_figure)

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts = (
        filtered_data[timestamp_column]
        .dt.day_of_week.value_counts()
        .reindex(range(7), fill_value=0)
    )
    weekday_data = pd.DataFrame(
        {
            "Weekday": weekday_names,
            "Pickups": weekday_counts.to_numpy(),
        }
    )
    weekday_figure = px.bar(
        weekday_data,
        x="Weekday",
        y="Pickups",
        title="Pickups by weekday",
        labels={"Pickups": "Number of rides"},
    )
    st.plotly_chart(weekday_figure)

    if latitude_column and longitude_column:
        map_data = filtered_data[[latitude_column, longitude_column]].copy()
        map_data[latitude_column] = pd.to_numeric(
            map_data[latitude_column], errors="coerce"
        )
        map_data[longitude_column] = pd.to_numeric(
            map_data[longitude_column], errors="coerce"
        )
        map_data = map_data.dropna()
        map_data = map_data[
            map_data[latitude_column].between(-90, 90)
            & map_data[longitude_column].between(-180, 180)
        ]

        if not map_data.empty:
            # Limit only the displayed points so large uploads remain responsive.
            if len(map_data) > 5000:
                map_data = map_data.sample(n=5000, random_state=42)
                st.caption("The map displays a reproducible sample of up to 5,000 valid points.")

            map_figure = px.scatter_map(
                map_data,
                lat=latitude_column,
                lon=longitude_column,
                zoom=2,
                height=500,
                title="Valid pickup locations",
                map_style="open-street-map",
            )
            st.plotly_chart(map_figure)
        else:
            st.info("Pickup coordinate columns exist, but no valid coordinate pairs were found.")
    else:
        st.info(
            "No recognized pickup latitude/longitude pair was found, so the map is not shown."
        )

    return filtered_data


def show_data_quality(raw_data, cleaned_data, cleaning_summary, timestamp_column):
    """Show source information, missing values, and the cleaning summary."""
    st.subheader("Uploaded data")
    st.write(
        f"**Rows:** {len(raw_data):,} · **Columns:** {len(raw_data.columns):,} "
        f"· **Pickup-time column:** `{timestamp_column}`"
    )
    st.dataframe(raw_data.head(), hide_index=True)

    column_information = pd.DataFrame(
        {
            "Column": raw_data.columns,
            "Data type": [str(raw_data[column].dtype) for column in raw_data.columns],
            "Missing values (before)": [
                int(raw_data[column].isna().sum()) for column in raw_data.columns
            ],
            "Missing values (after)": [
                int(cleaned_data[column].isna().sum()) for column in raw_data.columns
            ],
        }
    )
    st.subheader("Columns, data types, and missing values")
    st.dataframe(column_information, hide_index=True)

    st.subheader("Before-and-after cleaning summary")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Original rows", f"{cleaning_summary['original_rows']:,}")
    summary_columns[1].metric(
        "Exact duplicates removed", f"{cleaning_summary['duplicate_rows']:,}"
    )
    summary_columns[2].metric(
        "Invalid pickup times removed",
        f"{cleaning_summary['invalid_timestamp_rows']:,}",
    )
    summary_columns[3].metric("Cleaned rows", f"{cleaning_summary['cleaned_rows']:,}")
    st.caption(
        "Rows with an invalid pickup time are removed because time-based charts and "
        "the demand model require a valid pickup timestamp. Other missing fields "
        "are reported but not guessed or filled."
    )


def main():
    """Load a CSV, clean it, and display the four dashboard tabs."""
    st.title("🚕 Uber Rides: Predictive Analytics Dashboard")
    st.write(
        "Explore ride patterns, review data quality, and evaluate an hourly demand "
        "model using your own CSV data."
    )

    st.sidebar.header("Ride data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload a ride dataset with a pickup date/time column.",
    )

    try:
        if uploaded_file is not None:
            raw_data = pd.read_csv(uploaded_file)
            dataset_label = uploaded_file.name
        elif DEFAULT_CSV.exists():
            raw_data = pd.read_csv(DEFAULT_CSV)
            dataset_label = DEFAULT_CSV.name
            st.sidebar.caption(f"Using the supplied sample: {dataset_label}")
        else:
            st.info("Upload a CSV file in the sidebar to get started.")
            return
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError,
    ) as error:
        st.error(f"Could not read the CSV file: {error}")
        return

    if raw_data.empty or len(raw_data.columns) == 0:
        st.error("The CSV has no data rows or columns. Upload a non-empty CSV file.")
        return

    timestamp_column = find_timestamp_column(raw_data)
    if timestamp_column is None:
        st.error(
            "No pickup date/time column was found. Expected a column such as "
            "`pickup_time`, `pickup_datetime`, `Date/Time`, or `timestamp`. "
            "Check the CSV header and upload again."
        )
        st.write("Columns found:", list(raw_data.columns))
        return

    cleaned_data, cleaning_summary = clean_data(raw_data, timestamp_column)
    st.sidebar.caption(f"Dataset: {dataset_label}")
    st.sidebar.caption(
        f"{len(raw_data):,} input rows · {len(cleaned_data):,} cleaned rows"
    )

    if cleaned_data.empty:
        st.error(
            "No usable rides remain after removing duplicates and invalid pickup "
            "timestamps. Check the timestamp values in the CSV."
        )
        show_data_quality(raw_data, cleaned_data, cleaning_summary, timestamp_column)
        return

    st.sidebar.download_button(
        "Download cleaned CSV",
        data=cleaned_data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_uber_rides.csv",
        mime="text/csv",
    )

    latitude_column, longitude_column = find_coordinate_columns(cleaned_data)
    overview_tab, correlation_tab, prediction_tab, quality_tab = st.tabs(
        ["Overview", "Correlation Analysis", "Predictive Model", "Data Quality"]
    )

    with overview_tab:
        filtered_data = show_overview(
            cleaned_data,
            timestamp_column,
            latitude_column,
            longitude_column,
        )

    with correlation_tab:
        st.header("Correlation Analysis")
        show_correlation_analysis(cleaned_data)

    with prediction_tab:
        st.header("Hourly Demand Prediction")
        st.write(
            "The model predicts hourly ride counts from calendar and time-index "
            "features. The latest part of the selected date range is held out for testing."
        )
        show_predictive_model(filtered_data, timestamp_column)

    with quality_tab:
        st.header("Data Quality")
        show_data_quality(raw_data, cleaned_data, cleaning_summary, timestamp_column)
        st.subheader("Cleaned data preview")
        st.dataframe(cleaned_data.head(), hide_index=True)

if __name__ == "__main__":
    main()
