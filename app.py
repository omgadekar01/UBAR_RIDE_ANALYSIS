"""Interactive Uber ride analytics dashboard for a college mini-project.

The app reads ride records from an uploaded CSV or the bundled sample file,
cleans the pickup timestamps, and displays charts, correlations, model results,
and data-quality information in a Streamlit dashboard.
"""

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


# ---------------------------------------------------------------------------
# Reusable helpers: chart styling, column detection, and data preparation
# ---------------------------------------------------------------------------

def simplify_figure(figure, height=360, show_axes=True):
    """Apply the dashboard's shared visual style to a Plotly chart.

    INPUT:
        figure: A Plotly figure to format.
        height: Chart height in pixels.
        show_axes: Whether to apply the shared axis and grid styling.
    OUTPUT:
        The same Plotly figure, with the shared layout applied.
    """
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 48, "r": 20, "t": 52, "b": 42},
        title={"x": 0, "font": {"size": 17}},
        font={"size": 13},
    )
    if show_axes:
        figure.update_xaxes(showgrid=False, zeroline=False)
        figure.update_yaxes(
            showgrid=True,
            gridcolor="#E8EDF3",
            zeroline=False,
        )
    return figure


def normalize_name(column_name):
    """Convert a column name to lowercase letters separated by underscores.

    INPUT: A source column name.
    OUTPUT: A normalized string, such as "Pickup Time" becoming "pickup_time".
    """
    return re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower()).strip("_")


def find_timestamp_column(data):
    """Find the column containing each ride's pickup date and time.

    INPUT: A Pandas DataFrame loaded from the ride CSV.
    OUTPUT: The matching original column name, or None if no match is found.
    """
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
    """Remove duplicate rides and rows without a usable pickup timestamp.

    INPUT:
        raw_data: The original ride records as a Pandas DataFrame.
        timestamp_column: The column that contains pickup date/time values.
    OUTPUT:
        A pair containing the cleaned DataFrame and a dictionary of row counts
        used by the Data Quality tab.
    """
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


def get_analysis_columns(data):
    """Choose useful numeric measurement columns for correlation analysis.

    INPUT: A cleaned ride DataFrame.
    OUTPUT: Column names for varying numeric measurements; ID columns and
        columns with only one distinct value are excluded.
    """
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


# ---------------------------------------------------------------------------
# Analysis helpers and dashboard sections
# ---------------------------------------------------------------------------

def show_correlation_analysis(data):
    """Display the numeric correlation heatmap in Streamlit.

    INPUT: A cleaned ride DataFrame.
    OUTPUT: No Python value; displays a heatmap or an explanatory message.
    """
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
        title="Numeric feature correlations",
        labels={"x": "Measurement", "y": "Measurement", "color": "Correlation"},
    )
    simplify_figure(figure, height=420)
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Correlation describes how two measurements vary together. "
        "It does not prove that one causes the other."
    )


def make_hourly_demand(data, timestamp_column):
    """Aggregate ride records into chronological hourly pickup counts.

    INPUT:
        data: Ride records to aggregate.
        timestamp_column: The column containing parsed pickup timestamps.
    OUTPUT: A Pandas Series indexed by hour, with zeroes for hours without rides.
    """
    hourly_counts = (
        data.set_index(timestamp_column)
        .sort_index()
        .resample("h")
        .size()
        .asfreq("h", fill_value=0)
    )
    return hourly_counts


def create_time_features(hourly_counts):
    """Build calendar and sequence inputs for hourly demand prediction.

    INPUT: Hourly pickup counts indexed by timestamps.
    OUTPUT: A DataFrame with hour, weekday, month, weekend, and time-index
        features, one row per hourly count.
    """
    timestamps = hourly_counts.index
    features = pd.DataFrame(index=timestamps)
    features["hour"] = timestamps.hour
    features["day_of_week"] = timestamps.day_of_week
    features["month"] = timestamps.month
    features["is_weekend"] = (timestamps.day_of_week >= 5).astype(int)
    features["time_index"] = np.arange(len(hourly_counts))
    return features


def show_predictive_model(data, timestamp_column):
    """Train and evaluate hourly demand predictions in the Streamlit page.

    INPUT:
        data: Ride records in the date range selected on the Overview tab.
        timestamp_column: The column containing parsed pickup timestamps.
    OUTPUT: No Python value; displays evaluation metrics and model charts.
    """
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
        color_discrete_sequence=["#2563EB", "#F97316"],
        title="Actual vs predicted rides",
        labels={"Hour": "Pickup hour", "Ride count": "Number of rides"},
    )
    simplify_figure(prediction_figure)
    st.plotly_chart(prediction_figure, use_container_width=True)

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
        color_discrete_sequence=["#2563EB"],
        title="Feature importance",
        labels={"Importance": "Relative importance", "Feature": "Input feature"},
    )
    simplify_figure(importance_figure)
    st.plotly_chart(importance_figure, use_container_width=True)
    st.caption(
        "Feature importance shows which inputs the fitted forest used most. "
        "It does not show that a feature causes demand to change."
    )


def show_overview(data, timestamp_column):
    """Display date filters, summary cards, and pickup charts.

    INPUT:
        data: The cleaned ride DataFrame.
        timestamp_column: The column containing parsed pickup timestamps.
    OUTPUT: The DataFrame filtered to the selected date range. The same
        filtered records are passed to the predictive-model tab.
    """
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

    # Count all rides per calendar day for the daily trend chart.
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
        color_discrete_sequence=["#2563EB"],
    )
    simplify_figure(daily_figure)
    st.plotly_chart(daily_figure, use_container_width=True)

    # Count each hour once; reindex adds zero-count hours missing from the data.
    hour_counts = filtered_data[timestamp_column].dt.hour.value_counts().reindex(
        range(24), fill_value=0
    )
    # The hourly chart shows 6 AM through 5 PM; other charts keep all rides.
    daytime_hours = list(range(6, 18))
    daytime_data = pd.DataFrame(
        {
            "Hour": [
                f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'}"
                for hour in daytime_hours
            ],
            "Pickups": [hour_counts.loc[hour] for hour in daytime_hours],
        }
    )
    daytime_figure = px.bar(
        daytime_data,
        x="Hour",
        y="Pickups",
        title="Daytime pickups (6 AM–5 PM)",
        labels={"Pickups": "Number of rides"},
        color_discrete_sequence=["#F59E0B"],
    )
    simplify_figure(daytime_figure)
    st.plotly_chart(daytime_figure, use_container_width=True)
    st.caption(
        "This chart shows 6:00 AM–5:59 PM only. Nighttime pickups remain included "
        "in the other dashboard analysis."
    )

    # Reindex ensures weekday bars stay Monday-to-Sunday, even when a day is absent.
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
        color_discrete_sequence=["#2563EB"],
    )
    simplify_figure(weekday_figure)
    st.plotly_chart(weekday_figure, use_container_width=True)

    return filtered_data


def show_data_quality(raw_data, cleaned_data, cleaning_summary, timestamp_column):
    """Display source records, column details, and cleaning results.

    INPUT:
        raw_data: The original DataFrame before cleaning.
        cleaned_data: The cleaned DataFrame after duplicate and timestamp checks.
        cleaning_summary: Row counts returned by clean_data().
        timestamp_column: The detected pickup timestamp column name.
    OUTPUT: No Python value; displays data previews and quality information.
    """
    st.subheader("Uploaded data")
    st.write(
        f"**Rows:** {len(raw_data):,} · **Columns:** {len(raw_data.columns):,} "
        f"· **Pickup-time column:** `{timestamp_column}`"
    )
    st.dataframe(
        raw_data,
        hide_index=True,
        height=400,
    )

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
    """Run the Streamlit dashboard from file upload through analysis.

    INPUT: None directly; reads a CSV uploaded in the sidebar or the bundled
        default CSV file.
    OUTPUT: No Python value; renders the Overview, Correlation Analysis,
        Predictive Model, and Data Quality tabs in the browser.
    """
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
    st.sidebar.markdown(
        "**Input features**\n"
        "- CSV file with a header row\n"
        "- Required: pickup date/time column "
        "(`pickup_time`, `pickup_datetime`, or similar)\n"
        "- Optional: numeric measurements such as fare or distance\n\n"
        "**Output features**\n"
        "- Cleaned CSV download\n"
        "- Pickup charts and date filtering\n"
        "- Correlation analysis and hourly demand prediction\n"
        "- Data quality and cleaning summary"
    )

    # INPUT: Prefer the user's uploaded CSV; otherwise use the bundled sample.
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

    # Validate that the file has a recognizable pickup time before analysis.
    timestamp_column = find_timestamp_column(raw_data)
    if timestamp_column is None:
        st.error(
            "No pickup date/time column was found. Expected a column such as "
            "`pickup_time`, `pickup_datetime`, `Date/Time`, or `timestamp`. "
            "Check the CSV header and upload again."
        )
        st.write("Columns found:", list(raw_data.columns))
        return

    # Prepare the shared cleaned dataset used throughout the dashboard.
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

    # OUTPUT: Four tabs containing the dashboard's charts, model, and reports.
    overview_tab, correlation_tab, prediction_tab, quality_tab = st.tabs(
        ["Overview", "Correlation Analysis", "Predictive Model", "Data Quality"]
    )

    with overview_tab:
        filtered_data = show_overview(cleaned_data, timestamp_column)

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
