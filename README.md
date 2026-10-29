# Multi-Feature Predictive Analytics Dashboard for Uber Rides

A beginner-friendly Streamlit project that cleans ride records, explores pickup
patterns, calculates numerical correlations, and evaluates an hourly ride-demand
model.

## Project files

```text
AIDS_MINI_PROJECT/
|-- app.py
|-- requirements.txt
|-- README.md
|-- uber_trips_dataset_50k.csv
```

`uber_trips_dataset_50k.csv` is the dataset already supplied with this project.
It is kept under its original filename and loaded automatically when the app
starts. The dashboard also accepts another CSV from the sidebar. No sample data
or model scores are fabricated.

## What each file does

- **`app.py`** — loads and cleans the data, detects timestamp and pickup
  coordinate columns, builds four interactive tabs, and trains/evaluates the
  Random Forest model.
- **`requirements.txt`** — lists the Python packages used by the dashboard.
- **`README.md`** — installation, usage, test cases, viva questions, and report
  and presentation guidance.
- **`uber_trips_dataset_50k.csv`** — the provided 50,000-row ride dataset used
  as the default example.

The supplied CSV columns are `trip_id`, `driver_id`, `rider_id`, `city`,
`pickup_lat`, `pickup_lng`, `drop_lat`, `drop_lng`, `distance_km`,
`fare_amount`, `status`, `payment_method`, `pickup_time`, and `drop_time`.
The code uses the actual `pickup_time` column. Numeric identifiers are left out
of correlation analysis. Fare, distance, and passenger fields are not required.

## Install and run on Windows in VS Code

1. Install **Python 3.10 or newer** and **Visual Studio Code**. In the VS Code
   Extensions view, install the Microsoft **Python** extension.
2. In VS Code, choose **File → Open Folder...** and open this project folder.
   Open a terminal with **Terminal → New Terminal**.
3. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

4. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, either use the Command Prompt terminal:

   ```bat
   .venv\Scripts\activate.bat
   ```

   Or, if you choose to allow local scripts for your Windows user, run this
   once in PowerShell and then activate the environment:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

5. Install the project packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

   Streamlit prints a local URL (normally `http://localhost:8501`). Open it in
   your browser. Stop the server by pressing **Ctrl+C** in the terminal.

In VS Code, use **Ctrl+Shift+P → Python: Select Interpreter** and select the
interpreter inside this project's `.venv` if VS Code does not select it
automatically.

## Upload another CSV

Use the **Upload a CSV file** control in the left sidebar. The file must have a
header row and a pickup timestamp column. Common names such as `pickup_time`,
`pickup_datetime`, `Date/Time`, `datetime`, and `timestamp` are recognized.
Pickup latitude/longitude columns are optional; common forms such as
`pickup_lat`/`pickup_lng`, `pickup_latitude`/`pickup_longitude`, and `Lat`/`Lon`
are supported. The dashboard reports an error and lists the columns it found
when it cannot identify a pickup timestamp.

After loading, use the **Download cleaned CSV** button in the sidebar to save
the cleaned records. Exact duplicate rows and rows with missing or invalid
pickup timestamps are removed. Other missing fields are reported rather than
filled with guessed values. The Overview date filter is also used to select the
date range evaluated in the Predictive Model tab.

## Dashboard tabs

1. **Overview** — pickup totals, active days, average rides per active day,
   daily pickups, hour-of-day pickups, weekday pickups, and a map when valid
   pickup coordinates are available.
2. **Correlation Analysis** — interactive heatmap for varying numerical
   measurements. Fields named like IDs are excluded because their numeric
   ordering does not represent a measurement. Correlation does not prove
   causation.
3. **Predictive Model** — hourly ride counts, hour/day/month/weekend/time-index
   features, a chronological 80/20 split, actual-versus-predicted chart,
   feature importance, and MAE/RMSE/R-squared comparison with a training-mean
   baseline. R-squared is shown as unavailable when it cannot be computed
   meaningfully. The dashboard does not promise model accuracy.
4. **Data Quality** — original preview, column types, missing values before
   and after cleaning, cleaning counts, and a cleaned-data preview.

## Manual test cases and expected behavior

| Test | How to check | Expected behavior |
|---|---|---|
| Start with the supplied dataset | Run the app without uploading a file | The supplied `uber_trips_dataset_50k.csv` loads automatically; four tabs appear. |
| Check the input summary | Open **Data Quality** | The source preview, row/column counts, types, missing-value counts, duplicate count, invalid-time count, and cleaned-row count are shown. |
| Download cleaned records | Click **Download cleaned CSV** in the sidebar | A CSV named `cleaned_uber_rides.csv` downloads with duplicates and invalid pickup times removed. |
| Use the date filter | Select a smaller date interval in **Overview** | Overview charts and KPIs update; the predictive evaluation uses rides from that selected interval. |
| Check pickup charts | Open **Overview** | Daily, hourly (0–23), and Monday-to-Sunday pickup charts render with labels. |
| Check the supplied coordinates | Open **Overview** with the supplied CSV | A map is shown only if valid pickup coordinate pairs exist. Invalid coordinate values are excluded. |
| Upload Kaggle-style headers | Upload a CSV with `Date/Time`, `Lat`, `Lon`, and numeric fields | The pickup timestamp is detected; the map appears when coordinate values are valid. |
| Test duplicate removal | Upload a file containing an identical row twice | The exact duplicate count increases and the repeated record is removed. |
| Test invalid/missing times | Include a blank or unparseable value in the pickup-time column | Those rows are counted as invalid and excluded from cleaned analysis. |
| Test optional columns | Upload a timestamp-only CSV with no fare, distance, passenger, or coordinate fields | Time-based analysis still works; the map is omitted and no absent fields are invented. |
| Test identifier handling | Include numeric columns named `trip_id` or `driver_id` | These identifier columns do not appear in the correlation heatmap. |
| Test short date coverage | Select a range spanning fewer than 10 hourly buckets | The prediction tab explains that more hourly buckets are needed instead of showing invented metrics. |
| Test no usable rows | Upload a header-only file, an empty file, or records with invalid timestamps | The app reports an understandable error and does not attempt to fit the model. |
| Test missing timestamp header | Upload a non-empty CSV without a recognizable pickup time column | The app reports the problem and displays the headers it found. |
| Compare model with baseline | Open **Predictive Model** for a sufficiently long range | The chronological test-period results show the Random Forest beside the training-mean baseline; no accuracy claim is made in advance. |

## Viva questions and simple answers

1. **What is the aim of this project?**  
   To explore ride pickup data and demonstrate hourly demand prediction in an
   interactive dashboard.
2. **What is a CSV file?**  
   A plain-text table format where values in a row are separated by commas.
3. **Why do we use Pandas?**  
   Pandas helps load, inspect, clean, group, and summarize table-shaped data.
4. **Why is NumPy included?**  
   It provides numerical operations, such as creating the baseline predictions.
5. **What does Streamlit do?**  
   It turns Python code into a small interactive web application.
6. **What does Plotly do?**  
   It creates charts that users can interact with, such as zooming and hovering.
7. **What is data cleaning?**  
   It means checking and preparing raw data so analyses use valid, consistent
   records.
8. **Why remove duplicate rows?**  
   An exact repeated record can count the same ride more than once.
9. **Why remove rows with invalid pickup timestamps?**  
   The project needs a valid time to group rides by day, hour, and weekday.
10. **What is exploratory data analysis (EDA)?**  
    It is the process of summarizing and visualizing data to understand its
    patterns.
11. **What does a correlation value represent?**  
    It describes the direction and strength of a linear relationship between
    numerical measurements.
12. **Does correlation prove causation?**  
    No. Two fields can move together without one causing the other.
13. **Why exclude identifiers from correlations?**  
    Numeric IDs label records; their numerical order usually has no meaningful
    relationship to ride measurements.
14. **What is a Random Forest regressor?**  
    It combines predictions from many decision trees to estimate a numerical
    value, such as rides per hour.
15. **What is a feature in this model?**  
    An input used to make a prediction, such as hour, weekday, month, weekend
    status, or time index.
16. **Why use a chronological split?**  
    It trains on earlier hours and tests on later hours, which better reflects
    predicting future demand.
17. **What is MAE?**  
    Mean Absolute Error: the average size of prediction errors, in rides per
    hour.
18. **What is RMSE?**  
    Root Mean Squared Error: an error measure in rides that gives larger misses
    more weight.
19. **What does R-squared mean?**  
    It compares the model with a mean-only reference. A negative score can mean
    the model did worse than that reference on the test data.
20. **Why compare against a baseline?**  
    The baseline helps show whether the more complex model improves on a simple
    prediction.
21. **What is feature importance?**  
    It estimates how much the fitted Random Forest relied on each input feature;
    it does not prove cause and effect.
22. **Can this model guarantee future ride demand?**  
    No. It is an educational model evaluated on historical held-out data, and
    future conditions can change.

## Project report outline

1. **Title page** — project title, student name, department, guide, and date.
2. **Abstract** — short summary of the problem, approach, and dashboard.
3. **Introduction and problem statement** — why understanding ride demand can
   be useful.
4. **Objectives and scope** — data cleaning, EDA, correlation, prediction, and
   dashboard features.
5. **Tools and technologies** — Python, Pandas, NumPy, Plotly, Scikit-learn,
   and Streamlit.
6. **Dataset description** — cite the actual dataset source; list its real
   columns, record count, and date coverage after checking the CSV.
7. **Methodology** — timestamp detection, duplicate removal, invalid-time
   handling, hourly aggregation, features, chronological split, and baseline.
8. **System design** — describe the CSV-to-cleaning-to-analysis-to-dashboard
   flow.
9. **Results and screenshots** — include screenshots from the running app and
   copy the actual model metrics shown by your run. Do not invent scores.
10. **Testing** — summarize the manual test cases above and their observed
    outcomes.
11. **Limitations and future work** — data quality, missing external factors,
    model limits, and possible improvements.
12. **Conclusion and references** — summarize the work and cite the data and
    library documentation you actually used.

## Short presentation script

> Good morning. My project is the Multi-Feature Predictive Analytics Dashboard
> for Uber Rides. It uses a CSV of ride records. The dashboard first checks the
> data, removes exact duplicate records, converts the pickup-time column to a
> datetime, and excludes records with invalid pickup times from time-based
> analysis. The Overview tab displays daily, hourly, and weekday patterns and
> shows a pickup map when valid coordinates are present. The Correlation tab
> compares suitable numeric measurements and excludes identifier columns.
> The Predictive Model tab groups rides into hours, creates calendar features,
> trains a Random Forest on earlier hours, and evaluates it on later hours
> against a simple mean baseline. I report the observed MAE, RMSE, and
> R-squared rather than claiming accuracy in advance. Finally, the Data Quality
> tab and download button make the cleaning steps visible and reusable. Thank
> you.
