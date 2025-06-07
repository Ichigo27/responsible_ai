import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os
import sys

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_manager import ConfigManager

st.set_page_config(page_title="Responsible AI Metrics Dashboard", layout="wide")

st.title("Responsible AI")

# Load configuration and determine dataset path
try:
    config_manager = ConfigManager()
    app_config = config_manager.get_config("app_config")
    dashboard_config = app_config.get("DASHBOARD", {})
    dataset_path = dashboard_config.get("DATASET_PATH", "data/dashboard")
    dataset_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dataset_path, "dashboard_data.jsonl")
except Exception as e:
    st.error(f"Error loading configuration: {str(e)}")
    dataset_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/dashboard/dashboard_data.jsonl")
    st.warning(f"Using default dataset path: {dataset_file}")

# Check if the dataset file exists
if not os.path.exists(dataset_file):
    st.error(f"Dashboard dataset not found at: {dataset_file}")
    st.info("Upload data using the API endpoints: POST /api/v1/dashboard/dataset/replace or /append")
    st.stop()

timestamp = os.path.getmtime(dataset_file)
last_modified = datetime.fromtimestamp(timestamp)
formatted_date = last_modified.strftime("%Y-%m-%d %H:%M:%S")

# Load the dataset
try:
    with open(dataset_file, "r") as f:
        jsonl_content = f.read().splitlines()

    # Parse the JSONL content
    data = []
    for line in jsonl_content:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            st.error(f"Error parsing line in dataset")
            continue

    if not data:
        st.error("No valid data found in the file.")
        st.stop()

except Exception as e:
    st.error(f"Error reading dashboard dataset: {str(e)}")
    st.stop()

# =====================================================================
# DATA LOADING AND PROCESSING
# =====================================================================

# Convert to DataFrame and extract necessary fields
processed_data = []
for item in data:
    # Extract application name, date, and prompt_id from the id
    id_parts = item.get("id", "unknown-01:01:2000-0").split("-")
    if len(id_parts) >= 3:
        app_name = id_parts[0]
        date_str = id_parts[1]
        prompt_id = id_parts[2]

        # Convert date string to datetime
        try:
            date = datetime.strptime(date_str, "%d:%m:%Y")
        except ValueError:
            st.warning(f"Invalid date format in ID: {item.get('id')}")
            date = None

        # Extract metrics
        metrics = item.get("metrics", {})
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        entry = {"application": app_name, "date": date, "prompt_id": prompt_id, "prompt": prompt, "response": response}

        # Add metrics
        for metric_name, metric_data in metrics.items():
            if metric_name != "explainability":
                entry[f"{metric_name}_score"] = metric_data.get("score", 0)
                entry[f"{metric_name}_threshold"] = metric_data.get("threshold", 0)
                entry[f"{metric_name}_passed"] = metric_data.get("passed", False)
                entry[f"{metric_name}_reason"] = metric_data.get("reason", "")

                # Extract additional data if available
                if "additional_data" in metric_data:
                    for add_key, add_value in metric_data["additional_data"].items():
                        entry[f"{metric_name}_{add_key}"] = add_value

        # Add explainability separately if it exists
        if "explainability" in metrics:
            entry["has_explainability"] = True
            entry["explainability"] = metrics["explainability"]
        else:
            entry["has_explainability"] = False

        # Add prompt and response length
        entry["prompt_length"] = len(prompt.split())
        entry["response_length"] = len(response.split())

        processed_data.append(entry)

df = pd.DataFrame(processed_data)

# Filter out rows with invalid dates
df = df.dropna(subset=["date"])

if df.empty:
    st.error("No valid data entries after processing.")
    st.stop()

# =====================================================================
# METRIC NAME MAPPING
# =====================================================================

# Map original metric names to more readable display names
metric_display_names = {
    "hallucination": "Factual Accuracy",
    "toxicity": "Content Safety",
    "bias_fairness": "Fairness & Inclusion",
    "relevance": "Response Relevance",
    "helpfulness": "User Helpfulness",
    "harmfulness": "Harm Prevention",
    "coherence": "Logical Coherence",
    "completeness": "Information Completeness",
    "quality": "Overall Quality",
    "correctness": "Answer Correctness",
    "appropriateness": "Content Appropriateness",
    "accuracy": "Technical Accuracy",
    "sentiment": "Sentiment Analysis",
    "consistency": "Response Consistency",
    "ethics": "Ethical Alignment",
    "language_quality": "Language Quality",
    "groundedness": "Source Groundedness",
    "harmful_content": "Safety Compliance",
    "creativity": "Creativity Score",
    "clarity": "Response Clarity",
    "pii": "PII Detection",
    "privacy": "Privacy Compliance",
    "jailbreak": "Jailbreak Detection",
    "offensive": "Offensive Content",
    "explainability": "Model Explainability",
}


# Function to get display name for any metric
def get_display_name(metric_name):
    """Convert original metric name to a display-friendly name."""
    if metric_name in metric_display_names:
        return metric_display_names[metric_name]
    else:
        # Default formatting for metrics not in our mapping
        return metric_name.replace("_", " ").title()


# Function to get original metric name from display name
def get_original_name(display_name):
    """Get the original metric name from a display name."""
    for orig, disp in metric_display_names.items():
        if disp == display_name:
            return orig

    # If not found in mapping, try to reverse the default formatting
    return display_name.replace(" ", "_").lower()


# Create color mapping by metric category
metric_categories = {
    "Content Safety": ["toxicity", "harmful_content", "jailbreak", "harmfulness", "offensive"],
    "Response Relevance": ["relevance", "coherence", "quality", "completeness", "clarity", "language_quality"],
    "Factual Accuracy": ["hallucination", "factuality", "accuracy", "correctness", "groundedness", "consistency"],
    "Fairness & Inclusion": ["bias_fairness", "ethics", "appropriateness"],
}

# Create color scheme
category_colors = {
    "Content Safety": "rgba(255, 99, 71, 0.8)",  # Red tones
    "Response Relevance": "rgba(65, 105, 225, 0.8)",  # Blue tones
    "Factual Accuracy": "rgba(50, 205, 50, 0.8)",  # Green tones
    "Fairness & Inclusion": "rgba(255, 165, 0, 0.8)",  # Orange tones
}


# Get color for a metric
def get_metric_color(metric_name):
    for category, metrics in metric_categories.items():
        if metric_name in metrics:
            return category_colors[category]
    return "rgba(128, 128, 128, 0.8)"  # Default gray


# Extract unique applications
applications = sorted(df["application"].unique())

if not applications:
    st.error("No applications found in the dataset.")
    st.stop()

# Extract min and max dates for filter
min_date = df["date"].min()
max_date = df["date"].max()

# =====================================================================
# FILTERS SECTION
# =====================================================================

# Create sidebar for filters
st.sidebar.header("Filters")

# Application filter - Without "All" option, default to first application alphabetically
selected_app = st.sidebar.selectbox("Select Application", applications, index=0)

# Date range filter with date_input widgets
min_date_value = min_date.date()
max_date_value = max_date.date()

start_date = st.sidebar.date_input("Start Date", value=min_date_value, min_value=min_date_value, max_value=max_date_value)

end_date = st.sidebar.date_input("End Date", value=max_date_value, min_value=min_date_value, max_value=max_date_value)

# Ensure end_date is not before start_date
if end_date < start_date:
    st.sidebar.error("End date must be after start date.")
    end_date = start_date

# Convert date inputs to datetime for filtering
start_datetime = pd.Timestamp(start_date)
end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Set to end of day

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["application"] == selected_app]  # Always filter by application
filtered_df = filtered_df[(filtered_df["date"] >= start_datetime) & (filtered_df["date"] <= end_datetime)]

# Get list of metrics
metric_names = [col.replace("_score", "") for col in filtered_df.columns if col.endswith("_score") and not col.startswith("explainability")]

# Calculate threshold values
thresholds = {}
for metric in metric_names:
    if f"{metric}_threshold" in filtered_df.columns:
        thresholds[metric] = filtered_df[f"{metric}_threshold"].iloc[0]

## =====================================================================
# CONTENT STATISTICS
# =====================================================================
st.header("Dashboard Overview")

# Calculate overall pass rate
pass_columns = [f"{metric}_passed" for metric in metric_names]
overall_pass_rate = 0

if pass_columns and len(pass_columns) > 0:
    filtered_df["all_passed"] = filtered_df[pass_columns].all(axis=1)
    overall_pass_rate = filtered_df["all_passed"].mean() * 100

# Create a two-column layout for stats
col1, col2 = st.columns(2)

with col1:
    # Donut chart for pass rate
    if overall_pass_rate > 0:
        fig_donut = go.Figure(
            go.Pie(
                values=[overall_pass_rate, 100 - overall_pass_rate], labels=["Pass", "Fail"], hole=0.7, marker_colors=["green", "lightgrey"], textinfo="none", showlegend=False
            )
        )

        fig_donut.add_annotation(text=f"{overall_pass_rate:.1f}%", x=0.5, y=0.5, font_size=28, showarrow=False)

        fig_donut.update_layout(title_text="Overall Pass Rate", height=250, margin=dict(t=30, b=30, l=10, r=10))

        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Pass rate data not available")

with col2:
    # Failure distribution per metric as a pie chart
    if pass_columns and len(pass_columns) > 0:
        # Count failures by metric
        failure_counts = {}
        colors = []

        for metric in metric_names:
            if f"{metric}_passed" in filtered_df.columns:
                failures = (~filtered_df[f"{metric}_passed"]).sum()
                if failures > 0:  # Only include metrics with failures
                    failure_counts[get_display_name(metric)] = failures
                    # Use the same color scheme as the main dashboard
                    colors.append(get_metric_color(metric))

        if failure_counts:
            # Create pie chart
            failure_labels = list(failure_counts.keys())
            failure_values = list(failure_counts.values())

            fig_failures = go.Figure(
                data=[
                    go.Pie(
                        labels=failure_labels,
                        values=failure_values,
                        textinfo="percent",  # Change to show percentage text
                        textfont=dict(size=12),  # Increase text size for readability
                        marker=dict(colors=colors),
                        showlegend=True,  # Explicitly enable legend for the trace
                        insidetextorientation="radial",  # Align text radially for better fit
                    )
                ]
            )

            fig_failures.update_layout(
                title="Failure Distribution by Metric",
                height=250,
                margin=dict(t=30, b=30, l=10, r=120),  # Increased right margin for legend
                showlegend=True,  # Enable legend in layout
                legend=dict(
                    orientation="v",
                    x=1.05,
                    y=0.5,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.8)",  # Add semi-transparent background
                    bordercolor="lightgrey",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(fig_failures, use_container_width=False)  # Use False to respect margins
        else:
            st.success("No failures detected across any metrics!")

# Display date range and sample count in caption
total_samples = len(filtered_df)
date_range = f"{min_date.strftime('%b %d, %Y')} to {max_date.strftime('%b %d, %Y')}"
st.caption(f"Data period: {date_range} | Total Samples: {total_samples:,} | Filter: {selected_app}")

# =====================================================================
# OVERVIEW GRAPH - ALL METRICS
# =====================================================================

st.header("Overview - All Metrics")

# Prepare data for the overview graph
daily_metrics = filtered_df.groupby(filtered_df["date"].dt.date).agg({f"{metric}_score": "mean" for metric in metric_names}).reset_index()

# Create a figure with all metrics
fig_overview = go.Figure()

# Add lines for each metric
for i, metric in enumerate(metric_names):
    color = get_metric_color(metric)

    # Find which category this metric belongs to
    category = "Other"
    for cat, metrics in metric_categories.items():
        if metric in metrics:
            category = cat.capitalize()
            break

    fig_overview.add_trace(
        go.Scatter(
            x=daily_metrics["date"],
            y=daily_metrics[f"{metric}_score"],
            mode="lines+markers",
            name=get_display_name(metric),
            line=dict(width=2, color=color),
            marker=dict(size=8),
            legendgroup=category,
            legendgrouptitle_text=category,
        )
    )

    # Add threshold line
    if metric in thresholds:
        fig_overview.add_trace(
            go.Scatter(
                x=daily_metrics["date"],
                y=[thresholds[metric]] * len(daily_metrics),
                mode="lines",
                line=dict(dash="dash", color=color, width=1),
                name=f"{get_display_name(metric)} Threshold",
                opacity=0.7,
                legendgroup=category,
                showlegend=True,  # Changed from False to True to show in legend
            )
        )

# Add pass rate as a secondary y-axis
daily_pass_rates = filtered_df.groupby(filtered_df["date"].dt.date).agg({f"{metric}_passed": "mean" for metric in metric_names}).reset_index()

# Calculate overall pass rate (all metrics passed)
pass_columns = [f"{metric}_passed" for metric in metric_names]
if pass_columns and len(pass_columns) > 0:
    # Calculate all_passed field
    filtered_df["all_passed"] = filtered_df[pass_columns].all(axis=1)

    # Calculate overall pass rate for each date
    daily_pass_rates["overall_pass_rate"] = filtered_df.groupby(filtered_df["date"].dt.date)["all_passed"].mean()

    # Only add the trace if there is valid data
    if not daily_pass_rates["overall_pass_rate"].isna().all():  # Check if we have any non-NA values
        fig_overview.add_trace(
            go.Scatter(
                x=daily_pass_rates["date"],
                y=daily_pass_rates["overall_pass_rate"],
                mode="lines+markers",
                name="Overall Pass Rate",
                line=dict(width=3, color="black"),
                marker=dict(size=10),
                yaxis="y2",
                legendgroup="Pass Rate",
                legendgrouptitle_text="Pass Rate",
            )
        )

# Always position the legend on the right side
legend_right_margin = 300  # Default large margin

# Adjust margin based on number of metrics and thresholds
num_legend_items = len(metric_names) * 2  # Each metric + threshold
if num_legend_items <= 10:
    legend_right_margin = 200
elif num_legend_items <= 20:
    legend_right_margin = 250
else:
    legend_right_margin = 300

fig_overview.update_layout(
    title="All Metrics Performance Over Time",
    xaxis_title="Date",
    yaxis=dict(title="Score", range=[0, 1.1]),
    yaxis2=dict(title="Overall Pass Rate", overlaying="y", side="right", range=[0, 1.1], showgrid=False),
    height=550,  # Slightly increased height
    margin=dict(r=legend_right_margin, t=50, b=50),  # Dynamic right margin
    hovermode="x unified",
    showlegend=True,
    legend=dict(
        x=1.05,
        y=0.5,
        xanchor="left",
        yanchor="middle",
        font=dict(size=7),  # Smaller font
        bordercolor="LightGrey",
        borderwidth=1,
        tracegroupgap=2,  # Minimal gap between legend groups
    ),
)

st.plotly_chart(fig_overview, use_container_width=True)

# =====================================================================
# METRICS OVERVIEW SECTION
# =====================================================================

st.header("Metrics Overview")

# Group by date and calculate metrics
daily_metrics = filtered_df.groupby(filtered_df["date"].dt.date).agg({f"{metric}_score": "mean" for metric in metric_names}).reset_index()

# Create individual plots for each metric
for metric in metric_names:
    fig = go.Figure()

    # Add line for the metric
    fig.add_trace(
        go.Scatter(
            x=daily_metrics["date"],
            y=daily_metrics[f"{metric}_score"],
            mode="lines+markers",
            name=get_display_name(metric),
            line=dict(width=3, color=get_metric_color(metric)),
            fill="tozeroy",
            fillcolor=f"rgba({get_metric_color(metric).replace('rgba(', '').replace(')', '').split(',')[0]}, {get_metric_color(metric).replace('rgba(', '').replace(')', '').split(',')[1]}, {get_metric_color(metric).replace('rgba(', '').replace(')', '').split(',')[2]}, 0.2)",
        )
    )

    # Add threshold line
    if metric in thresholds:
        fig.add_trace(
            go.Scatter(
                x=daily_metrics["date"],
                y=[thresholds[metric]] * len(daily_metrics),
                mode="lines",
                line=dict(dash="dash", color="red", width=2),
                name=f"Threshold ({thresholds[metric]})",
            )
        )

    fig.update_layout(
        title=f"{get_display_name(metric)} Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        height=300,
        margin=dict(l=0, r=100, t=40, b=0),  # Add right margin for legend
        hovermode="x unified",
        showlegend=True,  # Explicitly show legend
        legend=dict(
            x=1.02,  # Position legend outside of plot
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
            bordercolor="LightGrey",
            borderwidth=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# PASS/FAIL ANALYSIS SECTION
# =====================================================================

st.header("Pass/Fail Analysis")

pass_rates = {}
for metric in metric_names:
    if f"{metric}_passed" in filtered_df.columns:
        pass_rates[get_display_name(metric)] = filtered_df[f"{metric}_passed"].mean() * 100

if pass_rates:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=list(pass_rates.keys()),
            y=list(pass_rates.values()),
            text=[f"{rate:.1f}%" for rate in pass_rates.values()],
            textposition="outside",
            marker_color=["green" if rate > 95 else "yellowgreen" if rate > 90 else "orange" if rate > 80 else "red" for rate in pass_rates.values()],
            name="Pass Rate",
        )
    )

    # Add custom legend items explaining color coding
    fig2.add_trace(go.Bar(x=[None], y=[None], name="Excellent (>95%)", marker_color="green", showlegend=True, legendgroup="threshold_explanation"))
    fig2.add_trace(go.Bar(x=[None], y=[None], name="Good (90-95%)", marker_color="yellowgreen", showlegend=True, legendgroup="threshold_explanation"))
    fig2.add_trace(go.Bar(x=[None], y=[None], name="Fair (80-90%)", marker_color="orange", showlegend=True, legendgroup="threshold_explanation"))
    fig2.add_trace(go.Bar(x=[None], y=[None], name="Poor (<80%)", marker_color="red", showlegend=True, legendgroup="threshold_explanation"))

    fig2.update_layout(
        title="Pass Rate by Metric",
        xaxis_title="Metric",
        yaxis_title="Pass Rate (%)",
        yaxis=dict(range=[0, 105]),
        height=450,  # Increased height
        margin=dict(r=120, t=70, b=50),  # Add margins
        showlegend=True,
        legend=dict(
            title="Performance Tiers", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1
        ),
    )

    st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
# EXPLAINABILITY ANALYSIS SECTION
# =====================================================================

st.header("Explainability Analysis")

# Add multiselect for metrics filtering
selected_metrics_display = st.multiselect("Show examples that failed these metrics:", options=[get_display_name(m) for m in metric_names], default=[])

# Convert display names back to original metric names
selected_metrics = [get_original_name(m) for m in selected_metrics_display]

# Apply metrics filter if any metrics are selected
if selected_metrics:
    metric_filtered_entries = filtered_df.copy()
    for metric in selected_metrics:
        metric_filtered_entries = metric_filtered_entries[metric_filtered_entries[f"{metric}_passed"] == False]
    explainable_entries = metric_filtered_entries[metric_filtered_entries.get("has_explainability", False) == True]
else:
    # If no metrics selected, show all entries with explainability
    explainable_entries = filtered_df[filtered_df.get("has_explainability", False) == True]

if not explainable_entries.empty:
    # Dropdowns for date and prompt_id
    col1, col2 = st.columns(2)  # Keep the dropdowns in columns

    with col1:
        # First dropdown for date selection
        available_dates = sorted(explainable_entries["date"].dt.date.unique())
        selected_date = st.selectbox("Select a date:", options=available_dates, format_func=lambda d: d.strftime("%Y-%m-%d"))

    # Filter by selected date
    date_filtered = explainable_entries[explainable_entries["date"].dt.date == selected_date]

    with col2:
        # Second dropdown for prompt_id selection
        if not date_filtered.empty:
            available_prompts = date_filtered["prompt_id"].unique()
            selected_prompt_id = st.selectbox("Select a sample ID:", options=available_prompts)

    # Get the selected sample
    if not date_filtered.empty and selected_prompt_id in date_filtered["prompt_id"].values:
        sample = date_filtered[date_filtered["prompt_id"] == selected_prompt_id].iloc[0]

        # Create two columns for info and donut chart
        sample_col1, sample_col2 = st.columns(2)

        with sample_col1:
            st.subheader("Sample Information")
            # Basic information about the sample
            st.write(f"**Application:** {sample['application']}")
            st.write(f"**Date:** {sample['date'].strftime('%Y-%m-%d')}")
            st.write(f"**Sample ID:** {sample['prompt_id']}")

        with sample_col2:
            # Metrics summary section
            st.subheader("Metrics Summary")

            # Count passed metrics
            total_metrics = len(metric_names)
            # Create two columns for metric details
            met_col1, met_col2 = st.columns(2)  # Keep the metrics in columns for better display

            # Distribute metrics across columns
            metrics_per_column = (total_metrics + 1) // 2

            for i, metric in enumerate(metric_names):
                # Determine which column to use
                col = met_col1 if i < metrics_per_column else met_col2

                # Get display name for the metric
                display_name = get_display_name(metric)

                # Check if metric passed or failed
                passed = sample.get(f"{metric}_passed", False)
                score = sample.get(f"{metric}_score", 0)
                threshold = sample.get(f"{metric}_threshold", 0)

                # Create color-coded metric display with score and threshold
                with col:
                    st.markdown(
                        f"**{display_name}**: "
                        f"<span style='color:{'green' if passed else 'red'}; font-weight:bold'>"
                        f"{'✓ PASS' if passed else '✗ FAIL'}</span> "
                        f"(Score: {score:.2f}, Threshold: {threshold:.2f})",
                        unsafe_allow_html=True,
                    )

        col1, col2 = st.columns(2)

        with col1:
            # Display metadata if available
            if sample.get("metadata"):
                st.subheader("Metadata")
                for key, value in sample["metadata"].items():
                    st.write(f"**{key}:** {value}")
            else:
                st.subheader("Prompt")
                st.code(sample["prompt"], language=None)
            # Display response
            st.subheader("Response")
            st.code(sample["response"], language=None)

        with col2:
            # Display explanation section last
            st.subheader("Explanation")
            explainability = sample["explainability"]

            if isinstance(explainability, dict) and "explanation" in explainability:
                explanation = explainability["explanation"]

                if isinstance(explanation, dict):
                    if "reasoning" in explanation:
                        st.write("**Reasoning:**")
                        st.write(explanation["reasoning"])
                else:
                    # If explanation is not a dict, display as is
                    st.write(explanation)
            else:
                st.write("Explanation data not available in the expected format.")
    else:
        st.info("No explainability data found for the selected date and sample ID.")
else:
    st.info(f"No explainability data found matching the selected criteria{' and failed metrics' if selected_metrics else ''}.")

# =====================================================================
# TEMPORAL ANALYSIS AND TRENDS
# =====================================================================

st.header("Temporal Analysis and Trends")
st.write("Analyzing how metrics change over time and identifying patterns")

# Calculate moving averages and trends
window_size = max(3, len(daily_metrics) // 10)  # Adaptive window size
moving_avg_data = daily_metrics.copy()

for metric in metric_names:
    # Calculate 7-day moving average
    moving_avg_data[f"{metric}_ma"] = moving_avg_data[f"{metric}_score"].rolling(window=min(7, len(moving_avg_data)), min_periods=1).mean()

    # Calculate trend (slope of linear regression)
    if len(moving_avg_data) >= 3:
        x = np.arange(len(moving_avg_data))
        y = moving_avg_data[f"{metric}_score"].values

        if len(x) == len(y) and len(x) > 1:
            slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
            moving_avg_data[f"{metric}_trend"] = slope[0]
        else:
            moving_avg_data[f"{metric}_trend"] = 0
    else:
        moving_avg_data[f"{metric}_trend"] = 0

# Create date_ordinal column for trend analysis
moving_avg_data["date_ordinal"] = pd.to_datetime(moving_avg_data["date"]).map(lambda x: x.toordinal())

# Create tabs for different temporal analyses
trend_tabs = st.tabs(["Moving Averages", "Trend Analysis", "Anomaly Detection"])

with trend_tabs[0]:
    # Moving averages visualization
    st.subheader("Moving Averages")

    selected_metric_ma_display = st.selectbox("Select metric for moving average analysis:", options=[get_display_name(m) for m in metric_names], key="ma_metric")

    # Convert back to original metric name
    orig_metric_ma = get_original_name(selected_metric_ma_display)

    fig_ma = go.Figure()

    # Add raw data
    fig_ma.add_trace(
        go.Scatter(
            x=moving_avg_data["date"],
            y=moving_avg_data[f"{orig_metric_ma}_score"],
            mode="markers",
            name=f"Daily {selected_metric_ma_display} Score",
            marker=dict(size=8, color="lightgray"),
        )
    )

    # Add moving average
    fig_ma.add_trace(
        go.Scatter(x=moving_avg_data["date"], y=moving_avg_data[f"{orig_metric_ma}_ma"], mode="lines", name="Moving Average (7-day)", line=dict(width=3, color="blue"))
    )

    # Add threshold
    if orig_metric_ma in thresholds:
        fig_ma.add_trace(
            go.Scatter(
                x=moving_avg_data["date"],
                y=[thresholds[orig_metric_ma]] * len(moving_avg_data),
                mode="lines",
                line=dict(dash="dash", color="red", width=2),
                name=f"Threshold ({thresholds[orig_metric_ma]})",
            )
        )

    fig_ma.update_layout(
        title=f"Moving Average Trend for {selected_metric_ma_display}",
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        height=400,
        margin=dict(r=100, t=50),  # Add right margin
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", title="Data Series", bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1),
    )

    st.plotly_chart(fig_ma, use_container_width=True)

with trend_tabs[1]:
    # Trend analysis
    st.subheader("Trend Analysis")

    # Calculate overall trends for each metric
    trend_data = []
    for metric in metric_names:
        if f"{metric}_trend" in moving_avg_data.columns:
            trend = moving_avg_data[f"{metric}_trend"].iloc[0]
            # Calculate total change over period
            first_score = moving_avg_data[f"{metric}_score"].iloc[0]
            last_score = moving_avg_data[f"{metric}_score"].iloc[-1]
            total_change = last_score - first_score

            trend_data.append(
                {
                    "Metric": get_display_name(metric),
                    "Trend": trend,
                    "Total Change": total_change,
                    "Direction": "Improving" if total_change > 0 else "Declining" if total_change < 0 else "Stable",
                }
            )

    trend_df = pd.DataFrame(trend_data)

    if not trend_df.empty:
        # Create a horizontal bar chart showing trends
        fig_trend = go.Figure()

        fig_trend.add_trace(
            go.Bar(
                y=trend_df["Metric"],
                x=trend_df["Total Change"],
                orientation="h",
                marker_color=["green" if change > 0 else "red" if change < 0 else "gray" for change in trend_df["Total Change"]],
                text=[f"{change:.4f}" for change in trend_df["Total Change"]],
                textposition="outside",
                name="Total Change",
            )
        )

        fig_trend.update_layout(
            title="Total Change in Metrics Over Period",
            xaxis_title="Change in Score (End - Start)",
            yaxis_title="",
            height=400,
            margin=dict(r=120, t=50),  # Add right margin
            showlegend=True,
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", title="Direction", bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1),
        )

        # Add legend items for direction categories
        fig_trend.add_trace(go.Bar(x=[None], y=[None], name="Improving", marker_color="green", showlegend=True))
        fig_trend.add_trace(go.Bar(x=[None], y=[None], name="Declining", marker_color="red", showlegend=True))
        fig_trend.add_trace(go.Bar(x=[None], y=[None], name="Stable", marker_color="gray", showlegend=True))

        st.plotly_chart(fig_trend, use_container_width=True)

        # Linear regression plot for selected metric
        selected_metric_trend_display = st.selectbox("View detailed trend for:", options=trend_df["Metric"], key="trend_metric")

        # Convert back to original metric name
        orig_metric_trend = get_original_name(selected_metric_trend_display)

        # Fix for regression plot - Use ordinal dates
        fig_regression = px.scatter(
            moving_avg_data,
            x="date_ordinal",
            y=f"{orig_metric_trend}_score",
            trendline="ols",
            trendline_color_override="red",
            title=f"Trend Analysis for {selected_metric_trend_display}",
            labels={f"{orig_metric_trend}_score": "Score"},  # Better label for legend
        )

        # Rename the traces for better legend
        fig_regression.data[0].name = "Daily Score"
        if len(fig_regression.data) > 1:
            fig_regression.data[1].name = "Trend Line"

        # Update x-axis to show actual dates instead of ordinal values
        date_range = moving_avg_data["date"].tolist()
        fig_regression.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[pd.to_datetime(d).toordinal() for d in date_range],
                ticktext=[d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else pd.to_datetime(d).strftime("%Y-%m-%d") for d in date_range],
                title="Date",
            ),
            yaxis_title="Score",
            height=400,
            margin=dict(r=100, t=50, b=100),  # Add margins
            showlegend=True,
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1),
        )

        # Improve tick label rotation for better readability
        fig_regression.update_xaxes(tickangle=45)

        st.plotly_chart(fig_regression, use_container_width=True)

with trend_tabs[2]:
    # Anomaly detection
    st.subheader("Anomaly Detection")

    selected_metric_anomaly_display = st.selectbox("Select metric for anomaly detection:", options=[get_display_name(m) for m in metric_names], key="anomaly_metric")

    # Convert back to original metric name
    orig_metric_anomaly = get_original_name(selected_metric_anomaly_display)

    # Calculate Z-scores for anomaly detection
    scores = moving_avg_data[f"{orig_metric_anomaly}_score"]
    mean = scores.mean()
    std = scores.std() if scores.std() > 0 else 0.01  # Avoid division by zero
    z_scores = (scores - mean) / std

    # Define anomalies as points with z-score > 2 or < -2
    anomalies = abs(z_scores) > 2

    fig_anomaly = go.Figure()

    # Add regular points
    fig_anomaly.add_trace(
        go.Scatter(
            x=moving_avg_data.loc[~anomalies, "date"],
            y=moving_avg_data.loc[~anomalies, f"{orig_metric_anomaly}_score"],
            mode="markers",
            name="Normal Values",
            marker=dict(size=8, color="blue"),
        )
    )

    # Add anomalies
    if anomalies.any():
        fig_anomaly.add_trace(
            go.Scatter(
                x=moving_avg_data.loc[anomalies, "date"],
                y=moving_avg_data.loc[anomalies, f"{orig_metric_anomaly}_score"],
                mode="markers",
                name="Anomaly",
                marker=dict(size=12, color="red", symbol="star"),
            )
        )

    # Add trend line
    fig_anomaly.add_trace(
        go.Scatter(x=moving_avg_data["date"], y=moving_avg_data[f"{orig_metric_anomaly}_ma"], mode="lines", name="Moving Average", line=dict(color="gray", width=2))
    )

    # Add bands for expected range (±2 std)
    fig_anomaly.add_trace(go.Scatter(x=moving_avg_data["date"], y=moving_avg_data[f"{orig_metric_anomaly}_ma"] + 2 * std, mode="lines", line=dict(width=0), showlegend=False))

    fig_anomaly.add_trace(
        go.Scatter(
            x=moving_avg_data["date"],
            y=moving_avg_data[f"{orig_metric_anomaly}_ma"] - 2 * std,
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(200, 200, 200, 0.2)",
            fill="tonexty",
            name="Expected Range (±2σ)",
        )
    )

    fig_anomaly.update_layout(
        title=f"Anomaly Detection for {selected_metric_anomaly_display}",
        xaxis_title="Date",
        yaxis_title="Score",
        height=500,
        hovermode="closest",
        margin=dict(r=120, t=50),  # Add right margin
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", title="Data Points", bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1),
    )

    # Display the anomaly chart
    st.plotly_chart(fig_anomaly, use_container_width=True)

    # Show table of anomalies
    if anomalies.any():
        st.write("### Anomalous Data Points")
        anomaly_table = moving_avg_data.loc[anomalies, ["date", f"{orig_metric_anomaly}_score"]]
        anomaly_table.columns = ["Date", "Score"]
        anomaly_table = anomaly_table.sort_values("Date")
        st.dataframe(anomaly_table)
    else:
        st.info("No anomalies detected for this metric.")

# =====================================================================
# FAILURE MODE ANALYSIS
# =====================================================================

st.header("Failure Mode Analysis")
st.write("Understanding patterns and root causes of failures")

# Create tabs for different failure analyses
failure_tabs = st.tabs(["Failure Categories", "Co-occurrence", "Pareto Analysis"])

with failure_tabs[0]:
    st.subheader("Failure Categories")

    # Count failures by metric
    failure_counts = {}
    for metric in metric_names:
        if f"{metric}_passed" in filtered_df.columns:
            failures = (~filtered_df[f"{metric}_passed"]).sum()
            failure_counts[get_display_name(metric)] = failures

    if failure_counts:
        # Sort by count
        failure_counts = {k: v for k, v in sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)}

        fig_failures = go.Figure(
            [go.Bar(x=list(failure_counts.keys()), y=list(failure_counts.values()), text=list(failure_counts.values()), textposition="auto", name="Failures")]
        )

        fig_failures.update_layout(
            title="Number of Failures by Metric Type",
            xaxis_title="Metric",
            yaxis_title="Number of Failures",
            height=400,
            showlegend=True,
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1),
        )

        st.plotly_chart(fig_failures, use_container_width=True)

        # Show as percentage of total
        total_prompts = len(filtered_df)
        failure_percentages = {k: (v / total_prompts) * 100 for k, v in failure_counts.items()}

        fig_failure_pct = go.Figure(
            [
                go.Bar(
                    x=list(failure_percentages.keys()),
                    y=list(failure_percentages.values()),
                    text=[f"{p:.1f}%" for p in failure_percentages.values()],
                    textposition="auto",
                    marker_color="indianred",
                    name="Failure Rate",
                )
            ]
        )

        fig_failure_pct.update_layout(
            title="Percentage of Samples that Failed Each Metric",
            xaxis_title="Metric",
            yaxis_title="Failure Rate (%)",
            height=400,
            yaxis=dict(range=[0, 100]),
            showlegend=True,
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1),
        )

        st.plotly_chart(fig_failure_pct, use_container_width=True)

with failure_tabs[1]:
    st.subheader("Failure Co-occurrence")

    # Create co-occurrence matrix for failures
    if len(metric_names) > 1:
        fail_matrix = np.zeros((len(metric_names), len(metric_names)))

        for i, metric1 in enumerate(metric_names):
            for j, metric2 in enumerate(metric_names):
                if f"{metric1}_passed" in filtered_df.columns and f"{metric2}_passed" in filtered_df.columns:
                    # Count co-occurrences of failures
                    if i == j:
                        # Diagonal shows total failures for this metric
                        fail_matrix[i, j] = (~filtered_df[f"{metric1}_passed"]).sum()
                    else:
                        # Count how often metrics fail together
                        fail_together = ((~filtered_df[f"{metric1}_passed"]) & (~filtered_df[f"{metric2}_passed"])).sum()
                        fail_matrix[i, j] = fail_together

        # Create a heatmap of co-occurrences
        fig_cooccur = px.imshow(
            fail_matrix,
            x=[get_display_name(m) for m in metric_names],
            y=[get_display_name(m) for m in metric_names],
            color_continuous_scale="Reds",
            title="Failure Co-occurrence Matrix",
            text_auto=".0f",
        )

        fig_cooccur.update_layout(
            xaxis_title="Metric",
            yaxis_title="Metric",
            height=500,
            margin=dict(r=120, t=50, b=50),  # Add margins
            coloraxis=dict(colorbar=dict(title="Number of Co-occurring Failures", title_side="right", thickness=15, len=0.6, y=0.5)),  # Fixed property  # Center vertically
        )

        st.plotly_chart(fig_cooccur, use_container_width=True)

        # Calculate conditional probabilities
        conditional_probs = np.zeros((len(metric_names), len(metric_names)))

        for i, metric1 in enumerate(metric_names):
            total_failures1 = (~filtered_df[f"{metric1}_passed"]).sum()

            if total_failures1 > 0:
                for j, metric2 in enumerate(metric_names):
                    if i != j:
                        # P(metric2 fails | metric1 fails)
                        conditional_probs[i, j] = fail_matrix[i, j] / total_failures1

        # Create a heatmap of conditional probabilities
        fig_cond_prob = px.imshow(
            conditional_probs,
            x=[get_display_name(m) for m in metric_names],
            y=[get_display_name(m) for m in metric_names],
            color_continuous_scale="Blues",
            title="P(Column Fails | Row Fails)",
            text_auto=".2f",
            range_color=[0, 1],
        )

        fig_cond_prob.update_layout(
            xaxis_title="Result Metric",
            yaxis_title="Condition Metric",
            height=500,
            margin=dict(r=120, t=50, b=50),  # Add margins
            coloraxis=dict(colorbar=dict(title="Conditional Probability", title_side="right", thickness=15, len=0.6, y=0.5)),  # Fixed property  # Center vertically
        )

        st.plotly_chart(fig_cond_prob, use_container_width=True)
    else:
        st.info("Co-occurrence analysis requires at least two metrics.")

with failure_tabs[2]:
    st.subheader("Pareto Analysis")

    # Create sub-tabs for different Pareto analyses
    pareto_tabs = st.tabs(["By Application", "By Metric"])

    with pareto_tabs[0]:
        st.subheader("Pareto Analysis by Application")

        # Count failures by application using the FULL dataset (df) instead of filtered_df
        app_failures = {}
        for app in df["application"].unique():  # Use df instead of filtered_df
            app_df = df[df["application"] == app]  # Filter the full dataset by application
            failure_count = 0

            # Count total failures across all metrics for this application
            for metric in metric_names:
                if f"{metric}_passed" in app_df.columns:
                    failure_count += (~app_df[f"{metric}_passed"]).sum()

            app_failures[app] = failure_count

        # Sort by number of failures (descending)
        app_failures = {k: v for k, v in sorted(app_failures.items(), key=lambda item: item[1], reverse=True)}

        # Rest of the Pareto Analysis code remains the same
        if app_failures:
            # Create Pareto chart
            fig_app_pareto = go.Figure()
            
            # Add bar chart
            fig_app_pareto.add_trace(go.Bar(x=list(app_failures.keys()), y=list(app_failures.values()), name="Failures", marker_color="indianred"))

            # Add cumulative percentage line
            values = list(app_failures.values())
            cumsum = np.cumsum(values)
            total = cumsum[-1] if cumsum.size > 0 else 0
            percentage = [100 * cs / total if total > 0 else 0 for cs in cumsum]

            fig_app_pareto.add_trace(
                go.Scatter(
                    x=list(app_failures.keys()),
                    y=percentage,
                    mode="lines+markers",
                    name="Cumulative %",
                    yaxis="y2",
                    line=dict(color="royalblue", width=3),
                    marker=dict(size=10),
                )
            )

            # Draw 80% reference line
            fig_app_pareto.add_shape(type="line", x0=-0.5, y0=80, x1=len(app_failures) - 0.5, y1=80, line=dict(color="red", width=2, dash="dash"), yref="y2")

            fig_app_pareto.update_layout(
                title="Pareto Analysis: Failures by Application",
                xaxis_title="Application",
                yaxis=dict(title="Number of Failures", side="left"),
                yaxis2=dict(
                    title="Cumulative Percentage",
                    side="right",
                    overlaying="y",
                    tickmode="array",
                    tickvals=[0, 20, 40, 60, 80, 100],
                    ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
                    range=[0, 100],
                    showgrid=False,
                ),
                height=500,
                margin=dict(r=120, t=50),  # Add right margin
                showlegend=True,
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1),
            )

            st.plotly_chart(fig_app_pareto, use_container_width=True)

            # Calculate the vital few (applications causing 80% of failures)
            vital_few = []
            cumulative_percent = 0
            for app, count in app_failures.items():
                cumulative_percent += (count / total * 100) if total > 0 else 0
                vital_few.append(app)
                if cumulative_percent >= 80:
                    break

            if vital_few:
                st.write(f"### The 'vital few' applications to focus on:")
                st.write(
                    f"These {len(vital_few)} applications ({(len(vital_few) / len(app_failures) * 100):.1f}% of all applications) account for approximately 80% of all failures:"
                )
                for app in vital_few:
                    st.write(f"- {app}: {app_failures[app]} failures")
        else:
            st.info("No failure data available for applications.")

    with pareto_tabs[1]:
        st.subheader("Pareto Analysis by Metric")

        # Count failures by metric
        metric_failures = {}
        for metric in metric_names:
            if f"{metric}_passed" in filtered_df.columns:
                failure_count = (~filtered_df[f"{metric}_passed"]).sum()
                metric_failures[get_display_name(metric)] = failure_count

        # Sort by number of failures (descending)
        metric_failures = {k: v for k, v in sorted(metric_failures.items(), key=lambda item: item[1], reverse=True)}

        if metric_failures:
            # Create Pareto chart
            fig_metric_pareto = go.Figure()

            # Add bar chart
            fig_metric_pareto.add_trace(go.Bar(x=list(metric_failures.keys()), y=list(metric_failures.values()), name="Failures", marker_color="indianred"))

            # Add cumulative percentage line
            values = list(metric_failures.values())
            cumsum = np.cumsum(values)
            total = cumsum[-1] if cumsum.size > 0 else 0
            percentage = [100 * cs / total if total > 0 else 0 for cs in cumsum]

            fig_metric_pareto.add_trace(
                go.Scatter(
                    x=list(metric_failures.keys()),
                    y=percentage,
                    mode="lines+markers",
                    name="Cumulative %",
                    yaxis="y2",
                    line=dict(color="royalblue", width=3),
                    marker=dict(size=10),
                )
            )

            # Draw 80% reference line
            fig_metric_pareto.add_shape(type="line", x0=-0.5, y0=80, x1=len(metric_failures) - 0.5, y1=80, line=dict(color="red", width=2, dash="dash"), yref="y2")

            fig_metric_pareto.update_layout(
                title="Pareto Analysis: Failures by Metric Type",
                xaxis_title="Metric",
                yaxis=dict(title="Number of Failures", side="left"),
                yaxis2=dict(
                    title="Cumulative Percentage",
                    side="right",
                    overlaying="y",
                    tickmode="array",
                    tickvals=[0, 20, 40, 60, 80, 100],
                    ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
                    range=[0, 100],
                    showgrid=False,
                ),
                height=500,
                margin=dict(r=120, t=50),  # Add right margin
                showlegend=True,
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1),
            )

            st.plotly_chart(fig_metric_pareto, use_container_width=True)

            # Calculate the vital few (metrics causing 80% of failures)
            vital_few = []
            cumulative_percent = 0
            for metric, count in metric_failures.items():
                cumulative_percent += (count / total * 100) if total > 0 else 0
                vital_few.append(metric)
                if cumulative_percent >= 80:
                    break

            if vital_few:
                st.write(f"### The 'vital few' metrics to focus on:")
                st.write(
                    f"These {len(vital_few)} metrics ({(len(vital_few) / len(metric_failures) * 100):.1f}% of all metrics) account for approximately 80% of all failures:"
                )
                for metric in vital_few:
                    st.write(f"- {metric}: {metric_failures[metric]} failures")
        else:
            st.info("No failure data available for metrics.")

# =====================================================================
# FAILED EXAMPLES SECTION
# =====================================================================

st.header("Failed Examples")

failed_metrics_display = st.multiselect(
    "Show examples that failed these metrics:", options=[get_display_name(m) for m in metric_names], default=[get_display_name(metric_names[0])] if metric_names else []
)

# Convert display names back to original metric names
failed_metrics = [get_original_name(m) for m in failed_metrics_display]

if failed_metrics:
    failed_df = filtered_df.copy()
    for metric in failed_metrics:
        failed_df = failed_df[failed_df[f"{metric}_passed"] == False]

    if not failed_df.empty:
        for i, row in failed_df.iterrows():
            with st.expander(f"Example ID: {row['prompt_id']} (App: {row['application']})"):
                st.write(f"**Metadata:** Prompt length: {row['prompt_length']} words, Response length: {row['response_length']} words")
                st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d')}")

                st.write("**Failed Metrics:**")
                for metric in failed_metrics:
                    st.write(f"- **{get_display_name(metric)}** (Score: {row[f'{metric}_score']:.2f}, Threshold: {row[f'{metric}_threshold']:.2f})")
                    st.write(f"  *Reason:* {row.get(f'{metric}_reason', 'No reason provided')}")
    else:
        st.info("No examples found that failed the selected metrics.")

# Show dataset information
st.sidebar.header("Dataset Information")
st.sidebar.success(f"Dataset loaded!")
st.sidebar.info(f"Last modified on: {formatted_date}")
st.sidebar.markdown("### Metric Categories")

# Add legend for metric categories in sidebar
for category, color in category_colors.items():
    st.sidebar.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<div style='width: 15px; height: 15px; background-color: {color}; margin-right: 10px;'></div>"
        f"<div>{category.capitalize()}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
