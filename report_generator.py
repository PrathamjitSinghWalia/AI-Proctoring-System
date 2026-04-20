import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF
from datetime import datetime
import os

# ============================
# COLOR MAPPING FOR EVENTS
# ============================
EVENT_COLORS = {
    "LOOKING_AWAY": "#FFA500",    # Orange
    "EYES_CLOSED": "#FFD700",     # Yellow
    "MULTIPLE_FACES": "#FF4500",  # Red-Orange
    "NO_FACE": "#DC143C",         # Crimson
    "PHONE_DETECTED": "#8B0000",  # Dark Red
    "LAPTOP_DETECTED": "#800080", # Purple
    "BOOK_DETECTED": "#4169E1",   # Royal Blue
}

def load_session_data(csv_path):
    """
    Load session CSV into a Pandas DataFrame.
    DataFrame is like an Excel spreadsheet in Python —
    rows are events, columns are timestamp/event_type/severity etc.
    """
    df = pd.read_csv(csv_path)

    # Convert timestamp column from string to actual datetime objects
    # This allows us to do time calculations on it
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def generate_event_bar_chart(df, output_path):
    """
    Generate a bar chart showing count of each event type.
    This gives a quick visual summary of what happened.
    """
    # Count occurrences of each event type
    # value_counts() is a Pandas function that counts unique values
    event_counts = df['event_type'].value_counts()

    # Create figure and axis
    # figsize controls the size of the chart in inches
    fig, ax = plt.subplots(figsize=(10, 5))

    # Get colors for each bar
    colors = [EVENT_COLORS.get(event, "#888888") for event in event_counts.index]

    # Draw the bars
    bars = ax.bar(event_counts.index, event_counts.values, color=colors)

    # Add count labels on top of each bar
    for bar, count in zip(bars, event_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Labels and title
    ax.set_title("Suspicious Event Frequency", fontsize=14, fontweight='bold')
    ax.set_xlabel("Event Type", fontsize=11)
    ax.set_ylabel("Number of Occurrences", fontsize=11)
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Bar chart saved: {output_path}")


def generate_timeline_chart(df, output_path):
    """
    Generate a timeline showing WHEN events happened during the session.
    X axis = time into exam, Y axis = event type
    This shows patterns - e.g. cheating mostly at the start vs end
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Get unique event types
    event_types = df['event_type'].unique()

    # Calculate minutes from session start for each event
    session_start = df['timestamp'].min()
    df['minutes_elapsed'] = (df['timestamp'] - session_start).dt.total_seconds() / 60

    # Plot each event as a vertical line on the timeline
    for i, event_type in enumerate(event_types):
        events = df[df['event_type'] == event_type]
        color = EVENT_COLORS.get(event_type, "#888888")

        ax.scatter(events['minutes_elapsed'],
                  [i] * len(events),
                  c=color, s=100, zorder=3, label=event_type)

        # Draw horizontal grid line for this event
        ax.axhline(y=i, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    # Labels
    ax.set_yticks(range(len(event_types)))
    ax.set_yticklabels(event_types, fontsize=9)
    ax.set_xlabel("Time into Exam (minutes)", fontsize=11)
    ax.set_title("Suspicious Event Timeline", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Timeline chart saved: {output_path}")


def generate_pdf_report(csv_path, student_name=None):
    """
    Main function — loads data, generates charts, creates PDF report.
    """
    # Load data
    df = load_session_data(csv_path)

    # Get student name from data if not provided
    if student_name is None:
        student_name = "Unknown Student"

    # Session info
    session_start = df['timestamp'].min()
    session_end = df['timestamp'].max()
    duration_minutes = (session_end - session_start).total_seconds() / 60
    total_score = df['cumulative_score'].max()
    total_events = len(df)

    # Determine risk level based on score
    if total_score < 10:
        risk_level = "LOW"
        risk_color = (0, 150, 0)       # Green
    elif total_score < 25:
        risk_level = "MEDIUM"
        risk_color = (255, 140, 0)     # Orange
    else:
        risk_level = "HIGH"
        risk_color = (200, 0, 0)       # Red

    # Create reports folder
    os.makedirs("reports", exist_ok=True)

    # Generate chart image paths
    session_id = os.path.basename(csv_path).replace("session_", "").replace(".csv", "")
    bar_chart_path = f"reports/bar_{session_id}.png"
    timeline_path = f"reports/timeline_{session_id}.png"
    pdf_path = f"reports/report_{session_id}.pdf"

    # Generate charts
    generate_event_bar_chart(df, bar_chart_path)
    generate_timeline_chart(df, timeline_path)

    # ============================
    # BUILD PDF
    # ============================
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(30, 30, 30)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(0, 8)
    pdf.cell(210, 12, "AI PROCTORING SYSTEM - SESSION REPORT", align='C')

    # Reset color
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 38)

    # Student Info Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "SESSION INFORMATION", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Student Name:     {student_name}", ln=True)
    pdf.cell(0, 7, f"Session ID:       {session_id}", ln=True)
    pdf.cell(0, 7, f"Date:             {session_start.strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(0, 7, f"Start Time:       {session_start.strftime('%H:%M:%S')}", ln=True)
    pdf.cell(0, 7, f"Duration:         {duration_minutes:.1f} minutes", ln=True)
    pdf.cell(0, 7, f"Total Events:     {total_events}", ln=True)

    pdf.ln(5)

    # Risk Score Box
    pdf.set_fill_color(*risk_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, f"  SUSPICION SCORE: {total_score}   |   RISK LEVEL: {risk_level}", 
             ln=True, fill=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Event Breakdown
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "EVENT BREAKDOWN", ln=True)

    event_counts = df['event_type'].value_counts()
    pdf.set_font("Helvetica", "", 11)
    for event_type, count in event_counts.items():
        severity = df[df['event_type'] == event_type]['severity'].iloc[0]
        pdf.cell(0, 7, f"  {event_type:<25} {count:>3} occurrences   (severity: +{severity} each)", ln=True)

    pdf.ln(5)

    # Bar Chart
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "EVENT FREQUENCY CHART", ln=True)
    pdf.image(bar_chart_path, x=10, w=190)

    # New page for timeline
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "EVENT TIMELINE", ln=True)
    pdf.image(timeline_path, x=10, w=190)

    pdf.ln(5)

    # Written Summary
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "AUTOMATED SUMMARY", ln=True)
    pdf.set_font("Helvetica", "", 10)

    # Generate human readable summary
    summary_lines = []
    summary_lines.append(
        f"Student {student_name} was monitored for {duration_minutes:.1f} minutes. "
        f"A total of {total_events} suspicious events were recorded, "
        f"resulting in a cumulative suspicion score of {total_score}."
    )

    if "PHONE_DETECTED" in event_counts:
        summary_lines.append(
            f"A mobile phone was detected {event_counts['PHONE_DETECTED']} time(s) during the session. "
            f"This is a high severity violation."
        )

    if "MULTIPLE_FACES" in event_counts:
        summary_lines.append(
            f"Multiple faces were detected {event_counts['MULTIPLE_FACES']} time(s), "
            f"suggesting possible assistance from another person."
        )

    if "LOOKING_AWAY" in event_counts:
        summary_lines.append(
            f"The student looked away from the screen {event_counts['LOOKING_AWAY']} time(s). "
            f"Frequent gaze deviation may indicate reference to external materials."
        )

    summary_lines.append(
        f"Based on the analysis, this session is classified as {risk_level} RISK."
    )

    for line in summary_lines:
        pdf.multi_cell(0, 7, line)
        pdf.ln(2)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, f"Generated by AI Proctoring System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             align='C')

    # Save PDF
    pdf.output(pdf_path)
    print(f"\nPDF report saved: {pdf_path}")

    return pdf_path


# ============================
# TEST - run directly to generate report from latest session
# ============================
if __name__ == "__main__":
    import glob

    # Find the most recent session CSV
    csv_files = glob.glob("logs/session_*.csv")

    if not csv_files:
        print("No session logs found! Run main.py first.")
    else:
        # Get most recent file
        latest_csv = max(csv_files, key=os.path.getmtime)
        print(f"Generating report for: {latest_csv}")

        student_name = input("Enter student name for report: ")
        pdf_path = generate_pdf_report(latest_csv, student_name)
        print(f"\nReport ready: {pdf_path}")