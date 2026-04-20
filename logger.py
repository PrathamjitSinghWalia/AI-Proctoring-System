import csv
import os
from datetime import datetime

# ============================
# EVENT SEVERITY WEIGHTS
# These define how suspicious each event is
# Higher number = more suspicious
# ============================
EVENT_WEIGHTS = {
    "LOOKING_AWAY": 1,
    
    "MULTIPLE_FACES": 3,
    "NO_FACE": 2,
    "PHONE_DETECTED": 5,
    "LAPTOP_DETECTED": 4,
    "BOOK_DETECTED": 3,
}

class SessionLogger:
    """
    Handles all logging and scoring for a proctoring session.
    This is a CLASS - our first one in this project!
    A class is a blueprint that bundles related data and functions together.
    """

    def __init__(self, student_name="Student", session_id=None):
        """
        __init__ is called automatically when you create a SessionLogger object.
        It sets up everything needed for a new session.
        """
        self.student_name = student_name

        # Generate unique session ID using current timestamp
        # This ensures each session has a unique filename
        if session_id is None:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.session_id = session_id

        # Session start time
        self.start_time = datetime.now()

        # Suspicion score starts at 0
        self.suspicion_score = 0

        # List to store all events in memory
        self.events = []

        # Create logs folder if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # CSV file path for this session
        self.csv_path = f"logs/session_{self.session_id}.csv"

        # Create CSV file with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "event_type",
                "severity",
                "details",
                "cumulative_score"
            ])

        print(f"Session started: {self.session_id}")
        print(f"Logging to: {self.csv_path}")


    def log_event(self, event_type, details=""):
        """
        Log a single suspicious event.
        Called every time something suspicious is detected.
        """
        # Only log known event types
        if event_type not in EVENT_WEIGHTS:
            return

        # Get severity weight for this event
        severity = EVENT_WEIGHTS[event_type]

        # Add to suspicion score
        self.suspicion_score += severity

        # Current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store event in memory list
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "cumulative_score": self.suspicion_score
        }
        self.events.append(event)

        # Write to CSV file immediately
        # This ensures data is saved even if program crashes
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                event_type,
                severity,
                details,
                self.suspicion_score
            ])


    def get_score(self):
        """Return current suspicion score"""
        return self.suspicion_score


    def get_summary(self):
        """
        Returns a summary of the session.
        Counts how many times each event type occurred.
        """
        summary = {event_type: 0 for event_type in EVENT_WEIGHTS}

        for event in self.events:
            event_type = event["event_type"]
            if event_type in summary:
                summary[event_type] += 1

        # Calculate session duration
        duration = datetime.now() - self.start_time
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)

        return {
            "student_name": self.student_name,
            "session_id": self.session_id,
            "duration": f"{minutes}m {seconds}s",
            "total_score": self.suspicion_score,
            "event_counts": summary
        }


    def end_session(self):
        """Call this when exam ends to print final summary"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Student: {summary['student_name']}")
        print(f"Duration: {summary['duration']}")
        print(f"Total Suspicion Score: {summary['total_score']}")
        print("\nEvent Breakdown:")
        for event_type, count in summary['event_counts'].items():
            if count > 0:
                print(f"  {event_type}: {count} times")
        print("="*50)
        print(f"Full log saved to: {self.csv_path}")