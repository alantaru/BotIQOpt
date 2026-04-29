import logging
from datetime import datetime
import json
import os

class ErrorTracker:
    """Class to track and manage errors in the application"""
    
    def __init__(self):
        self.error_log = []
        self.error_file = "error_log.json"
        self.max_errors = 1000  # Maximum number of errors to keep in memory
        
        # Create error log file if it doesn't exist
        if not os.path.exists(self.error_file):
            with open(self.error_file, 'w') as f:
                json.dump([], f)

    def log_error(self, error_type: str, message: str):
        """Log an error with timestamp and details"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': message
        }
        
        # Add to in-memory log
        self.error_log.append(error_entry)
        
        # Maintain max size
        if len(self.error_log) > self.max_errors:
            self.error_log.pop(0)
            
        # Append to file
        try:
            with open(self.error_file, 'r+') as f:
                existing_errors = json.load(f)
                existing_errors.append(error_entry)
                f.seek(0)
                json.dump(existing_errors, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to write error to file: {str(e)}")

    def get_recent_errors(self, count: int = 10):
        """Get most recent errors"""
        return self.error_log[-count:]

    def clear_errors(self):
        """Clear all error logs"""
        self.error_log = []
        try:
            with open(self.error_file, 'w') as f:
                json.dump([], f)
        except Exception as e:
            logging.error(f"Failed to clear error log: {str(e)}")

    def analyze_errors(self):
        """Analyze error patterns and generate report"""
        error_counts = {}
        for error in self.error_log:
            error_type = error['type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
        return {
            'total_errors': len(self.error_log),
            'error_counts': error_counts,
            'most_common_error': max(error_counts, key=error_counts.get) if error_counts else None
        }

    def get_error_stats(self):
        """Get error statistics"""
        if not self.error_log:
            return {}
            
        first_error = self.error_log[0]['timestamp']
        last_error = self.error_log[-1]['timestamp']
        
        time_diff = (datetime.fromisoformat(last_error) - 
                    datetime.fromisoformat(first_error)).total_seconds()
        
        return {
            'first_error': first_error,
            'last_error': last_error,
            'error_rate': len(self.error_log) / time_diff if time_diff > 0 else 0
        }
