import subprocess
import time
import os
import signal
import sys

def run_bot(mode, duration=30):
    print(f"\n--- Testing mode: {mode} ---")
    cmd = [
        sys.executable, 
        "main.py", 
        "--mode", mode, 
        "--config", "config.ini",
        "--assets", "EURUSD",
        "--debug"
    ]
    
    # We will use mock environment variables to trigger simulation if needed
    # or rely on the mock objects we built in integration tests.
    # However, for a "real" operational test, we should run the script and see if it crashes.
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        # Let it run for a while
        for _ in range(duration):
            if process.poll() is not None:
                break
            time.sleep(1)
            
        # Send termination signal to stop gracefully
        if process.poll() is None:
            print(f"Stopping bot in {mode} mode...")
            if sys.platform == "win32":
                # For Windows, we use taskkill to signal termination
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
            else:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
            
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"Bot in {mode} mode timed out and was killed.")
    
    stdout, stderr = process.communicate()
    
    # Print a summary of logs if it failed
    if process.returncode != 0:
        print("\n--- STDOUT (Last 20 lines) ---")
        print("\n".join(stdout.splitlines()[-20:]))
        print("\n--- STDERR ---")
        print(stderr)
        return False
    
    print(f"Mode {mode} completed successfully.")
    return True

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    
    modes = ["download", "learning", "test"] # We avoid 'real' unless we are sure it's safe
    
    success = True
    for mode in modes:
        if not run_bot(mode):
            success = False
            break
            
    if success:
        print("\nAll operational tests passed!")
        sys.exit(0)
    else:
        print("\nOperational tests failed.")
        sys.exit(1)
