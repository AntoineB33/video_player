import threading
import time

# This will store the latest user input
user_input = None
running = True

def input_thread():
    global user_input, running
    while running:
        text = input()  # This will block this thread, but not the main loop
        user_input = text
        if text.lower() == "quit":
            running = False

# Start the input listener thread
thread = threading.Thread(target=input_thread, daemon=True)
thread.start()

# Main loop
counter = 0
while running:
    counter += 1
    print(f"Loop iteration {counter}...")
    if user_input:
        print(f"You typed: {user_input}")
        user_input = None  # reset after processing
    time.sleep(1)

print("Program ended.")
