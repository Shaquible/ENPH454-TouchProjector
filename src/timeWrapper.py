import time

def time_restricted(func):
    # This will store the timestamp of the last call to the function
    last_called = [0]  # Using a list so it can be mutable inside the decorator

    def wrapper(*args, **kwargs):
        # Get the current time
        current_time = time.time()
        
        # Check if 10 seconds have passed since the last call
        if current_time - last_called[0] < 10:
            print("You must wait 10 seconds between calls.")
            return
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Update the last called time
        last_called[0] = current_time
        return result

    return wrapper



# Example usage
@time_restricted
def my_function():
    print("Function called!")



