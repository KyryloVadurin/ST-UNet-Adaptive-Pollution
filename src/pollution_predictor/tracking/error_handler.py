import traceback
from functools import wraps

def safe_execution(func):
    """
    Decorator for robust error handling in the AI pipeline.
    Captures stack traces and re-raises exceptions to notify high-level orchestrators.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Execute the wrapped function logic
            return func(*args, **kwargs)
        except Exception as e:
            # Log critical failure details and stack trace
            print(f"\n[CRITICAL ERROR] in {func.__name__}: {str(e)}")
            traceback.print_exc()
            
            # Re-raise the exception to prevent silent failures in training loops
            raise e
    return wrapper