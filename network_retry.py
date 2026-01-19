"""
Network retry utility for handling temporary network failures.
Provides a retry wrapper that can be used for uploads and API calls.
"""

import time
import functools
from typing import Callable, Any, Optional


def with_network_retry(
    max_attempts: int = 15,
    retry_delay: int = 60,
    log_callback: Optional[Callable[[str], None]] = None
):
    """
    Decorator that adds retry logic to network operations.
    
    Args:
        max_attempts: Maximum number of attempts (default: 15)
        retry_delay: Delay in seconds between retries (default: 60)
        log_callback: Optional callback function for logging messages
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @with_network_retry(max_attempts=15, retry_delay=60)
        def upload_file():
            # Upload logic here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract log_callback from kwargs if provided, otherwise use the one from decorator
            func_log_callback = kwargs.get('log_callback') or log_callback
            
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # If the function returns a boolean (success/failure indicator)
                    if isinstance(result, bool):
                        if result:
                            # Success - return immediately
                            if attempt > 1 and func_log_callback:
                                func_log_callback(f"✅ Operation succeeded on attempt {attempt}")
                            return result
                        else:
                            # Function returned False - treat as failure
                            if attempt < max_attempts:
                                if func_log_callback:
                                    func_log_callback(f"⚠️ Attempt {attempt}/{max_attempts} failed. Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                continue
                            else:
                                if func_log_callback:
                                    func_log_callback(f"❌ All {max_attempts} attempts failed. Giving up.")
                                return False
                    else:
                        # For non-boolean returns, assume success if no exception was raised
                        if attempt > 1 and func_log_callback:
                            func_log_callback(f"✅ Operation succeeded on attempt {attempt}")
                        return result
                        
                except Exception as e:
                    # Exception occurred - retry if we have attempts left
                    if func_log_callback:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if attempt < max_attempts:
                            func_log_callback(f"⚠️ Attempt {attempt}/{max_attempts} failed with {error_type}: {error_msg}")
                            func_log_callback(f"   Retrying in {retry_delay} seconds...")
                        else:
                            func_log_callback(f"❌ All {max_attempts} attempts failed. Last error: {error_type}: {error_msg}")
                    
                    if attempt >= max_attempts:
                        # Re-raise the exception on the last attempt
                        raise
                    
                    time.sleep(retry_delay)
            
            # Should not reach here, but return False as fallback
            return False
            
        return wrapper
    return decorator


def retry_operation(
    operation: Callable,
    max_attempts: int = 15,
    retry_delay: int = 60,
    log_callback: Optional[Callable[[str], None]] = None,
    operation_name: str = "Operation"
) -> Any:
    """
    Function wrapper for adding retry logic to a callable operation.
    Useful when you can't use a decorator.
    
    Args:
        operation: The callable to execute with retry logic
        max_attempts: Maximum number of attempts (default: 15)
        retry_delay: Delay in seconds between retries (default: 60)
        log_callback: Optional callback function for logging messages
        operation_name: Name of the operation for logging purposes
        
    Returns:
        The result of the operation
        
    Example:
        result = retry_operation(
            lambda: upload_file(file_path),
            max_attempts=15,
            retry_delay=60,
            log_callback=print,
            operation_name="File upload"
        )
    """
    for attempt in range(1, max_attempts + 1):
        try:
            result = operation()
            
            # If the function returns a boolean (success/failure indicator)
            if isinstance(result, bool):
                if result:
                    # Success - return immediately
                    if attempt > 1 and log_callback:
                        log_callback(f"✅ {operation_name} succeeded on attempt {attempt}")
                    return result
                else:
                    # Function returned False - treat as failure
                    if log_callback:
                        if attempt < max_attempts:
                            log_callback(f"⚠️ {operation_name} attempt {attempt}/{max_attempts} failed. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            log_callback(f"❌ {operation_name}: All {max_attempts} attempts failed. Giving up.")
                    return False
            else:
                # For non-boolean returns, assume success if no exception was raised
                if attempt > 1 and log_callback:
                    log_callback(f"✅ {operation_name} succeeded on attempt {attempt}")
                return result
                
        except Exception as e:
            # Exception occurred - retry if we have attempts left
            if log_callback:
                error_type = type(e).__name__
                error_msg = str(e)
                if attempt < max_attempts:
                    log_callback(f"⚠️ {operation_name} attempt {attempt}/{max_attempts} failed with {error_type}: {error_msg}")
                    log_callback(f"   Retrying in {retry_delay} seconds...")
                else:
                    log_callback(f"❌ {operation_name}: All {max_attempts} attempts failed. Last error: {error_type}: {error_msg}")
            
            if attempt >= max_attempts:
                # Re-raise the exception on the last attempt
                raise
            
            time.sleep(retry_delay)
    
    # Should not reach here, but return False as fallback
    return False

