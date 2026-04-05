import logging
import functools
import time
import traceback
from typing import Callable, Any, Optional
from flask import has_request_context, request


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def log_call(logger: Optional[logging.Logger] = None, log_args: bool = False, log_result: bool = False) -> Callable:
    """
    Decorator to log function calls with entry/exit messages.

    Args:
        logger: Logger instance (if None, creates one from function module)
        log_args: Whether to log function arguments
        log_result: Whether to log function return value

    Returns:
        Decorated function

    Example:
        @log_call(log_args=True, log_result=True)
        def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = _get_request_id()
            func_name = func.__qualname__

            log_msg = f"[{request_id}] → {func_name}"
            if log_args:
                log_msg += f" | args={args}, kwargs={kwargs}"

            logger.debug(log_msg)

            try:
                result = func(*args, **kwargs)
                exit_msg = f"[{request_id}] ← {func_name}"
                if log_result:
                    exit_msg += f" | result={result}"
                logger.debug(exit_msg)
                return result
            except Exception as e:
                logger.error(f"[{request_id}] ✗ {func_name} | error={str(e)}")
                raise

        return wrapper
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None, threshold_ms: float = 0) -> Callable:
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (if None, creates one from function module)
        threshold_ms: Only log if execution time exceeds this threshold (in milliseconds)

    Returns:
        Decorated function

    Example:
        @log_execution_time(threshold_ms=100)
        def expensive_operation():
            pass
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = _get_request_id()
            func_name = func.__qualname__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                if elapsed_ms >= threshold_ms:
                    logger.info(f"[{request_id}] {func_name} took {elapsed_ms:.2f}ms")

                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(f"[{request_id}] {func_name} failed after {elapsed_ms:.2f}ms | {str(e)}")
                raise

        return wrapper
    return decorator


def log_errors(logger: Optional[logging.Logger] = None, include_traceback: bool = True) -> Callable:
    """
    Decorator to log exceptions with optional traceback.

    Args:
        logger: Logger instance (if None, creates one from function module)
        include_traceback: Whether to include full traceback in error logs

    Returns:
        Decorated function

    Example:
        @log_errors(include_traceback=True)
        def risky_operation():
            pass
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = _get_request_id()
            func_name = func.__qualname__

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"[{request_id}] Exception in {func_name}: {type(e).__name__}: {str(e)}"

                if include_traceback:
                    logger.error(error_msg + "\n" + traceback.format_exc())
                else:
                    logger.error(error_msg)
                raise

        return wrapper
    return decorator


def log_database_call(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator specifically for database operations.
    Logs query execution time and row counts.

    Args:
        logger: Logger instance (if None, creates one from function module)

    Returns:
        Decorated function

    Example:
        @log_database_call()
        def get_user_by_id(user_id):
            pass
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = _get_request_id()
            func_name = func.__qualname__
            start_time = time.time()

            logger.debug(f"[{request_id}] DB: {func_name} starting")

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                row_info = ""
                if isinstance(result, (list, tuple)):
                    row_info = f" | {len(result)} rows"

                logger.info(f"[{request_id}] DB: {func_name} completed in {elapsed_ms:.2f}ms{row_info}")
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(f"[{request_id}] DB: {func_name} failed after {elapsed_ms:.2f}ms | {str(e)}")
                raise

        return wrapper
    return decorator


def log_service_call(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator for service layer operations.
    Logs entry, exit, and execution time.

    Args:
        logger: Logger instance (if None, creates one from function module)

    Returns:
        Decorated function

    Example:
        @log_service_call()
        def process_query(query: str):
            pass
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = _get_request_id()
            func_name = func.__qualname__
            start_time = time.time()

            logger.debug(f"[{request_id}] SERVICE: {func_name} starting")

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"[{request_id}] SERVICE: {func_name} completed in {elapsed_ms:.2f}ms")
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(f"[{request_id}] SERVICE: {func_name} failed after {elapsed_ms:.2f}ms | {str(e)}")
                raise

        return wrapper
    return decorator


def log_api_call(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator for API endpoints.
    Logs HTTP method, path, and response status.

    Args:
        logger: Logger instance (if None, creates one from function module)

    Returns:
        Decorated function

    Example:
        @app.route('/api/endpoint')
        @log_api_call()
        def api_handler():
            pass
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if has_request_context():
                request_id = _get_request_id()
                logger.debug(
                    f"[{request_id}] API: {request.method} {request.path} "
                    f"from {request.remote_addr}"
                )

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.info(
                        f"[{request_id}] API: {request.method} {request.path} "
                        f"completed in {elapsed_ms:.2f}ms"
                    )
                    return result
                except Exception as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.error(
                        f"[{request_id}] API: {request.method} {request.path} "
                        f"failed after {elapsed_ms:.2f}ms | {str(e)}"
                    )
                    raise
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator


def _get_request_id() -> str:
    """Get request ID from Flask context or generate one."""
    if has_request_context():
        return request.headers.get('X-Request-ID', request.remote_addr)
    return "BACKGROUND"
