import functools


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin_time = time.perf_counter()
        res = func(*args, **kwargs)
        time_used = time.perf_counter() - begin_time
        print(f'{func.__name__}| {time_used} sec')
        return res
    return wrapper