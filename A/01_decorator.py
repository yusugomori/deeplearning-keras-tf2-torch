def decorator(fn):
    def wrapper(*args, **kwargs):
        print('decorator')
        fn(*args, **kwargs)
    return wrapper


@decorator
def main():
    print('main')


main()
