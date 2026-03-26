
def filter_even_squares(numbers):
    result = []
    for n in numbers:
        if n % 2 == 0:
            result.append(n * n)
    return result

#prepisi ovo samo sa generatorom
def gen_filter_even_squares(numbers):
    