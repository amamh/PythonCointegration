import datetime
from pprint import pprint
import pandas as pd
import json
import decorator
from functools import wraps
import inspect
from os import path


# TODO: extend str type
# class str(str):
#     def myformat(self, format, *args):
#         def typebased(input):
#             if type(input) is pd.DataFrame:
#                 return input.to_string()
#             else:
#                 return input

#         myargs = [typebased(a) for a in *args]
#         return format.format(myargs)

def myformat(format, *args):
    def typebased(input):
        if type(input) is pd.DataFrame:
            return input.to_string()
        else:
            return input

    myargs = [typebased(a) for a in args]
    return format.format(*myargs)


def persist_to_file(file_name='cache.dat'):
    def decorator(function):
        try:
            cache = json.load(open(file_name, 'r'))
        except (IOError, ValueError):
            cache = {}

        for k in cache.keys():
            try:
                # is this a DataFrame?
                val = pd.read_json(cache[k])
                cache[k] = val
            except:
                pass

        # TODO: Handle multiple params
        @wraps(function)
        def wrapper(param):
            if param not in cache:
                val = function(param)
                cache[param] = val
                if type(val) is pd.DataFrame: # needs special function:
                    cache[param] = val.to_json(date_format='iso')
                json.dump(cache, open(file_name, 'w'))
                return val
            return cache[param]
        return wrapper
    return decorator


def timeit(logfile="main.log"):
    def real_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            f = open(logfile, 'a')
            start = datetime.datetime.now()
            val = function(*args, **kwargs)
            end = datetime.datetime.now()
            ms = end - start
            f.write("TIMING: {0} took {1} ms\n".format(function.__name__, ms.microseconds/1000))
            f.close()
            return val

        return wrapper
    return real_decorator


def logit(logfile=None):
    (frame, filename, line_number, function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[1]
    if logfile is None:
        callername = path.split(filename)[1]
        logfile = str.join('.', callername.split('.')[:-1])+".log" if '.' in callername else callername+'.log'

    def real_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            f = open(logfile, 'a')
            val = function(*args, **kwargs)
            argnames = function.__code__.co_varnames

            f.write("\n\n")
            f.write("===+++ {0} DUMP +++===\n\n".format(function.__name__))
            f.write("=== Input\n\n")

            for i in range(len(args)):
                f.write(myformat("{0}:\n{1}\n", argnames[i], args[i]))

            f.write("\n")
            f.write("=== Output\n\n")

            # f.writelines("{0}:\n\n{1}".format(val.__name__, val))
            f.write(myformat("{0}", val))
            f.write("\n")

            f.write("===--- {0} DUMP ---===\n\n".format(function.__name__))
            f.close()
            return val

        return wrapper
    return real_decorator
