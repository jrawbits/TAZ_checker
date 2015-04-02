
import numpy
import pandas as pd

def recToArray(arr):
    return arr.view((float,len(arr.dtype.names)))

def toArray(fiter, varlist):
    return [tuple([row[var] if var in row.keys() and row[var] is not None else 0 for var in varlist]) for row in fiter]

def updateRow(row, field, val):
    try:
        val = float(val)
    except ValueError:
        pass
    row['properties'][field] = val

def addResult(fiter, fields, value_array, array_fields):
    [updateRow(outrow, field, row[arr_field]) for outrow,row in zip(fiter.data_parsed['features'],value_array) for field,arr_field in zip(fields,array_fields)]

#This function converts a pandas dataframe to an ndarray
def df_to_ndarray(df):
    col_names = df.dtypes.index.values
    fmts = df.dtypes.values
    dt = numpy.dtype({'names':col_names,'formats':fmts})
    return numpy.array([tuple(x) for x in df.values],dt)

def leftjoin(xdata,ydata,byvars):
    xdata = pd.DataFrame(xdata)
    ydata = pd.DataFrame(ydata)
    xdata = pd.merge(xdata,ydata,on=byvars,how='left')
    return df_to_ndarray(xdata)

def addMatchResult(fiter, fields, match_field, value_array, array_fields, match_array_field):
    for outrow in fiter.data_parsed['features']:
        curr_taz = int(outrow['properties'][match_field])
        match_row = value_array[value_array[match_array_field]==curr_taz]
        outrow['properties'] = {}
        outrow['properties'][match_field] = curr_taz
        for f1,f2 in zip(fields,array_fields):
            outrow['properties'][f1] = float(match_row[f2])



