import numpy as np
import data_interface as iface

def chkBinTotal(chk_array, bincols, totcol, tol=0.0):
    tmp = chk_array[bincols]
    tmp = iface.recToArray(tmp)
    tmp = chk_array[totcol] - np.sum(tmp,axis=1)
    tmp = np.absolute(tmp)
    retval = np.zeros(len(tmp))
    retval[tmp>tol] = 1
    return retval

def chkRange(chk_array, minval, maxval):
    retval = np.zeros(len(chk_array))
    retval[chk_array<minval] = -1
    retval[chk_array>maxval] = 1
    return retval

def pctChange(chk_array,basecol,altcol):
    return (chk_array[altcol][chk_array[basecol]>0]-chk_array[basecol][chk_array[basecol]>0])*100/chk_array[basecol][chk_array[basecol]>0]


