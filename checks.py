import numpy as np
import data_interface as iface

def chkBinTotal(chk_array, bincols, totcol, tol=0.0, labels=None):
    tmp = chk_array[bincols]
    tmp = iface.recToArray(tmp)
    tmp = chk_array[totcol] - np.sum(tmp,axis=1)
    tmp = np.absolute(tmp)
    if labels == None:
        retval = np.zeros(len(tmp))
        retval[tmp>tol] = 1
    else:
        retval = np.repeat(np.array(labels[0],dtype='|S50'),len(tmp))
        retval[tmp>tol] = labels[1]
    return retval

def chkRange(chk_array, minval, maxval, labels=None):
    if labels == None:
        retval = np.zeros(len(chk_array))
        retval[chk_array<minval] = -1
        retval[chk_array>maxval] = 1
    else:
        if (isinstance(minval,float) or isinstance(minval,int)) and (isinstance(maxval,float) or isinstance(maxval,int)):
            retval = np.repeat(np.array(labels[1] + ' [' + str(minval) + ', ' + str(maxval) + ']',dtype='|S50'),len(chk_array))
        else:
            retval = np.repeat(np.array(labels[1],dtype='|S50'),len(chk_array))
        retval[chk_array<minval] = labels[0]
        retval[chk_array>maxval] = labels[2]
    return retval

def pctChange(chk_array,basecol,altcol):
    return (chk_array[altcol][chk_array[basecol]>0]-chk_array[basecol][chk_array[basecol]>0])*100/chk_array[basecol][chk_array[basecol]>0]


