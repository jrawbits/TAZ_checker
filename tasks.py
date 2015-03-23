from celery.task import task
import json
import datetime
from collections import namedtuple
import decimal
import math
import os
from django.core.serializers.json import DjangoJSONEncoder
import logging
logger=logging.getLogger(__name__)
from NMTK_apps.helpers.config_iterator import ConfigIterator
import numpy as np
import pandas as pd
import numpy.lib.recfunctions as recfuntions
import data_interface as iface
import checks

@task(ignore_result=False)
def performModel(input_files,
                 tool_config,
                 client,
                 subtool_name=False):
    '''
    This is where the majority of the work occurs related to this tool.
    The data_file is the input (probably spatial) data.
    The job_setup is a JSON object that contains the job configuration.
    The auth_key is the auth_key as passed by the NMTK server.  It needs
    to be used to pass things like status or results back to the NMTK server as 
    the job is processed.
    
    There's no model (yet) for this particular job - everything happens with the
    uploaded data file.  Later we can store/manage data for the job.
    
    The assumption here is that all relevant authorization/authentication has 
    happened prior to this job being run.
    '''
    logger=performModel.get_logger()
    logger.debug('Job input files are: %s', input_files)

    try:
        setup=json.loads(open(input_files['config'][0]).read()).get('analysis settings',{})
        # the name used for the File config object for this tool, so we can 
        # read the file from the config.
        file_namespace='data'
        failures=[]
        if (not isinstance(setup, (dict))):
            failures.append('Please provide analysis settings')
        else:
            logger.debug("Loaded config: %s", setup)
            file_iterator=ConfigIterator(input_files, file_namespace, setup)

            if subtool_name.lower()=='scn_compare':
                file_iterator2=ConfigIterator(input_files, 'data2', setup)
    except:
        logger.exception('Failed to parse config file or data file.')
        failures.append('Invalid job configuration')

    if failures:
        for f in failures:
            logger.debug("Failure: %s",f)
        client.updateResults(payload={'errors': failures },
                             failure=True)
    else:    
        client.updateStatus('Parameter & data file validation complete.')
        try:
            if subtool_name.lower()=='taz_checker':
                min = None
                max = None
                param_iterator=ConfigIterator(input_files, 'parameters', setup)
                if param_iterator.iterable:
                    raise Exception('Parameters cannot be iterable')
                else:
                    parameters=param_iterator.data
                    bintol = parameters.get('bintol_param')
                    regional_medinc = parameters.get('medinc_param')
                    regional_nonwrk_pct = parameters.get('nonwrk_param')
                    regional_chu5_pct = parameters.get('chu5_param')
                thresh_iterator=ConfigIterator(input_files, 'thresholds', setup)
                if thresh_iterator.iterable:
                    raise Exception('Thresholds cannot be iterable')
                else:
                    thresholds=thresh_iterator.data

                req_vars = ['households','population','avgincome','vehicles','employment',
                            'hh_size1','hh_size2','hh_size3','hh_size4','hh_size5','hh_size6',
                            'hh_wrk1','hh_wrk2','hh_wrk3','hh_wrk4',
                            'hh_inc1','hh_inc2','hh_inc3','hh_inc4',
                            'hh_lfcyc1','hh_lfcyc2','hh_lfcyc3',
                            'hh_veh1','hh_veh2','hh_veh3','hh_veh4',
                            'empcat1','empcat2','empcat3','empcat4','empcat5',
                            'pop_age1','pop_age2','pop_age3',
                            'enr_k_6','enr_7_12','enr_col']
                data_array = iface.toArray(file_iterator, req_vars)
                dt = np.dtype([(var,'float') for var in req_vars])
                data_array = np.array(data_array,dt)
                chkcols = ['chk_bin_hhsize','chk_bin_hhwrk','chk_bin_hhinc','chk_bin_hhveh','chk_bin_hhlfcyc','chk_bin_empcat',
                           'chk_avginc_ratio','chk_percapita_veh','chk_perwrk_veh','chk_pop_hhsize',
                           'regchk_empwrk','regchk_wrkage','regchk_schenr']
                for newcol in chkcols:
                    data_array = addCol(data_array,newcol,0.0)
                data_array = addCol(data_array,'temp',0.0)
                data_array = addCol(data_array,'workers',0.0)
                data_array = addCol(data_array,'pop_hhsize',0.0)
                data_array['workers'] = np.sum(iface.recToArray(data_array[['hh_wrk1','hh_wrk2','hh_wrk3','hh_wrk4']]),axis=1)
                data_array['pop_hhsize'] = data_array['hh_size1']+2*data_array['hh_size2']+3*data_array['hh_size3']+4*data_array['hh_size4']+5*data_array['hh_size5']+6*data_array['hh_size6']

                #TAZ level checks
                data_array['chk_bin_hhsize'] = checks.chkBinTotal(data_array,['hh_size1','hh_size2','hh_size3','hh_size4','hh_size5','hh_size6'],'households',bintol)
                data_array['chk_bin_hhwrk'] = checks.chkBinTotal(data_array,['hh_wrk1','hh_wrk2','hh_wrk3','hh_wrk4'],'households',bintol)
                data_array['chk_bin_hhinc'] = checks.chkBinTotal(data_array,['hh_inc1','hh_inc2','hh_inc3','hh_inc4'],'households',bintol)
                data_array['chk_bin_hhlfcyc'] = checks.chkBinTotal(data_array,['hh_lfcyc1','hh_lfcyc2','hh_lfcyc3'],'households',bintol)
                data_array['chk_bin_hhveh'] = checks.chkBinTotal(data_array,['hh_veh1','hh_veh2','hh_veh3','hh_veh4'],'households',bintol)
                data_array['chk_bin_empcat'] = checks.chkBinTotal(data_array,['empcat1','empcat2','empcat3','empcat4','empcat5'],'employment',bintol)
                data_array['chk_avginc_ratio'] = checks.chkRange(data_array['avgincome']/regional_medinc,thresholds.get('avginc_ratio_threshmin'),thresholds.get('avginc_ratio_threshmax'))
                data_array['chk_percapita_veh'] = checks.chkRange(data_array['vehicles']/data_array['population'],thresholds.get('percap_veh_threshmin'),thresholds.get('percap_veh_threshmax'))
                data_array['chk_perwrk_veh'] = checks.chkRange(data_array['vehicles']/data_array['workers'],thresholds.get('perwrk_veh_threshmin'),thresholds.get('perwrk_veh_threshmax'))
                data_array['chk_pop_hhsize'] = checks.chkRange(data_array['pop_hhsize'],0,data_array['population'])

                #Regional checks
                data_array['regchk_empwrk'] = checks.chkRange(np.array([np.sum(data_array['employment'])/np.sum(data_array['workers'])]),thresholds.get('empwrk_ratio_threshmin'),thresholds.get('empwrk_ratio_threshmax'))
                data_array['regchk_wrkage'] = checks.chkRange(np.array([np.sum(data_array['workers'])]),0,np.sum(data_array['pop_age2']+data_array['pop_age3'])*regional_nonwrk_pct/100)
                data_array['regchk_schenr'] = checks.chkRange(np.array([np.sum(data_array['enr_k_6']+data_array['enr_7_12'])]),0,np.sum(data_array['pop_age1'])*(1-regional_chu5_pct/100))

                result_cols = [setup['results'][col]['value'] for col in chkcols]
                iface.addResult(file_iterator,result_cols,data_array,chkcols)
                
            elif subtool_name.lower()=='scn_compare':
                thresh_iterator=ConfigIterator(input_files, 'thresholds', setup)
                if thresh_iterator.iterable:
                    raise Exception('Thresholds cannot be iterable')
                else:
                    thresholds=thresh_iterator.data

                req_vars = ['tazid','households','population','vehicles']
                data_array = iface.toArray(file_iterator, req_vars)
                dt = np.dtype([(var,'float') for var in req_vars])
                data_array = np.array(data_array,dt)
                req_vars = ['tazid2','households2','population2','vehicles2']
                data_array2 = iface.toArray(file_iterator2, req_vars)
                req_vars = ['tazid','households2','population2','vehicles2']
                dt = np.dtype([(var,'float') for var in req_vars])
                data_array2 = np.array(data_array2,dt)
                
                chkcols = ['pctchg_hh','pctchg_pop','pctchg_veh']
                for newcol in chkcols:
                    data_array = addCol(data_array,newcol,0.0)

                #print data_array.dtype.fields
                data_array = iface.leftjoin(data_array,data_array2,'tazid')

                data_array['pctchg_hh'][data_array['households']>0] = checks.pctChange(data_array,'households','households2')
                data_array['pctchg_pop'][data_array['population']>0] = checks.pctChange(data_array,'population','population2')
                data_array['pctchg_veh'][data_array['vehicles']>0] = checks.pctChange(data_array,'vehicles','vehicles2') 
                result_cols = [setup['results'][col]['value'] for col in chkcols]
                match_arr_col = 'tazid'
                match_result_col = setup['data'][match_arr_col]['value']
                iface.addMatchResult(file_iterator,result_cols,match_result_col,data_array,chkcols,match_arr_col)
          
        except Exception, e:
            # if anything goes wrong we'll send over a failure status.
            print e
            logger.exception('Job Failed with Exception!')
            client.updateResults(payload={'errors': [str(e),] },
                                 failure=True)
        # Since we are updating the data as we go along, we just need to return
        # the data with the new column (results) which contains the result of the 
        # model.
        #result_field=setup['results']['chk_bin_hhsize']['value']
        #units='HH size bins total check'
        client.updateResults(result_field=None,
                             units=None,
                             result_file='data',
                             files={'data': ('data.{0}'.format(file_iterator.extension),
                                             file_iterator.getDataFile(), 
                                             file_iterator.content_type)
                                    })
    for namespace, fileinfo in input_files.iteritems():
        os.unlink(fileinfo[0])        

def addCol(arr, col, factor):
    return recfuntions.append_fields(arr,col,np.ones(len(arr))*factor,usemask=False)
