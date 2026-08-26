# -*- coding: utf-8 -*-
from openpyxl import Workbook, load_workbook
from datetime import datetime
from sklearn.utils.validation import check_is_fitted
from pandas import ExcelWriter

#interns functions
from ..functions.utils import is_dataframe, is_series, is_namedtuple, is_dict

def save(obj, 
         excel_writer=None, 
         engine='openpyxl',  
         verbose=True, 
         **kwargs):
    """
    Print results for general factor analysis model in an Excel sheet

    Parameters
    ----------
    obj : class
        A fitted factor analysis model.

    excel_writer: path-like, file-like
        File path or existing ExcelWriter.

    verbose: bool, default = True
        If True, print message.

    **kwargs: dict
        Additionals parameters for `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html>`_.

    Returns
    -------
    NoneType    
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #all possible attributes
    all_attr = ["call_","col_","col_sup_","eig_","freq_","freq_sup_",
                "group_","group_sup_","ind_","ind_sup_","partial_axes_",
                "quali_var_","quali_var_sup_","quanti_var_","quanti_var_sup_",
                "quali_sup_","quanti_sup_","row_","row_sup_","var_"]
    
    if is_namedtuple(obj):
        name, obj_attr = obj.__class__.__name__.replace("Result",""), list(obj._asdict().keys())
    else:
        check_is_fitted(obj)
        #model attribute
        name, obj_attr = obj.__class__.__name__, [s for s in dir(obj) if s[0].isalpha() and s.endswith("_")]
    #find intersection
    attr = [x for x in obj_attr if x in all_attr]

    if excel_writer is None:
        excel_writer = "{}_{}.xlsx".format(name,datetime.today().strftime('%Y-%m-%d'))

    #initialize empty dictionary
    stats = {}
    for i in attr:
        vals = getattr(obj, i)
        #convert to pd.DataFrame if pd.Series
        if is_series(vals): stats = {**stats, **{i.replace("_","") : vals.to_frame()}}
        if is_dataframe(vals): stats = {**stats, **{i.replace("_","") : vals}}
        if is_dict(vals):
            for j in list(vals.keys()):
                if is_series(vals[j]): stats = {**stats, **{f"{i}{j}" : vals[j].to_frame()}}
                if is_dataframe(vals[j]): stats = {**stats, **{f"{i}{j}" : vals[j]}}
        if is_namedtuple(vals):
            vals_i = vals._asdict()
            for j in list(vals_i.keys()):
                if is_series(vals_i[j]): stats = {**stats, **{f"{i}{j}" : vals_i[j].to_frame()}}
                if is_dataframe(vals_i[j]): stats = {**stats, **{f"{i}{j}" : vals_i[j]}}
                if is_namedtuple(vals_i[j]):
                    vals_ij = vals_i[j]._asdict()
                    for k in list(vals_ij.keys()):
                        if is_series(vals_ij[k]): stats = {**stats, **{f"{i}{j}_{k}" : vals_ij[k]}}
                        if is_dataframe(vals_ij[k]): stats = {**stats, **{f"{i}{j}_{k}" : vals_ij[k]}}

    #create empty xlsx file
    wb = Workbook()
    wb.save(filename=excel_writer)
    with ExcelWriter(excel_writer, engine=engine, mode='a', if_sheet_exists='replace') as writer:
        for i in list(stats.keys()): stats[i].to_excel(writer, sheet_name=i, **kwargs)
    wb = load_workbook(excel_writer)
    wb.remove(wb['Sheet'])
    wb.save(excel_writer)
    if verbose: print("All the results are in the file {}".format(excel_writer))