# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from sklearn.utils.validation import check_is_fitted

#intern function
from ..functions.method_desc import method_desc
from ..functions.utils import is_namedtuple

def summary(obj, 
            digits = 4, 
            nbelt = 10, 
            ncp = 3, 
            detailed=False, 
            to_markdown=False, 
            tablefmt = "simple", 
            **kwargs):
    """
    Printing summaries of general factor analysis model

    Parameters
    ----------
    obj : class
        A fitted factor analysis model.

    digits : int, default = 4
        The number of decimal printed.

    nbelt : int, default = 10
        The number of element.

    ncp : int, default = 3. 
        The number of components.

    detailed : bool, default = False
        To print detailed summaries.

    to_markdown : bool, default = False
        To print summaries in `markdown <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html>`_-friendly format. Requires the `tabulate <https://pypi.org/project/tabulate/>`_. package.

    tablefmt : str, default = "simple"
        The table format.

    **kwargs: Any
        Additionals parameters. These parameters will be passed to `tabulate <https://pypi.org/project/tabulate/>`_.

    Returns
    -------
    NoneType

    Examples
    --------
    >>> from scientisttools.datasets import children
    >>> from scientisttools import CA, summary
    >>> clf = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
    >>> clf.fit(children)
    CA(col_sup=(5,6,7),row_sup=(14,15,16,17),sup_var=8)
    >>> summary(clf)
    """
    # define extract fn
    def func_extract(dicts, 
                     digits=4, 
                     ncp=3, 
                     nbelt=10, 
                     to_markdown=False, 
                     tablefmt="simple", 
                     **kwargs):
        """
        Extract summaries in attributes objects

        Parameters
        ----------
        dicts : dict:
            A dictionnary containing informations.

        digits : int, default = 4
            The number of decimal printed.

        nbelt : int, default = 10
            The number of element.

        ncp : int, default = 3. 
            The number of components.

        to_markdown : bool, default = False
            To print summaries in `markdown <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html>`_-friendly format. Requires the `tabulate <https://pypi.org/project/tabulate/>`_. package.

        tablefmt : str, default = "github"
            The table format.

        **kwargs : Any
            Additionals parameters. These parameters will be passed to `tabulate <https://pypi.org/project/tabulate/>`_.

        Returns
        -------
        infos: DataFrame
            Summay informations
        """
        names, infos = list(dicts.keys()), DataFrame().astype("float")
        for i in range(ncp):
            if "coord" in names: 
                infos = concat((infos, dicts["coord"].iloc[:,i]),axis=1)
            if "vtest" in names: 
                infos = concat((infos, dicts["vtest"].iloc[:,i].to_frame("vtest")),axis=1)
            if "cos2" in names: 
                infos = concat((infos, dicts["cos2"].iloc[:,i].to_frame("cos2")),axis=1)
            if "contrib" in names: 
                infos = concat((infos, dicts["contrib"].iloc[:,i].to_frame("ctr")),axis=1)
        infos = infos.head(nbelt).round(decimals=digits)
        if to_markdown: 
            infos = infos.to_markdown(tablefmt=tablefmt,**kwargs)
        return infos

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if is_namedtuple(obj):
        name = obj.__class__.__name__.replace("Result","")
    else:
        check_is_fitted(obj)
        name = obj.__class__.__name__

    #set number of components and number of elements
    ncp, nbelt = min(ncp,obj.call_.ncp), min(nbelt,obj.call_.X.shape[0])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print(f"                     {method_desc(name=name)} - Results                     ")

    if obj.__class__.__name__ not in ("CANCORR","CCA","COIA","Procrustes"):
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #importance of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"eig_"):
            print("\nImportance of components:")
            eig = obj.eig_.iloc[:ncp,:].round(decimals=digits)
            if to_markdown: 
                eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
            print(eig)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #matrix of rotation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"rotmat_"):
            print("\nMatrix of rotation:")
            rotmat = obj.rotmat_.round(decimals=digits)
            if to_markdown: 
                rotmat = rotmat.to_markdown(tablefmt=tablefmt,**kwargs)
            print(rotmat)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for goups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "group_") and "coord" in list(obj.group_._fields):
            group_ = obj.group_._asdict()
            first = list(group_.keys())[0]
            text = f"\nGroups (the {nbelt} first):" if group_[first].shape[0] >= nbelt else "\nGroups:"
            print(text)
            infos = func_extract(dicts=group_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

            if "coord_sup" in list(obj.group_._fields):
                text = f"\nSupplementary groups (the {nbelt} first):" if group_[first].shape[0] >= nbelt else "\nSupplementary groups:"
                print(text)
                infos = DataFrame().astype("float")
                for i in range(ncp):
                    infos = concat((infos, group_["coord_sup"].iloc[:,i]),axis=1)
                    if "cos2_sup" in list(obj.group_._fields):
                        infos = concat((infos, group_["cos2_sup"].iloc[:,i].to_frame("cos2")),axis=1)
                infos = infos.head(nbelt).round(decimals=digits)
                if to_markdown: 
                    infos = infos.to_markdown(tablefmt=tablefmt,**kwargs)
                print(infos)

            if "lambd" in list(group_.keys()):
                lambd = group_["lambd"]
                text = f"\nSpecific variances of groups (the {nbelt} first):" if lambd.shape[0] >= nbelt else "\nSpecific variances of groups:"
                print(text)
                lambd = lambd.iloc[:nbelt,:ncp].round(decimals=digits)
                if to_markdown: 
                    lambd = lambd.to_markdown(tablefmt=tablefmt,**kwargs)
                print(lambd)

            if "expl_var" in list(group_.keys()):
                expl_var = group_["expl_var"]
                text = f"\nPercentages of total variance recovered associated with each dimension (the {nbelt} first):" if expl_var.shape[0] >= nbelt else "\nPercentages of total variance recovered associated with each dimension:"
                print(text)
                expl_var = expl_var.iloc[:nbelt,:ncp].round(decimals=digits)
                if to_markdown: 
                    expl_var = expl_var.to_markdown(tablefmt=tablefmt,**kwargs)
                print(expl_var)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #resulys for individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"ind_") and "coord" in list(obj.ind_._fields):
            ind_ = obj.ind_._asdict()
            text = f"\nIndividuals (the {nbelt} first):" if obj.call_.X.shape[0] >= nbelt else "\nIndividuals:"
            print(text)
            infos = func_extract(dicts=ind_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"ind_sup_") and "coord" in list(obj.ind_sup_._fields):
            ind_sup_= obj.ind_sup_._asdict()
            text = f"\nSupplementary individuals (the {nbelt} first):" if len(obj.call_.ind_sup) >= nbelt else "\nSupplementary individuals:"
            print(text)
            infos = func_extract(dicts=ind_sup_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #resulys for rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"row_"):
            row_ = obj.row_._asdict()
            first = list(row_.keys())[0]
            text = f"\nRows (the {nbelt} first):" if row_[first].shape[0] >= nbelt else "\nRows:"
            print(text)
            infos = func_extract(dicts=row_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for supplementary rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"row_sup_"):
            row_sup_= obj.row_sup_._asdict()
            first = list(row_sup_.keys())[0]
            text = f"\nSupplementary rows (the {nbelt} first):" if row_sup_[first].shape[0] >= nbelt else "\nSupplementary rows:"
            print(text)
            infos = func_extract(dicts=row_sup_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #resulys for columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"col_"):
            col_ = obj.col_._asdict()
            first = list(col_.keys())[0]
            text = f"\nColumns (the {nbelt} first):" if col_[first].shape[0] >= nbelt else "\nColumns:"
            print(text)
            infos = func_extract(dicts=col_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for supplementary columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"col_sup_"):
            col_sup_= obj.col_sup_._asdict()
            first = list(col_sup_.keys())[0]
            text = f"\nSupplementary columns (the {nbelt} first):" if col_sup_[first].shape[0] >= nbelt else "\nSupplementary columns:"
            print(text)
            infos = func_extract(dicts=col_sup_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "quanti_var_"):
            quanti_var_ = obj.quanti_var_._asdict()
            first = list(quanti_var_.keys())[0]
            text = f"\nContinuous (the {nbelt} first):" if quanti_var_[first].shape[0] >= nbelt else "\nContinuous:"
            print(text)
            infos = func_extract(dicts=quanti_var_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for frequencies
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "freq_"):
            freq_ = obj.freq_._asdict()
            first = list(freq_.keys())[0]
            text = f"\nFrequencies (the {nbelt} first):" if freq_[first].shape[0] >= nbelt else "\nFrequencies:"
            print(text)
            infos = func_extract(dicts=freq_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "levels_"):
            levels_ = obj.levels_._asdict()
            first = list(levels_.keys())[0]
            text = f"\nCategories (the {nbelt} first):\n" if levels_[first].shape[0] >= nbelt else "\nCategories:"
            print(text)
            infos = func_extract(dicts=levels_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "quali_var_"):
            quali_var_ = obj.quali_var_._asdict()
            first = list(quali_var_.keys())[0]
            text = f"\nCategorical variables (the {nbelt} first):" if quali_var_[first].shape[0] >= nbelt else "\nCategorical variables:"
            print(text)
            infos = func_extract(dicts=quali_var_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for supplementary groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "group_sup_"):
            group_sup_ = obj.group_sup_._asdict()
            first = list(group_sup_.keys())[0]
            text = f"\nSupplementary groups (the {nbelt} first):" if group_[first].shape[0] >= nbelt else "\nSupplementary groups:"
            print(text)
            infos = func_extract(dicts=group_sup_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #esults for supplementary continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"quanti_var_sup_"):
            quanti_var_sup_ = obj.quanti_var_sup_._asdict()
            first = list(quanti_var_sup_.keys())[0]
            text = f"\nSupplementary continuous variables (the {nbelt} first):" if quanti_var_sup_[first].shape[0] >= nbelt else "\nSupplementary continuous variables:"
            print(text)
            infos = func_extract(dicts=quanti_var_sup_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #esults for supplementary frequencies
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"freq_sup_"):
            freq_sup_ = obj.freq_sup_._asdict()
            first = list(freq_sup_.keys())[0]
            text = f"\nSupplementary frequencies (the {nbelt} first):" if freq_sup_[first].shape[0] >= nbelt else "\nSupplementary frequencies:"
            print(text)
            infos = func_extract(dicts=freq_sup_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for supplementary levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "levels_sup_"):
            levels_sup_ = obj.levels_sup_._asdict()
            first = list(levels_sup_.keys())[0]
            text = f"\nSupplementary categories (the {nbelt} first):" if levels_sup_[first].shape[0] >= nbelt else "\nSupplementary categories:"
            print(text)
            infos = func_extract(dicts=levels_sup_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for supplementary qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "quali_var_sup_"):
            quali_var_sup_ = obj.quali_var_sup_._asdict()
            first = list(quali_var_sup_.keys())[0]
            text = f"\nSupplementary categorical variables (the {nbelt} first):" if quali_var_sup_[first].shape[0] >= nbelt else "\nSupplementary categorical variables:"
            print(text)
            infos = func_extract(dicts=quali_var_sup_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for instrumental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "iv_"):
            iv_ = obj.iv_._asdict()
            first = list(iv_.keys())[0]
            text = f"\nInstrumental variables (the {nbelt} first):" if iv_[first].shape[0] >= nbelt else "\nInstrumental variables:"
            print(text)
            infos = func_extract(dicts=iv_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)
    elif obj.__class__.__name__ == "CANCORR":
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Correlation among the original dataset
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if detailed:
            print(f"\nCorrelations Among the VAR Variables - {obj.call_.name_group[0]}:")
            xcorr = obj.corr_.xcorr.round(decimals=digits)
            if to_markdown: 
                xcorr = xcorr.to_markdown(tablefmt=tablefmt,**kwargs)
            print(xcorr)

            print(f"\nCorrelations Among the WITH Variables - {obj.call_.name_group[1]}:")
            ycorr = obj.corr_.ycorr.round(decimals=digits)
            if to_markdown: 
                ycorr = xcorr.to_markdown(tablefmt=tablefmt,**kwargs)
            print(ycorr)

            print(f"\nCorrelations Between the VAR Variables - {obj.call_.name_group[0]} and the WITH Variables - {obj.call_.name_group[1]}:")
            xycorr = obj.corr_.xycorr.round(decimals=digits)
            if to_markdown: 
                xycorr = xycorr.to_markdown(tablefmt=tablefmt,**kwargs)
            print(xycorr)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #importance of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("\nImportance of components:")
        eig = obj.eig_.iloc[:ncp,:].round(decimals=digits)
        if to_markdown: 
            eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
        print(eig)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Canonical Correlation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("\nCanonical Correlation:")
        cancorr = obj.cancorr_.round(decimals=digits)
        if to_markdown: 
            cancorr = cancorr.to_markdown(tablefmt=tablefmt,**kwargs)
        print(cancorr)

        if detailed:
            print("\nMultivariate Statistics:")
            manova_ = obj.manova_
            for i in range(len(manova_._fields)):
                print(f"{manova_[i].header}:")
                stats = manova_[i].statistic
                if to_markdown: 
                    stats = stats.to_markdown(tablefmt=tablefmt,**kwargs)
                print(stats)

        for i in range(2):
            print(f"\nRaw Canonical Coefficients for {obj.call_.name_group[i]}:")
            cancoef = obj.cancoef_[i]
            if to_markdown: 
                cancoef = cancoef.to_markdown(tablefmt=tablefmt,**kwargs)
            print(cancoef)

        for i in range(2):
            print(f"\nIndividuals coordinates for {obj.call_.name_group[i]}:")
            ind_coord = obj.ind_[i].head(nbelt)
            if to_markdown: 
                ind_coord = ind_coord.to_markdown(tablefmt=tablefmt,**kwargs)
            print(ind_coord)
        
        # Canonical Structure Correlations
        xquanti_var_coord, yquanti_var_coord = obj.quanti_var_
        print(f"\nCorrelations Between the {obj.call_.name_group[0]} and Their Canonical Variables")
        xxcoord = xquanti_var_coord.xscores.head(nbelt)
        if to_markdown: 
            xxcoord = xxcoord.to_markdown(tablefmt=tablefmt,**kwargs)
        print(xxcoord)

        print(f"\nCorrelations Between the {obj.call_.name_group[1]} and Their Canonical Variables")
        yycoord = yquanti_var_coord.yscores.head(nbelt)
        if to_markdown: 
            yycoord = yycoord.to_markdown(tablefmt=tablefmt,**kwargs)
        print(yycoord)

        print(f"\nCorrelations Between the {obj.call_.name_group[0]} and The Canonical Variables of the {obj.call_.name_group[1]}")
        xycoord = xquanti_var_coord.yscores.head(nbelt)
        if to_markdown:
            xycoord = xycoord.to_markdown(tablefmt=tablefmt,**kwargs)
        print(xycoord)

        print(f"\nCorrelations Between the {obj.call_.name_group[1]} and The Canonical Variables of the {obj.call_.name_group[0]}")
        yxcoord = yquanti_var_coord.xscores.head(nbelt)
        if to_markdown:
            yxcoord = yxcoord.to_markdown(tablefmt=tablefmt,**kwargs)
        print(yxcoord)

        if hasattr(obj, "ind_sup_"):
            for i in range(2):
                print(f"\nSupplementary individuals coordinates for {obj.call_.name_group[i]}:")
                ind_sup_coord = obj.ind_sup_[i].head(nbelt)
                if to_markdown: 
                    ind_sup_coord = ind_sup_coord.to_markdown(tablefmt=tablefmt,**kwargs)
                print(ind_sup_coord)
    elif obj.__class__.__name__ == "CCA":
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #importance of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("\nImportance of components:")
        for i, k in enumerate(obj.eig_._fields):
            print(f"{k}:")
            eig = obj.eig_[i].iloc[:ncp].to_frame().round(decimals=digits)
            if to_markdown: 
                eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
            print(eig)
            print("")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # coordinates for the rows
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print(f"Rows:")
        for i, k in enumerate(obj.eig_._fields):
            print(f"{k}:")
            coord = obj.row_.coord[i].iloc[:nbelt,:ncp]
            if to_markdown: 
                coord = coord.to_markdown(tablefmt=tablefmt,**kwargs)
            print(coord)
            print("")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # coordinates for the columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        text = f"\nColumns (the {nbelt} first):" if obj.col_.coord[0].shape[0] >= nbelt else "\nColumns:"
        print(text)
        for i, k in enumerate(obj.eig_._fields):
            print(f"{k}:")
            coord = obj.col_.coord[i].iloc[:nbelt,:ncp]
            if to_markdown: 
                coord = coord.to_markdown(tablefmt=tablefmt,**kwargs)
            print(coord)
            print("")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # coordinates for environmental variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        text = f"\nContinuous (the {nbelt} first):" if obj.quanti_var_.coord.shape[0] >= nbelt else "\nContinuous:"
        print(text)
        coord = obj.quanti_var_.coord.iloc[:nbelt,:ncp]
        if to_markdown: 
            coord = coord.to_markdown(tablefmt=tablefmt,**kwargs)
        print(coord)
        print("")
    elif obj.__class__.__name__ == "COIA":
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #importance of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("\nImportance of components:")
        eig = obj.eig_.iloc[:ncp,:].round(decimals=digits)
        if to_markdown: 
                eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
        print(eig)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for goups: coinertia and RV
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print(f"\nRV coefficients between groups:")
        group = obj.group_.RV.round(decimals=digits)
        if to_markdown:
                group = group.to_markdown(tablefmt=tablefmt,**kwargs)
        print(group)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #resulys for individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        text = f"\nIndividuals (the {nbelt} first):" if obj.call_.X.shape[0] >= nbelt else "\nIndividuals:"
        print(text)
        for i, g in enumerate(obj.call_.name_group):
            print(f"{g}:")
            infos = func_extract(dicts=obj.ind_[i]._asdict(),digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)
            print("")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for supplementary individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj,"ind_sup_"):
            text = f"\nSupplementary individuals (the {nbelt} first):" if len(obj.call_.ind_sup) >= nbelt else "\nSupplementary individuals:"
            print(text)
            for i, g in enumerate(obj.call_.name_group):
                print(f"{g}:")
                infos = func_extract(dicts=obj.ind_sup_[i]._asdict(),digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
                print(infos)
                print("")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "quanti_var_"):
            quanti_var_ = obj.quanti_var_._asdict()
            first = list(quanti_var_.keys())[0]
            text = f"\nContinuous (the {nbelt} first):" if quanti_var_[first].shape[0] >= nbelt else "\nContinuous:"
            print(text)
            infos = func_extract(dicts=quanti_var_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for frequencies
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "freq_"):
            freq_ = obj.freq_._asdict()
            first = list(freq_.keys())[0]
            text = f"\nFrequencies (the {nbelt} first):" if freq_[first].shape[0] >= nbelt else "\nFrequencies:"
            print(text)
            infos = func_extract(dicts=freq_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "levels_"):
            levels_ = obj.levels_._asdict()
            first = list(levels_.keys())[0]
            text = f"\nCategories (the {nbelt} first):\n" if levels_[first].shape[0] >= nbelt else "\nCategories:"
            print(text)
            infos = func_extract(dicts=levels_,digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
            print(infos)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for qualitative variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if hasattr(obj, "quali_var_"):
            text = f"\nCategorical variables (the {nbelt} first):" if obj.quali_var_.contrib.shape[0] >= nbelt else "\nCategorical variables:"
            print(text)
            for i, g in enumerate(obj.call_.name_group):
                print(f"{g}:")
                infos = func_extract(dicts=obj.quali_var_[i]._asdict(),digits=digits,ncp=ncp,nbelt=nbelt,to_markdown=to_markdown,tablefmt=tablefmt,**kwargs)
                print(infos)
                print("")
    elif obj.__class__.__name__ == "Procrustes":
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #importance of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("\nImportance of components:")
        eig = obj.eig_.iloc[:ncp,:].round(decimals=digits)
        if to_markdown: 
            eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
        print(eig)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        text = f"\nIndividuals (the {nbelt} first):" if obj.call_.X.shape[0] >= nbelt else "\nIndividuals:"
        print(text)
        for i, g in enumerate(obj.call_.name_group):
            print(f"{g}:")
            coord = obj.ind_[i].coord.iloc[:min(nbelt,obj.ind_[i].coord.shape[0]),:ncp]
            if to_markdown:
                coord = coord.to_markdown(tablefmt=tablefmt,**kwargs)
            print(coord)
            print("")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #results for continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("\nContinuous:")
        for i, g in enumerate(obj.call_.name_group):
            print(f"{g}:")
            coord = obj.quanti_var_[i].coord.iloc[:min(nbelt,obj.quanti_var_[i].coord.shape[0]),:ncp]
            if to_markdown: 
                coord = coord.to_markdown(tablefmt=tablefmt,**kwargs)
            print(coord)
            print("")




        

        



        

            





