

class PMFA:
    """
    Procustes Multiple Factor Analysis (PMFA)

    Performns Procustes Multiple Factor Analysis (PMFA) in the sense <>_


    Parameters
    ----------
    
    
    
    
    
    
    
    """

def predictMFA(
        obj,X
):
    """
    Predict projection for new individuals with Multiple Factor Analysis
   
    Performs the coordinates, squared cosinus, square distance to origin and partial coordinates of new individuals with Multiple Factor Analysis.

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`, :class:`~scientisttools.MFACT`.

    X : DataFrame of shape (n_rows, n_columns)
        Input data in which to look for variables with which to predict. ``X`` must contain columns with the same names as the original data.
    
    Returns
    -------
    result : predictMCAResult
        An object containing all the results for the new individuals, with the following attributes:
    
        coord : DataFrame of shape (n_rows, ncp) 
            The coordinates for the new individuals.
        cos2 : DataFrame of shape (n_rows, ncp) 
            The squared cosinus for the new individuals.
        dist2 : Series of shape (n_rows,)
            The squared distance to origin for the new individuals.

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import MFA, predictMFA
    >>> group_type = ["n"]+["s"]*5
    >>> mfa = MFA(ncp=5,group=group,group_type=group_type,col_w_mfa=None,name_group=group_name,num_group_sup=[0,5])
    >>> mfa.fit(wine)
    MFA()
    >>> #predict on active elements
    >>> X_active = wine.drop(columns=["Overall.quality","Typical","Label","Soil"])
    >>> predict = predictMFA(mfa, X=X_active)
    >>> predict.coord #coordinate of new individuals
    >>> predict.cos2 #squared cosinus of new individuals
    >>> predict.dist2 #squared distance to origin of new individuals
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class MFA, MFACT
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (obj.__class__.__name__ in ("MFA", "MFACT")):
        raise TypeError("'obj' must be an object of class MFA, MFACT")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #prediction input check
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X = predict_first_check(obj,X)
     
    #initialize
    Xcod = None 
    for g, cols in obj.call_.group.items():
        fa = obj.separate_analyses_[g]
        if fa.__class__.__name__ == "PCA":
            #check if all columns are numerics
            check_is_all_numeric_dtype(X[cols])
            Xcols = X[cols]
        elif fa.__class__.__name__ == "MCA":
            #check if all columns are object or category
            check_is_all_object_or_category_dtype(X[cols])
            #revaluated data
            Xcols_quali = revalue(X[cols])
            #test if X contains all active categorics
            new = [x for x in unique(Xcols_quali.values) if x not in fa.call_.dummies.columns]
            if len(new) > 0:
                raise ValueError("The following categories are not in the active dataset: "+",".join(new))
            #disjunctive table
            Xcols = tab_disjunctive(Xcols_quali,dummies_cols=fa.call_.dummies.columns)
        else:
            #split X
            split_Xcols = splitmix(X[cols])
            Xcols_quanti, Xcols_quali, n_cols_quanti, n_cols_quali = split_Xcols.quanti, split_Xcols.quali, split_Xcols.k1, split_Xcols.k2
            #initialization
            Xcols = None
            if n_cols_quanti > 0:
                if fa.call_.k1 != n_cols_quanti:
                    raise TypeError("The number of quantitative variables must be the same")
                Xcols = concat_empty(Xcols,Xcols_quanti,axis=1)
            #check
            if n_cols_quali > 0:
                if fa.call_.k2 != n_cols_quali:
                    raise TypeError("The number of qualitative variables must be the same")
                #revalue
                Xcols_quali = revalue(Xcols_quali)
                #test if X contains all active categorics
                new = [x for x in unique(Xcols_quali.values) if x not in fa.call_.dummies.columns]
                if len(new) > 0:
                    raise ValueError("The following categories are not in the active dataset: "+",".join(new))
                #disjunctive table
                dummies = tab_disjunctive(X=Xcols_quali,dummies_cols=fa.call_.dummies.columns)
                #concatenate
                Xcols = concat_empty(Xcols,dummies,axis=1)
        #concatenate
        Xcod = concat_empty(Xcod,Xcols,axis=1)

    #standardize according to MFA program and non normed PCA program
    Z = Xcod.sub(obj.call_.center,axis=1).div(obj.call_.scale,axis=1).sub(obj.call_.z_center,axis=1)
    #statistics for news individuals
    predict = predict_sup(X=Z,Y=obj.svd_.V[:,:obj.call_.ncp],weights=obj.call_.col_w,axis=0)
    #partiels coordinates for new individuals
    coord_partiel = None
    for g, cols in obj.call_.columns_dict.items():
        data_partiel = DataFrame(tile(obj.call_.z_center.values,(X.shape[0],1)),index=Z.index,columns=Z.columns)
        data_partiel[cols] = Z[cols]
        partial_coord = (len(list(obj.call_.group.keys()))*data_partiel.sub(obj.call_.z_center,axis=1)).mul(obj.call_.col_w,axis=1).dot(obj.svd_.V[:,:obj.call_.ncp])
        partial_coord.columns = MultiIndex.from_tuples([(g,c) for c in predict["coord"].columns])
        coord_partiel = concat_empty(coord_partiel,partial_coord,axis=1)
    #add to dictionary
    predict["coord_partiel"] = partial_coord
    #convert to namedtuple
    return namedtuple("predictMFAResult",predict.keys())(*predict.values())

def supvarMFA(
        obj,X,group,group_type,name_group=None
):
    """
    Supplementary variables in Multiple Factor Analysis (MFA)
    
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Multiple Factor Analysis (MFA)

    Parameters
    ----------
    obj : class
        An object of class :class:`~scientisttools.MFA`, :class:`~scientisttools.MFAcat`, :class:`~scientisttools.MFAmix`.

    X : Dataframe of shape (n_rows, n_columns)
        The supplementary groups

    group: list, tuple
        The number of variables in each supplementary group.

    group_type : the type of variables in each supplementary group. Possible values are : 
        * "c" or "s" for quantitative variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (quantitative and qualitative variables)

    name_group_sup : a list or a tuple containing the name of the supplementary groups (by default, None and the group are named Gr1, Gr2 and so on)

    Returns
    -------
    result : supvarMFAResult
        An object with the following attributes: 

        group : group
            An object containing the results of the supplementary groups, with the following attributes:
            
            coord : DataFrame of shape (n_groups_sup, n_groups_sup)
                Coordinates of the supplementary groups.
            cos2 : DataFrame of shape (n_groups_sup, n_groups_sup)
                Square cosinus of the supplementary groups.
            dist2 : Series of shape (n_groups_sup,)
                Square distance to origin of the supplementary groups.
            Lg : DataFrame of shape (n_groups_actifs + n_groups_sup, n_groups_actifs + n_groups_sup)
                Lg coefficients.
            RV : DataFrame of shape (n_groups_actifs + n_groups_sup, n_groups_actifs + n_groups_sup)
                RV coefficients
        
        partial_axes : partial_axes
            An object containing the results of the supplementary groups partial axes, with the following attributes:

            coord : DataFrame of shape (ncp, ncp)
                Coordinates of the supplementary partial axes.
            cos2 : DataFrame of shape (ncp, ncp)
                Square cosinus of the supplementary partial axes.

        quanti : quanti_sup
            An object containing the results of the supplementary quantitatives variables, with the following attributes:

            coord : DataFrame of shape (n_quanti_sup, ncp)
                Coordinates of the supplementary quantitatives variables.
            cos2 : DataFrame of shape (n_quanti_sup, ncp)
                Square cosinus of the supplementary quantitatives variables.
        
        quali : quali_sup
            An object containing the results of the supplementary qualitatives/categories variables, with the following attributes:

            coord : DataFrame of shape (n_levels_sup, ncp)
                Coordinates of the supplementary levels.
            cos2 : DataFrame of shape (n_levels_sup, ncp)
                square cosinus of the supplementary levels.
            vtest : DataFrame of shape (n_levels_sup, ncp)
                Value-test of the supplementary levels.
            dist : Series of shape (n_levels_sup,)
                Square distance to origin of the supplementary levels.
            eta2 : DataFrame of shape (n_quali_sup, ncp)
                Square correlation ratio of the supplementary qualitative variables.
            coord_partiel : DataFrame of shape (n_levels_sup, ncp)
                Partiel coordinates of the supplementary qualitatives variables

    Examples
    --------
    
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class MFA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "MFA":
        raise TypeError("'obj' must be an object of class MFA")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if group is None
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if group is None:
        raise ValueError("'group' must be assigned.")
    elif not (isinstance(group, (list,tuple,ndarray,Series))):
        raise ValueError("'group' must be a 1d array-like with the number of variables in each group")
    else:
        nb_elt_group = [int(x) for x in group]
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if group type in not None
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if group_type is None:
        raise ValueError("'group_type' must be assigned")
        
    if len(group) != len(group_type):
        raise TypeError("Not convenient group definition")
        
    # Assigned supplementary group name
    if name_group is None:
        group_name = ["Gr"+str(x+1) for x in range(len(nb_elt_group))]
    elif not isinstance(name_group,(list,tuple,ndarray,Series)):
        raise TypeError("'name_group' must be a 1d array-like with name of group")
    else:
        group_name = [x for x in name_group]
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if group name is an integer
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for i in range(len(group_name)):
        if isinstance(group_name[i],(int,float)):
            group_name[i] = "Gr"+str(i+1)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #assigned group name to label
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    group_dict = OrderedDict()
    j = 0
    for i in range(len(nb_elt_group)):
        X_group = X.iloc[:,(j):(j+nb_elt_group[i])]
        group_dict[group_name[i]] = list(X_group.columns)
        j += nb_elt_group[i]
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #separate general factor analysis for supplementary groups
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    model = OrderedDict()
    for g, cols in group_dict.items():
        if all(is_numeric_dtype(X[k]) for k in cols):
            if group_type[group_name.index(g)] not in ["c","s"]:
                raise TypeError("For quantitative variables 'group_type' should be one of 'c', 's'")
            standardize = False if group_type[group_name.index(g)] == "c" else True
            fa = PCA(standardize=standardize,ncp=obj.call_.ncp,row_w=obj.call_.row_w)   
        elif all(check_is_object_or_category_dtype(X[q]) for q in cols):
            if group_type[group_name.index(g)]!="n":
                raise TypeError("For qualitative variables 'group_type' should be 'n'")
            fa = MCA(ncp=obj.call_.ncp,row_w=obj.call_.row_w)     
        else:
            if group_type[group_name.index(g)] != "m":
                raise TypeError("For mixed variables 'group_type' should be 'm'")
            fa = FAMD(ncp=obj.call_.ncp,row_w=obj.call_.row_w)   
        model[g] = fa.fit(X[cols])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #extract elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #standardized data
    Z = concat((model[g].call_.Z for g in list(group_dict.keys())),axis=1)
    #columns
    columns_dict = {g : list(model[g].call_.Z.columns) for g in list(group_dict.keys())}
    #eigenvalues for all models
    eigvals = {g : model[g].eig_.iloc[:,0].values for g in list(model.keys())}
    #fisrt eigen values for all models
    first_eigvals = Series([eigvals[g][0] for g in list(model.keys())],index=list(model.keys()))
    #number of components in all models
    nb_comps = Series([model[g].call_.ncp for g in list(model.keys())],index=list(model.keys()))
    #number of columns in each groups
    nb_cols = Series([len(columns_dict[g]) for g in list(group_dict.keys())],index=list(group_dict.keys()))
    #columns weights in each groups 
    col_weights = concat((model[g].call_.col_w for g in list(group_dict.keys())),axis=0)
    #set variables weights for multiple factor analysis
    col_w_mfa = Series(array([x*(1/y) for x,y in zip(col_weights,array(list(chain(*[repeat(i,k) for i, k in zip(first_eigvals,nb_cols)]))))]),index=Z.columns,name="weight")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #group informations : Lg and RV coefficients
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Lg coefficients between supplementary groups
    Lg = DataFrame(index=list(group_dict.keys()),columns=list(group_dict.keys())).astype("float")
    for g1, cols1 in columns_dict.items():
        for g2, cols2 in columns_dict.items():
            Lg.loc[g1,g2] = function_lg(X=Z[cols1],Y=Z[cols2],xcol_w=col_w_mfa[cols1],ycol_w=col_w_mfa[cols2],row_w=obj.call_.row_w)
    
    #Lg coefficients between active and supplementary groups
    Lg2 = DataFrame(index=list(obj.call_.group.keys()),columns=list(obj.call_.group.keys())).astype("float")
    Lg = concat((Lg2,Lg),axis=1)
    for g1, cols1 in obj.call_.columns_dict.items():
        for g2, cols2 in columns_dict.items(): 
            Lg.loc[g1,g2] = function_lg(X=obj.call_.Z[cols1],Y=Z[cols2],xcol_w=obj.call_.col_w[cols1],ycol_w=col_w_mfa[cols2],row_w=obj.call_.row_w)
            Lg.loc[g2,g1] = Lg.loc[g1,g2]

    #add Lg coefficients for active groups
    for g1 in list(obj.call_.group.keys()):
        for g2 in list(obj.call_.group.keys()):
            Lg.loc[g1,g2] = obj.group_.Lg.loc[g1,g2]
            Lg.loc[g2,g1] = Lg.loc[g1,g2]

    #calculate RV coefficients
    RV = coeffRV(X=Lg)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #group informations : 
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #group squared distance to origin
    group_sqdisto = Series([sum(eigvals[g]**2)/first_eigvals[g]**2 for g in list(group_dict.keys())],index=list(group_dict.keys()),name="Sq. Dist.")
    #group factor coordinates
    group_coord = DataFrame(index=list(group_dict.keys()),columns=obj.ind_.coord.columns).astype("float")
    for g, cols in columns_dict.items():
        for i, d in enumerate(obj.ind_.coord.columns):
            group_coord.loc[g,d] = function_lg(X=obj.ind_.coord[d],Y=Z[cols],xcol_w=[1/obj.eig_.iloc[i,0]],ycol_w=col_w_mfa[cols],row_w=obj.call_.row_w)
    #cos2 of groups
    group_sqcos = group_coord.pow(2).div(group_sqdisto,axis=0)
    #convert to namedtuple
    group_ = namedtuple("group_sup",["coord","cos2","dist2","Lg","RV"])(group_coord,group_sqcos,group_sqdisto,Lg,RV)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #partial axes informations: coordinates, squared cosinus
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #partial axes coordinates
    partial_axes_coord = None
    for g in list(columns_dict.keys()):
        coord = wcorrcoef(concat((obj.ind_.coord,model[g].ind_.coord),axis=1),weights=obj.call_.row_w).iloc[:obj.call_.ncp,obj.call_.ncp:]
        coord.columns = MultiIndex.from_tuples([(g,c) for c in coord.columns])
        partial_axes_coord = concat_empty(partial_axes_coord,coord,axis=1)
    #partial axes square cosinus
    partial_axes_cos2 = partial_axes_coord.pow(2)

    #partial axes correlation between
    all_coord = None
    for g in list(group_dict.keys()):
        data = model[g].ind_.coord
        data.columns = MultiIndex.from_tuples([(g,c) for c in data.columns])
        all_coord = concat_empty(all_coord,data,axis=1)
    #reorder according to group_name
    all_coord = all_coord.reindex(columns=all_coord.columns.reindex(group_name, level=0)[0])
    #correlation between
    cor_between = wcorrcoef(all_coord,weights=obj.call_.row_w)
    #convert to namedtuple
    partial_axes = namedtuple("partial_axes",["coord","cos2","cor_between"])(partial_axes_coord,partial_axes_cos2,cor_between)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #split X
    split_X = splitmix(X=X)
    X_quanti, X_quali, n_rows, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary quantitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quanti > 0:
        #standardization : z_ik = (x_ik - m_k)/s_k
        Z_quanti = recodecont(X=X_quanti).Z
        #statistics for supplementary quantitative variables
        quanti_ = predict_sup(X=Z_quanti,Y=obj.svd_.U[:,:obj.call_.ncp],weights=obj.call_.row_w,axis=1)
        #delete dist2
        del quanti_['dist2']
        #convert to namedtuple
        quanti = namedtuple("quanti_sup",quanti_.keys())(*quanti_.values())
    else:
        quanti = None
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #statistics for supplementary qualitative variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if n_quali > 0:
        #recode supplementary qualitative variables
        rec = recodecat(X_quali)
        X_quali, dummies = rec.X, rec.dummies
        #conditional average of standardized data
        X_levels = conditional_wmean(X=obj.call_.Z,Y=X_quali,weights=obj.call_.row_w)
        #standardization: z_ik = x_ik - m_k
        Z_levels = X_levels.sub(obj.call_.z_center, axis=1)
        #statistics for supplementary levelsp
        quali_ = predict_sup(X=Z_levels,Y=obj.svd_.V[:,:obj.call_.ncp],weights=obj.call_.col_w,axis=0)
        #proportion for the supplementary levels
        p_k = dummies.mul(obj.call_.row_w,axis=0).sum(axis=0)
        #vtest for the supplementary levels
        quali_["vtest"] = quali_["coord"].mul(sqrt((n_rows-1)/(1/p_k).sub(1)),axis=0).div(obj.svd_.vs[:obj.call_.ncp],axis=1)
        #eta2 for the supplementary qualitative variables
        quali_["eta2"] = function_eta2(X=X_quali,Y=obj.ind_.coord,weights=obj.call_.row_w,excl=None)
        #partiel coordinates for qualitative variables
        quali_var_sup_coord_partiel = None
        for g in list(obj.call_.group.keys()):
            quali_sup_coord_partiel = conditional_wmean(X=obj.ind_.coord_partiel[g],Y=X_quali,weights=obj.call_.row_w)
            quali_sup_coord_partiel.columns = MultiIndex.from_tuples([(g,c) for c in quali_sup_coord_partiel.columns])
            quali_var_sup_coord_partiel = concat_empty(quali_var_sup_coord_partiel,quali_sup_coord_partiel,axis=1)
        #add to dictionary
        quali_["coord_partiel"] = quali_var_sup_coord_partiel
        #convert to namedtuple
        quali = namedtuple("quali_sup",quali_.keys())(*quali_.values())

    #convert to namedtuple
    return namedtuple("supvarMFAResult",["group","partial_axes","quanti","quali"])(group_,partial_axes,quanti,quali)


def predictDMFA(
        obj,X
):
    """
    Predict projection for new individuals with Dual Multiple Factor Analysis (DMFA)
    
    Performs the coordinates, squared cosinus and squared distance to origin of new individuals with Dual Multiple Factor Analysis (DMFA)

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.DMFA`.

    X : DataFrame of shape (n_samples, n_columns)
        Input data in which to look for variables with which to predict. ``X`` must contain columns with the same names as the original data.
    
    Returns
    -------
    result : predictDMFAResult
        An object containing all the results for the new individuals, with the following attributes:
    
        coord : DataFrame of shape (n_samples, ncp) 
            The coordinates for the new individuals.
        cos2 : DataFrame of shape (n_samples, ncp) 
            The squared cosinus for the new individuals.
        dist2 : Series of shape (n_samples,)
            The squared distance to origin for the new individuals.

    Examples
    --------
    >>> from scientisttools.datasets import iris, housevotes84
    >>> from scientisttools import DMFA, predictDMFA
    >>>> #DMFA for quantitative variables
    >>> dmfa = DMFA(group=4)
    >>> dmfa.fit(iris)
    DMFA(group=4)
    >>> #predict for the new individuals
    >>> predict = predictDMFA(dmfa,iris)
    >>> predict.coord #coord for the new individuals
    >>> predict.cos2 #cos2 for the new individuals
    >>> predict.dist2 #dist2 for the new individuals
    >>> #DMFA for qualitative variables
    >>> dmfa = DMFA(scale_unit=True,ncp=2,group=0)
    >>> dmfa.fit(housevotes84)
    DMFA(group=0,ncp=2,scale_unit=True)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class DMFA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "DMFA":
        raise TypeError("'obj' must be an object of class DMFA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #prediction input check
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X = predict_first_check(obj,X)
    #split X into x and y
    y, x = X[obj.call_.group], X.drop(columns=[obj.call_.group])
    
    if all(is_numeric_dtype(x[k]) for k in x.columns):
        xcod = x.copy()
    elif all(check_is_object_or_category_dtype(x[k]) for k in x.columns):
        x = revalue(x)
        #test if x contains all active categorics
        new = [x for x in unique(x.values) if x not in obj.call_.dummies.columns]
        if len(new) > 0:
            raise ValueError("The following categories are not in the active dataset: "+",".join(new))
        xcod = tab_disjunctive(X=x,dummies_cols=obj.call_.dummies.columns).astype(float).mul(obj.call_.M.loc[y.values,:].values,axis=1)
    
    #standardization: z_ikl = (x_ikl - m_kl)/s_kl
    zs = xcod.sub(obj.call_.center.loc[y.values,:].values).div(obj.call_.scale.loc[y.values,:].values)
    #standardization: z_ik = (x_ik  - m_k)/s_k
    z = zs.sub(obj.call_.z_center,axis=1).div(obj.call_.z_scale,axis=1)
    #statistics for news individuals
    predict = predict_sup(X=z,Y=obj.svd_.V,weights=obj.call_.col_w,axis=0)
    #convert to namedtuple
    return namedtuple("predictDMFAResult",predict.keys())(*predict.values())

def supvarDMFA(
        obj,X
):
    """
    Supplementary variables in Dual Multiple Factor Analysis (DMFA)
    
    Performs the coordinates, squared cosinus and squared distance to origin of supplementary variables with Dual Multiple Factor Analysis (DMFA)

    Parameters
    ----------
    obj : class 
        An object of class :class:`~scientisttools.DMFA`.

    X : array-like of shape (n_samples,) or (n_samples, n_columns)
        Input data of supplementary variables (quantitative and/or qualitative).
    
    Returns
    -------
    result : supvarDMFAResult
        An object containing the results for supplementary variables, with the following attributes: 

        quanti : None or quanti_sup
            An object containing all the results of the supplementary quantitative variables, with the following attributes:

            coord : DataFrame of shape (n_quanti_sup, ncp) 
                The coordinates of the supplementary quantitative variables.
            cos2 : DataFrame of shape (n_quanti_sup, ncp) 
                The squared cosinus of the supplementary quantitative variables.
        
        quali : None or quali_sup
            An object containing all the results of the supplementary qualitative variables/levels, with the following attributes:

            coord :  DataFrame of shape (n_levels, ncp) 
                The coordinates of the supplementary levels.
            cos2 :  DataFrame of shape (n_levels, ncp) 
                The squares cosinus of the supplementary levels.
            vtest :  DataFrame of shape (n_levels, ncp) 
                The value-test of the supplementary levels.
            dist2 : Series of shape (n_levels,)
                The squared distance to origin of the supplementary levels.
            eta2 :  DataFrame of shape (n_quali_sup, ncp) 
                The squared correlation ratio of the supplementary qualitative variables.

    Examples
    --------
    >>> from scientisttools.datasets import iris
    >>> from scientisttools import DMFA, supvarDMFA
    >>> dmfa = DMFA(group=4)
    >>> dmfa.fit(iris)
    DMFA(group=4)
    >>> #supplementary quantitative variables
    >>> sup_var_predict = supvarDMFA(res_dmfa, iris.iloc[:,:4])
    >>> quanti_sup = sup_var_predict.quanti
    >>> quanti_sup.coord #coordinates for the supplementary quantitative variables
    >>> quanti_sup.cos2 #cos2 for the supplementary quantitative variables
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class DMFA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "DMFA":
        raise TypeError("'obj' must be an object of class DMFA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert Series to DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X = convert_series_to_dataframe(X)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X)

    #split X
    split_X = splitmix(X=X)
    X_quanti, X_quali, n_samples, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

    #statistics for supplementary quantitative variables
    if n_quanti > 0:
        #fill NA with mean
        X_quanti = fill_na(X_quanti)
        #conditional weighted average
        center = conditional_wmean(X=X_quanti,Y=obj.call_.y,weights=obj.call_.row_w)
        #conditional weighted standard deviation
        if obj.scale_unit:
            scale = conditional_wstd(X=X_quanti,Y=obj.call_.y,weights=obj.call_.row_w)
        else:
            scale = DataFrame(ones((obj.call_.y.nunique(),n_quanti)),columns=X_quanti.columns,index=obj.group_.coord.index)
        #standardization: z_ikl = (x_ikl - m_kl)/s_kl
        z = X_quanti.sub(center.loc[obj.call_.y.values,:].values).div(scale.loc[obj.call_.y.values,:].values)
        #standardization: z_ik = (x_ik - m_k)/s_k
        z_center = average(z,axis=0,weights=obj.call_.row_w)
        z_scale = array([sqrt(cov(z.iloc[:,k],aweights=obj.call_.row_w,ddof=0)) for k in range(n_quanti)])
        zs = z.sub(z_center,axis=1).div(z_scale,axis=1)
        #statistics for supplementary quantitative variables
        quanti_ = predict_sup(X=zs,Y=obj.svd_.U,weights=obj.call_.row_w,axis=1)
        #delete dist2
        del quanti_['dist2']
        #partiel coordinates for supplementary quantitative variables
        quanti_coord_partiel = None
        for g, rows in obj.call_.group_dict.items():
            corr_sup_g = wcorrcoef(concat_empty(z.loc[rows,:],obj.ind_.coord.loc[rows,:],axis=1),weights=obj.call_.row_w[rows]/sum(obj.call_.row_w[rows])).iloc[:n_quanti,n_quanti:]
            corr_sup_g.columns = MultiIndex.from_tuples([(g,j) for j in obj.eig_.index[:obj.call_.ncp]])
            quanti_coord_partiel = concat_empty(quanti_coord_partiel,corr_sup_g,axis=1)
        quanti_["coord_partiel"] = quanti_coord_partiel
        #convert to namedtuple
        quanti = namedtuple("quanti_sup",quanti_.keys())(*quanti_.values())
    else:
        quanti = None

    #statistics for supplementary qualitative variables/levels
    if n_quali > 0:
        #create new qualitative columns
        X_quali_new = concat((concat((X_quali[x],obj.call_.y),axis=1).apply(lambda x: ''.join(x),axis=1) for x in X_quali.columns),axis=1)
        X_quali_new.columns = ["{}{}".format(x,obj.call_.group) for x in X_quali.columns]
        #concatenate
        X_quali = concat((X_quali,X_quali_new),axis=1)
        #recode qualitative variables
        rec = recodecat(X=X_quali)
        X_quali, dummies = rec.X, rec.dummies
        #compute conditional weighted average
        x_levels = conditional_wmean(X=obj.call_.z,Y=X_quali,weights=obj.call_.row_w)
        #standardization : z_ik = (x_ik - m_k)/s_k
        z_levels = x_levels.sub(obj.call_.z_center,axis=1).div(obj.call_.z_scale,axis=1)
        #statistics for supplementary levels
        quali_ = predict_sup(X=z_levels,Y=obj.svd_.V,weights=obj.call_.col_w,axis=0)
        #proportion for the supplementary levels
        p_k = dummies.mul(obj.call_.row_w,axis=0).sum(axis=0)
        #vtest for the supplementary levels
        quali_["vtest"] = quali_["coord"].mul(sqrt((n_samples-1)/(1/p_k).sub(1)),axis=0).div(obj.svd_.vs[:obj.call_.ncp],axis=1)
        #eta2 for the supplementary qualitative variables
        quali_["eta2"] = function_eta2(X=X_quali,Y=obj.ind_.coord,weights=obj.call_.row_w,excl=None)
        #convert to namedtuple
        quali = namedtuple("quali_sup",quali_.keys())(*quali_.values())
    else:
        quali = None

    #convert to namedtuple
    return namedtuple("supvarDMFAResult",["quanti","quali"])(quanti,quali)