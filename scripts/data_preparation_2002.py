import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from variable_cleaner import clean_ess_data

def load_data(
        data_path,
        codebook_path,
        country,
        random_state,
        test_size=0.3,
        nonresponse_dict=None
):
    """
    MAKE DOCSTRING
    """
    # Read in data
    df = pd.read_csv(data_path)

    # Filter the dataframe
    df = df[df['cntry'].isin(country)].copy()
    df = df[df['ctzcntr'] == 1] # eligible national election voters (citizenship)
    df = df[df['yrbrn']>= 18] # eligible national election voters (age)
    df['vote'] = df['vote'].replace([3,7,8,9], np.nan) # drop non-response values for target
    df = df[df['vote'].notna()]

    # Drop irrelevant columns
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop_existing)

    # Switch non-response values to NaN
    df = clean_ess_data(df, codebook_path, nonresponse_dict)

    # Map vote values to 0 and 1
    df['vote'] = df['vote'].map({1.0: 1, 2.0: 0})

    # Prepare features
    exclude_from_predictors = ['vote', 'anweight']  
    predictor_cols = [col for col in df.columns if col not in exclude_from_predictors]
    X = df[predictor_cols].copy()
    y = df['vote'].copy()

    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    #scaler = StandardScaler()

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, df


# Columns to drop: country-specific variables and/or irrelevant

# Party voted variables 
party_voted_cols = [
    'prtvtat', 'prtvtbe', 'prtvtch', 'prtvtcz', 'prtvde1', 'prtvde2', 'prtvtdk',
    'prtvtes', 'prtvtfi', 'prtvtfr', 'prtvtgb', 'prtvtgr', 'prtvthu', 'prtvtie',
    'prtvtil', 'prtvtit', 'prtvtlu', 'prtvtnl', 'prtvtno', 'prtvtpl', 'prtvtpt',
    'prtvtse', 'prtvtsi'
]

# Party feel closer to variables 
party_close_cols = [
    'prtclat', 'prtclbe', 'prtclch', 'prtclcz', 'prtclde', 'prtcldk',
    'prtcles', 'prtclfi', 'prtclfr', 'prtclgb', 'prtclgr', 'prtclhu',
    'prtclie', 'prtclil', 'prtclit', 'prtcllu', 'prtclnl', 'prtclno',
    'prtclpl', 'prtclpt', 'prtclse', 'prtclsi'
]

# Member of party variables
party_member_cols = [
    'prtmbat', 'prtmbbe', 'prtmbch', 'prtmbcz', 'prtmbde', 'prtmbdk',
    'prtmbes', 'prtmbfi', 'prtmbfr', 'prtmbgb', 'prtmbgr', 'prtmbhu',
    'prtmbie', 'prtmbil', 'prtmbit', 'prtmblu', 'prtmbnl', 'prtmbno',
    'prtmbpl', 'prtmbpt', 'prtmbse', 'prtmbsi'
]

# Sports/organisation variables
sports_cols = [
    'sptcref', 'sptcna', 'sptcnn', 'sptcmmb', 'sptcptp', 'sptcdm',
    'sptcvw', 'sptcfrd', 'cltofrd', 'trufrd', 'prfofra', 'cnsofrd',
    'hmnofrd', 'epaofrd', 'rlgofrd', 'prtyfrd', 'setofrd', 'sclcfrd',
    'othvfrd', 'cltoref', 'cltona', 'cltonn', 'cltommb', 'cltoptp',
    'cltodm', 'cltovw', 'truref', 'truna', 'trunn', 'trummb', 'truptp',
    'trudm', 'truvw', 'prforef', 'prfona', 'prfonn', 'prfommb', 'prfoptp',
    'prfodm', 'prfovw', 'cnsoref', 'cnsona', 'cnsonn', 'cnsommb', 'cnsoptp',
    'cnsodm', 'cnsovw', 'hmnoref', 'hmnona', 'hmnonn', 'hmnommb', 'hmnoptp',
    'hmnodm', 'hmnovw', 'epaoref', 'epaona', 'epaonn', 'epaommb', 'epaoptp',
    'epaodm', 'epaovw', 'rlgoref', 'rlgona', 'rlgonn', 'rlgommb', 'rlgoptp',
    'rlgodm', 'rlgovw', 'prtyref', 'prtyna', 'prtynn', 'prtymmb', 'prtyptp',
    'prtydm', 'prtyvw', 'setoref', 'setona', 'setonn', 'setommb', 'setoptp',
    'setodm', 'setovw', 'sclcref', 'sclcna', 'sclcnn', 'sclcmmb', 'sclcptp',
    'sclcdm', 'sclcvw', 'othvref', 'othvna', 'othvnn', 'othvmmb', 'othvptp',
    'othvdm', 'othvvw', 'impfml', 'impfrds', 'implsrt', 'imppol', 'impwrk',
    'imprlg', 'impvo'
]

# Household grid variables
house_cols = [
    'gndr', 'gndr2', 'gndr3', 'gndr4', 'gndr5', 'gndr6', 'gndr7', 'gndr8',
    'gndr9', 'gndr10', 'gndr11', 'gndr12', 'gndr13', 'gndr14', 'gndr15', 
    'yrbrn2', 'yrbrn3', 'yrbrn4', 'yrbrn5', 'yrbrn6', 'yrbrn7', 'yrbrn8', 
    'yrbrn9', 'yrbrn10', 'yrbrn11', 'yrbrn12', 'yrbrn13', 'yrbrn14', 'yrbrn15',
    'rshipa2', 'rshipa3', 'rshipa4', 'rshipa5', 'rshipa6', 'rshipa7', 'rshipa8',
    'rshipa9', 'rshipa10', 'rshipa11', 'rshipa12', 'rshipa13', 'rshipa14', 'rshipa15'
] 

# Respondent's education variables
education_cols = [
    'edlvbe', 'edlvch', 'edlvcz', 'edlvdk', 'edlves', 'edlvfr', 'edlvgb',
    'edlvgr', 'edlvhu', 'edlvie', 'edlvil', 'edlvit', 'edlvlu', 'edlvnl',
    'edlvno', 'edlvpl', 'edlvpt', 'edlvse', 'wrkctrhu', 'iscoco', 'nacer1'
]

# Partner variables
partner_education_cols = [
    'iscocop', 'occf14ie', 'occm14ie', 'martlfr'
]

# Country specific region variables
region_cols = [
    'regionat', 'regionbe', 'regioach', 'regioncz', 'regionde', 'regiondk',
    'regiones', 'regionfi', 'regionfr', 'regiongb', 'regiongr', 'regionhu',
    'regionie', 'regionil', 'regionit', 'regionlu', 'regionlu', 'regionnl',
    'regionno', 'regionpl', 'regionpt', 'regionse', 'regionsi', 'intewde'
]

# Administrative and metadata variables
admin_cols = [
    'inwdd', 'inwmm', 'inwyr', 'inwshh', 'inwsmm', 'inwemm', 'inwehh',
    'inwtm', 'spltadm', 'supqadm', 'name', 'essround', 'edition', 'proddate',
    'idno', 'cntry', 'dweight', 'pspwght', 'pweight'
]

# Combine all columns to drop
cols_to_drop = (party_voted_cols + party_close_cols + 
                party_member_cols + sports_cols +
                house_cols + education_cols + 
                partner_education_cols + region_cols +
                admin_cols
                )


