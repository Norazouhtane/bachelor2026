import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
#from scripts.variable_cleaner import clean_ess_data
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
    df = df[df['agea']>= 18] # eligible national election voters (age)
    df['vote'] = df['vote'].replace([3,7,8,9], np.nan) # drop non-response values for target
    df = df[df['vote'].notna()]

    # Calculate anweight
    df['anweight'] = df['pspwght']*df['pweight']

    # Drop irrelevant columns
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop_existing)

    # Switch non-response values to NaN
    df = clean_ess_data(df, codebook_path, nonresponse_dict)

    # Map vote values to 0 and 1
    df['vote'] = df['vote'].map({1: 1, 2: 0})

    # Prepare features
    exclude_from_predictors = ['vote', 'anweight', 'pspwght', 'pweight']  
    predictor_cols = [col for col in df.columns if col not in exclude_from_predictors]
    X = df[predictor_cols].copy()
    y = df['vote'].copy()

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # One hot encode categorical columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols:
        one_hot = ce.OneHotEncoder(cols=categorical_cols, use_cat_names=True)
        X_train = one_hot.fit_transform(X_train)
        X_test = one_hot.transform(X_test)

    return X_train, X_test, y_train, y_test, df


# Columns to drop: country-specific variables and/or irrelevant for 2004

# Party voted variables 
party_voted_cols = [
    'prtvtat', 'prtvtabe', 'prtvtch', 'prtvtcz', 'prtvade1', 'prtvade2',
    'prtvtdk', 'prtvtee', 'prtvtaes', 'prtvtfi', 'prtvtfr', 'prtvtgb',
    'prtvtagr', 'prtvthu', 'prtvtie', 'prtvtis', 'prtvtait', 'prtvtlu',
    'prtvtanl', 'prtvtno', 'prtvtpl', 'prtvtpt', 'prtvtse', 'prtvtasi',
    'prtvtsk', 'prtvttr', 'prtvtua'
]

# Party feel closer to variables 
party_close_cols = [
    'prtclat', 'prtclabe', 'prtclch', 'prtclcz', 'prtclade', 'prtcldk',
    'prtclee', 'prtclaes', 'prtclfi', 'prtclfr', 'prtclgb', 'prtclagr',
    'prtclhu', 'prtclie', 'prtclis', 'prtclait', 'prtcllu', 'prtclanl',
    'prtclno', 'prtclapl', 'prtclapt', 'prtclse', 'prtclsk', 'prtcltr',
    'prtclua', 'prtclasi'
]

# Member of party variables
party_member_cols = [
    'prtmbat', 'prtmbabe', 'prtmbch', 'prtmbcz', 'prtmbdk', 'prtmbade',
    'prtmbee', 'prtmbaes', 'prtmbfi', 'prtmbfr', 'prtmbgb', 'prtmbagr',
    'prtmbhu', 'prtmbie', 'prtmbis', 'prtmbait', 'prtmblu', 'prtmbanl',
    'prtmbno', 'prtmbapl', 'prtmbpt', 'prtmbse', 'prtmbasi', 'prtmbsk',
    'prtmbtr', 'prtmbua', 'mmbprty'
]

# Social, social behaviour and political variables
social_cols = [
    'happy', 'sclmeet', 'sclact', 'dscrrce', 'dscrntn', 'dscrrlg', 'dscrlng', 
    'dscretn', 'dscrage', 'dscrgnd', 'dscrsex', 'dscrdsb', 'dscroth', 'dscrdk', 
    'dscrref', 'dscrnap', 'dscrna', 'inmdisc', 'lnghoma', 'lnghomb', 'fbrncnt',
    'mbrncnt', 'poldcs', 'prtyban', 'scnsenv', 'wrkprty', 'wrkorg',
    'rlgatnd', 'pray'
] #'health', 'hlthhmp', 'dscrgrp' kept

# Health variables
health_cols = [
    'mdlswgt', 'mdhair', 'mdmemo', 'mdhappy', 'mdsexlf', 'hltherb', 'mdsdeff', 
    'prfmddc', 'ddprsmd', 'usmdprs', 'usmprse', 'advsthr', 'prcsthr', 'advhach', 
    'prchach', 'advslep', 'prcslep', 'advbach', 'prcbach', 'chsrgp', 'prfsmdc', 
    'tmcnsdc', 'illcure', 'pplcure', 'pprlydc', 'psmdcpr', 'follwdc', 'dsplvpr',
    'dckptrt', 'dctreql', 'dcdisc', 'ptnrlcq', 'dcadmms', 'dcdfcwr',
]   #'cgtsmok', 'medtrun' missing, 

# Economic morality
moral_cols = [
    'ctzhlpo', 'scbevts', 'ctzchtx', 'tstrprh', 'tstfnch', 'tstpboh',
    'rprochg', 'fodcncl', 'bnkfldl', 'scndhfl', 'pboafvr', 'wrytrdh',
    'pyavtxw', 'slcnflw', 'flinsrw', 'pbofvrw', 'mnyacth', 'olwmsop',
    'ignrlaw', 'bsnprft', 'frmwktg', 'cmprcti', 'frdbnft', 'kptchng',
    'payavtx', 'slcnsfl', 'musdocm', 'flinsr', 'pbofvr', 'flgvbnf'
]

# Household grid variables
house_cols = [
    'hhmmb', 'gndr2',  'gndr3',  'gndr4',  'gndr5',  'gndr6',  'gndr7',  'gndr8',
    'gndr9',  'gndr10', 'gndr11', 'gndr12', 'gndr13', 'gndr14', 'gndr15', 'gndr16', 
    'gndr17', 'gndr18', 'yrbrn2',  'yrbrn3',  'yrbrn4',  'yrbrn5',  'yrbrn6',  'yrbrn7',
    'yrbrn8',  'yrbrn9',  'yrbrn10', 'yrbrn11', 'yrbrn12', 'yrbrn13', 'yrbrn14', 
    'yrbrn15', 'yrbrn16', 'yrbrn17', 'yrbrn18', 'rshipa2',  'rshipa3',  'rshipa4',  
    'rshipa5',  'rshipa6',  'rshipa7', 'rshipa8',  'rshipa9',  'rshipa10', 'rshipa11', 
    'rshipa12', 'rshipa13', 'rshipa14', 'rshipa15', 'rshipa16', 'rshipa17', 'rshipa18',
]

# Socio-demographic variables
socio_demo_cols = [
    'domicil', 'hhmodwl', 'rmhhus', 'brwmny', 'partner', 'pphincr', 'marital', 
    'martlfr', 'lvghw', 'lvgoptn', 'lvgptn', 'lvgptne', 'icmsw', 'dvrcdev', 
    'chldhm', 'chldhhe', 'moalv', 'faalv', 'intewde', 
] 

# Respondent's education variables
education_cols = [
    'edlvbe', 'edlvach', 'edlvcz', 'edlvde', 'edlvadk', 'edlvee',
    'edlvaes', 'edlvfr', 'edlvagb', 'edlvgr', 'edlvahu', 'edlvie',
    'edlvait', 'edlvlu', 'edlvnl', 'edlvno', 'edlvapl', 'edlvpt',
    'edlvse', 'edlvsk', 'edlvua', 'edufld', 'atncrse', 'wkdcpce'
] #'edulvla', 'eisced' kept

# Partner's education/employment variables
partner_education_cols = [
    'pdwrkp', 'edctnp', 'uemplap', 'uemplip', 'dsbldp', 'rtrdp',
    'cmsrvp', 'hswrkp', 'dngothp', 'dngdkp', 'dngnapp', 'dngrefp',
    'dngnap', 'crpdwkp', 'emprelp', 'wkhtotp', 'emplnop', 'jbspvp', 
    'njbspvp', 'wkdcorp', 'ioactp'
] #'edulvlpa', 'mnactp' kept

# Father's education/employment variables
father_education_cols = [
    'emprf14', 'emplnof', 'jbspvf', 'occf14a'
] 

# Mother's education/employment variables
mother_education_cols = [
    'emprm14', 'emplnom', 'jbspvm', 'occm14a'
] 

# Family work and well-being variables
family_work_cols = [
    'gdsprt', 'clmrlx', 'actvgrs', 'lfintr', 'frshrst',
    'wmcpwrk', 'mnrsphm', 'mnrgtjb', 'prntghr', 'prrfmly',
    'yrlvptn', 'dsgrhwk', 'dsgrmny', 'dsgrpwk', 'gwhhprc', 
    'gwdvhwk', 'wkengtp', 'wkovtmp', 'ptnwkwe', 'hwktwd1', 
    'hwkpwd1', 'hwkpwdp', 'hwktwe1', 'hwkpwe1', 'hwkpwep',
    'hwktwd2', 'hwkpwd2', 'hwktwe2', 'hwkpwe2', 'tngdohm', 
    'hwkmono', 'chdohwk', 'hwkstrs', 'hmeqphw', 'lkafohh', 
    'updhlrl', 'updhlrp', 'cld12hh', 'cldcrot', 'cldcrmr',
    'cldnhh', 'cldnhhn', 'cldnhhg', 'clnhhyb', 'clnhhbo', 
    'clnhhby', 'cldnhhd', 'clfncsp', 'clhwksp', 'clfncrc', 
    'clhwkrc', 'plnchld', 'jbcoedu', 'jbedyrs', 'jblrn',
    'vrtywrk', 'jbrqlrn', 'jbscr', 'wgdpeft', 'hlpcowk', 
    'dcsfwrk', 'hlthrwk', 'wrkhrd', 'nevdnjb', 'oprtad',
    'nbsrsp', 'bsmw', 'ppwwkp', 'yrcremp', 'trndnjb', 
    'wrkspv', 'smbtjoba', 'rpljbde', 'tmtowrk','wrkengt', 
    'wkovrtm', 'wrkwe', 'wrywprb', 'trdawrk', 'jbprtfp', 
    'pfmfdjb', 'dfcnswk', 'grspay', 'netpay', 'payprd',
    'stdlvl', 'prmpls', 'quclss', 'tchtruf', 'stdtruf', 
    'tchints', 'tchlcrt', 'stdask', 'stpvtfm', 'fmpvtst', 
    'tchlp', 'stdhrsw', 'stdmcdo', 'crspce', 'rtryr', 
    'wntrtr', 'ipjbscr', 'ipjbhin', 'ipjbprm', 'ipjbini', 
    'ipjbwfm', 'wkhsch', 'fstjbyr', 'yrspdwk', 'flthmcc', 
    'fthcncr', 'ptmhmcc', 'pthcncr',
]

# Region variables for other countries
reg_cols = [
    'regiontr', 'regionua', 'regiongb', 'regiondk', 'regionit',
    'regionlu', 'regionno', 'regioncz', 'regioach', 'regionis',
]

 
# Administrative and metadata variables
admin_cols = [
    'name', 'essround', 'edition', 'proddate', 'idno', 'cntry',
    'dweight', 'inwtm', 'inwdd', 'inwmm', 'inwyr', 'inwshh', 'inwsmm', 
    'inwehh', 'inwemm', 'intewde',
    'spltadma', 'supqad1', 'supqad2', 'supqdd', 'supqmm', 'supqyr',
    'icomdng', 'icmnac', 'icempl', 'icemplr', 'icnopfm', 'icgndr',
    'icchld', 'icag45y', 'icagu70', 'iccldnh', 'icptnwk', 'ichwk1', 
    'ichwk2', 'icomdnp', 'icptn'
]

# Variables with codes not needed
code_cols = [
    'iscoco', 'iscocop', 'nacer11',
]


# Combine all columns to drop
cols_to_drop = (party_voted_cols + party_close_cols + party_member_cols
                + social_cols + health_cols + moral_cols + house_cols
                + socio_demo_cols + education_cols + partner_education_cols
                + father_education_cols + mother_education_cols + family_work_cols
                + reg_cols + admin_cols + code_cols
                )


