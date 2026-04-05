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
    df = df[df['agea']>= 18] # eligible national election voters (age)
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
    'prtvtat', 'prtvtebe', 'prtvtfbg', 'prtvtchr', 'prtvtccy', 'prtvtiee',
    'prtvtffi', 'prtvtffr', 'prtvtegr', 'prtvthhu', 'prtvteis', 'prtvteie',
    'prtvteil', 'prtvteit', 'prtvtblv', 'prtvclt1', 'prtvclt2', 'prtvclt3',
    'prtvtbme', 'prtvtinl', 'prtvtcno', 'prtvtfpl', 'prtvtept', 'prtvtbrs',
    'prtvtesk', 'prtvtgsi', 'prtvtges', 'prtvtese', 'prtvthch', 'prtvtdua',
    'prtvtdgb', 'prtvgde1', 'prtvgde2', 'prtvtdat', 'prtvtdse'
]

# Party feel closer to variables 
party_close_cols = [
    'prtcleat', 'prtclebe', 'prtclfbg', 'prtclbhr', 'prtclccy', 'prtcliee',
    'prtclqfi', 'prtclqfr', 'prtclegr', 'prtclihu', 'prtcleis', 'prtclfie',
    'prtclfil', 'prtclfit', 'prtclblv', 'prtclclt', 'prtclbme', 'prtclhnl',
    'prtclcno', 'prtcljpl', 'prtclgpt', 'prtclbrs', 'prtclesk', 'prtclgsi',
    'prtclhes', 'prtclese', 'prtclhch', 'prtcleua', 'prtcldgb', 'prtclgde', 
    'prtclgfr', 'prtclgfi'
]

# Religion present variables 
religion_present_cols = [
    'rlgdnbat', 'rlgdncy', 'rlgdnafi', 'rlgdnagr', 'rlgdnhu', 'rlgdnais',
    'rlgdnie', 'rlgdnlv', 'rlgdnlt', 'rlgdme', 'rlgdnanl', 'rlgdnno',
    'rlgdnapl', 'rlgdnapt', 'rlgdnrs', 'rlgdnask', 'rlgdnase', 'rlgdnach',
    'rlgdnaua', 'rlgdngb', 'rlgdnade'
]

# Religion past variables
religion_past_cols = [
    'rlgdebat', 'rlgdecy', 'rlgdeafi', 'rlgdeagr', 'rlgdehu', 'rlgdeais',
    'rlgdeie', 'rlgdelv', 'rlgdelt', 'rlgdeme', 'rlgdeanl', 'rlgdeno',
    'rlgdeapl', 'rlgdeapt', 'rlgders', 'rlgdeask', 'rlgdease', 'rlgdeach',
    'rlgdeaua', 'rlgdegb', 'rlgdeade'
]

# Health variables
health_cols = [
    'etfruit', 'eatveg', 'dosprt', 'cgtsmok', 'alcfreq', 'alcwkdy', 'alcwknd',
    'alcbnge', 'height', 'weighta', 'dshltgp', 'dshltms', 'dshltnt', 'dshltref',
    'dshltdk', 'dshltna', 'trhltacu', 'trhltacp', 'trhltcm', 'trhltch', 'trhltos',
    'trkltho', 'trhltht', 'trhlthy', 'trhltmt', 'trhltpt', 'trhltre', 'trhltsh',
    'trhltnt', 'trhltref', 'trhltdk', 'trhltna', 'hltprhc', 'hltprhb', 'hltprbp',
    'hltpral', 'hltprbn', 'hltprpa', 'hltprpf', 'hltprsd', 'hltprsc', 'hltprsh',
    'hltprdi', 'hltprnt', 'hltprref', 'hltprdk', 'hltprna', 'hltphhc', 'hltphhb',
    'hltphbp', 'hltphal', 'hltphbn', 'hltphpa', 'hltphpf', 'hltphsd', 'hltphsc',
    'hltphsh', 'hltphdi', 'hltphnt', 'hltphnap', 'hltphref', 'hltphdk', 'hltphna',
    'hltprca', 'cancfre', ''
]   #cigarette smoke # drink and binge alcohol #depressed #conflict household 

# Household grid variables
house_cols = [
    'gndr', 'gndr2', 'gndr3', 'gndr4', 'gndr5', 'gndr6', 'gndr7', 'gndr8',
    'gndr9', 'gndr10', 'gndr11', 'gndr12', 'yrbrn2', 'yrbrn3', 'yrbrn4', 'yrbrn5',
    'yrbrn6', 'yrbrn7', 'yrbrn8', 'yrbrn9', 'yrbrn10', 'yrbrn11', 'yrbrn12',
    'rshipa2', 'rshipa3', 'rshipa4', 'rshipa5', 'rshipa6', 'rshipa7', 'rshipa8',
    'rshipa9', 'rshipa10', 'rshipa11', 'rshipa12', 'rshpsgb', 'marstgb'
] #rshpsgb marstgb UK

# Respondent's education variables
education_cols = [
    'edlveat', 'edlvebe', 'edlvebg', 'edlvehr', 'edlvgcy', 'edlvdee',
    'edlvdfi', 'edlvdfr', 'edlvegr', 'edlvdahu', 'edlvdis', 'edlvdie',
    'edubil1', 'eduail2', 'edlvfit', 'edlvelv', 'edlvdlt', 'edlveme',
    'edlvenl', 'edlveno', 'edlvipl', 'edlvept', 'edlvdrs', 'edlvdsk',
    'edlvesi', 'edlvies', 'edlvdse', 'edlvdch', 'edlvdua', 'educgb1',
    'edubgb2', 'edagegb', 'edudde1', 'educde2'
]

# Partner's education variables
partner_education_cols = [
    'eiscedp', 'edlvpfat', 'edlvpebe', 'edlvpebg', 'edlvpehr', 'edlvpgcy',
    'edlvpdee', 'edlvpdfi', 'edlvpdfr', 'edlvpegr', 'edlvpdahu', 'edlvpdis',
    'edlvpdie', 'edupail2', 'edupbil1', 'edlvpfit', 'edlvpelv', 'edlvpdlt',
    'edlvpeme', 'edlvpenl', 'edlvpeno', 'edlvphpl', 'edlvpept', 'edlvpdrs',
    'edlvpdsk', 'edlvpesi', 'edlvphes', 'edlvpdse', 'edlvpdch', 'edlvpdua',
    'edupcgb1', 'edupbgb2', 'edagepgb', 'edupdde1', 'edupcde2'
]

# Father's education variables
father_education_cols = [
    'eiscedf', 'edlvfeat', 'edlvfebe', 'edlvfebg', 'edlvfehr', 'edlvfgcy',
    'edlvfdee', 'edlvfdfi', 'edlvfdfr', 'edlvfegr', 'edlvfdahu', 'edlvfdis',
    'edlvfdie', 'edufail2', 'edufbil1', 'edlvffit', 'edlvfelv', 'edlvfdlt',
    'edlvfeme', 'edlvfenl', 'edlvfeno', 'edlvfgpl', 'edlvfept', 'edlvfdrs',
    'edlvfdsk', 'edlvfesi', 'edlvfges', 'edlvfdse', 'edlvfdch', 'edlvfdua',
    'edufcgb1', 'edufbgb2', 'edagefgb', 'edufcde1', 'edufbde2'
]

# Mother's education variables
mother_education_cols = [
    'eiscedm', 'edlvmeat', 'edlvmebe', 'edlvmebg', 'edlvmehr', 'edlvmgcy',
    'edlvmdee', 'edlvmdfi', 'edlvmdfr', 'edlvmegr', 'edlvmdahu', 'edlvmdis',
    'edlvmdie', 'edumail2', 'edumbil1', 'edlvmfit', 'edlvmelv', 'edlvmdlt',
    'edlvmeme', 'edlvmenl', 'edlvmeno', 'edlvmgpl', 'edlvmept', 'edlvmdrs',
    'edlvmdsk', 'edlvmesi', 'edlvmges', 'edlvmdse', 'edlvmdch', 'edlvmdua',
    'edumcgb1', 'edumbgb2', 'edagemgb', 'edumcde1', 'edumbde2'
]

# Test questions
test_cols = [
    'testji1', 'testji2', 'testji3', 'testji4', 'testji5', 'testji6', 'testji7', 
    'testji8', 'testji9'
]

# Covid variables
covid_cols = [
    'respc19a', 'symtc19', 'symtnc19', 'vacc19'
]

# Administrative and metadata variables
admin_cols = [
    'recon', 'inwds', 'ainws', 'ainwe', 'binwe', 'cinwe', 'dinwe', 'einwe',
    'finwe', 'hinwe', 'iinwe', 'kinwe', 'rinwe', 'inwde', 'jinws', 'jinwe',
    'inwtm', 'mode', 'domain', 'prob', 'stratum', 'psu', 'name', 'essround', 
    'edition', 'proddate', 'idno', 'dweight', 'pweight', 'cntry', 'pspwght'
]

# Variables with codes as variables
code_cols = [
    'anctrya1', 'anctrya2', 'isco08', 'isco08p'
]

# Combine all columns to drop
cols_to_drop = (party_voted_cols + party_close_cols + 
                religion_present_cols + religion_past_cols + 
                health_cols + house_cols + education_cols + 
                partner_education_cols + father_education_cols + 
                mother_education_cols + test_cols + covid_cols +
                admin_cols + code_cols)


