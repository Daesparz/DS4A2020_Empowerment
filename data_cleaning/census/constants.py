WEEK_IDX_RANGE = range(1, 14)

DATES_FROM_WEEK_IDX = {
     1: ('2020-04-23', '2020-05-05'),
     2: ('2020-05-07', '2020-05-12'),
     3: ('2020-05-14', '2020-05-19'),
     4: ('2020-05-21', '2020-05-26'),
     5: ('2020-05-28', '2020-06-02'),
     6: ('2020-06-04', '2020-06-09'),
     7: ('2020-06-11', '2020-06-16'),
     8: ('2020-06-18', '2020-06-23'),
     9: ('2020-06-25', '2020-06-30'),
    10: ('2020-07-02', '2020-07-07'),
    11: ('2020-07-09', '2020-07-14'),
    12: ('2020-07-16', '2020-07-21'),
    13: ('2020-08-19', '2020-08-31'),
    14: ('2020-09-02', '2020-09-14'),
    15: ('2020-09-16', '2020-09-28'),
    16: ('2020-09-30', '2020-10-12'),
    17: ('2020-10-14', '2020-10-26'),
}

DEMOGRAPHICS_VARS = [
    'TBIRTH_YEAR', 'EGENDER', 'MS', 'EST_ST', 'REGION', 'EST_MSA',
    'RHISPANIC', 'RRACE',
    'EEDUC',
    'THHLD_NUMPER', 'THHLD_NUMKID', 'THHLD_NUMADLT',
    'INCOME',
]

ID_VAR = ['SCRAM']

WEEK_VAR = ['WEEK']

EIP_VARS = ['EIP', 'EIPSPND1',  'EIPSPND2', 'EIPSPND3', 'EIPSPND4',
           'EIPSPND5', 'EIPSPND6', 'EIPSPND7', 'EIPSPND8',
           'EIPSPND9', 'EIPSPND10', 'EIPSPND11', 'EIPSPND12', 'EIPSPND13']

FOOD_VARS = ['FOODCONF', 'TSPNDPRPD', 'TSPNDFOOD', 'SNAPMNTH1',
           'SNAPMNTH2', 'SNAPMNTH3', 'SNAPMNTH4', 'SNAPMNTH5',
           'SNAPMNTH6', 'SNAPMNTH7', 'SNAPMNTH8', 'SNAPMNTH9',
           'SNAPMNTH10', 'SNAPMNTH11', 'SNAPMNTH12', 'SNAP_YN',
           'WHEREFREE1', 'WHEREFREE2', 'WHEREFREE3', 'WHEREFREE4',
           'WHEREFREE5', 'WHEREFREE6', 'WHEREFREE7', 'FREEFOOD', 
           'FOODSUFRSN1', 'FOODSUFRSN2', 'FOODSUFRSN3', 'FOODSUFRSN4',
           'FOODSUFRSN5', 'CHILDFOOD', 'PRIFOODSUF', 'CURFOODSUF',
           'SPNDSRC8']

SHOPPING_VARS = ['WHYCHNGD1','WHYCHNGD2', 'WHYCHNGD3','WHYCHNGD4',
                 'WHYCHNGD5','WHYCHNGD6', 'WHYCHNGD7','WHYCHNGD8',
                 'WHYCHNGD9','WHYCHNGD10', 'WHYCHNGD11','WHYCHNGD12',
                 'CHNGHOW1', 'CHNGHOW2', 'CHNGHOW3', 'CHNGHOW4',
                 'CHNGHOW5', 'CHNGHOW6', 'CHNGHOW7', 'CHNGHOW8',
                 'CHNGHOW9', 'CHNGHOW10', 'CHNGHOW11', 'CHNGHOW12',
                 'EXPNS_DIF']

TELEWORK = ['TW_START']

TRIPS_VAR = ['FEWRTRIPS', 'FEWRTRANS', 'PLNDTRIPS', 'CNCLDTRPS']

HEALTH_VARS = [
    'HLTHSTATUS',
    'ANXIOUS', 'WORRY', 'INTEREST', 'DOWN'
]

WORK_VARS = [
    'WRKLOSS', 'EXPCTLOSS', 'ANYWORK', 'KINDWORK', 'RSNNOWRK', 'UNEMPPAY'
]

STATES =  ['Alabama',
          'Alaska',
          'Arizona',
          'Arkansas',
          'California',
          'Colorado',
          'Connecticut',
          'Delaware',
          'District of Columbia',
          'Florida',
          'Georgia',
          'Hawaii',
          'Idaho',
          'Illinois',
          'Indiana',
          'Iowa',
          'Kansas',
          'Kentucky',
          'Louisiana',
          'Maine',
          'Maryland',
          'Massachusetts',
          'Michigan',
          'Minnesota',
          'Mississippi',
          'Missouri',
          'Montana',
          'Nebraska',
          'Nevada',
          'New Hampshire',
          'New Jersey',
          'New Mexico',
          'New York',
          'North Carolina',
          'North Dakota',
          'Ohio',
          'Oklahoma',
          'Oregon',
          'Pennsylvania',
          'Rhode Island',
          'South Carolina',
          'South Dakota',
          'Tennessee',
          'Texas',
          'Utah',
          'Vermont',
          'Virginia',
          'Washington',
          'West Virginia',
          'Wisconsin',
          'Wyoming']
