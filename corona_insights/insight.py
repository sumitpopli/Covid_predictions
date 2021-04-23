import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster.hierarchy as sch

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

from sklearn.metrics import make_scorer,silhouette_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


us_states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','District of Columbia','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Puerto Rico','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']

def us_state_covid_chart(state):
    print(us_states)
    state_cases = pd.read_csv('../data/us_states.csv')
    state_cases['date'] = pd.to_datetime(state_cases['date'])
    state_cases['week'] = state_cases['date'].dt.strftime('%W')
    state_cases['drate'] = (state_cases['deaths']/state_cases['cases'])*100
    print(state_cases.columns)
    print(state_cases.head())
    print(state_cases.tail())
    print(state_cases.shape)
    print(len(us_states))

    if(len(state) ==0):
        state_data = state_cases.copy()
        print(state_data.head(10))
        # showing first 20
        fig, axs = plt.subplots(4, 5, sharey=True)
        plt.subplots_adjust(bottom=0.001,top = 0.98,wspace = 0.2)
        breakfree = False

        for xiter in range (0,4):
            for yiter in range(0,5):
                if((xiter * 5 + yiter) == len(us_states)):
                    breakfree = True
                    break
                #print("x = {0}, y = {1} state number {2} and name {3}".format(xiter, yiter, xiter * 5 + yiter, us_states[xiter * 5 + yiter]))
                title_str = us_states[xiter * 5 + yiter] + ' state covid chart'
                state_data = state_cases[state_cases['state'] == us_states[yiter*5+xiter]]
                axs[xiter, yiter].plot(state_data['week'], state_data['deaths'], label='deaths',color='red')
                axs[xiter, yiter].plot(state_data['week'], state_data['drate'], label='death rate', color='blue')
                axs[xiter, yiter].plot(state_data['week'], state_data['cases'], color='green', label='cases')
                axs[xiter, yiter].set_title(title_str)
            if(breakfree == True):
                break
        plt.show()

        # showing next 20
        fig, axs = plt.subplots(4, 5, sharey=True)
        plt.subplots_adjust(bottom=0.001, top=0.98, wspace=0.2)
        breakfree = False

        for xiter in range(0, 4):
            for yiter in range(0, 5):
                if ( (xiter * 5 + 20 + yiter) == len(us_states)):
                    breakfree = True
                    break
                '''
                print("x = {0}, y = {1} state number {2} and name {3}".format(xiter, yiter, xiter * 5 + 20 +  yiter,
                                                                              us_states[xiter * 5 + +20+yiter]))
               '''
                title_str = us_states[xiter * 5 +  20 + yiter] + ' state covid chart'
                state_data = state_cases[state_cases['state'] == us_states[xiter * 5 +  20 + yiter]]
                axs[xiter, yiter].plot(state_data['week'], state_data['deaths'], label='deaths', color='red')
                axs[xiter, yiter].plot(state_data['week'], state_data['drate'], label='death rate', color='blue')
                axs[xiter, yiter].plot(state_data['week'], state_data['cases'], color='green', label='cases')
                axs[xiter, yiter].set_title(title_str)
            if (breakfree == True):
                break
        plt.show()

        # showing last 10
        fig, axs = plt.subplots(2, 5, sharey=True)
        plt.subplots_adjust(bottom=0.001, top=0.98, wspace=0.2)
        breakfree = False

        for xiter in range(0, 2):
            for yiter in range(0, 5):
                if ((xiter * 5 + 40 + yiter) == len(us_states)):
                    breakfree = True
                    break
                '''
                print("x = {0}, y = {1} state number {2} and name {3}".format(xiter, yiter, xiter * 5 + 40 + yiter,
                                                                              us_states[xiter * 5 + +39+yiter]))
                '''
                title_str = us_states[xiter * 5 + 40 + yiter] + ' state covid chart'
                state_data = state_cases[state_cases['state'] == us_states[xiter * 5 + 40 + yiter]]
                axs[xiter, yiter].plot(state_data['week'], state_data['deaths'], label='deaths', color='red')
                axs[xiter, yiter].plot(state_data['week'], state_data['drate'], label='death rate', color='blue')
                axs[xiter, yiter].plot(state_data['week'], state_data['cases'], color='green', label='cases')
                axs[xiter, yiter].set_title(title_str)
            if (breakfree == True):
                break
        plt.show()

    else:
        state_data = state_cases[state_cases['state']== state]
        print(state_data.head(10))
        title_str = state + ' state covid chart'
        sns.lineplot(x=state_data['week'], y=state_data['deaths'], label='deaths')
        sns.lineplot(x=state_data['week'], y=state_data['cases'], color='blue', label='cases')
        plt.title(title_str, fontsize=18)
        plt.xlabel('week number', fontsize=16)
        plt.ylabel('no. of cases/deaths', fontsize=16)
        plt.show()

        g = sns.barplot(x='week', y='cases', data=state_data, color='yellow')
        sns.barplot(x='week', y='deaths', data=state_data, color='red')
        plt.title(title_str, fontsize=18)
        plt.show()

        sns.lineplot(x=state_data['week'], y=state_data['drate'], label='death rate')
        plt.title(title_str, fontsize=18)
        plt.show()


        #compare each state with national average



def national_covid_chart():
    #access the USA chart of deaths and cases over a period of time.
    national_cases = pd.read_csv('../data/us.csv')
    national_cases['date'] = pd.to_datetime(national_cases['date'])
    national_cases['week'] = national_cases['date'].dt.strftime('%W')


    nw = national_cases['week'].unique()
    national_week_means = pd.DataFrame({'week': nw})
    national_week_means['deaths_mean']= 0
    national_week_means['cases_mean']=0
    national_week_means['death_rate'] = 0

    loop_run = 0
    for witer in national_week_means['week']:
        print(witer)
        national_week_means.loc[loop_run, 'cases_mean']= national_cases[national_cases['week']==witer]['cases'].mean()
        national_week_means.loc[loop_run, 'deaths_mean'] = national_cases[national_cases['week'] == witer]['deaths'].mean()
        loop_run = loop_run + 1
    national_week_means['death_rate'] = (national_week_means['deaths_mean'] * 100) / national_week_means['cases_mean']

    #printing some basic data
    print(national_cases.columns)
    print(national_cases.head())
    print(national_cases.tail())
    print(national_cases.shape)

    sns.lineplot(x=national_cases['date'], y=national_cases['deaths'], label= 'deaths')
    sns.lineplot(x=national_cases['date'], y=national_cases['cases'],color= 'blue', label= 'cases')
    plt.title('national covid chart', fontsize=18)
    plt.xlabel('dates', fontsize=16)
    plt.ylabel('no. of cases/deaths', fontsize=16)
    plt.show()

    sns.lineplot(x='week', y='death_rate', data=national_week_means, color='orange')
    plt.title('national weekly death rate', fontsize=18)
    plt.show()

    print(national_week_means.loc[:,'death_rate'])
    return national_cases



def national_covid_prediction():
    national_cases = pd.read_csv('../data/us.csv')
    national_cases['date']=pd.to_datetime(national_cases['date'])
    print(national_cases.head)
    base_date = national_cases.iloc[0,0]
    national_cases['day']= (national_cases['date']- base_date)
    national_cases['day']= national_cases['day'] / np.timedelta64(1, 'D')

    predict_df = national_cases.iloc[:,1:4]
    print(predict_df.head)
    national_covid_cases_pr(predict_df)
    national_covid_deaths_pr(predict_df)
    national_covid_deathrate_pr(predict_df)
    national_covid_deathrate_svr(predict_df)



    return

def national_covid_cases_pr(df):
    #splitting the data in x and y
    df_day_X = df.iloc[:, 2].to_numpy()
    df_cases_Y = df.iloc[:,0].to_numpy()

    #model training time.
    pf_covid = pf(degree=2)
    X_poly = pf_covid.fit_transform(df_day_X.reshape(-1,1))

    lr_model = LinearRegression()
    lr_model.fit(X_poly, df_cases_Y)
    y_pred = lr_model.predict(X_poly)

    print("r2 score {}".format(r2_score(df_cases_Y,y_pred)))
    print("mean squared error score {}".format(mean_squared_error(df_cases_Y,y_pred)))
    print("mean absolute error score {}".format(mean_absolute_error(df_cases_Y, y_pred)))
    #print("estimator values {}".format(lr_model.estimators_))

    sns.scatterplot(x=df_day_X, y=df_cases_Y, color='blue', label='cases')
    sns.scatterplot(x=df_day_X, y = y_pred, color = 'red', label='predictions')
    plt.title('national cases covid chart', fontsize=18)
    plt.xlabel('days', fontsize=16)
    plt.ylabel('no. of cases', fontsize=16)
    plt.show()

    rem_days = np.arange(111, 365,1).reshape(-1,1)
    rem_poly = pf_covid.fit_transform(rem_days)
    rem_cases = lr_model.predict(rem_poly)

    sns.lineplot(x=df_day_X, y=df_cases_Y, color='blue', label='cases')
    sns.lineplot(x=rem_days.ravel(), y=rem_cases, color='red', label='predictions')
    plt.title('national covid cases prediction for the year', fontsize=18)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('days', fontsize=16)
    plt.ylabel('no. of cases', fontsize=16)
    plt.show()

    return

def national_covid_deaths_pr(df):
    #splitting the data in x and y
    df_day_X = df.iloc[:, 2].to_numpy()
    df_deaths_Y = df.iloc[:,1].to_numpy()

    #model training time.
    pf_covid = pf(degree=2)
    X_poly = pf_covid.fit_transform(df_day_X.reshape(-1,1))

    lr_model = LinearRegression()
    lr_model.fit(X_poly, df_deaths_Y)
    y_pred = lr_model.predict(X_poly)

    print("r2 score {}".format(r2_score(df_deaths_Y,y_pred)))
    print("mean squared error score {}".format(mean_squared_error(df_deaths_Y,y_pred)))
    print("mean absolute error score {}".format(mean_absolute_error(df_deaths_Y, y_pred)))
    #print("estimator values {}".format(lr_model.estimators_))

    sns.scatterplot(x=df_day_X, y=df_deaths_Y, color='blue', label='deaths')
    sns.scatterplot(x=df_day_X, y = y_pred, color = 'red', label='death predictions')
    plt.title('national covid death chart', fontsize=18)
    plt.xlabel('days', fontsize=16)
    plt.ylabel('no. of deaths', fontsize=16)
    plt.show()

    rem_days = np.arange(111, 365,1).reshape(-1,1)
    rem_poly = pf_covid.fit_transform(rem_days)
    rem_cases = lr_model.predict(rem_poly)

    sns.lineplot(x=df_day_X, y=df_deaths_Y, color='blue', label='deaths')
    sns.lineplot(x=rem_days.ravel(), y=rem_cases, color='red', label='death predictions')
    plt.title('national covid death prediction for the year', fontsize=18)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('days', fontsize=16)
    plt.ylabel('no. of death', fontsize=16)
    plt.show()

    return

def national_covid_deathrate_pr(df):
    df['death_rate'] = (df['deaths'] * 100) / df['cases']
    print(df.head(10))

    df_day_X = df.iloc[:, 2].to_numpy()
    df_dr_Y = df.iloc[:, 3].to_numpy()

    # model training time.
    pf_covid = pf(degree=2)
    X_poly = pf_covid.fit_transform(df_day_X.reshape(-1, 1))

    lr_model = LinearRegression()
    lr_model.fit(X_poly, df_dr_Y)
    y_pred = lr_model.predict(X_poly)

    print("r2 score {}".format(r2_score(df_dr_Y, y_pred)))
    print("mean squared error score {}".format(mean_squared_error(df_dr_Y, y_pred)))
    print("mean absolute error score {}".format(mean_absolute_error(df_dr_Y, y_pred)))
    # print("estimator values {}".format(lr_model.estimators_))

    sns.scatterplot(x=df_day_X, y=df_dr_Y, color='blue', label='deaths')
    sns.scatterplot(x=df_day_X, y=y_pred, color='red', label='death predictions')
    plt.title('national covid death chart', fontsize=18)
    plt.xlabel('days', fontsize=16)
    plt.ylabel('death rate', fontsize=16)
    plt.show()

    rem_days = np.arange(111, 365, 1).reshape(-1, 1)
    rem_poly = pf_covid.fit_transform(rem_days)
    rem_cases = lr_model.predict(rem_poly)

    sns.lineplot(x=df_day_X, y=df_dr_Y, color='blue', label='deaths')
    sns.lineplot(x=rem_days.ravel(), y=rem_cases, color='red', label='death predictions')
    plt.title('national covid death rate prediction for the year', fontsize=18)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('days', fontsize=16)
    plt.ylabel('rate of death', fontsize=16)
    plt.show()

    return

def national_covid_deathrate_svr(df):
    df['death_rate'] = (df['deaths'] * 100) / df['cases']
    print(df.head(10))

    df_day_X = df.iloc[:, 2].to_numpy()
    df_dr_Y = df.iloc[:, 3].to_numpy()

    covid_svr = SVR()
    covid_svr.fit(df_day_X.reshape(-1, 1), df_dr_Y)

    rem_days = np.arange(111, 365, 1).reshape(-1, 1)
    rem_dr = covid_svr.predict(rem_days)

    sns.lineplot(x=df_day_X, y=df_dr_Y, color='blue', label='deaths')
    sns.lineplot(x=rem_days.ravel(), y=rem_dr, color='red', label='death predictions')
    plt.title('national covid death prediction for the year', fontsize=18)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('days', fontsize=16)
    plt.ylabel('no. of death', fontsize=16)
    plt.show()

    return

def getcols4range(end: int, colprefix:str )->pd:
    no_of_cols = 0
    for no_of_cols in range (0, end):
        if((2**no_of_cols) >= end):
            break

    states_cols = pd.DataFrame(columns=np.arange(no_of_cols))
    states_cols = states_cols.add_prefix(colprefix)
    for val in range(0,end):
        bin_val = bin(val)
        place_holder = bin_val.find('0b')
        bin_val = bin_val[place_holder+2:].zfill(6)
        array = np.array(list(bin_val))
        states_cols.loc[val] = array
    state_name = np.array(us_states)
    state_name = state_name.reshape(-1,1)
    states_cols['statenames']=state_name
    return states_cols

def cleaup_state_population_data(states_covid_file_path: str ):
    df_states_population = pd.read_csv('../data/ACSDP1Y2018.DP05_data_with_overlays_2020-05-17T083055.csv',header=1)
    df_states_population.drop(df_states_population.columns[df_states_population.columns.str.contains('^Margin')], axis=1,inplace=True)
    df_states_population.drop(df_states_population.columns[df_states_population.columns.str.contains('^Percent')],axis=1, inplace=True)
    df_states_population.drop(columns=['id'], axis=1, inplace=True)

    #removing columns that are redundant
    cols = ['Estimate!!SEX AND AGE!!Total population!!Sex ratio (males per 100 females)',
       'Estimate!!SEX AND AGE!!Total population!!Under 18 years',
       'Estimate!!SEX AND AGE!!Total population!!16 years and over',
       'Estimate!!SEX AND AGE!!Total population!!18 years and over',
       'Estimate!!SEX AND AGE!!Total population!!21 years and over',
       'Estimate!!SEX AND AGE!!Total population!!62 years and over',
       'Estimate!!SEX AND AGE!!Total population!!65 years and over',
       'Estimate!!SEX AND AGE!!Total population!!18 years and over.1',
       'Estimate!!SEX AND AGE!!Total population!!18 years and over!!Male',
       'Estimate!!SEX AND AGE!!Total population!!18 years and over!!Female',
       'Estimate!!SEX AND AGE!!Total population!!18 years and over!!Sex ratio (males per 100 females)',
       'Estimate!!SEX AND AGE!!Total population!!65 years and over.1',
       'Estimate!!SEX AND AGE!!Total population!!65 years and over!!Male',
       'Estimate!!SEX AND AGE!!Total population!!65 years and over!!Female',
       'Estimate!!SEX AND AGE!!Total population!!65 years and over!!Sex ratio (males per 100 females)',
       'Estimate!!SEX AND AGE!!Total population!!65 years and over!!Male',
       'Estimate!!SEX AND AGE!!Total population!!65 years and over!!Female',
       'Estimate!!SEX AND AGE!!Total population!!65 years and over!!Sex ratio (males per 100 females)',
       'Estimate!!RACE!!Total population',
       'Estimate!!RACE!!Total population!!One race',
       'Estimate!!RACE!!Total population!!Two or more races',
       'Estimate!!RACE!!Total population!!One race.1',
       'Estimate!!CITIZEN, VOTING AGE POPULATION!!Citizen, 18 and over population',
       'Estimate!!CITIZEN, VOTING AGE POPULATION!!Citizen, 18 and over population!!Male',
       'Estimate!!CITIZEN, VOTING AGE POPULATION!!Citizen, 18 and over population!!Female']


    df_states_population.drop(columns=cols, axis=1, inplace=True)


    #take care of null and empty vals
    print('############################################################################################')
    df_states_population = df_states_population.replace(r'', np.NaN)
    df_states_population = df_states_population.replace('N', np.NaN)


    print(df_states_population.isna().sum().sum())

    nan_columns = df_states_population.columns[df_states_population.isnull().any()]
    for col in nan_columns:
        print(col, (df_states_population[col].isna().sum()*100)/df_states_population.shape[0] )

    #removing columns with more than 30 percent missing data
    cols = ['Estimate!!RACE!!Total population!!One race!!American Indian and Alaska Native!!Cherokee tribal grouping',
    'Estimate!!RACE!!Total population!!One race!!American Indian and Alaska Native!!Chippewa tribal grouping',
    'Estimate!!RACE!!Total population!!One race!!American Indian and Alaska Native!!Navajo tribal grouping',
    'Estimate!!RACE!!Total population!!One race!!American Indian and Alaska Native!!Sioux tribal grouping',
    'Estimate!!RACE!!Total population!!One race!!Native Hawaiian and Other Pacific Islander!!Native Hawaiian',
    'Estimate!!RACE!!Total population!!One race!!Native Hawaiian and Other Pacific Islander!!Guamanian or Chamorro',
    'Estimate!!RACE!!Total population!!One race!!Native Hawaiian and Other Pacific Islander!!Samoan',
    'Estimate!!RACE!!Total population!!One race!!Native Hawaiian and Other Pacific Islander!!Other Pacific Islander']

    df_states_population.drop(columns=cols, axis=1, inplace=True)

    df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Asian Indian'].fillna(df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Asian Indian'].astype(float).mean(),inplace =True)
    df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Other Asian'].fillna(df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Other Asian'].astype(float).mean(), inplace=True)
    df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Chinese'].fillna(df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Chinese'].astype(float).mean(), inplace=True)
    df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Filipino'].fillna(df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Filipino'].astype(float).mean(), inplace=True)
    df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Japanese'].fillna(df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Japanese'].astype(float).mean(), inplace=True)
    df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Korean'].fillna(df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Korean'].astype(float).mean(), inplace=True)
    df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Vietnamese'].fillna(df_states_population['Estimate!!RACE!!Total population!!One race!!Asian!!Vietnamese'].astype(float).mean(), inplace=True)

    #apply PCA
    #removing states information and scaling all the columns to get right PCA
    state_name = df_states_population['Geographic Area Name']
    df_states_population.drop(columns=['Geographic Area Name'], inplace = True,axis = 1)

    #applying scaler
    ss_states_pop = StandardScaler()
    df_states_population = pd.DataFrame(ss_states_pop.fit_transform(df_states_population),columns = df_states_population.columns)


    states_population_pca = PCA()
    states_population_pca.fit_transform(df_states_population)
    df_states_pca = pd.DataFrame(states_population_pca.components_, columns=df_states_population.columns)
    print('df states pca shape {0}. df states population shape {1}'.format(df_states_pca.shape, df_states_population.shape))
    df_states_pca.columns = df_states_population.columns
    states_evr = states_population_pca.explained_variance_ratio_
    states_ev = states_population_pca.explained_variance_

    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.plot(np.cumsum(states_evr))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()

    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.plot(states_ev)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()



    # PCA explained variance
    ratios = states_population_pca.explained_variance_ratio_.reshape(len(states_population_pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
    print(variance_ratios)
    #selecting no. of components to get maximum accuracy
    for comp in range(5, df_states_population.shape[1]):
        pca = PCA(n_components=comp)
        state_pop_PCA = pd.DataFrame(pca.fit_transform(df_states_population))
        # PCA components
        dimensions = ['Dimension {0}'.format(i) for i in range(1, len(pca.components_) + 1)]
        print(dimensions)
        components = pd.DataFrame(np.round(pca.components_, 4), columns=df_states_population.keys())
        components.index = dimensions

        # PCA explained variance
        ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
        variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
        variance_ratios.index = dimensions
        # print(variance_ratios['Explained Variance'].sum())
        if variance_ratios['Explained Variance'].sum() > 0.90:
            print('no. of features to retain {0} and sum of explained variance is {1}'.format(comp, variance_ratios[
                'Explained Variance'].sum()))
            break

    pca = PCA(n_components=comp)
    df_states_pca = pd.DataFrame(pca.fit_transform(df_states_population))
    df_states_pca.columns = ['state_pop1','state_pop2','state_pop3','state_pop4','state_pop5']

    #adding state names again for merging later on
    df_states_pca['Geographic Area Name'] = state_name
    print('############################################################################################')

    print(df_states_pca.shape)
    print(df_states_pca.head(5))

    state_cases = pd.read_csv('../data/us_states.csv')

    #replace dates with days
    state_cases['date'] = pd.to_datetime(state_cases['date'])
    print(state_cases.head)
    base_date = state_cases.iloc[0, 0]
    state_cases['day'] = (state_cases['date'] - base_date)
    state_cases['day'] = state_cases['day'] / np.timedelta64(1, 'D')


    '''
    df_merge = df_states_population.merge(state_cases,left_on='Geographic Area Name', right_on='state' )
    print(df_merge.columns)
    print(df_merge.head(10))
    '''
    #applying standard scaling and feature scaling

    #tackling states first
    df_states_bin = getcols4range(52,'states')

    #merging states column with main
    #df_merge = df_merge.merge(df_states_bin, left_on='Geographic Area Name', right_on='statenames')
    #df_merge.drop(columns=['fips', 'date', 'state','statenames','Geographic Area Name'], inplace=True)
    df_cases_encoded = state_cases.merge(df_states_bin, left_on='state', right_on='statenames')
    df_cases_encoded.drop(columns=['fips', 'date', 'statenames' ], inplace=True)
    #apply scaling
    #df_merge = df_merge.astype(np.float)
    temp_states_col = df_cases_encoded['state']
    df_cases_encoded.drop(columns=['state'], inplace=True)
    ss = StandardScaler()
    df_cases_encoded = pd.DataFrame(ss.fit_transform(df_cases_encoded),columns= df_cases_encoded.columns)
    df_cases_encoded['state']= temp_states_col

    #merging state covid df and state population df on state names
    df_cases_encoded = df_cases_encoded.merge(df_states_pca, left_on='state', right_on='Geographic Area Name')
    df_cases_encoded.drop(columns=['state','Geographic Area Name'], inplace=True)


    return  df_cases_encoded,temp_states_col

def apply_clustering(df_covid_population:pd,state_col:pd):

    #AgglomerativeClustering()
    dendrogram = sch.dendrogram(sch.linkage(df_covid_population, method='ward'))
    plt.show()

    #analyzing the clustering
    cluster_nos = [3,4,5,6]
    silhouette_scores = []
    for k in cluster_nos:
        ac_model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
        silhouette_scores.append(silhouette_score(df_covid_population, ac_model.fit_predict(df_covid_population)))
    plt.bar(cluster_nos, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('S(i)', fontsize=20)
    plt.show()

    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    model.fit(df_covid_population)
    labels = model.labels_

    df_covid_population['label']= labels
    df_covid_population['states']= state_col
    data = df_covid_population.groupby(['label','states'])
    for group, val in data:
        print(group)


if __name__ == "__main__":
    '''
    nation_df = national_covid_chart()
    national_covid_prediction()
    us_state_covid_chart('')
    '''
    us_state_covid_chart('')
    #cleanup population data and join with the main states covid cases
    df_state_covid_population, states_col = cleaup_state_population_data('../data/us_states.csv')
    apply_clustering(df_state_covid_population,states_col)