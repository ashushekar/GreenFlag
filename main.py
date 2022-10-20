"""
# Stolen phones in a nightclub
## Scenario
Your friend owns a nightclub, and the nightclub is suffering an epidemic
of stolen phones. At least one thief has been frequenting her club and
stealing her visitors&#39; phones. Her club has a licence scanner at its
entrance, that records the name and date-of-birth of everyone who enters
the club - so she should have the personal details of the thief or
thieves; it&#39;s just mixed in with the details of her honest customers. She
heard you call yourself a &quot;data scientist&quot;, so has asked you to come up
with a ranked list of up to 20 suspects to give to the police.
She&#39;s given you:
`visitor_log.csv` - details of who visited the club and on what day
(those visiting 2AM Tuesday are counted as visiting on Monday).
`theft_log.csv&#39; - a list of days on which thefts were reported to occur
(again, thefts after midnight are counted as the previous day - we&#39;re
being nice to you)
She wants from you:
- A list of ID details for the 20 most suspicious patrons, ranked from
most-suspicious to least-suspicious.
- If you think there are fewer than 20 thieves, a list of ID details for
everyone that you think is a thief.
- A brief explanation of your methodology.
## metadata

"""

import shutil
from itertools import combinations

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient
from scipy.stats import zscore
from plotly import graph_objects


def get_experiment_id():
    """Check if an experiment exists, if not create it"""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name('stolen_phones_program')
        return experiment, experiment.experiment_id
    except Exception as e:
        experiment = mlflow.get_experiment(mlflow.create_experiment('stolen_phones_program'))
        return experiment, experiment.experiment_id


def get_run_id():
    """Get the run id"""
    client = MlflowClient()
    experiment_id = get_experiment_id()[1]
    runs = client.search_runs(experiment_ids=experiment_id)
    return runs[0].info.run_id


def print_exp_info():
    """Prints the experiment info"""
    experiment_id = get_experiment_id()[1]
    client = MlflowClient()
    print(f'Experiment name: {client.get_experiment(experiment_id).name}')
    print(f'Experiment id: {experiment_id}')
    print(f'Experiment runs: {len(client.list_run_infos(experiment_id))}')


def del_exp():
    """Deletes the experiment with mlruns folder"""
    client = MlflowClient()
    experiment_id = get_experiment_id()[1]
    client.delete_experiment(experiment_id)
    shutil.rmtree('mlruns')


exp, exp_id = get_experiment_id()
print_exp_info()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def read_visitors():
    """Reads the visitor log and returns a dataframe"""
    df = pd.read_csv('data/Visitor Log.csv')
    df['visit_date'] = pd.to_datetime(df['visit_date'])

    # 2. Drop missing values and duplicates
    df = df[['name', 'dob', 'visit_date']].dropna().drop_duplicates()

    # 3. Sort by visit date
    df = df.sort_values(by='visit_date')

    return df


def read_thefts():
    """Reads the theft log and returns a dataframe"""
    df = pd.read_csv('data/Theft Log.csv', header=None)
    df.columns = ['theft_date']
    df['theft_date'] = pd.to_datetime(df['theft_date'])
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 1. Read in the visitors data
    visitors_df = read_visitors()
    thefts_df = read_thefts()

    # 2. Merge the two dataframes by date column
    # since there is the date record in both the datasets, we can merge them
    # on the date column
    merged_df = pd.merge(visitors_df,
                         thefts_df, how='left',
                         left_on='visit_date',
                         right_on='theft_date')

    # 3. Add 1 if theft occurred on that day or 0 if not
    merged_df['theft'] = np.where(merged_df['theft_date'].isnull(), 0, 1)

    # Now it is possible to group the data by the visitor name and dob,
    # then count how many times that visitor was at the club when there was a robbery.
    # Visualize the results using a bar chart with plotly.
    # 4. Group by visitor name and dob
    grouped_df = merged_df.groupby(['name', 'dob']).agg({'theft': 'sum'}).sort_values(by='theft',
                                                                                      ascending=False).reset_index()
    # Join name and dob columns
    grouped_df['name_dob'] = grouped_df['name'] + ' ' + grouped_df['dob']

    # add color column to the dataframe to color the bars
    grouped_df['color'] = np.where(grouped_df['theft'] > 20, 'red', 'blue')
    karen_df = merged_df[(merged_df['name'] == 'Karen Keeney') & (merged_df['dob'] == '25/12/1993')]

    # now plot with name as x axis and theft as y axis
    fig = px.bar(grouped_df, x='name_dob', y='theft', color='color')
    fig.update_layout(
        title='Number of thefts per visitor',
        xaxis_title='Visitor name and Date of Birth',
        yaxis_title='Number of thefts',
        showlegend=False
    )

    fig.add_annotation(
        x=0.7,
        y=0.8,
        text=f"Karen Keeney's thefts: {karen_df['theft'].sum()}",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="blue"
        ),
        align="center",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=1,
        opacity=0.8
    )
    fig.update_xaxes(nticks=5)
    fig.show()
    # Add plot to mlflow
    mlflow.log_figure(fig, 'thefts_per_visitor.png')
    # From the plot we can see that Karen Keeney is the visitor
    # who was most often at the club when there was a robbery

    # 5. Let's see how many times Karen was at the club when there was a robbery
    print(f"Karen was at the club {karen_df['theft'].sum()} times when there was a robbery")

    # Create density distribution plot with plotly
    merged_df.groupby('name').theft.sum().sort_values(ascending=False).plot(legend=False, kind='kde')
    plt.title('Distribution of thefts per visitor')
    plt.xlabel('Number of thefts')
    plt.ylabel('Density')
    plt.show(block=False)
    # Add plot to mlflow
    mlflow.log_figure(fig, 'density_distribution.png')
    # Looks like a normal distribution

    # 6. Lets count how many visitors were at the club when there was a robbery
    theft_sum_df = merged_df.groupby(['name', 'dob']).theft.sum().sort_values(ascending=False).reset_index()
    theft_sum_df = theft_sum_df[theft_sum_df['theft'] != 0]
    theft_sum_df.columns = ['name', 'dob', 'theft_count']
    print(f"There were {theft_sum_df.shape[0]} visitors at the club when there was a robbery")
    print(theft_sum_df.head(10))
    # save this information in a csv file and add it to mlflow
    theft_sum_df.to_csv('theft_count_during_robbery.csv', index=False)
    mlflow.log_artifact('theft_count_during_robbery.csv')

    # 7. Lets see how many visitors were at the club when there was no robbery
    no_theft_df = merged_df[merged_df['theft'] == 0]
    no_theft_df = no_theft_df.groupby(['name', 'dob']).theft.count().sort_values(ascending=False).reset_index()
    no_theft_df.columns = ['name', 'dob', 'no_theft_count']
    print(f"There were {no_theft_df.shape[0]} visitors at the club when there was no robbery")
    print(no_theft_df.head(10))
    # save this information in a csv file and add it to mlflow
    no_theft_df.to_csv('theft_count_during_no_robbery.csv', index=False)
    mlflow.log_artifact('theft_count_during_no_robbery.csv')

    # 8. lets see how much of them actually overlap
    # merge the two dataframes
    merged_theft_df = pd.merge(theft_sum_df, no_theft_df, how='inner', on=['name', 'dob'])
    print(
        f"There were {merged_theft_df.shape[0]} visitors who were at the club when there was a robbery and when there was no robbery")
    print(merged_theft_df.head(10))

    # join with original visitors dataframe
    stats_df = pd.merge(merged_df, theft_sum_df, how='inner', on=['name', 'dob'])
    stats_df = pd.merge(stats_df, no_theft_df, how='inner', on=['name', 'dob'])

    # fillna with 0
    stats_df = stats_df.fillna(0)

    # calculate the ratio of thefts to total visits
    stats_df['total_visits'] = stats_df['theft_count'] + stats_df['no_theft_count']
    stats_df['theft_freq'] = stats_df['theft_count'] / stats_df['total_visits']
    stats_df = stats_df[stats_df['theft'] != 0]

    # lets sort this in order of theft frequency
    stats_df = stats_df.sort_values(['theft_freq', 'theft_count'], ascending=False)

    # Now lets calculate zscore for the theft frequency
    numeric_cols = stats_df.drop(['theft_date', 'theft'], axis=1).select_dtypes(include=[np.number]).columns
    zscore_df = stats_df[numeric_cols].apply(zscore)
    zscore_df.columns = [f'{col}_zscore' for col in zscore_df.columns]
    zscore_df = zscore_df.sort_values(['theft_count_zscore', 'theft_freq_zscore'], ascending=False)
    # print(zscore_df.head(10))

    # merge zscore with stats_df
    final_df = stats_df.join(zscore_df).sort_values(['theft_count_zscore', 'theft_freq_zscore'], ascending=False)
    # save this information in a csv file and add it to mlflow
    final_df.to_csv('final_theft_stats.csv', index=False)

    # First lets select just the data when there was a theft and lets do the crosstab
    cross_tab = pd.crosstab(stats_df['name'], stats_df['visit_date'], margins=True)

    # We do the crosstab because it is easier to find out the gang members with data in this format!
    cross_tab_mat = cross_tab.reset_index().values

    # Lets count how many times two visitors were at the club on the same date, when a phone was stolen

    criminal_friends = []
    for r1, r2 in combinations([x for x in cross_tab_mat], 2):
        matches = [i for i, j in zip(r1, r2) if i == j and i != 0]
        criminal_friends.append((r1[0], r2[0], len(matches)))

    criminal_friends_ranking = sorted(criminal_friends, key=lambda x: x[2], reverse=True)
    print(criminal_friends_ranking[:10])

    # Lets select the top names as possible thiefs
    # I use 21 as a threshold because the number of names retrieved is less than 20
    criminal_names = set()
    for t in criminal_friends_ranking:
        if t[2] > 21:
            criminal_names.add(t[0])
            criminal_names.add(t[1])

    print(len(criminal_names))

    # Finally here the list of possible thiefs of the nightclub!!
    # If I have to choose, I would say that Karen Keeney is the boss! And the following names are most
    # likely her accomplices
    criminals = final_df[final_df.name.isin(criminal_names)].sort_values(['theft_count', 'theft_freq'],
                                                                       ascending=False)

    # save this information in a csv file and add it to mlflow
    criminals.to_csv('data/criminals.csv', index=False)
    mlflow.log_artifact('data/criminals.csv')

    # now get the list of possible thiefs
    possible_thiefs = criminals.name.unique()
    print(possible_thiefs)

    # del_exp()
