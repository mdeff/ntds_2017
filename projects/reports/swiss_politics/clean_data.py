import pandas as pd
import numpy as np


def get_voter_data():
    """
    Loads the 2015 Smartvote data and cleans it to output a pandas Dataframe containing the answers to the questions and
    the recommended party. Each party has an index coresponding to the second output variable, which is a list of party
    names.

    :return: selection: Table containing the answers for each voter and their party recommendation as well as canton
    :return: parties: Political parties and their respective indexes used for the voter recommendation
    """

    # Load Data from csv file containing Swiss election 2015 survey dataset
    df = pd.read_csv('../Data/SFE2015/sv_2015_recommendations_1x.csv', sep=';', na_values=-7)

    # Remove unnecessary columns
    df.drop(df.columns[list(range(5))+list(range(82, 166))+list(range(167, 217, 2))], axis=1, inplace=True)

    # Keep only the rows corresponding to people who answered all questions
    selection = df[df['n_answers'] == 75]
    selection.pop('n_answers')

    # Keep only the index of the best recommended party for each person
    selection = selection.assign(recommendation=[-1]*len(selection))
    for i in range(25):
        selection.loc[selection[selection.columns[76+i]] == 1, 'recommendation'] = i
    parties = [party[6:-5] for party in list(selection.columns[76:100])] + ['Others']
    selection.drop(selection.columns[list(range(76, 101))], axis=1, inplace=True)

    # Transform the 0-100 answer scheme to a -1 to 1 quantization
    quantizer = lambda x: x/50-1
    selection.iloc[:, list(range(1, 76))] = selection.iloc[:, list(range(1, 76))].applymap(quantizer)

    return selection, parties


def get_candidate_data():
    """
    Loads the 2015 Smartvote candidate data and cleans it to output a pandas Dataframe containing the answers to the
    questions and informations like party, gender, age, district (canton).

    :return:
    selecction: Table containing the answers for each candidate and their associated party as well as data such as
    affiliated party, gender, age and canton (district).
    """

    # Load Data from csv file containing Swiss election 2015 candidate dataset
    df = pd.read_csv('../Data/SFE2015/smartvote_C_2015_V2.0.csv', sep=';', na_values=-9)

    # Get inndex dictionary for the parties
    candidate_parties = pd.read_csv('../Data/SFE2015/candidate_parties.csv', sep=',', na_values=-9)

    # Remove unnecessary columns
    df.drop(df.columns[list(range(11))+list(range(17, 21)) + list(range(97, 107))], axis=1, inplace=True)

    # Keep only the rows corresponding to people who answered all questions
    selection = df[df['n_answers'] == 75]
    selection.pop('n_answers')

    # Adapt the index encoding the party to which the candidate is affiliated to fit the indexes used for the voter data
    index_dic = {}
    for i in candidate_parties['CandidatePartyIndex'].values:
        value = candidate_parties.loc[candidate_parties['CandidatePartyIndex'] == i]['VoterPartyIndex'].values[0]
        index_dic[i] = value
    translate_party_index = lambda candidate_index: index_dic[candidate_index]
    selection['party_REC'] = selection['party_REC'].apply(translate_party_index)

    # Transform the 0-100 answer scheme to a -1 to 1 quantization
    quantizer = lambda x: x/50-1
    selection.iloc[:, list(range(6, 81))] = selection.iloc[:, list(range(6, 81))].applymap(quantizer)

    return selection
