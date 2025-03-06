import pandas as pd
from rapidfuzz import process, fuzz
import os
# from .predict_model import run_predictions_for_date


def get_aggregated_data():
    # Path to the CSV file (relative to your current working directory)
    # Since you're calling this from 'routes', we need to go up one directory to 'app' and then into 'utils'
    data_file_path = os.path.join(os.path.dirname(__file__), 'aggregated_total_data.csv')
    
    try:
        # Using pandas to read the CSV
        df_data = pd.read_csv(data_file_path, index_col=False, low_memory=False)
        return df_data
    except FileNotFoundError:
        return None
def runner_main(df_input):
    input_file_path = './Input_Format.csv'
    # data_file_path = 'aggregated_total_data.csv'

    # Add low_memory=False to prevent the warning related to mixed types
    # df_input = pd.read_csv(input_file_path, index_col=False, low_memory=False)
    # df_data = pd.read_csv(data_file_path, index_col=False, low_memory=False)
    df_data = get_aggregated_data()
    
    df_data = df_data[
        (df_data['start_date'] == df_input['Match Date'][0]) &
        (df_data['match_type'] == df_input['Format'][0])
    ]
    
    squads = df_input['Squad'].unique()
    df_data = df_data[df_data['player_team'].isin(squads)]
    target_names = df_input['Player Name'].to_list()
    
    def map_name(name, source_names):
        match, score, _ = process.extractOne(name, source_names, scorer=fuzz.ratio)
        return match if score > 60 else None  
    
    mapped_names = [
        {'Target Name': target, 'Mapped Name': map_name(target, df_data['full_name'].tolist())}
        for target in target_names
    ]
    
    mapped_df = pd.DataFrame(mapped_names) 
    result_df = pd.merge(mapped_df, df_data[['full_name', 'player_id','player_team']], left_on='Mapped Name', right_on='full_name', how='left')
    result_df = result_df.drop(columns=['full_name'])
    
    inp = result_df['player_id'].to_list()
    # date = df_input['Match Date']
    # print(result_df)
    # print(inp)
    return result_df
    # print(date)
    # run_predictions_for_date(date, inp)

if __name__ == "__main__":
    runner_main()
