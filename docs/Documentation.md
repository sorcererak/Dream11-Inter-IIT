# data_download.py
Master function to download and populate `/data/raw/` with the cricsheet data, as well as `/data/interim/` with the interim csv datasheets generated. 

#### `execute_scraper()`
**Description:** The `execute_scraper` function automates the process of downloading, extracting, and organizing cricket match data from Cricsheet into a structured directory. It handles both the available JSON and CSV data formats, as well as downloading the separate CSV file containing player registry.  

The scraper can be tested in the `/src/scraper/cricsheet.py` file

**Input:** This function takes no arguments.  
**Output:** This function creates the following structure and files in the local filesystem:
- A directory containing unzipped JSON files from Cricsheet.  
- A directory containing unzipped CSV files from Cricsheet.  
- A CSV file for player metadata

#### `export_to_csv(batsmen, bowlers, match_attributes_parsed, batsman_order)`
**Purpose:** Appends match data for all players to the CSV file.  
**Inputs:**  
- `batsmen`: Batting stats dictionary.  
- `bowlers`: Bowling stats dictionary.  
- `match_attributes_parsed`: Parsed match-level attributes.  
- `batsman_order`: Batting order list.  

**Output:** Saves the current dataframe/JSON to the appropriate output file path.

#### `parse_innings_data(innings_data, players_in_match, match_attributes_parsed, player_ids, match_info)`
**Purpose:** Processes innings data to compute player stats and then exports to CSV.  
**Inputs:**  
- `innings_data`: List of innings details.  
- `players_in_match`: List of player names in the match.  
- `match_attributes_parsed`: Parsed match-level attributes.  
- `player_ids`: Player ID mappings.  
- `match_info`: Match metadata.  

**Output:** None

#### `json_generator()`
**Purpose:** Iterates through all JSON files in the directory and processes them into `total_data.csv` which is then saved in the `../data/interim` directory.  
**Input:** None  
**Output:** Returns 1 on successful completion.

#### `rename_date()`
**Purpose:** The `rename_date` function processes a CSV file containing cricket match data, modifies the 'date' column, and renames it to 'start_date'. The function ensures that if the match spans multiple days, only the date of first day is retained in the column. Finally, the function saves the updated data back to the CSV file.

**Inputs:**  
No direct inputs to the function. The function operates on the CSV file located at: `../data/interim/total_data.csv`

**Outputs:**  
Updates the `total_data.csv` file by:
- Splitting the date column based on the delimiter " - " and keeping only the first date.
- Renaming the column `date` to `start_date`.

#### `matchwise_data_generator()`
**Description:** The `matchwise_data_generator` function processes cricket match data by iterating through JSON files and corresponding CSV files, aggregating match statistics, and merging the data into a unified dataframe.

**Inputs:**  
This function does not take direct arguments. It operates on files located in directories specified by the global variables `json_dir` and `csv_dir`.

**Outputs:**  
A combined dataframe containing match wise data, which is saved as a CSV file at the path defined by `output_path_mw_overall`.

#### `adding_names()`
**Description:** The `adding_names` function aggregates player statistics from match data and incorporates player details such as full name, batting style, bowling style, playing role, and team information. It processes data from multiple CSV files, performs statistical aggregation, and saves the final combined data to a new CSV file.
The scraper can be tested in the `/src/scrapers/cricinfo_scraper.py` file

**Inputs:**  
This function does not take direct arguments. It operates on files located in the following paths:
- `/data/raw/cricksheet/people.csv` (player information).
- `/data/interim/total_data.csv` (match-level data).

**Outputs:**  
A CSV file located at the required output path, which contains the aggregated player statistics along with additional player details (e.g., full name, teams).


# feature_generation.py

#### `rolling_dot_balls_features()`
**Description:** The `rolling_dot_balls_features` function calculates the dot ball percentage for a bowler using rolling windows of different sizes (3, 7, and 12 matches). It computes the metrics by applying a rolling sum to the bowled and dot balls, sorted by match date.

**Inputs:**  
- `group`: A DataFrame containing bowler statistics for a specific player, with columns like `start_date`, `balls_bowled`, and `dot_balls_as_bowler`.  
- `n1`: The window size (in number of matches) for the first dot ball percentage calculation (default is 3).  
- `n2`: The window size (in number of matches) for the second dot ball percentage calculation (default is 7).  
- `n3`: The window size (in number of matches) for the third dot ball percentage calculation (default is 12).  

**Outputs:**  
A DataFrame with additional columns representing the dot ball percentage for the respective rolling window sizes:
- Dot ball percentage for the rolling window of size `n1`.
- Dot ball percentage for the rolling window of size `n2`.
- Dot ball percentage for the rolling window of size `n3`.

#### `longtermfeatures_dot_balls()`
**Description:** The `longtermfeatures_dot_balls` function calculates the long-term dot ball percentage for a bowler using an expanding window. It also computes the overall dot ball percentage and the variance of dot ball percentage over time.

**Inputs:**  
- `group`: A DataFrame containing bowler statistics, including columns like `start_date`, `balls_bowled`, and `dot_balls_as_bowler`.

**Outputs:**  
A DataFrame with additional columns:
- `longterm_dot_ball_percentage`: Long-term dot ball percentage using an expanding sum.
- `dot_ball_percentage`: Dot ball percentage for each match.
- `longterm_var_dot_ball_percentage`: Variance of the dot ball percentage over time.

#### `calculate_rolling_batting_stats_test()`
**Description:** The `calculate_rolling_batting_stats_test` function calculates rolling batting statistics such as averages, strike rates, and boundary percentages for a player over different rolling window sizes. It applies conditional logic based on a minimum number of balls faced and computes these metrics for three different window sizes: `n1`, `n2`, and `n3`.

**Inputs:**  
- `group`: A DataFrame containing batting statistics, including `runs_scored`, `balls_faced`, `player_out`, `fours_scored`, and `sixes_scored`.  
- `n1` (default=3): Window size for the first rolling window.  
- `n2` (default=7): Window size for the second rolling window.  
- `n3` (default=12): Window size for the third rolling window.  
- `min_balls` (default=20): Minimum number of balls faced to calculate valid strike rates.

**Outputs:**  
A DataFrame with additional columns:
- Batting averages over the windows
- Strike rates over the windows
- Boundary percentages over the windows

#### `calculate_rolling_bowling_stats_test()`
**Description:** The `calculate_rolling_bowling_stats_test` function calculates rolling bowling statistics such as bowling averages, economy rates, strike rates, and an updated CBR for a player over different rolling window sizes. It also computes fielding points based on catches, stumpings, and run outs.

**Inputs:**  
- `group`: A DataFrame containing bowling statistics, including `runs_conceded`, `wickets_taken`, `balls_bowled`, `balls_per_over`, `catches_taken`, `stumpings_done`, `run_out_direct`, and `run_out_throw`.  
- `n1` (default=3): Window size for the first rolling window.  
- `n2` (default=7): Window size for the second rolling window.  
- `n3` (default=12): Window size for the third rolling window.  

**Outputs:**  
A DataFrame with additional columns:
- Bowling averages over the windows
- Economy rates over the windows
- Bowling strike rates over the windows
- A computed bowling CBR value based on the rolling statistics from `n2`.
- Fielding points calculated from rolling aggregates of fielding events (catches, stumpings, and run outs).

#### `calculate_alpha_batsmen_score()`
**Description:**  
The `calculate_alpha_batsmen_score` function calculates the tailored α_batsmen_score for Dream11 point prediction in ODIs over multiple rolling time horizons (n1, n2, n3). It considers factors such as runs scored, strike rate, boundary counts, half-centuries, centuries, and ducks to compute a performance score for each player.

**Inputs:**  
- `group`: A DataFrame containing batting statistics such as `runs_scored`, `strike_rate_n1`, `sixes_scored`, `fours_scored`, `half_centuries_cumsum`, `centuries_cumsum`, and `rolling_ducks_n1`.  
- `n1` (default=3): Window size for the first rolling window (time horizon 1).  
- `n2` (default=7): Window size for the second rolling window (time horizon 2).  
- `n3` (default=12): Window size for the third rolling window (time horizon 3).  

**Outputs:**  
A DataFrame with additional columns:
- Rolling averages of runs scored over the windows.
- Rolling averages of strike rate over the windows
- Rolling averages of sixes scored over the windows
- Rolling averages of fours scored over the windows
- Rolling sums of half-centuries scored over the windows
- Rolling sums of centuries scored over the windows
- Rolling sums of ducks over the windows
- The computed α_batsmen_score for each time window, considering runs, strike rate, boundaries, half-centuries, centuries, and ducks.

#### `calculate_alpha_bowler_score()`
**Description:**  
The `calculate_alpha_bowler_score` function calculates the tailored α_bowler_score for Dream11 point prediction in ODIs over multiple rolling time horizons (n1, n2, n3). It factors in wickets, bowling average, strike rate, economy rate, and maidens to compute a performance score for each bowler.

**Inputs:**  
- `group`: A DataFrame containing bowling statistics.
- `n1`: Window size for the first rolling window (time horizon 1).  
- `n2`: Window size for the second rolling window (time horizon 2).  
- `n3`: Window size for the third rolling window (time horizon 3).  

**Outputs:**  
A DataFrame with additional columns:
- Rolling averages of wickets taken over the windows
- Rolling averages of bowling average over the windows
- Rolling averages of bowling strike rate over the windows
- Rolling averages of economy rate over the windows
- Rolling sums of maidens bowled over the windows
- The computed α_bowler_score for each time horizon, considering wickets, strike rate, economy rate, maidens, and bowling average.

#### `Class FeatureGeneration`
**Description:** Dataframe wrapper class which contains feature generation methods and formulae. Encorporates customisation based on match format.
**Input:**  
- `mw_overall`: Match-wise overview data. 
- `mw_pw_profile`: Match-wise, player-wise data.
- `match_format`: `'Test'`, `'ODI'` or `'T20'`


#### `process_country_and_homeaway()`
**Description:**  
This function processes cricket match data to determine the country of the venue (`country_ground`) and whether a player's team played in a home, away, or neutral match (`home_away`). It utilizes geonames and pycountry libraries for mapping cities to countries.

**Updates:**  
Updates `self.mw_pw_profile` by adding:
- `country_ground`: The country of the ground where the match was played.
- `home_away`: Whether the match was played at home, away, or a neutral venue.


#### `calculate_match_level_venue_stats()`
**Description:**  
Calculates venue-level match statistics (ARPO, boundary percentages, strike rates, and pitch classification) using rolling and cumulative metrics grouped by venue and match type.

**Updates:**  
Updates `self.mw_pw_profile` with:
- `ARPO_venue`: Average Runs Per Over at the venue.
- `Boundary_Percentage_venue`: Percentage of runs from boundaries.
- `BSR_venue`: Batting strike rate at the venue.
- `AFIS_venue`: Average first-innings score at the venue.
- `Pitch_Type`: Classification as Bowling-Friendly, Neutral, or Batting-Friendly.

#### `avg_of_opponent()`
**Description:**  
This function calculates the average fantasy score of opponents faced by a player in cricket matches, based on their role (batter, bowler, all-rounder, or neither). The scores are computed using data grouped by player ID, match type, and opposition team.

**Updates:**  
Updates `self.mw_pw_profile` with a new column `avg_of_opponent`

#### `calculate_fantasy_score()`
**Description:**  
This function calculates the actual fantasy score obtained by player in the match on the basis of different conditions used by Dream 11.
It is formed using the sum of fantasy_score batting and fantasy_score bowling of a player.


**Updates:**  
Updates `self.mw_pw_profile` with a new column `fantasy_score_total`

#### `calculate_rolling_fantasy_score()`
**Description**
This function uses the fantasy score total column added to the dataframe to make some rolling features depending on the period specified for rolling.

**Updates**
Updates `self.mw_pw_profile` with new columns of the form `avg_fantasy_score_{n}` .

#### `sena_sub_countries()`
**Description**
This function tries to penalize or reward batsman and bowlers based on the playing location. For the match_location the country_ground column is used which is generated in mw_overall using the city of match played.

**Updates**
Updates a permutation of columns with batsman and bowler of sena playing in sub-continent or vice-versa.
# train_model.py

#### `train_models_regression`
**Description:**  
This function trains multiple regression models on the provided training data.

**Inputs:**
- `X_train`: The training features.
- `y_train`: The target target feature score for training the models.

**Outputs/Updates:**
- `trained_models`: A dictionary containing `model_name`:`model` pairs.

#### `train_models_classification()`

**Description:**  
This function trains multiple classification models on the provided training data. It includes models such as XGBoost, Logistic Regression, and CatBoost classifiers, and returns the trained models.

**Inputs:**
- `X_train`: The training features.
- `y_train`: The target feature score for training the models.

**Outputs/Updates:**
- `trained_models`: A dictionary containing `model_name`:`model` pairs.

#### `iterative_training()`

**Description:**  
This function trains regression and classification models iteratively on the input data in batches.

**Inputs:**
- `X_train`: The training features for regression.
- `y`: The target variable for regression.
- `X_trainc`: The training features for classification.
- `yc`: The target variable for classification.
- `test`: The test data used for making predictions.

**Outputs/Updates:**
- `trained_modelsr`: The trained regression models.
- `trained_modelsc`: The trained classification models.
- `combined_predictions`: A dataframe containing the combined predictions from both regression and classification models for each iteration.

#### `train_and_save_model_odi`

**Description:**  
End-to-end function that trains models for ODI matches using historical data, processes the data, and saves the trained models and neural network weights to a file. Documentation for `ODI`, `Test` and `T20` match types follow the same.

**Inputs:**
- `train_start_date`: The start date for filtering the training data.
- `train_end_date`: The end date for filtering the training data.

**Outputs/Updates:**
Saves the following
- `trained_modelscc`: The trained classification models.
- `trained_modelsrr`: The trained regression models.
- `neural_weights`: The trained weights of the neural network.


# predict_model.py

#### `predict_scores_c()`

**Description**  
Generates predicted scores for a set of test data (`X_test`) using multiple trained classification models stored in a dictionary (`trained_models`). The function ensures compatibility between models and handles exceptions during prediction. Predicted scores are returned in a new DataFrame with each model's predictions in separate columns.
Similar structure is used for regression and 

**Inputs**  
- `trained_models`: A dictionary where keys are model names (strings) and values are dictionaries containing:
  - `model`: The trained model object (e.g., from scikit-learn).
  - `columns`: A list of feature names to retain in the test dataset for prediction.
- `X_test`: The test dataset containing all potential features.

**Outputs**  
- `test_data`: A DataFrame where each column corresponds to the predicted scores for the respective model from `trained_models`. The column names are formatted as `<model_name>_predicted_score`.

#### `predict_scores_odi()`

**Description**  
Generates predicted scores for a set of test data (`X_test`) using multiple trained regression models stored in a dictionary (`trained_models`). The function predicts scores using the `predict` method of each model and returns the results in a DataFrame, where each model's predictions are stored in separate columns.

**Inputs**  
- `trained_models`: A dictionary where keys are model names (strings) and values are dictionaries containing:
  - `model`: The trained regression model object (e.g., from scikit-learn).
  - `columns`: A list of feature names to retain in the test dataset for prediction.
- `X_test`: The test dataset containing all potential features.

**Outputs**  
- `test_data`: A DataFrame where each column corresponds to the predicted scores for the respective model from `trained_models`. The column names are formatted as `<model_name>_predicted_score`.

#### `generate_predictions_t20()`

**Description**  
Generates predictions for T20 matches using the saved pre-trained models and test data filtered by date range. It processes the input data, prepares feature vectors, and calculates predictions for each player in the dataset. Results are saved as a CSV file.

**Inputs**  
- `train_start_date`: Start date of the training period in the format YYYY-MM-DD.
- `train_end_date`: End date of the training period in the format YYYY-MM-DD.
- `test_start_date`: Start date of the testing period in the format YYYY-MM-DD.
- `test_end_date`: End date of the testing period in the format YYYY-MM-DD.

**Outputs**  
- A CSV file containing predictions for T20 matches, saved to path.

Similarly we use generate_predictions_odi() and generate_predictions_test() to generate an output CSV for all of these formats between our testing period.

# final_runner.py

#### `process_matches()`

**Description**
Processes match data for a given match type by merging prediction data and player profiles, selecting top players based on predicted and fantasy scores, and calculating evaluation metrics. The output is a CSV file containing detailed information about the best-performing players and their respective scores for each match.

** Inputs**
- **`match_type`**:  
  Specifies the type of match (e.g., T20, ODI) for which the data should be processed. This determines the file paths used for input and output.

**Outputs**
- A CSV file containing:
  - Match details (e.g., match date, participating teams).
  - Top 11 players for each match based on:
    - Predicted scores.
    - Fantasy scores.
  - Metrics comparing total predicted and fantasy scores, including the Mean Absolute Error (MAE).

**Process Details**

#### Step 1: File Paths
Constructs file paths for:
- Input prediction data (`predictions_<match_type>.csv`).
- Player profile data (`mw_pw_profiles.csv`).
- Output processed data (`final_output_<match_type>.csv`).

#### Step 2: Load Data
Reads:
- **Prediction Data** (`df_pred`): Contains player predictions for the match.
- **Player Profile Data** (`df_data`): Includes player details such as `start_date`, `player_team`, and `full_name`.

#### Step 3: Merge Data
Combines prediction and profile data using `match_id` and `player_id` to create a single DataFrame with comprehensive match and player information.

### Step 4: Process Matches
Loops through all unique matches in the dataset:
1. **Extract Match Details**:
   - Match date.
   - Participating teams (`Team 1` and `Team 2`).

2. **Ensure Team Representation**:
   - Selects top players such that both teams are represented.
   - Sorts players by scores (predicted or fantasy) and selects:
     - 1 player from each team.
     - The remaining players based on highest scores, ensuring a total of 11.

3. **Select Top Players**:
   - **Predicted Scores**: Selects top 11 players by `predicted_score`.
   - **Fantasy Scores**: Selects top 11 players by `fantasy_score_total`.

4. **Compile Results**:
   - Player names and scores for the top 11 players in both categories.
   - Total fantasy points for both teams.
   - Absolute difference between predicted and fantasy scores as `MAE`.

#### Step 5: Save Results
Combines results for all matches into a single DataFrame and saves it to a CSV file.

**Example Output Columns**
- `Match Date`
- `Team 1`
- `Team 2`
- `Predicted Player 1`, `Predicted Player 1 Points`, ..., `Predicted Player 11`, `Predicted Player 11 Points`
- `Dream Team Player 1`, `Dream Team Player 1 Points`, ..., `Dream Team Player 11`, `Dream Team Player 11 Points`
- `Total Dream Team Points`
- `Total Predicted Players Fantasy Points`
- `Total Points MAE`

** Usage**
Call the function with the desired match type:

```python
final_results = process_matches("t20")

# DreamAI Web App Developer Documentation

The DreamAI web app merges machine learning, natural language processing, and user-centric design to deliver a seamless predictive fantasy cricket experience. Key features include LLM-based Dream Team summarization, an AI Assistant Bot, and multilingual generative AI support.

---
## Overall Project structure

```plaintext
ProductUI
├── backend
├── docker-compose.yaml
├── frontend
└── main.py
```

---

## 1. Backend Technical Architecture(FastAPI)

The backend of the **ProductUI** project is built using **FastAPI** and follows a modular architecture. Below is an outline of the key components:

### Key Components

1. **API (`/api`)**  
   - Defines FastAPI routes and endpoint logic (e.g., `/users`, `/products`).
   
2. **Chatbot (`/chatbot`)**  
   - Handles chatbot-specific functionality and integrations.
   
3. **Database (`/db`)**  
   - Manages database models, migrations, and interactions (via ORM or raw SQL).
   
4. **Models (`/model`)**  
   - Contains machine learning models and inference logic.
   
5. **UI Models (`/product_ui_model`)**  
   - Defines data structures related to UI features and components.
   
6. **Schemas (`/schemas`)**  
   - Pydantic models for request and response validation.
   
7. **Services (`/services`)**  
   - Contains business logic and external service integrations.
   
8. **Utils (`/utils`)**  
   - Helper functions for common tasks like logging, error handling, etc.
   
9. **`main.py`**  
   - Initializes the FastAPI app, loads routes, and configures the application.

### Data Flow in the Backend

1. **Client Request** →
2. **Route Handler (`api/`)** →
3. **Request Validation (`schemas/`)** →
4. **Business Logic (`services/`)** →
5. **Database Interaction (`db/`)** →
6. **Machine Learning (`model/`)** →
7. **Response Preparation (`schemas/`)** →
8. **Send Response**


### Backend Structure

```plaintext

backend                     	# Backend logic and services
├── Dockerfile               	# Docker setup for backend
├── app
│   ├── api                  	# API endpoints and routes
│   ├── chatbot              	# Chatbot related code
│   ├── db                   	# Database models and migrations
│   ├── model                	# Machine learning models
│   ├── product_ui_model     	# UI-related product models
│   ├── schemas              	# Data validation schemas
│   ├── services             	# Service layer for business logic
│   └── utils                	# Utility functions
├── entrypoint.sh            	# Script to start the app
├── execute_sql_from_file.py 	# Execute SQL commands from file
├── main.py                  	# Main application logic
├── requirements.txt         	# Python dependencies
├── scripts                  	# Helper scripts for data processing
├── sql                      	# SQL queries and scripts
├── tests                    	# Unit and integration tests
│   ├── test_api             	# API endpoint tests
│   └── test_main.py         	# Main logic tests

```



---

This architecture promotes separation of concerns, scalability, and maintainability, ensuring a clean and efficient flow from client request to server response.


---

## 2. Frontend Technical Architecture

### Frontend Architecture

- **React 18.3**: Utilizes the latest React features, including concurrent rendering for a smooth user experience.
- **Component Structure**: Modular design with reusable components to enhance maintainability and scalability.
- **State Management**: React Context for global state management, ensuring seamless data flow across components.

### UI Libraries & Integration
- **Material UI**: Provides a set of customizable UI components and a consistent theme across the app.
- **Tailwind CSS**: A utility-first CSS framework used for responsive and efficient styling.
- **Material Tailwind**: Enhances Material Design components with Tailwind CSS.
- **PrimeReact**: Advanced UI components to enhance the user interface.
- **Chart.js**: Used for visualizing match data, providing interactive charts and graphs.
- **React DnD**: Enables drag-and-drop functionality within the app for user interaction.
- **React Joyride**: Adds interactive user tours to guide users through the app.

### Data Flow
```bash
User Input → React Components → Context API → Data Processing → UI Update
```

### Project Structure

```plaintext
├── assets
│   └── (Folder for image assets)
├── component
│   ├── ImportCSV.jsx        	# Handles CSV file import functionality
│   ├── Loading.jsx          	# Displays a loading screen or indicator
│   ├── MatchDetailsCard.jsx 	# Displays match details in a card layout
│   ├── Navbar.jsx           	# Implements the navigation bar
│   ├── playerCard.jsx       	# Displays player information in a card
│   ├── Playerlist.jsx       	# Lists all players in the match
│   └── TeamSelection.jsx    	# Allows users to select teams for the match
├── HomePage
│   ├── AllMatches.jsx       	# Displays a list of all matches
│   ├── Calendar.jsx         	# Displays the match calendar
│   ├── Header.jsx           	# Displays the header of the homepage
│   ├── MatchCard.jsx        	# Displays match details in a compact card
│   ├── HowToPlay.jsx        	# Provides instructions on how to play
│   └── NoMatches.jsx        	# Displays a message when no matches are available
├── pages
│   ├── CustomMatchCSV.jsx   	# Handles custom match creation
│   ├── dreamTeam.jsx        	# Displays Playground (user's dream team)
│   ├── HomePage.jsx         	# Main homepage component
│   ├── MatchDetails.jsx     	# Displays detailed match information
│   └── Starterpage.jsx     	# Displays the starter page of the app
├── constants.jsx            	# Holds constant values used across the app
├── App.jsx
└── main.jsx
```

---

## 3. Routes

### 1. **Landing Page (/)**
- **Purpose**: Entry point to the application.
- **Key Features**:
  - About Us: Information about the platform, including its goals, features, and background.
  - FAQs: Frequently Asked Questions to help users understand how to use the platform.
  - Get Started: Prominent CTA to navigate users to the homepage (`/home`).

### 2. **Home Page (/home)**
- **Purpose**: The main hub for exploring matches, interacting with match data, and accessing other areas.
- **Key Features**:
  - Featured Matches: Top 20 matches with key information.
  - All Matches: List of all matches with initial 5 matches displayed.
  - Navigation: Access to other sections such as match details, team selection, and custom match features.

### 3. **Select Match (/teamSelect)**
- **Purpose**: Allows users to select two teams for comparison or simulation.
- **Key Features**:
  - Team Selection: Choose two teams.
  - Schedule Options: Pick an existing match or create a custom match.
  - Custom Match Page: Create unique team compositions and simulate matches.

### 4. **Match Details (/custommatch/:id)**
- **Purpose**: Provides detailed information about a specific match.
- **Key Features**:
  - Full Player Rosters: Displays all players with their statistics.
  - Custom Squad Creation: Combine players from different teams and analyze match performance.
  - Match Simulation: Simulate matches with different scenarios and strategies.

### 5. **Custom Match Input (/custommatch)**
- **Purpose**: Dedicated page for creating matches from scratch.
- **Key Features**:
  - CSV Upload: Upload a CSV file with player and team data.
  - Match Customization: Set match details like date, format, and team composition.
  - Simulate and Explore: Simulate matches and explore different outcomes.

### 6. **Playground (/dreamTeam)**
- **Purpose**: Central interactive feature for exploring curated teams, analyzing players, and experimenting with strategies.
- **Key Features**:
  - Dream Team: AI-curated teams with Dream Scores to visualize player potential.
  - GenAI Description: Explains team composition and predicts player performance.
  - Match Insights: Additional match details like pitch conditions and weather forecasts.
  - Player Profiles: Interactive player cards with career stats and achievements.

---

## 4. Features

### Select Match
- **Purpose**: Compare or simulate two teams.
- **Key Features**:
  - **Team Selection**: Choose two teams from the list.
  - **Schedule Options**: Pick an existing match or create a custom match.
  - **Custom Match Creation**: Build and simulate unique matches by selecting teams and analyzing player data.

### Custom Match
- **Purpose**: Create and simulate custom matches with full player rosters.
- **Key Features**:
  - **View Player Rosters**: Displays all players in selected teams.
  - **Create Custom Squads**: Combine players from different teams.
  - **Simulate Fantasy Matches**: Experiment with different strategies and player combinations.

### Custom Input
- **Purpose**: Allows users to upload CSV files for custom team creation.
- **Key Features**:
  - **CSV File Format**: Includes player names, squad details, match dates, and format information.
  - **Match Customization**: Users can customize teams, set match dates, and configure simulation formats.

### Playground (Dream Team)
- **Purpose**: Interactive feature for exploring curated teams and strategies.
- **Key Features**:
  - **Dream Team**: AI-generated teams with Dream Scores.
  - **GenAI Description**: Explains team composition and projected performance.
  - **Match Insights**: Includes pitch conditions, weather forecasts, and more.
  - **Player Profiles**: Interactive cards with player career stats and achievements.

---
## 5. Features

### **SelectMatch**
This feature allows users to select two teams from a list to compare or compete against each other. Users can choose a match from the existing schedules or create their own custom match.
- **Team Selection**: Choose two teams from a comprehensive list.
- **Schedule Options**: Pick an existing match or create a custom match.
- **Custom Match Creation**: Build and simulate unique matches by selecting teams and analyzing player data.

### **Custom Match**
- **View Player Rosters**: Displays full rosters of selected teams.
- **Create Custom Squads**: Users can create and combine teams using player data.
- **Simulate Fantasy Matches**: Experiment with different strategies by simulating matches with custom teams.

### **Custom Input**
Allows users to upload CSV files to create custom teams and players from scratch. The file format should include player names, squad details, match dates, and format information.

### **Playground (Dream Team)**
The Playground serves as the heart of the product, offering an immersive, interactive environment for users to explore their teams and strategies:
- **Dream Team**: AI-generated teams with Dream Scores to visualize player potential.
- **GenAI Description**: Provides explanations of the team composition and predictions for player performances.
- **Match Insights**: Provides additional match details such as pitch conditions and weather forecasts.
- **Player Profiles**: Interactive player cards with career stats, skills, and achievements.

---

##6.  API Documentation

### **Teams**

1. **Get All Teams**
   - **Endpoint**: `GET /team/`
   - **Response**: A list of all teams in the system.

2. **Get Team By Name**
   - **Endpoint**: `GET /team/{team_name}`
   - **Path Parameter**: `team_name` (required)
   - **Response**: Details of the team.

3. **Get Matches By Team Name**
   - **Endpoint**: `GET /match/team/{team_name}`
   - **Path Parameter**: `team_name` (required)
   - **Response**: List of matches.

4. **Get Player Lifetime Stats**
   - **Endpoint**: `GET /player/cricketers_lifetime_stats/{player_id}`
   - **Path Parameter**: `player_id` (required)
   - **Response**:
 	- `200 OK`: Returns the lifetime statistics for the player.
 	- `422 Validation Error`: If the player ID is invalid.

5. **Get Player Stats for Multiple Players**
   - **Endpoint**: `POST /player/player_stats/all`
   - **Request Body**:
 	- `match_id` (required): The match ID.
 	- `player_ids` (required): A list of player IDs.
   - **Response**:
 	- `200 OK`: Returns the stats for the specified players.
 	- `422 Validation Error`: If the request body is invalid or if any player ID is incorrect.

6. **Search Players by Team Name**
   - **Endpoint**: `GET /player/search_players/{team_name}`
   - **Path Parameter**: `team_name` (required)
   - **Response**:
 	- `200 OK`: Returns a list of players in the specified team.
 	- `422 Validation Error`: If the team name is invalid.

### **AI**

1. **Chat**
   - **Endpoint**: `POST /ai/chat`
   - **Request Body**:
 	- `message` (required): The message to send to the AI.
   - **Response**:
 	- `200 OK`: Returns the AI's response to the message.
 	- `422 Validation Error`: If the request body is invalid or missing the message parameter.

2. **Text to Speech**
   - **Endpoint**: `POST /ai/audio`
   - **Request Body**:
 	- `message` (required): The text message to convert to speech.
 	- `target_language_code` (required): The language code (e.g., "en" for English).
   - **Response**:
 	- `200 OK`: Returns an audio file created from the text.
 	- `422 Validation Error`: If the request body is invalid, or the language code is not supported.

3. **Get Match Description**
   - **Endpoint**: `POST /ai/description`
   - **Request Body**:
 	- `match_type` (required): The type of the match (e.g., Test, ODI, T20).
 	- `player_ids` (required): A list of player IDs involved in the match.
   - **Response**:
 	- `200 OK`: Returns a description of the match, including insights and potential strategies based on the players and match type.
 	- `422 Validation Error`: If the request body is malformed, or if player IDs or match type are incorrect.

### **Error Responses**

- **422 Validation Error**: This error occurs when a request contains invalid or missing parameters, such as invalid IDs or incorrectly formatted data. The response body will include an error message describing the issue.

**Example Response**:
```json
{
  "error": "Invalid player ID or match type."
}
```

---

##  7. GenAI Features

### **LLM-Based Dream Team Summarization**

**Objective**: Generate detailed summaries to explain the logic behind Dream Team creation.
- **Core Inputs**: SHAP values, historical player statistics, and top features identified by the trained models.
- **Integration**:
  - Backend API processes machine learning model outputs.
  - Frontend presents structured summaries in a user-friendly format.

### **DreamAI Assistant BOT**

**Objective**: Address user queries, provide cricket insights, and simplify app interactions.
- **Architecture**:
  - **RAG-Based Agents**: Retrieve relevant information for system-related queries.
  - **Database Query Agents**: Translate user questions into SQL queries, fetch data, and present answers in a structured format.
  - **General LLM Agents**: Enable smooth, conversational interactions.
- **Integration**:
  - Backend microservices for each agent type.
  - Interactive chatbot interface on the frontend.
  - **RAG-Based Chatbot** was curated and trained for our specific use-case from scratch, trained on match data statistics according to the defined time periods: Before 2024-06-30. Hence, online search is disabled to adhere to app-specific constraints.

### **Multilingual Generative AI Support**

**Objective**: Provide inclusive support by breaking language barriers through speech generation in various Indian languages.
- **Integration**:
  - **Sarvam AI model** processes text-to-speech tasks (though limited by lifetime credit limits).
  - **Dynamic language selection** incorporated in the frontend.
## 7. Performance and Scalability

The DreamAI Web App has been designed with a robust architecture to deliver lightning-fast responses and seamlessly handle the demands of an expanding user base. Built for peak efficiency, it ensures both individual user satisfaction and the capacity to support a thriving community of fantasy cricket enthusiasts. Here's how DreamAI tackles performance and scalability:

### 1. Streamlined Query Processing
- **Single Endpoint Efficiency**: By centralizing all interactions to a single endpoint, `/api/chat`, we reduce network overhead and minimize processing latency.  
  We run two separate models with a single endpoint:
  - One model handles intelligent response generation based on user queries.
  - The other model is responsible for making Endpoint-to-DB calls to retrieve player statistic data.
 
- **Asynchronous Task Handling**: Non-blocking requests allow multiple processes, like data retrieval and audio generation, to run concurrently, ensuring a smooth user experience.

### 2. AI Acceleration
- **Pre-trained Model Caching**: Frequently used models, such as SHAP for explainability and RAG for retrieval, are cached in memory to avoid repetitive initializations, speeding up response times.
 
- **Dynamic Resource Allocation**: Critical tasks, such as multilingual audio snippet generation, are prioritized dynamically based on user demand to maintain rapid response times.

### 3. Minimal Latency Visualizations
- **Client-Side Rendering**: Graphs and charts are generated on the client side using pre-fetched JSON data, reducing server load and improving perceived speed.
 
- **React-Window Loading**: Prioritizes critical information first while loading additional insights in the background, ensuring that users can interact with the app while other data continues to load.

- **Local Storage Utilization**: Saved and unsaved data are stored in local storage to avoid unnecessary API calls, enhancing performance by reducing network load.

