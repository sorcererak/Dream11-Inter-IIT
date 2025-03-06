export default function CleanPlayerType(player_type) {
  const mapping = {
    "Batting Allrounder": "batting",
    "Wicketkeeper Batter": "batting",
    "Bowling Allrounder": "bowling",
    "Middle order Batter": "batting",
    Batter: "batting",
    "Top order Batter": "batting",
    Bowler: "bowling",
    Allrounder: "batting",
    "Opening Batter": "batting",
    Wicketkeeper: "batting",
  };
  return mapping[player_type];
}
