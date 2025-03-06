import React from "react";
import styles from "../css/HeadToHeadCard.module.css";

const HeadToHeadCard = ({ headdata }) => {
  const { 
    team_a, 
    team_b, 
    avg_batting_a, 
    avg_batting_b, 
    avg_wickets_a, 
    avg_wickets_b, 
    winner, 
    winner_a, 
    winner_b, 
    draws 
  } = headdata;

  // Determine the dominance message based on the winner
  const dominanceMessage =
    winner === team_a
      ? `${team_a} has been more dominant than ${team_b}.`
      : winner === team_b
      ? `${team_b} has been more dominant than ${team_a}.`
      : `${team_a} and ${team_b} are evenly matched.`;

  return (
    <div className={styles.card}>
      {/* Heading */}
      <h3 className={styles.heading}>In the last 5 matches</h3>

      {/* Teams Row */}
      <div className={styles.row}>
        {/* Team A */}
        <div className={styles.teamColumn}>
          {team_a && <h3 className={styles.teamName}>{team_a}</h3>}
          {winner_a !== undefined && <p className={styles.stat}>Wins: {winner_a}</p>}
          {avg_batting_a !== undefined && <p className={styles.stat}>Avg Batting Score: {avg_batting_a}</p>}
          {avg_wickets_a !== undefined && <p className={styles.stat}>Avg Wickets Lost: {avg_wickets_a}</p>}
        </div>

        {/* Team B */}
        <div className={styles.teamColumn}>
          {team_b && <h3 className={styles.teamName}>{team_b}</h3>}
          {winner_b !== undefined && <p className={styles.stat}>Wins: {winner_b}</p>}
          {avg_batting_b !== undefined && <p className={styles.stat}>Avg Batting Score: {avg_batting_b}</p>}
          {avg_wickets_b !== undefined && <p className={styles.stat}>Avg Wickets Lost: {avg_wickets_b}</p>}
        </div>
      </div>

      {/* Dominance Message */}
      <div className={styles.dominanceMessage}>
        <p>{dominanceMessage}</p>
      </div>

      {/* Draws (conditionally render if present) */}
      {draws !== undefined && (
        <div className={styles.drawsMessage}>
          <p>Draws: {draws}</p>
        </div>
      )}
    </div>
  );
};

export default HeadToHeadCard;
