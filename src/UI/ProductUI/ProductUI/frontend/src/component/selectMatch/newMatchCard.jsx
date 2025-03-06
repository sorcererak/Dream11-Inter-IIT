import { CalendarToday, LocationOn } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import styles from "./newMatchCard.module.css";

const NewMatchCard = ({ match, team_info }) => {
  console.log(match);
  console.log(team_info);
  const navigate = useNavigate();
  const teams = match?.teams || [];
  const team1 = teams[0] || "Team 1";
  const team2 = teams[1] || "Team 2";

  const fixedVenue = match?.venue ? match.venue.replace(/({|})/g, "") : "";
  const parsedVenue = fixedVenue.split(",");
  const venue = parsedVenue.join(" ");

  const matchDate = match?.dates?.[0] || "Date not available";
  const eventName = match?.event_name || "Event name not available";  // Add event name

  const handleCardClick = () => {
    // Pass the necessary data as state when navigating to the match details page
  
    console.log("match ok", team_info)
    navigate(`/matchdetails/${match.match_id}`, {
      state: {
        matchDate,
       team_info,
      },
    });
  };

  return (
    <div className={styles["match-card"]}>
      <h2 className={styles["match-type"]}>
        {match?.event_name || "Name Not Available"}
      </h2>

      {/* Display event name */}
      {/* <h3 className={styles["event-name"]}>{eventName}</h3>   */}

      <div className={styles.logos}>
        <div className={styles.team}>
          <img
            className={styles["team-logo"]}
            alt={`${team1} logo`}
            src={team_info[0]?.url}
          />
          {/* <p className={styles["team-name"]}>{team1}</p> */}
        </div>
        <span className={styles.vs}>vs</span>
        <div className={styles.team}>
          <img
            className={styles["team-logo"]}
            alt={`${team2} logo`}
            src={team_info[1]?.url}
          />
          {/* <p className={styles["team-name"]}>{team2}</p> */}
        </div>
      </div>

      <div className={styles["match-details"]}>
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            gap: "20px",
          }}
        >
          <p>
            <CalendarToday className={styles.icon} />
            {matchDate}
          </p>
          <p>Format: {match?.match_type || "Match Type Not Available"}</p>
        </div>

        <p>
          <LocationOn className={styles.icon} />
          {venue}
        </p>
      </div>
      <div className={styles.buttonDiv}>
        <button
          onClick={
            handleCardClick}
        >
          View Details
        </button>
        <button
          onClick={() => {
            navigate(`/dreamTeam/${match.match_id}`);
          }}
        >
          Create Team
        </button>
      </div>
    </div>
  );
};

export default NewMatchCard;
