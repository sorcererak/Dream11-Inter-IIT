import styles from "../../css/HomePage/MatchCard.module.css";
import { CalendarToday, LocationOn, SportsCricket } from "@mui/icons-material"; // Use the appropriate icon for cricket
import { useNavigate } from "react-router-dom";

const MatchCard = ({ match }) => {
  const navigate = useNavigate();
  const teams = match?.teams || [];
  const team1 = teams[0] || "Team 1";
  const team2 = teams[1] || "Team 2";

  // Clean up the venue and remove the quotes
  const fixedVenue = match?.venue ? match.venue.replace(/({|})/g, "").replace(/"/g, "") : "";
  const parsedVenue = fixedVenue.split(",");
  const stadium = parsedVenue[0] || "Stadium not available"; // Stadium part
  const city = match.city || ""; // City part, leave empty if not available

  const matchDate = match?.dates?.[0] ? new Date(match.dates[0]) : null;
  const formattedDate = matchDate
    ? matchDate.toLocaleDateString("en-GB", {
        day: "numeric",
        month: "long",
        year: "numeric",
      })
    : "Date not available";

  const handleCardClick = () => {
    navigate(`/matchdetails/${match.match_id}`, {
      state: {
        matchDate: formattedDate,
        team1Logo: match?.team_info.teamAinfo.url, 
        team2Logo:match?.team_info.teamBinfo.url,
      }
    });
  };

  return (
    <div className={styles.container}>
      <div className={styles["match-card"]} onClick={handleCardClick}>
        <h2 className={styles["match-type"]}>
          {match?.event_name || "Match Type Not Available"}
        </h2>
        
        <div className={styles["match-date"]}>
          {/* <CalendarToday className={styles.icon} /> */}
          {match?.match_type || {formattedDate}} Match
        </div>
        <div className={styles.separator}></div> {/* Separator line */}
        <div className={styles.logos}>
          <div className={styles.team}>
            <img className={styles["team-logo"]} alt={`${team1} logo`} src={match?.team_info.teamAinfo.url} />
            <p className={styles["team-name"]}>{team1}</p>
          </div>
          <span className={styles.vs}>vs</span>
          <div className={styles.team}>
            <img className={styles["team-logo"]} alt={`${team2} logo`} src={match?.team_info.teamBinfo.url} />
            <p className={styles["team-name"]}>{team2}</p>
          </div>
        </div>
        <div className={styles["match-details"]}>
          <p>
            <LocationOn className={styles.icon} />
            <span className={styles.ellipsis}>{city}</span>
          </p>
          <p>
            <SportsCricket className={styles.icon} />
            <span className={styles.ellipsis}>{stadium}</span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default MatchCard;
