import { CalendarToday, LocationOn, SportsCricket } from "@mui/icons-material";
import styles from "../css/MatchDetailsCard.module.css";

const MatchDetailsCard = ({ match }) => {
  const fixedVenue = match?.venue
    ? match.venue.replace(/({|})/g, "").replace(/"/g, "")
    : "";
  const parsedVenue = fixedVenue.split(",");
  const stadium = parsedVenue[0] || "Stadium not available"; // Stadium part
  const city = match.city || "";  // City part, leave empty if not available

  const matchDate = match?.dates?.[0] ? new Date(match.dates[0]) : null;
  const formattedDate = matchDate
    ? matchDate.toLocaleDateString("en-GB", {
      day: "numeric",
      month: "long",
      year: "numeric",
    })
    : "Date not available";

  const teams = match?.teams || [];
  const team1 = teams[0] || "Team 1";
  const team2 = teams[1] || "Team 2";

  const umpires = match?.umpires || [];
  const referee = match?.referees || "Referee not available";
  const eventName = match?.event_name || "Event Name Not Available";

  return (
    <div className={styles.container}>
      <div className={styles["match-card"]}>
        {/* Event Name */}
        <h3 className={styles["event-name"]}>{eventName}</h3>

        {/* Match Type */}
        <h3 className={styles["match-type"]}>
          {match?.match_type || "Match Type Not Available"}
        </h3>

        {/* Match Date */}
        <div className={styles["match-date"]}>
          <CalendarToday className={styles.icon} />
          {formattedDate}
        </div>

        {/* Separator */}
        <div className={styles.separator}></div>

        {/* Team Logos */}
        <div className={styles.logos}>
          <div className={styles.team}>
            <img
              className={styles["team-logo"]}
              alt={`${team1} logo`}
              src={match?.teamAicon}
            />
            <p className={styles["team-name"]}>{team1}</p>
          </div>
          <span className={styles.vs}>vs</span>
          <div className={styles.team}>
            <img
              className={styles["team-logo"]}
              alt={`${team2} logo`}
              src={match?.teamBicon}
            />
            <p className={styles["team-name"]}>{team2}</p>
          </div>
        </div>

        {/* Match Details */}
        <div className={styles["match-details"]}>
          {city && (
            <p>
              <LocationOn className={styles.icon} />
              {city}
            </p>
          )}
          
          {stadium && (
            <p>
              <SportsCricket className={styles.icon} />
              {stadium}
            </p>
          )}
        </div>

        {/* Separator */}
        <div className={styles.separator}></div>

        {/* Referee and Umpires */}

        <div className={styles["officials"]}>
          <p className={styles["referee"]}>
            <strong>Referee:</strong>{referee}</p>
          <p className={styles["umpires"]}>
            <strong>Umpires:</strong> {umpires.join(", ") || "Umpires not available"}
          </p>
        </div>

      </div>
    </div>
  );
};

export default MatchDetailsCard;