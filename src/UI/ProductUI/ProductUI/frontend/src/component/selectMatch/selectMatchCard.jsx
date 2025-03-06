import { useEffect, useState } from "react";
import MatchCard from "../HomePage/MatchCard.jsx";
import styles from "../../css/cardStack.module.css";
import HorizontalCalendar from "./horizontalCalendar.jsx";
import Loading from "../common/Loading.jsx";
import { BASE_URL } from "../../constants.jsx";
import { useNavigate } from "react-router-dom";
import NewMatchCard from "./newMatchCard.jsx";

const SelectMatchCard = ({ teamA, teamB }) => {
  const [matchDate, setMatchDate] = useState(null);
  const [teamInfo, setTeamInfo] = useState(null);
  const [allMatches, setAllMatches] = useState(null);
  const [filteredMatches, setFilteredMatches] = useState(null);
  const [notEmpty, setNotEmpty] = useState(false);

  const navigate = useNavigate();

  const customMatch = () => {
    console.log(matchDate);
    // const formattedDate = matchDate.toISOString().split("T")[0]; // Format as 'YYYY-MM-DD'
    const formattedDate = matchDate
      ? matchDate.toLocaleDateString("en-CA")
      : new Date().toLocaleDateString("en-CA");
    navigate(`/custommatch`, {
      state: {
        teamAdata: { name: teamA.name, url: teamA.url },
        teamBdata: { name: teamB.name, url: teamB.url },
        matchDate: formattedDate,
      },
    });
  };

  const formDreamTeam = (match_id) => {
    navigate(`/dreamTeam/${match_id}`);
  };

  useEffect(() => {
    if (filteredMatches && filteredMatches.length > 0) {
      setNotEmpty(true);
    } else {
      setNotEmpty(false);
    }
  }, [filteredMatches]);

  // Fetch matches when teamA and teamB change
  useEffect(() => {
    const dataFeatch = async () => {
      try {
        const payload = {
          team_name2: teamA.name,
          team_name1: teamB.name,
        };
        const response = await fetch(`${BASE_URL}/match/team/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(
            `HTTP Error: ${response.status} - ${response.statusText}`
          );
        }
        const data = await response.json();
        const dist = data.data.filter(
          (obj, index, self) =>
            index === self.findIndex((t) => t.match_id === obj.match_id)
        );
        setAllMatches(dist);
        setTeamInfo(data.team_info);
      } catch (error) {
        alert("We encountered an issue. Please try again later.");
        console.error("Error fetching teams:", error);
      }
    };
    dataFeatch();
  }, [teamA.name, teamB.name]);

  useEffect(() => {
    if (allMatches && matchDate) {
      const filtered = allMatches.filter((match) => {
        // Loop through each date in the 'dates' array and check if any matches the selected date
        const matchHasSelectedDate = match.dates.some((date) => {
          const matchDateString = new Date(date).toLocaleDateString();
          const selectedDateString = matchDate.toLocaleDateString();
          return matchDateString === selectedDateString;
        });
        return matchHasSelectedDate;
      });
      setFilteredMatches(filtered);
    }
  }, [matchDate, allMatches]);

  return allMatches === null ? (
    <Loading />
  ) : (
    <div className={styles.calenderMatchCardWrapper}>
      <div className={styles.calender} data-tour-id="scrolling-calendar">
        <h3 className={styles.headmatch}>Select Match</h3>
        <HorizontalCalendar
          initialDate={new Date()}
          setMatchDate={setMatchDate}
        />
      </div>
      <div className={`${styles.matchCardList} hide-scrollbar`}>
        {/* If there are no matches between the two teams */}
        {allMatches.length === 0 && !filteredMatches && (
          <div className={styles.noMatchesDiv}>
            <p>
              No matches have been played between <b>{teamA.name}</b> and <b>{teamB.name}</b>.
            </p>
            <p>You can create a custom match by selecting a date.</p>
          </div>
        )}

        {/* If a date is selected, show matches for that date */}
        {matchDate && filteredMatches && filteredMatches.length > 0 ? (
          <div
            className={`${styles.matchCardList} hide-scrollbar`}
            style={{ width: "100%" }}
          >
            <h4>Matches on Selected Date</h4>
            {filteredMatches.map((match) => (
              <NewMatchCard
                key={match.match_id}
                match={match}
                formDreamTeam={formDreamTeam}
                team_info={teamInfo}
              />
            ))}
          </div>
        ) : matchDate ? (
          <div className={styles.customMatchDiv}>
            <p>No matches available for the selected date.</p>
            <button className={styles.customMatchBtn} onClick={customMatch}>
              Create Custom Match
            </button>
          </div>
        ) : null}
        {notEmpty ? null : matchDate ? (
          <h4>Other Matches</h4>
        ) : (
          <h4>All Matches</h4>
        )}
        {allMatches.map((match) => (
          <NewMatchCard
            key={match.match_id}
            match={match}
            formDreamTeam={formDreamTeam}
            team_info={teamInfo}
          />
        ))}
      </div>
    </div>
  );
};

export default SelectMatchCard;
