import React, { useEffect, useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import Playerlist from "../component/Playerlist";
import Navbar from "../component/Navbar";
import MatchDetailsCard from "../component/MatchDetailsCard";
import styles from "../css/MatchDetails.module.css";
import batsman_img from "../assets/matchdetails.png";
import { BASE_URL } from "../constants";
import { Button } from "@mui/material"; // Import the Button component
import Loading from "../component/common/Loading";
import WeatherCard from "../component/common/weatherCard";
import PitchCard from "../component/Pitchcard";
import HeadToHeadCard from "../component/HeadToHead";

const MatchDetails = () => {
  const { match_id } = useParams();
  const location = useLocation();
  const [teamAPlayers, setTeamAPlayers] = useState([]);
  const [info, setInfo] = useState("");
  const [pitchdata, setpitchdata] = useState("");
  const [headdata, setheaddata] = useState(""); // Holds the wins data
  const [teamBPlayers, setTeamBPlayers] = useState([]);
  const [carddata, setCardData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const state = location.state || {};
  const isStateTypehome =
    state.hasOwnProperty("matchDate") && state.hasOwnProperty("team1Logo");

  useEffect(() => {
    const fetchMatchDetails = async () => {
      try {
        const response = await fetch(
          `${BASE_URL}/match/matchdetails/${match_id}`,
          { method: "GET" }
        );
        const data = await response.json();
        console.log("Data", data);
        if (data.pitch && data.pitch !== "") {
          setpitchdata("Grass");
          // setpitchdata(data.pitch);
        } else {
          setpitchdata("Grass");
        }
        
        // If data.wins exists, set it as head-to-head data
        if (data.wins) {
          setheaddata(data.wins);  // Pass the actual wins data here
        }

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        if (data.status === "ok") {
          const match = data.matchdetails;

          if (data.teamA && data.teamB) {
            setTeamAPlayers(data.teamA);
            setTeamBPlayers(data.teamB);
          } else {
            setError("Team data is missing or undefined.");
            return;
          }

          setCardData({
            venue: match.venue,
            teamAicon: data?.team_info?.teamA?.url || "",
            teamBicon: data?.team_info?.teamB?.url || "",
            match_type: match.match_type,
            event_name: match.event_name,
            umpires: match.umpires,
            referees: match.match_referees,
            teams: match.teams,
            dates: match.dates,
          });
        } else {
          setError(data.message || "Something went wrong");
        }
      } catch (error) {
        console.log(error);
      } finally {
        setLoading(false);
      }
    };

    fetchMatchDetails();
  }, [match_id]); // Include match_id in the dependency array to trigger effect

  if (loading) {
    return <Loading />;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  const handleGenerateDreamTeam = () => {
    navigate(`/dreamTeam/${match_id}`);
  };

  return (
    <div>
      <Navbar />
      <div className={styles.matchDetailsContainer}>
        {/* Team A Card */}
        <div className={styles.teamCard}>
          <div className={styles.teamHeader}>
            <div className={styles.team}>
              <img
                src={carddata?.teamAicon}
                alt="Team A Logo"
                className={styles.teamLogo}
                style={{ backgroundColor: "white" }}
              />
              <span className={styles.teamName}>{carddata?.teams[0]}</span>
            </div>
          </div>
          <Playerlist playerdata={teamAPlayers} />
        </div>

        {/* Weather and Pitch Section */}
        <div className={styles.weatherAndPitchContainer}>
          <div className={styles.weathercard}>
            <WeatherCard
              matchId={match_id}
              setEffect={setInfo}
              showEffect={true}
            />
          </div>
          <div className={styles.pitchcard}>
            <PitchCard pitch={pitchdata} />
          </div>
        </div>

        {/* Match Details Card */}
        <div className={styles.matchDetailsCardwithimg}>
          <MatchDetailsCard
            match={carddata}
            className={styles.matchDetailsCard}
          />
          {/* Head-to-Head Card */}
          <img src={batsman_img} alt="Batsman" className={styles.batsmanImg} />

          <div className={styles.headToHeadContainer}>
            <HeadToHeadCard headdata={headdata} /> {/* Use headdata here */}
          </div>

          <Button
            variant="contained"
            color="error"
            sx={{
              position: "absolute",
              bottom: "20px",
              left: "50%",
              transform: "translateX(-50%)",
              padding: "10px 20px",
              fontSize: "16px",
              borderRadius: "10px",
            }}
            onClick={handleGenerateDreamTeam}
          >
            Generate Dream Team
          </Button>
        </div>

        {/* Team B Card */}
        <div className={styles.teamCard}>
          <div className={styles.teamHeader}>
            <div className={styles.team}>
              <img
                src={carddata?.teamBicon}
                alt="Team B Logo"
                className={styles.teamLogo}
                style={{ backgroundColor: "white" }}
              />
              <span className={styles.teamName}>{carddata?.teams[1]}</span>
            </div>
          </div>
          <Playerlist playerdata={teamBPlayers} />
        </div>
      </div>
    </div>
  );
};

export default MatchDetails;
