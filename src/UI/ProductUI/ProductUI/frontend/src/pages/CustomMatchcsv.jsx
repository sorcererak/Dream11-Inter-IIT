import React, { useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import {
  Grid,
  Card,
  Typography,
  Button,
  Box,
  MenuItem,
  Select,
  Alert,
} from "@mui/material";
import { motion, AnimatePresence } from "framer-motion";
import PlayerSearch from "../component/selectMatch/PlayerSearch";
import PlayerCard from "../component/playerCard";
import Navbar from "../component/Navbar";
import ImportCSV from "../component/ImportCSV";
import batsmanimg from "../assets/batsman.png";
import { useEffect } from "react";
import Joyride from "react-joyride";
import CustomStyles from "../component/Tourstyles";
import ReadOnlyDate from "../component/common/readOnlyDate";

export default function CustomMatch() {
  const { state } = useLocation();
  const { teamAdata = {}, teamBdata = {}, matchDate } = state || {};
  const [teamAInfo, setTeamAInfo] = useState({});
  const [teamBInfo, setTeamBInfo] = useState({});
  const [isDisabled, setIsDisabled] = useState(true);
  const [selectedDate, setSelectedDate] = useState(matchDate);
  const [selectedMatchType, setSelectedMatchType] = useState("");

  const [teamA, setTeamA] = useState(() => {
    const savedTeamA = localStorage.getItem("selectedteamA");
    return savedTeamA ? JSON.parse(savedTeamA) : Array(11).fill(null);
  });

  const [teamB, setTeamB] = useState(() => {
    const savedTeamB = localStorage.getItem("selectedteamB");
    return savedTeamB ? JSON.parse(savedTeamB) : Array(11).fill(null);
  });

  const [assignedPlayers, setAssignedPlayers] = React.useState(() => {
    const savedAssignedPlayers = localStorage.getItem("assignedPlayers");
    return savedAssignedPlayers ? JSON.parse(savedAssignedPlayers) : {};
  });

  const [alert, setAlert] = useState({
    message: "",
    severity: "",
    show: false,
  });

  const navigate = useNavigate();

  //....tour....
  const [tourCompleted, setTourCompleted] = useState(false);
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const stepstour = [

    // {
    //   target: '[data-tour-id="match-type"]',
    //   content: "In the custom match option , you can select the type of match you want to play.",
    //   disableBeacon: true,
    // }
    //,

    {
      target: '[data-tour-id="player-search"]',
      content: "To select the 22 players for Team A and Team B, you can directly import them from a CSV file.",
      disableBeacon: true,
    },
    {
      target: '[data-tour-id="view-players"]',
      content: "You can view the 22 players in this section after selecting them from the dropdown or importing from a CSV file.",
    },
    {
      target: '[data-tour-id="generate-team"]',
      content: "Finally, generate your dream team by clicking on this button!",
    },    
  ];

  useEffect(() => {
    if (state?.continueTour && !tourCompleted) {
      setRun(true);
      console.log("state", state.continueTour);
      // Clear the state after starting the tour
      navigate(location.pathname, { replace: true });
      console.log("state", state.continueTour);
    }
  }, [state, location.pathname, navigate]);

  function hasNullElement(team) {
    return team.some((element) => element === null);
  }

  useEffect(() => {
    if (hasNullElement(teamA) || hasNullElement(teamB)) {
      setIsDisabled(true);
    } else {
      setIsDisabled(false);
    }
  }, [teamA, teamB]);

  useEffect(() => {
    localStorage.removeItem("positions");
    localStorage.removeItem("offFieldPlayers");
    localStorage.removeItem("modelOutput");
    localStorage.removeItem("dreamPoints");
    localStorage.removeItem("assignedPlayers");
    localStorage.removeItem("selectedteamA");
    localStorage.removeItem("selectedteamB");
    localStorage.removeItem("teamA");
    localStorage.removeItem("teamB");
    localStorage.removeItem("savedID");
  }, []);

  const handleJoyrideCallback = (data) => {
    const { action, index, type } = data;

    if (type === "step:after") {
      // Handle navigation at the end of the last step
      if (index === stepstour.length - 1 && action === "next") {
        console.log("next");
        setRun(false);
        setTourCompleted(true);
      } else {
        console.log("index", index);
        setStepIndex(index + 1);
      }
    }

    // Handle "skip" action
    if (type === "tour:end" && action === "skip") {
      setRun(false);
      setStepIndex(0);
    }

    // End of the tour
    if (type === "tour:end" && action !== "skip") {
      setRun(false);
      setStepIndex(0);
      setTourCompleted(true);
    }
  };

  //......tour.....

  // redirect to dream team page
  const generateDreamTeam = () => {
    navigate("/dreamTeam", {
      state: {
        teamA: teamA,
        teamB: teamB,
        match_date: selectedDate,
        match_type: selectedMatchType,
      },
    });
  };

  const handleAddToTeam = (player, team) => {
    const isPlayerInTeamA = teamA.some((p) => p && p.key === player.key);
    const isPlayerInTeamB = teamB.some((p) => p && p.key === player.key);

    if (isPlayerInTeamA || isPlayerInTeamB) {
      setAlert({
        message: "This player is already in a team!",
        severity: "info",
        show: true,
      });
      return;
    }

    const updateTeamState = (prevTeam) => {
      const emptyIndex = prevTeam.findIndex((p) => p === null);
      if (emptyIndex !== -1) {
        const newTeam = [...prevTeam];
        newTeam[emptyIndex] = player;
        return newTeam;
      } else {
        setAlert({
          message: `Team ${team} is already full!`,
          severity: "info",
          show: true,
        });
        return prevTeam;
      }
    };

    if (team === "A") {
      setTeamA((prev) => updateTeamState(prev));
    } else {
      setTeamB((prev) => updateTeamState(prev));
    }

    // Mark player as assigned
    if (
      (team == "A" && teamA.filter((player) => player !== null).length < 11) ||
      (team == "B" && teamB.filter((player) => player !== null).length < 11)
    ) {
      setAssignedPlayers((prev) => ({ ...prev, [player.key]: true }));
    }
  };

  const handleRemoveFromTeam = (playerKey, team) => {
    const updateTeam = team === "A" ? [...teamA] : [...teamB];
    const playerIndex = updateTeam.findIndex((p) => p && p.key === playerKey);

    if (playerIndex !== -1) {
      updateTeam[playerIndex] = null;
      team === "A" ? setTeamA(updateTeam) : setTeamB(updateTeam);

      setAssignedPlayers((prev) => ({ ...prev, [playerKey]: false }));
    }
  };

  const handleCloseAlert = () => {
    setAlert({ ...alert, show: false });
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.1 } },
  };

  const mapping = (player) => {
    return {
      name: player.full_name,
      key: player.key_cricinfo,
      type: player.playing_role,
      profileImage: player.img_src_url,
      bgImage: player.bg_image_url,
      player_id: player.player_id,
    };
  };

  const handlePlayersLoaded = (response) => {
    if (response.status === "ok") {
      console.log("response", response);
      const { teamA, teamB } = response;
      const teamAInfo = response.team_info.teamA;
      const teamBInfo = response.team_info.teamB;
      setTeamA(teamA.map((player) => mapping(player)));
      setTeamB(teamB.map((player) => mapping(player)));
      setTeamAInfo(teamAInfo);
      setTeamBInfo(teamBInfo);
      setSelectedDate(response.match_date);
      setSelectedMatchType(response.match_type);
      console.log("date", response.match_date);
      console.log("type", response.match_type);
      console.log("teamAinfo", teamAInfo);
      console.log("teamBinfo", teamBInfo);
      console.log(
        "teamAPlayers",
        teamA.map((player) => mapping(player))
      );
      console.log(
        "teamBPlayers",
        teamB.map((player) => mapping(player))
      );
    } else {
      setAlert({
        message: response.message || "Failed to load players.",
        severity: "error",
        show: true,
      });
    }
  };

  return (
    <div>
      {run && (
        <Joyride
          locale={{
            skip: "End Tour",
            last: "Finish",
          }}
          steps={stepstour}
          run={run}
          stepIndex={stepIndex}
          continuous
          callback={handleJoyrideCallback}
          showSkipButton
          styles={CustomStyles}
          hideBackButton
          disableScrolling={false}
        />
      )}
      <Navbar />

      <Box
        sx={{
          position: "relative",
          minHeight: "100vh",
          background: "var(--bg)",
        }}
      >
        <Box
          sx={{
            position: "absolute",
            bottom: 0,
            right: -50,
            width: "50vh",
            height: "50vh",
            backgroundImage: `url(${batsmanimg})`,
            backgroundSize: "contain",
            backgroundRepeat: "no-repeat",
            zIndex: 0,
            opacity: 0.3,
          }}
        />
        <Box
          sx={{
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            padding: { xs: "2 2 2 0", md: 4 },
            gap: { xs: 2, md: 4 },
          }}
        >
          {/* Left Section */}
          <Box
            sx={{
              width: { xs: "100%", md: "45%" },
              background: "rgba(255, 255, 255, 0.5)",
              boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.37)",
              backdropFilter: "blur(10px)",
              borderRadius: "10px",
              border: "1px solid rgba(255, 255, 255, 0.18)",
              marginTop: "30px",
              padding: 2,
              display: "flex",
              flexDirection: "column",
              justifyContent: "flex-start",
              alignItems: "center",
            }}
          >
            {/* Create Match Text */}
            <Typography
              variant="h4"
              sx={{
                fontWeight: "bold",
                textShadow: "2px 2px 4px rgba(0, 0, 0, 0.5)",
                mb: 3,
                marginBottom: "45px",
              }}
            >
              Create Match
            </Typography>


            {/* Team Logos and Names */}
            {(teamAInfo?.url || teamAdata?.url) && (
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  gap: 2,
                  marginBottom: 4,
                }}
              >
                {/* Team A */}
                <Box
                  sx={{
                    height: 100,
                    background: "transparent",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    alignItems: "center",
                    border: "none",
                  }}
                >
                  <img
                    src={teamAInfo?.url || teamAdata?.url}
                    alt={teamAInfo?.name || teamAdata?.name}
                    style={{
                      maxHeight: 110,
                      maxWidth: 110,
                      marginBottom: "10px",
                      // borderRadius: "50%",
                      // boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.2)",
                      objectFit: "contain",
                    }}
                  />
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: "bold",
                      // textShadow: "1px 1px 3px rgba(0, 0, 0, 0.6)",
                      color: "black",
                      fontSize: { xs: "1rem", sm: "1.2rem", md: "1.5rem" },
                      margin: "0",
                      padding: "0",
                      marginBottom: "10px",
                    }}
                  >
                    {teamAInfo?.name || teamAdata?.name}
                  </Typography>
                </Box>

                {/* VS */}
                <Typography
                  variant="h5"
                  sx={{
                    color: "var(--primary)",
                    fontWeight: "bold",
                    fontSize: { xs: "1.5rem", sm: "2rem", md: "2.5rem" },
                    // textShadow: "1px 1px 3px rgba(0, 0, 0, 0.4)",
                  }}
                >
                  VS
                </Typography>

                {/* Team B */}
                <Box
                  sx={{
                    height: 100,
                    background: "transparent",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    alignItems: "center",
                    border: "none",
                  }}
                >
                  <img
                    src={teamBInfo?.url || teamBdata?.url}
                    alt={teamBInfo?.name || teamBdata?.name}
                    style={{
                      maxHeight: 110,
                      maxWidth: 110,
                      marginBottom: "10px",
                      objectFit: "contain",
                      // borderRadius: "50%",
                      // boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.2)",
                    }}
                  />
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: "bold",
                      // textShadow: "1px 1px 3px rgba(0, 0, 0, 0.6)",
                      color: "#000",
                      marginBottom: "10px",
                      fontSize: { xs: "1rem", sm: "1.2rem", md: "1.5rem" },
                    }}
                  >
                    {teamBInfo?.name || teamBdata?.name}
                  </Typography>
                </Box>
              </Box>
            )}

            {matchDate && (
              <div
                style={{
                  boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.2)",
                  borderRadius: "10px",
                  width: "100%",
                  padding: "3% 3% 5%",
                }}
              >
                <ReadOnlyDate value={matchDate} />
                <div data-tour-id="match-type" style={{ width: "100%" }}>
                  <Select
                    value={selectedMatchType}
                    variant="outlined"
                    displayEmpty // Ensures the placeholder is displayed when value is empty
                    onChange={(e) => setSelectedMatchType(e.target.value)}
                    sx={{ width: "100%", marginBottom: 3 }}
                  >
                    <MenuItem value="" disabled>
                      Select Match Type
                    </MenuItem>
                    <MenuItem value="T20">IT20</MenuItem>
                    <MenuItem value="Test">Test</MenuItem>
                    <MenuItem value="ODI">ODI</MenuItem>
                    <MenuItem value="T20">T20</MenuItem>
                    <MenuItem value="Test">MDM</MenuItem>
                    <MenuItem value="ODI">ODM</MenuItem>
                  </Select>
                </div>

                <div style={{ width: "100%" }}>
                  <PlayerSearch
                    teamA={teamAInfo?.name || teamAdata?.name}
                    teamB={teamBInfo?.name || teamBdata?.name}
                    onAddToTeam={handleAddToTeam}
                    assignedPlayers={assignedPlayers}
                  />
                </div>
              </div>
            )}
            {matchDate && <Typography sx={{ mt: 2, mb: 2 }}>OR</Typography>}

            {/* Import CSV */}
            <div data-tour-id="player-search"
              style={{
                boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.2)",
                borderRadius: "10px",
                width: "100%",
                padding: "3% 3% 5%",
              }}
            >
              {isDisabled && teamAInfo?.name && (
                <PlayerSearch
                  teamA={teamAInfo?.name}
                  teamB={teamBInfo?.name}
                  onAddToTeam={handleAddToTeam}
                  assignedPlayers={assignedPlayers}
                />
              )}
              <ImportCSV onPlayersLoaded={handlePlayersLoaded} />
            </div>

            <div data-tour-id="generate-team" style={{ width: "100%" }}>
              <Box sx={{ textAlign: "center", mt: 2, width: "100%" }}>
                <Button
                  variant="contained"
                  color="error"
                  disabled={isDisabled}
                  onClick={generateDreamTeam}
                  style={{ width: "50%" }}
                >
                  GENERATE TEAM
                </Button>
              </Box>
            </div>
          </Box>

          {/* Right Section */}
          <Box sx={{ width: "100%", marginTop: "64px", zIndex: "2" }}>
            {alert.show && (
              <Alert severity={alert.severity} onClose={handleCloseAlert}>
                {alert.message}
              </Alert>
            )}
            <Grid container spacing={2}>
              {/* Team A */}
              <Grid item xs={12}>
                <Typography variant="h5" align="center" gutterBottom sx={{
                  fontWeight: "bold",
                  // textShadow: "1px 1px 3px rgba(0, 0, 0, 0.6)",
                  color: "#000",
                  fontSize: { xs: "1rem", sm: "1.2rem", md: "1.5rem" },
                }}>
                  {teamAInfo.name}
                </Typography>
              </Grid>
              {teamA.map((player, index) => (
                <Grid item xs={6} sm={4} md={2} key={`teamA-${index}`}>
                  <motion.div
                    variants={cardVariants}
                    initial="hidden"
                    animate="visible"
                  >
                    {player ? (
                      <PlayerCard
                        name={player?.full_name || player?.name}
                        points={player?.key_cricinfo || player?.dreamPoints}
                        bgImage={player?.bg_image_url || player?.bgImage}
                        profileImage={
                          player?.img_src_url || player?.profileImage
                        }
                        type={player.type}
                        isInField={true}
                        onRemove={() => handleRemoveFromTeam(player.key, "A")}
                      />
                    ) : (
                      <div
                        data-tour-id="view-players"
                        style={{
                          height: "15vh",
                          width: "100%",
                          display: "flex",
                          justifyContent: "center",
                          alignItems: "center",
                          background: "transparent",
                          boxShadow: "0 8px 32px 0 rgba( 31, 38, 135, 0.37 )",
                          borderRadius: "10px",
                          border: "1px solid rgba( 255, 255, 255, 0.18 )",
                        }}
                      ></div>
                    )}
                  </motion.div>
                </Grid>
              ))}

              {/* Team B */}
              <Grid item xs={12}>
                <Typography variant="h5" align="center" gutterBottom sx={{
                  fontWeight: "bold",
                  // textShadow: "1px 1px 3px rgba(0, 0, 0, 0.6)",
                  color: "#000",
                  fontSize: { xs: "1rem", sm: "1.2rem", md: "1.5rem" },
                }}>
                  {teamBInfo.name}
                </Typography>
              </Grid>
              {teamB.map((player, index) => (
                <Grid item xs={6} sm={4} md={2} key={`teamB-${index}`}>
                  <motion.div
                    variants={cardVariants}
                    initial="hidden"
                    animate="visible"
                  >
                    {player ? (
                      <PlayerCard
                        name={player?.full_name || player?.name}
                        points={player?.key_cricinfo || player?.dreamPoints}
                        bgImage={player?.bg_image_url || player?.bgImage}
                        profileImage={
                          player?.img_src_url || player?.profileImage
                        }
                        isInField={true}
                        type={player.type}
                        onRemove={() => handleRemoveFromTeam(player.key, "B")}
                      />
                    ) : (
                      <div
                        style={{
                          height: "15vh",
                          display: "flex",
                          justifyContent: "center",
                          alignItems: "center",
                          background: "transparent",
                          boxShadow: "0 8px 32px 0 rgba( 31, 38, 135, 0.37 )",
                          borderRadius: "10px",
                          border: "1px solid rgba( 255, 255, 255, 0.18 )",
                        }}
                      ></div>
                    )}
                  </motion.div>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Box>
      </Box>
    </div>
  );
}
