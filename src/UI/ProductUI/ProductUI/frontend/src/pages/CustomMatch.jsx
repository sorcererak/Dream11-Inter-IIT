import React, { useState, useEffect } from "react";
import {
  useParams,
  useLocation,
  generatePath,
  useNavigate,
} from "react-router-dom";
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
import ReadOnlyDate from "../component/common/readOnlyDate";
import batsmanimg from "../assets/batsman.png";

export default function CustomMatch() {
  const { date } = useParams();
  const matchDate = new Date(date);
  const { state } = useLocation();
  const { teamAdata = {}, teamBdata = {} } = state || {};
  const navigate = useNavigate();

  if (!teamAdata || !teamBdata) {
    return <p>Error: Missing team data!</p>;
  }

  const [matchType, setMatchType] = useState("");
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

  useEffect(() => {
    localStorage.setItem("selectedteamA", JSON.stringify(teamA));
    localStorage.setItem("selectedteamB", JSON.stringify(teamB));
    localStorage.setItem("assignedPlayers", JSON.stringify(assignedPlayers));
  }, [teamA, teamB, assignedPlayers]);

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

  const generateDreamTeam = () => {
    navigate("/dreamTeam", {
      state: {
        teamA: teamA,
        teamB: teamB,
        match_date: matchDate,
        match_type: matchType,
      },
    });
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

  const handleChange = (event) => {
    setMatchType(event.target.value); // Save the selected value
  };

  const handleCloseAlert = () => {
    setAlert({ ...alert, show: false });
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.1 } },
  };

  const handlePlayersLoaded = (loadedPlayers) => {
    setPlayers(loadedPlayers);
  };

  console.log(teamA);
  console.log(assignedPlayers);

  return (
    <div>
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
              width: { sx: "100%", md: "33%" },
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
            <Typography
              variant="h4"
              sx={{
                fontWeight: "bold",
                textShadow: "2px 2px 4px rgba(0, 0, 0, 0.5)",
                mb: 3,
              }}
            >
              Select Players
            </Typography>
            <ReadOnlyDate value={matchDate} />
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: "min(40px, 9vw)",
                marginBottom: 4,
                paddingTop: "70px",
                flexDirection: "row",
              }}
            >
              {/* Team A */}
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 100,
                  height: 150,
                }}
              >
                <Box
                  sx={{
                    width: 100,
                    height: 100,
                    borderRadius: "50%",
                    background: "white",
                    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.2)",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                  }}
                >
                  <img
                    src={teamAdata.url}
                    alt={teamAdata.name}
                    style={{ width: "80%", height: "80%", objectFit: "cover" }} // Ensures the image fits within the circle
                  />
                </Box>
                <Typography variant="h6" sx={{ marginTop: 1 }}>
                  {teamAdata.name}
                </Typography>
              </Box>

              {/* Team B */}
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 100,
                  height: 150,
                }}
              >
                <Box
                  sx={{
                    width: 100,
                    height: 100,
                    borderRadius: "50%",
                    background: "white",
                    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.2)",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                  }}
                >
                  <img
                    src={teamBdata.url}
                    alt={teamBdata.name}
                    style={{ width: "80%", height: "80%", objectFit: "cover" }} // Ensures the image fits within the circle
                  />
                </Box>
                <Typography variant="h6" sx={{ marginTop: 1 }}>
                  {teamBdata.name}
                </Typography>
              </Box>
            </Box>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Select Match Type
            </Typography>
            <Select
              value={matchType}
              onChange={handleChange}
              defaultValue=""
              variant="outlined"
              sx={{ width: "100%", marginBottom: 3 }}
              required
            >
              <MenuItem value="Test">Test</MenuItem>
              <MenuItem value="ODI">ODI</MenuItem>
              <MenuItem value="T20">T20</MenuItem>
              <MenuItem value="T20">T20I</MenuItem>
              <MenuItem value="Test">MDM</MenuItem>
              <MenuItem value="ODI">ODM</MenuItem>
            </Select>
            <Typography variant="h6" color="#333" sx={{ mb: 1 }}>
              Search for Player
            </Typography>
            <PlayerSearch
              onAddToTeam={handleAddToTeam}
              teamA={teamAdata.name}
              teamB={teamBdata.name}
              assignedPlayers={assignedPlayers}
            />
            <Box sx={{ textAlign: "center", mt: 5 }}>
              <Button
                variant="contained"
                color="success"
                onClick={generateDreamTeam}
              >
                GENERATE TEAM
              </Button>
            </Box>
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
                <Typography variant="h5" align="center" gutterBottom>
                  TEAM A
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
                        name={player.name}
                        points={player.dreamPoints}
                        bgImage={player.bgImage}
                        profileImage={player.profileImage}
                        player_id={player.player_id}
                        isInField={true}
                        onRemove={() => handleRemoveFromTeam(player.key, "A")}

                      />
                    ) : (
                      <div
                        style={{
                          height: "20vh",
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
                <Typography variant="h5" align="center" gutterBottom>
                  TEAM B
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
                        name={player.name}
                        points={player.dreamPoints}
                        bgImage={player.bgImage}
                        profileImage={player.profileImage}
                        isInField={true}
                        onRemove={() => handleRemoveFromTeam(player.key, "B")}
                      />
                    ) : (
                      <div
                        style={{
                          height: "20vh",
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
            </Grid>
          </Box>
        </Box>
      </Box>
    </div>
  );
}
