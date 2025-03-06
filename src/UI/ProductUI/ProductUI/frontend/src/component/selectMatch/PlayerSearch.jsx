import * as React from "react";
import TextField from "@mui/material/TextField";
import CircularProgress from "@mui/material/CircularProgress";
import Avatar from "@mui/material/Avatar";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import { FixedSizeList as List } from "react-window";
import { BASE_URL } from "../../constants";
import { Close } from "@mui/icons-material";

export default function PlayerSearch({
  teamA,
  teamB,
  onAddToTeam,
  assignedPlayers,
}) {
  const [open, setOpen] = React.useState(false);
  const [options, setOptions] = React.useState([]);
  const [loading, setLoading] = React.useState(false);
  const [buttonLoading, setButtonLoading] = React.useState({});
  const [searchTerm, setSearchTerm] = React.useState("");

  const parsePlayers = (playersData) => {
    return Object.values(playersData).map((player) => ({
      name: player.full_name,
      key: player.key_cricinfo,
      // dreamPoints: Math.floor(Math.random() * 100), // Random points for demo
      type: player.playing_role,
      profileImage: player.img_src_url,
      bgImage: player.bg_image_url,
      player_id: player.player_id,
    }));
  };

  React.useEffect(() => {
    localStorage.removeItem("teamA");
    localStorage.removeItem("teamB");

    const fetchAndStorePlayersData = async () => {
      setLoading(true);
      console.log(teamA, teamB);
      try {
        const responseA = await fetch(
          `${BASE_URL}/player/search_players/${teamA}`
        );
        const dataA = await responseA.json();

        const responseB = await fetch(
          `${BASE_URL}/player/search_players/${teamB}`
        );
        const dataB = await responseB.json();

        if (dataA.status === "ok" && dataB.status === "ok") {
          const playersTeamA = parsePlayers(dataA.players);
          const playersTeamB = parsePlayers(dataB.players);

          localStorage.setItem("teamA", JSON.stringify(playersTeamA));
          localStorage.setItem("teamB", JSON.stringify(playersTeamB));
          console.log(playersTeamA, playersTeamB);

          setOptions([...playersTeamA, ...playersTeamB]);
        }
      } catch (error) {
        console.error("Failed to fetch player data", error);
      } finally {
        setLoading(false);
      }
    };
    fetchAndStorePlayersData();
  }, [teamA, teamB]);

  const handleOpen = () => {
    setOpen(true);
  };

  const handleClear = () => {
    setOpen(false);
    setSearchTerm(""); // Clear search term
  };

  const handleAddToTeam = (player, team) => {
    setButtonLoading((prevState) => ({
      ...prevState,
      [player.key]: { ...prevState[player.key], [team]: true },
    }));

    setTimeout(() => {
      onAddToTeam(player, team);
      setButtonLoading((prevState) => ({
        ...prevState,
        [player.key]: { ...prevState[player.key], [team]: false },
      }));
    }, 1000);
  };

  const filteredOptions = options.filter((option) =>
    option.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const Row = ({ index, style }) => {
    const option = filteredOptions[index];
    const isAssigned = assignedPlayers[option.key] || false;

    return (
      <Box
        component="li"
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          py: 1,
          px: 2,
          borderBottom: "1px solid #ddd",
          ...style,
        }}
      >
        <Grid container spacing={2} alignItems="center">
          <Grid
            item
            xs={2}
            sx={{
              display: "flex",
              justifyContent: "center",
              textWrap: "nowrap",
              color: "#173bb0",
            }}
          >
            <Button
              variant="outlined"
              size="small"
              sx={{ textTransform: "none" }}
              onClick={(event) => {
                event.stopPropagation();
                handleAddToTeam(option, "A");
              }}
              disabled={isAssigned}
            >
              Add to A
            </Button>
          </Grid>
          <Grid
            item
            xs={8}
            sx={{ display: "flex", alignItems: "center", padding: 0 }}
          >
            <Avatar
              src={option.profileImage}
              alt={option.name}
              sx={{ width: 40, height: 40, marginRight: 2 }}
            />
            <Box>
              <div style={{ fontSize: "16px", fontWeight: "bold" }}>
                {option.name}
              </div>
              <div style={{ fontSize: "15px", color: "gray" }}>
                {option.type}
              </div>
              {/* <div style={{ fontSize: "12px", color: "gray" }}>
                Dream Points: {option.dreamPoints}
              </div> */}
            </Box>
          </Grid>
          <Grid
            item
            xs={2}
            sx={{
              display: "flex",
              justifyContent: "flex-end",
              padding: 0,
              textWrap: "nowrap",
              color: "#173bb0",
            }}
          >
            <Button
              variant="outlined"
              size="small"
              sx={{ textTransform: "none" }}
              onClick={(event) => {
                event.stopPropagation();
                handleAddToTeam(option, "B");
              }}
              disabled={isAssigned}
            >
              Add to B
            </Button>
          </Grid>
        </Grid>
      </Box>
    );
  };

  return (
    <Box sx={{ width: "100%" }}>
      <TextField
        label="Search Players"
        placeholder="Search..."
        fullWidth
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)} // Update searchTerm state
        InputProps={{
          endAdornment: (
            <>
              {loading && <CircularProgress color="inherit" size={20} />}
              <Close
                fontSize="small"
                onClick={handleClear}
                style={{ cursor: "pointer" }}
              />
            </>
          ),
        }}
        onClick={handleOpen}
      />
      {open && (
        <List
          height={200}
          itemCount={filteredOptions.length}
          itemSize={80}
          width="100%"
        >
          {Row}
        </List>
      )}
    </Box>
  );
}
