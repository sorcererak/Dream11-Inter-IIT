/* eslint-disable react/prop-types */
// import Fuse from "fuse.js";
import { useState } from "react";
import "./selectTeamCard.css";
import { motion } from "framer-motion";
import { FaTimes } from "react-icons/fa";
import { TextField, InputAdornment } from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";

const allFavTeams = [
  {
    name: "India",
    url: "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/Cricket_India_Crest.svg/800px-Cricket_India_Crest.svg.png",
  },
  {
    name: "Australia",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/h5apsohhlfsn9ydrivoe",
  },
  {
    name: "England",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/pnintojotbm9ll60cczx",
  },
  {
    name: "South Africa",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/kaefiv6vudqhj1vbqq0h",
  },
  {
    name: "New Zealand",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/ftjoqz45fxlacmrej5oc",
  },
  {
    name: "Pakistan",
    url: "https://upload.wikimedia.org/wikipedia/commons/a/ad/Pakistan_cricket_team_logo.png",
  },
  {
    name: "West Indies",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/vwd5gtsu96yeq44kntsv",
  },
  {
    name: "Sri Lanka",
    url: "https://upload.wikimedia.org/wikipedia/en/3/32/Sri_Lanka_Cricket_Logo.jpg",
  },
  {
    name: "Bangladesh",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/jxmhkysemy0gkkmjwmct",
  },
  {
    name: "Afghanistan",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/iwjk0vb7agpa7tyhuzea",
  },
  {
    name: "Ireland",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/j5p9gymd8zwbdt102sta",
  },
  {
    name: "Zimbabwe",
    url: "https://images.icc-cricket.com/image/upload/t_q-best/prd/vz1jpphfrjvbuowwrrtt",
  },
  {
    name: "Scotland",
    url: "https://upload.wikimedia.org/wikipedia/en/thumb/6/64/ScotlandMenCricketLogo.svg/308px-ScotlandMenCricketLogo.svg.png",
  },
  {
    name: "Netherlands",
    url: "https://upload.wikimedia.org/wikipedia/en/thumb/8/86/Logo_of_cricket_Netherlands.png/248px-Logo_of_cricket_Netherlands.png",
  },
  {
    name: "United States of America",
    url: "https://usacricket.org/wp-content/uploads/2018/03/usacricket-logo.svg",
  },
];

const teamIcon = (url, name) => {
  return (
    <div key={name} className="team-icon">
      <img
        alt={name}
        src={url}
        style={{
          height: "80px",
          width: "80px",
          objectFit: "contain",
        }}
      />
      <p>{name}</p>
    </div>
  );
};

const selectedTeamCard = (imageUrl, teamName, onClose) => {
  return (
    <div
      className="selected-team-card"
      style={{
        width: "70%",
        maxHeight: "60vh",
        backgroundImage: `url(${imageUrl})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        overflowY: "hidden",
      }}
    >
      <div className="glass-overlay">
        <div
          onClick={onClose}
          style={{
            position: "absolute",
            top: "10px",
            right: "10px",
            borderRadius: "50%",
            cursor: "pointer",
            overflowY: "hidden",
          }}
        >
          <FaTimes size={20} color="black" />
        </div>

        <img
          alt={teamName}
          src={imageUrl}
          style={{
            height: "auto",
            width: "250px",
            variant: "rectangle",
          }}
        />
        <h2 className="team-name-title">{teamName}</h2>
      </div>
    </div>
  );
};

const TeamSearchCard = ({ setTeam, moveCard, id, remove, allTeams }) => {
  const [searchTeam, setSearchTeam] = useState("");
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [isTeamSet, setIsTeamSet] = useState(false);

  const teams = allTeams.filter((item) => !remove.includes(item.name));
  const favTeams = allFavTeams.filter((item) => !remove.includes(item.name));

  const deselectTeam = () => {
    setTeam("");
    moveCard(false);
    setIsTeamSet(false);
    setSelectedTeam(null);
  };

  const selectTeamIcon = (team) => {
    return (
      <div
        key={team.name}
        onClick={() => {
          setTeam(team);
          moveCard(true);
          setIsTeamSet(true);
          setSelectedTeam(team);
        }}
      >
        {teamIcon(team.url, team.name)}
      </div>
    );
  };

  const searchDiv = () => {
    return (
      <div className="search-card" id={id}>
        <div className="search-bar" data-tour-id="search-team">
          <TextField
            variant="outlined"
            fullWidth
            value={searchTeam}
            aria-label="Search Teams"
            placeholder={remove === "" ? `Select your team` : `${remove} vs`}
            onChange={(e) => setSearchTeam(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
          />
        </div>
        {filteredTeams.length === 0 && searchTeam.length > 0 && (
          <p className="no-teams-error">
            No teams found, here are the featured teams:
          </p>
        )}
        {filteredTeams.length === 0 && searchTeam.length === 0 && (
          <p className="fav-teams-title">Featured Teams</p>
        )}
        <div
          className="teams-grid"
          style={{
            scrollBehavior: "smooth",
            maxHeight: "46vh",
            overflowY: "auto",
          }}
        >
          {filteredTeams.length > 0
            ? filteredTeams.map((team) => selectTeamIcon(team))
            : favTeams.map((team) => selectTeamIcon(team))}
        </div>
      </div>
    );
  };

  const filteredTeams =
    searchTeam.length === 0
      ? []
      : teams.filter((team) =>
        team.name.toLowerCase().includes(searchTeam.toLowerCase())
      );

  return (
    <div>
      {isTeamSet ? (
        <>
          {/* Fade-out Search Div */}
          <motion.div
            key="searchDiv"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
            }}
            initial={{ opacity: 1 }}
            animate={{ opacity: 0, visibility: "hidden", scale: 0.9 }}
            transition={{ duration: 0.5 }}
          >
            {searchDiv()}
          </motion.div>

          {/* Fade-in Selected Team Card */}
          <motion.div
            key="selectedTeamCard"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {selectedTeamCard(
              selectedTeam.url,
              selectedTeam.name,
              deselectTeam
            )}
          </motion.div>
        </>
      ) : (
        <motion.div
          key="searchDiv"
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
          }}
        >
          {searchDiv()}
        </motion.div>
      )}
    </div>
  );
};

export default TeamSearchCard;