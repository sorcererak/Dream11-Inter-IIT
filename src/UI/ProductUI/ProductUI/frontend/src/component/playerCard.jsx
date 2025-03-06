/* eslint-disable react/prop-types */
import { useState } from "react";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import Typography from "@mui/material/Typography";
import Avatar from "@mui/material/Avatar";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import ClearIcon from "@mui/icons-material/Clear";
import SportsCricketIcon from "@mui/icons-material/SportsCricket";
import Tooltip from "@mui/material/Tooltip";
import PlayerPopOut from "./playerStats/popUp";

const getWordColor = (word) => {
  // Hardcoded colors for specific words
  const specialColors = {
    "Middle order Batter": "#FF6347", // Tomato Red
    "Top order Batter": "#FFB07C", // Peach
  };

  // Return the hardcoded color if the word matches
  if (specialColors[word]) {
    return specialColors[word];
  }

  // Default color logic for other words
  const colors = [
    "#F05D5E", // Darker Medium Pink
    "#FFB347", // Darker Warm Yellow
    "#4CB0A6", // Darker Teal Green
    "#F2740A", // Darker Vibrant Orange
    "#8A64D1", // Darker Soft Purple
    "#FF618F", // Darker Rose Pink
    "#4AA8C1", // Darker Aqua Blue
    "#6F92D9", // Darker Cornflower Blue
    "#F2D04A", // Darker Soft Gold
    "#B39EB5", // Darker Thistle (Lavender)
  ];
  const hash = word
    .split("")
    .reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return colors[hash % colors.length];
};

export default function PlayerCard({
  name,
  dreamPoints,
  bgImage,
  profileImage,
  onRemove,
  onAddToField,
  type,
  isInField,
  teamIconUrl,
  team,
  matchId,
  player_id,
  description,
  isCap,
  isVcap,
}) {
  const [isVisible, setIsVisible] = useState(false);
  const handleClick = () => {
    setIsVisible(player_id ? !isVisible : false);
  };

  const firstName = name
    ?.toUpperCase()
    ?.split(" ")
    .slice(0, -1)
    .join(" ");
  const lastName = name
    ?.toUpperCase()
    ?.split(" ")
    .slice(-1)
    .join(" ");

  return (
    <div>
      <Card
        sx={{
          width: "20vh",
          minWidth: "100px",
          maxWidth: "200px",
          // height: "20vh",
          // minHeight: "150px",
          position: "relative",
          wordWrap: "break-word",
          fontSize: "10px",
          backgroundColor: "#DBD4D4",
          borderRadius: "10px",
          boxShadow: 3,
          cursor: onRemove ? "pointer" : null,
        }}
        onClick={handleClick}
      >
        <Box sx={{ position: "relative" }}>
          <CardMedia
            sx={{ height: 70, position: "relative" }}
            image={bgImage}
            title={name}
          />
          <Box
            sx={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              backgroundColor: "rgba(0, 0, 0, 0.5)",
            }}
          />
          <Avatar
            alt={name}
            src={profileImage}
            sx={{
              position: "absolute",
              bottom: 10,
              right: 10,
              width: 50,
              height: 50,
              border: "2px solid white",
            }}
          />
          {(isCap || isVcap) && (
            <Box
              sx={{
                position: "absolute",
                bottom: -7,
                right: 1,
                backgroundColor: isCap ? "gold" : "silver", // Dynamic color
                color: "purple",
                fontSize: "0.6rem",
                fontWeight: "bold",
                padding: "2px 6px",
                borderRadius: "5px",
                textTransform: "uppercase",
                boxShadow: 1,
                zIndex: 3,
              }}
            >
              {isCap ? "Captain" : "Vice-Captain"}
            </Box>
          )}
          {isInField && onRemove && (
            <IconButton
              onClick={onRemove}
              sx={{
                margin: 0,
                padding: 0,
                color: "#fff",
                position: "absolute",
                top: "-0.5%",
                left: "-0.5%",
              }}
            >
              <ClearIcon />
            </IconButton>
          )}
        </Box>
        <CardContent sx={{ textAlign: "left" }}>
          <Tooltip title={name} arrow>
            <Typography
              gutterBottom
              variant="h6"
              component="div"
              sx={{
                fontSize: "0.8rem",
                fontWeight: "800",
                // color: "#fff",
                overflow: "hidden",
                whiteSpace: "nowrap",
                textOverflow: "ellipsis",
              }}
            >
              {name}
            </Typography>
          </Tooltip>
          <Box
            sx={{
              display: "flex",
              flexDirection: "row",
              justifyContent: "space-between",
            }}
          >
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                alignItems: "start",
                gap: "10px",
              }}
            >
              {dreamPoints && (
                <Typography
                  variant="p"
                  sx={{
                    // color: "white",
                    fontSize: "0.6rem",
                    lineHeight: "0.2rem",
                  }}
                >
                  Dream Points:
                  <span
                    style={{
                      fontSize: "1rem",
                      lineHeight: "0.2rem",
                      marginLeft: "5px",
                    }}
                  >
                    {dreamPoints.toFixed(0)}
                  </span>
                </Typography>
              )}
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "row",
                  gap: "1px",
                  overflow: "hidden",
                }}
              >
                {type &&
                  // Check if it's a special type
                  ([
                    "Middle order Batter",
                    "Top order Batter",
                    "Opening Batter",
                  ].includes(type) ? (
                    <Typography
                      variant="p"
                      sx={{
                        backgroundColor: getWordColor(type), // Consistent color for the whole string
                        display: "inline-block",
                        padding: 0.5,
                        border: `1px solid ${getWordColor(type)}`,
                        borderRadius: "5px",
                      }}
                    >
                      {type}
                    </Typography>
                  ) : (
                    // For other types, split into words and color individually
                    type.split(" ").map((word, index) => (
                      <Typography
                        key={index}
                        variant="p"
                        sx={{
                          backgroundColor: getWordColor(word), // Color for each word
                          display: "inline", // Ensure words are inline
                          marginRight: 1, // Add spacing between words
                          padding: 0.3,
                          border: `1px solid ${getWordColor(word)}`,
                          borderRadius: "5px",
                        }}
                      >
                        {word}
                      </Typography>
                    ))
                  ))}
              </Box>
            </Box>
            <Tooltip title={"Add player to your squad"}>
              <Box>
                {!isInField && (
                  <IconButton
                    onClick={onAddToField}
                    sx={{ margin: 0, padding: 0, color: "" }}
                  >
                    <SportsCricketIcon />
                  </IconButton>
                )}
              </Box>
            </Tooltip>
          </Box>
        </CardContent>
      </Card>

      {/* Use the PlayerPopOut component */}
      {isVisible && (
        <PlayerPopOut
          isVisible={isVisible}
          onClose={handleClick}
          name={name}
          bgImage={bgImage}
          profileImage={profileImage}
          teamIconUrl={teamIconUrl}
          team={team}
          firstName={firstName}
          lastName={lastName}
          matchId={matchId}
          playerId={player_id}
          description={description}
        />
      )}
    </div>
  );
}
