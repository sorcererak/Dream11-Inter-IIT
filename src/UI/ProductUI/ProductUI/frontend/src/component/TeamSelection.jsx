// import * as React from "react";
// import Box from "@mui/material/Box";
// import PlayerSearch from "../component/PlayerSearch";
// import PlayerCard from "./playerCard";

// export default function TeamSelection() {
//   const [teamA, setTeamA] = React.useState([]);
//   const [teamB, setTeamB] = React.useState([]);

//   const handleAddToTeam = (player, team) => {
//     if (team === "A") {
//       setTeamA((prev) => [...prev, player]);
//     } else if (team === "B") {
//       setTeamB((prev) => [...prev, player]);
//     }
//   };

//   const handleRemoveFromTeam = (playerKey, team) => {
//     if (team === "A") {
//       setTeamA((prev) => prev.filter((player) => player.key !== playerKey));
//     } else if (team === "B") {
//       setTeamB((prev) => prev.filter((player) => player.key !== playerKey));
//     }
//   };

//   return (
//     <Box sx={{ display: "flex", flexDirection: "row", gap: 4, p: 2 }}>
//       {/* Search Component */}
//       <Box sx={{ flex: 1 }}>
//         <PlayerSearch onAddToTeam={handleAddToTeam} />
//       </Box>

//       {/* Team A */}
//       <Box sx={{ flex: 1 }}>
//         <h3>Team A</h3>
//         {teamA.map((player) => (
//           <PlayerCard
//             key={player.key}
//             name={player.name}
//             points={player.dreamPoints}
//             bgImage={player.bgImage}
//             profileImage={player.profileImage}
//             isInField={true}
//             onRemove={() => handleRemoveFromTeam(player.key, "A")}
//           />
//         ))}
//       </Box>

//       {/* Team B */}
//       <Box sx={{ flex: 1 }}>
//         <h3>Team B</h3>
//         {teamB.map((player) => (
//           <PlayerCard
//             key={player.key}
//             name={player.name}
//             points={player.dreamPoints}
//             bgImage={player.bgImage}
//             profileImage={player.profileImage}
//             isInField={true}
//             onRemove={() => handleRemoveFromTeam(player.key, "B")}
//           />
//         ))}
//       </Box>
//     </Box>
//   );
// }
