import React, { useEffect, useState } from "react";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import ListItemAvatar from "@mui/material/ListItemAvatar";
import Avatar from "@mui/material/Avatar";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import { AnimatePresence, motion } from "framer-motion";

// Animation transition settings
const transition = { type: "spring", stiffness: 500, damping: 30 };

export default function AlignItemsList({ playerdata }) {
  const [items, setItems] = useState([]);

  // Add items one by one with a delay
  useEffect(() => {
    // Clear the list before adding new items
    setItems([]);

    let timeouts = [];
    playerdata.forEach((item, index) => {
      const timeout = setTimeout(() => {
        setItems((prev) => [...prev, item]); // Add item sequentially
      }, index * 500); // Delay each item by 500ms
      timeouts.push(timeout);
    });

    return () => {
      // Clear timeouts on component unmount
      timeouts.forEach((timeout) => clearTimeout(timeout));
    };
  }, [playerdata]);

  return (
    <List sx={{ width: "100%", maxWidth: 360, bgcolor: "background.paper" }}>
      <AnimatePresence>
        {items.map((item, index) => (
          <motion.div
            key={item.player_id} // Ensure unique key for each player
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={transition}
          >
            <ListItem alignItems="flex-start">
              <ListItemAvatar>
                <Avatar alt={item.full_name} src={item.img_src_url} />
              </ListItemAvatar>
              <ListItemText
                primary={item.full_name} // Use full_name for player name
                secondary={
                  <React.Fragment>
                    <Typography
                      component="span"
                      variant="body2"
                      sx={{ color: "text.primary", display: "inline" }}
                    >
                      {item.playing_role}
                    </Typography>
                  </React.Fragment>
                }
              />
            </ListItem>
            {index < items.length - 1 && (
              <Divider variant="inset" component="li" />
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </List>
  );
}
