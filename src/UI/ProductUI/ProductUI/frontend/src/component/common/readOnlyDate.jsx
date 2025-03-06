import React from "react";
import { TextField, styled } from "@mui/material";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import dayjs from "dayjs";

// Styled TextField with custom styles
const StyledTextField = styled(TextField)(({ theme }) => ({
  "& .MuiInputBase-root": {
    backgroundColor: theme.palette.background.default,
    borderRadius: "8px",
    "&.Mui-disabled": {
      color: theme.palette.text.primary, 
      WebkitTextFillColor: theme.palette.text.primary, 
    },
  },
  "& .MuiOutlinedInput-notchedOutline": {
    borderColor: theme.palette.divider,
  },
}));

const ReadOnlyDate = ({ value }) => {
  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <StyledTextField
        value={dayjs(value).format("YYYY-MM-DD")} 
        InputProps={{
          readOnly: true,
        }}
        fullWidth
        variant="outlined"
        sx={{
          input: {
            textAlign: "center", 
            fontWeight: "bold",
            fontSize: "1rem", 
          },
        }}
      />
    </LocalizationProvider>
  );
};

export default ReadOnlyDate;
