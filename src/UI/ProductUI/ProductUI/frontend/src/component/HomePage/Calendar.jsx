import dayjs from "dayjs"; // Import Day.js
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { DateCalendar } from "@mui/x-date-pickers/DateCalendar";
import { Box, Typography } from "@mui/material";
import calendar from "../../assets/HomePage/date.svg";
import styles from "../../css/HomePage/Calendar.module.css";

const theme = createTheme({
  typography: {
    fontFamily: "Poppins, Arial, sans-serif", // Use Poppins as primary font
  },
  components: {
    MuiPickersDay: {
      styleOverrides: {
        root: {
          borderRadius: "50%", // Circular shape for selected dates
          padding: "8px", // Consistent padding
          "&:hover": {
            outline: "2px solid var(--calendar)",
            backgroundColor: "rgba(255, 0, 0, 0.1)",
            cursor: "pointer",
          },
          "&.Mui-selected": {
            backgroundColor: "var(--calendar)",
            color: "#ffffff",
            fontWeight: "bold",
            "&:hover": {
              backgroundColor: "var(--calendar)",
            },
          },
          "&.Mui-selected:focus": {
            backgroundColor: "var(--calendar)",
            outline: "none",
          },
        },
      },
    },
    MuiDateCalendar: {
      styleOverrides: {
        root: {
          // padding: "24px", 
          backgroundColor: "white",
          fontFamily: "Poppins, Arial, sans-serif",
          borderRadius: "10px",
          boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
          // transition: "box-shadow 0.3s ease-in-out", // Smooth transition on hover
          // "&:hover": {
          //   boxShadow: "0px 6px 12px rgba(0, 0, 0, 0.3)", // Stronger shadow on hover
          // },
        },
      },
    },
    MuiPickersCalendarHeader: {
      styleOverrides: {
        root: {
          padding: "0 16px",
        },
        switchViewButton: {
          color: "var(--calendar)",
          outline: "none",
        },
        label: {
          fontWeight: "bold",
          color: "var(--calendar)",
        },
      },
    },
    MuiPickersYear: {
      styleOverrides: {
        yearButton: {
          display: "flex",
          alignItems: "center", // Center align the year text
          justifyContent: "center", // Center align the year text
          padding: "6px 12px", // Padding for better spacing
          borderRadius: "8px", // Rounded corners for the year button
          "&.Mui-selected": {
            backgroundColor: "var(--calendar)",
            color: "#ffffff",
            fontWeight: "bold",
            outline: "none",
          },
          "&:hover": {
            outline: "2px solid var(--calendar)", // calendar outline on hover
            backgroundColor: "rgba(255, 0, 0, 0.1)", // Light calendar background on hover
            cursor: "pointer", // Cursor pointer for better UX
          },
          "&.Mui-selected:focus": {
            backgroundColor: "var(--calendar)",
            outline: "none",
          },
        },
      },
    },
  },
});

export default function Calendar({ selectedDate, setSelectedDate, minDate }) {
  const handleDateChange = (newDate) => {
    setSelectedDate(newDate ? newDate.toDate() : null); // Ensure compatibility with Date object
  };

  return (
    <ThemeProvider theme={theme}>
      <LocalizationProvider dateAdapter={AdapterDayjs}>
        <Box className={styles.container}>
          <Box className={styles.heading}>
            <img src={calendar} alt="Calendar Icon" className={styles.icon} />
            <div className={styles.title}>
              Select Date
            </div>
          </Box>
          <DateCalendar
            className={styles.calendar}
            value={dayjs(selectedDate)} // Convert selectedDate to Day.js object
            onChange={handleDateChange}
            minDate={dayjs(minDate)} // Convert minDate to Day.js object
          />
        </Box>
      </LocalizationProvider>
    </ThemeProvider>
  );
}
