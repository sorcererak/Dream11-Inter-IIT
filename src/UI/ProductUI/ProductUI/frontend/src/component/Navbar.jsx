import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import Button from "@mui/material/Button";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import FormControl from "@mui/material/FormControl";
import logo from "../assets/landing_page/dream11logo.svg";
import dayjs from "dayjs";
import MenuIcon from "@mui/icons-material/Menu";
import CloseIcon from "@mui/icons-material/Close";

const Navbar = () => {
  const [language, setLanguage] = useState("hi-IN");
  const [navColor, setNavColor] = useState("rgba(255, 255, 255, 0.1)");
  const [isMobile, setIsMobile] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const location = useLocation();
  const todayDate = dayjs().format("YYYY-MM-DD");

  // Load the name from local storage when the component mounts
  useEffect(() => {
    const savedLang = localStorage.getItem("lang");
    if (savedLang) {
      setLanguage(savedLang);
    }
  }, []);

  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
    localStorage.setItem("lang", event.target.value);
  };

  useEffect(() => {
    if (location.pathname === "/") {
      setNavColor("rgba(255, 255, 255, 0.1)");
    } else {
      setNavColor("rgba(31, 31, 31, 0.9)");
    }
  }, [location.pathname]);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 900);
    handleResize(); // Set initial state
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const links = [
    { path: "/teamSelect", label: "Select Match" },
    { path: `/custommatch`, label: "Create Match" },
    { path: "/home", label: "Home" },
  ];

  return (
    <>
      <nav
        style={{
          display: "flex",
          width: "100vw",
          position: "fixed",
          zIndex: "3000",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 20px",
          background: navColor,
          boxShadow: "0 4px 8px 0 rgba(31, 38, 135, 0.37)",
          backdropFilter: "blur(11px)",
          WebkitBackdropFilter: "blur(11px)",
          borderRadius: "0 0 15px 15px", // Apply radius only to the bottom corners
          border: "1px solid rgba(255, 255, 255, 0.18)",
          transition: "background-color 0.3s ease-in-out",
        }}
      >
        <div>
          <Link to="/">
            <img
              alt="logo"
              src={logo}
              style={{ width: "140px", height: "35px" }}
            />
          </Link>
        </div>

        {!isMobile ? (
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <p style={{ color: "red" }}>Your AI Voice</p>
            <FormControl size="small">
              <Select
                value={language}
                onChange={handleLanguageChange}
                style={{
                  color: "red", // Text color of the Select
                  minWidth: "120px",
                  backgroundColor: "black", // Background color of the Select
                  fontSize: "14px", // Font size for the Select
                  borderRadius: "6px", // Border radius
                  border: "1px solid red", // Red border color
                  paddingRight: "20px", // Ensure there's enough space for the icon
                  outline: "none", // Remove the blue outline when focused
                }}
                MenuProps={{
                  PaperProps: {
                    style: {
                      backgroundColor: "black", // Set background of the dropdown menu to black
                      color: "red", // Set text color inside the menu to red
                    },
                  },
                }}
                IconComponent={() => (
                  <span
                    style={{
                      color: "red", // Red icon color
                      fontSize: "12px", // Optional: adjust the size of the icon
                      marginRight: "0px", // Adjust margin on the right of the icon
                    }}
                  >
                    ▼
                  </span>
                )}
              >
                <MenuItem
                  value="en-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  English
                </MenuItem>
                <MenuItem
                  value="hi-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  हिन्दी
                </MenuItem>
                <MenuItem
                  value="bn-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  বাংলা
                </MenuItem>
                <MenuItem
                  value="kn-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  ಕನ್ನಡ
                </MenuItem>
                <MenuItem
                  value="ml-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  മലയാളം
                </MenuItem>
                <MenuItem
                  value="mr-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  मराठी
                </MenuItem>
                <MenuItem
                  value="od-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  ଓଡ଼ିଆ
                </MenuItem>
                <MenuItem
                  value="pa-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  ਪੰਜਾਬੀ
                </MenuItem>
                <MenuItem
                  value="ta-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  தமிழ்
                </MenuItem>
                <MenuItem
                  value="te-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  తెలుగు
                </MenuItem>
                <MenuItem
                  value="gu-IN"
                  style={{ color: "red", backgroundColor: "black" }}
                >
                  ગુજરાતી
                </MenuItem>
              </Select>
            </FormControl>

            {links.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                style={{ textDecoration: "none" }}
                data-tour-id={link.path}
              >
                <Button
                  variant="contained"
                  style={{
                    backgroundColor:
                      location.pathname === link.path
                        ? "var(--red)"
                        : "var(--dark-red)", // Reverse the background color
                    color: location.pathname === link.path ? "black" : "white", // Reverse text color
                    textTransform: "none",
                    // fontWeight: "bold",
                    boxShadow:
                      location.pathname === link.path
                        ? "0px 4px 10px rgba(0,0,0,0.3)"
                        : "none",
                    transition: "all 0.3s ease",
                    fontSize: "14px", // Reduce font size
                    padding: "6px 12px", // Reduce padding to make the button smaller
                    borderRadius: "6px", // Optional: reduce border radius to make the button sharper
                    fontFamily: "Poppins, sans-serif", // Set font to Poppins
                  }}
                >
                  {link.label}
                </Button>
              </Link>
            ))}
          </div>
        ) : (
          <div onClick={toggleSidebar} style={{ cursor: "pointer" }}>
            <MenuIcon style={{ color: "white", fontSize: "32px" }} />
          </div>
        )}
      </nav>

      {isMobile && isSidebarOpen && (
        <div
          style={{
            position: "fixed",
            top: 0,
            right: 0, // Sidebar opens from the right
            width: "75%",
            height: "100%",
            background: "rgba(31, 31, 31, 0.95)",
            boxShadow: "-2px 0 5px rgba(0, 0, 0, 0.5)", // Shadow on the left of the sidebar
            zIndex: 3000,
            padding: "20px",
            display: "flex",
            flexDirection: "column",
            gap: "20px",
            transform: isSidebarOpen ? "translateX(0)" : "translateX(100%)", // Smooth transition
            transition: "transform 0.3s ease-in-out", // Smooth transition effect
          }}
        >
          <div
            onClick={toggleSidebar}
            style={{ cursor: "pointer", alignSelf: "flex-start" }}
          >
            <CloseIcon style={{ color: "white", fontSize: "32px" }} />
          </div>

          <FormControl size="small" style={{ marginBottom: "20px" }}>
            <Select
              value={language}
              onChange={handleLanguageChange}
              style={{
                backgroundColor: "white",
                borderRadius: "8px",
              }}
            >
              <MenuItem value="en">English</MenuItem>
              <MenuItem value="hi">हिन्दी</MenuItem>
              <MenuItem value="fr">Français</MenuItem>
              <MenuItem value="es">Español</MenuItem>
            </Select>
          </FormControl>

          {links.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              onClick={toggleSidebar}
              style={{
                textDecoration: "none",
                color: "white",
                fontSize: "18px",
                fontWeight: location.pathname === link.path ? "bold" : "normal",
              }}
            >
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </>
  );
};

export default Navbar;
