import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Joyride from "react-joyride";
import dayjs from "dayjs";
import FeaturedMatches from "../component/HomePage/FeaturedMatches";
import AllMatches from "../component/HomePage/AllMatches";
import HowToPlay from "../component/HomePage/HowToPlay";
import Calendar from "../component/HomePage/Calendar";
import Header from "../component/HomePage/Header";
import Navbar from "../component/Navbar";
import MatchDataFetch from "../component/helper/fetchMatch";
import { BASE_URL } from "../constants";
import styles from "../css/HomePage/HomePage.module.css";
import CustomStyles from "../component/Tourstyles";

const HomePage = () => {
  const [allMatchData, setAllMatchData] = useState([]);
  const [featuredMatchData, setFeaturedMatchData] = useState([]);
  const [selectedDate, setSelectedDate] = useState(new Date(2024, 6, 6));
  const [run, setRun] = useState(false); // Tour running state
  const [stepIndex, setStepIndex] = useState(0); // Current step index
  const [tourCompleted, setTourCompleted] = useState(false); // Track if tour is completed
  const navigate = useNavigate();

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
  const steps = [
    {
      target: '[data-tour-id="matches"]',
      content:
        "Here are the featured matches of the day. You can also browse through all the available matches. Click on a match to view its details.",
      disableBeacon: true,
      placement: "top-right"
    },
    {
      target: '[data-tour-id="calendar"]',
      content:
        "Select the date of the match you want to play. Matches for that day and future matches will be displayed accordingly.",
    },
    {
      target: '[data-tour-id="select-match"]',
      content: "Click here to select a match to create your Dream11 team.",
    },
    {
      target: '[data-tour-id="create-match"]',
      content: "Alternatively, you can create a match with your desired squad.",
    },
    {
      target: '[data-tour-id="/teamSelect"]',
      content:
        "You can also select a match and create your desired Dream11 team from here.",
    },
  ];  

  // Fetch matches data whenever the selected date changes
  useEffect(() => {
    const formattedDate = dayjs(selectedDate).format("YYYY-MM-DD");
    const urlFeature = `${BASE_URL}/match/date/featured?date=${formattedDate}`;
    const urlAll = `${BASE_URL}/match/date/all?date=${formattedDate}`;

    MatchDataFetch(urlFeature, setFeaturedMatchData);
    MatchDataFetch(urlAll, setAllMatchData);
  }, [selectedDate]);

  const handleJoyrideCallback = (data) => {
    const { action, index, type } = data;

    if (type === "step:after") {
      // Handle navigation at the end of the last step
      if (index === steps.length - 1 && action === "next") {
        // console.log("next")
        setRun(false);
        setTourCompleted(true);
        navigate("/teamSelect", { state: { continueTour: true } });
      } else {
        // console.log("index")
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

  const handleStartTour = () => {
    if (!tourCompleted) {
      setTimeout(() => setRun(true), 500); // Small delay to ensure components render
    }
  };

  return (
    <div className={styles.container}>
      <Joyride
      
       locale={{
        skip: "End Tour", 
        last: "Next",  
      }}
        steps={steps}
        run={run}
        stepIndex={stepIndex}
        continuous
        callback={handleJoyrideCallback}
        showSkipButton
        styles={CustomStyles}
        hideBackButton 
      disableScrolling={false}
    
      />
      <Navbar className="navbar" />
      <Header className="header" />
      <div className={styles.nonheader}>
        <div
          className={`${styles.matchlist} featured-matches`}
          data-tour-id="matches"
        >
          <FeaturedMatches matches={featuredMatchData} />
          <AllMatches matches={allMatchData} className="all-matches" />
        </div>

        <div className={`${styles.sidebar}`} data-tour-id="calendar">
          <Calendar
            selectedDate={selectedDate}
            setSelectedDate={setSelectedDate}
            minDate={new Date(2024, 6, 1)}
            className="calendar"
          />
          <HowToPlay startTour={handleStartTour} />
        </div>
      </div>
    </div>
  );
};

export default HomePage;
