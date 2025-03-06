import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import styles from "../../css/cardStack.module.css";
import TeamSearchCard from "./selectTeamCard.jsx";
import { GiAmericanFootballHelmet } from "react-icons/gi";
import { BiCricketBall } from "react-icons/bi";
import { useNavigate } from "react-router-dom";
import Loading from "../common/Loading.jsx";
import ProgressBar from "./progressBar.jsx";
import SelectMatchCard from "./selectMatchCard.jsx";
import { BASE_URL } from "../../constants.jsx";
import Navbar from "../Navbar.jsx";
import Joyride from "react-joyride";
import CustomStyles from "../Tourstyles.jsx";
const CardStack = () => {
  const [firstCardMoved, setFirstCardMoved] = useState(false);
  const [secondCardMoved, setSecondCardMoved] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [firstTeam, setFirstTeam] = useState(null);
  const [secondTeam, setSecondTeam] = useState(null);

  const [allTeams, setallTeams] = useState(null);
  const navigate = useNavigate();
  //....tour....
  const { state } = useLocation();
  const [tourCompleted, setTourCompleted] = useState(false);
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const stepstour = [
    {
      target: '[data-tour-id="search-team"]',
      content: "Search for the teams you would like to play a match with and select them.",
      disableBeacon: true,
    },
    {
      target: '[data-tour-id="scrolling-calendar"]',
      content: "Select a date to play a match. If a match is scheduled, choose to continue with your current Dream11 squad or customize it. If no match is scheduled, create your own by selecting 22 players.",
      placement: "left"
    },
  ];  
  useEffect(() => {
    if (state?.continueTour && !tourCompleted) {
      setRun(true);
console.log("state", state.continueTour)
    // Clear the state after starting the tour
    navigate(location.pathname, { replace: true }); 
    console.log("state", state.continueTour)
    }
  }, [state, location.pathname, navigate]);
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
  const handleStartTour = () => {
    if (!tourCompleted) {
      setTimeout(() => setRun(true), 500); // Small delay to ensure components render
    }
  };

  const handleJoyrideCallback = (data) => {
    const { action, index, type } = data;

    if (type === "step:after") {
      // Handle navigation at the end of the last step
      if (index === stepstour.length - 1 && action === "next") {
      // console.log("next")
        setRun(false);
        setTourCompleted(true);
        navigate("/dreamTeam/1426757", { state: { continueTour: true } });
      } else {
        console.log("index", index);
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

  //......tour.....

  useEffect(() => {
    const dataFetch = async () => {
      try {
        const response = await fetch(`${BASE_URL}/team/teams/`, {
          method: "GET",
        });
        const data = await response.json();
        setallTeams(data.data);
      } catch (error) {
        alert("We encountered an issue. Please try again later.");
        console.error("Error fetching teams:", error);
        navigate("/home");
      }
    };

    dataFetch();
  }, []);

  const steps = [
    {
      label: "Team 1",
      icon: <GiAmericanFootballHelmet className={styles.progressBarIcons} />,
    },
    {
      label: "Team 2",
      icon: <GiAmericanFootballHelmet className={styles.progressBarIcons} />,
    },
    {
      label: "Match",
      icon: <BiCricketBall className={styles.progressBarIcons} />,
    },
  ];

  useEffect(() => {
    secondCardMoved && firstCardMoved
      ? setCurrentStep(1)
      : secondCardMoved || firstCardMoved
      ? setCurrentStep(0)
      : setCurrentStep(-1);
  }, [firstTeam, secondTeam]);

  return allTeams == null ? (
    <Loading />
  ) : (
    <div>
      {run && (<Joyride
       locale={{
        skip: "End Tour", 
        last: "Next",  
      }}
        steps={stepstour}
        run={run}
        stepIndex={stepIndex}
        continuous
        callback={handleJoyrideCallback}
        showSkipButton
        styles={CustomStyles}
        hideBackButton
        disableScrolling={false}
      />
      )}
      <Navbar />
      {/* <div className={styles.backgroundCover}></div> */}
      <motion.div
        className={`${styles.cardStackContainer} hide-scrollbar`}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        {/* Progress Bar */}
        <div className={styles.progressBarWrapper}>
          <ProgressBar currentStep={currentStep} steps={steps} />
        </div>
        {/* <p className={styles.find}>
          Let's find a Match
        </p> */}
        <div className={`${styles.cardContainer} hide-scrollbar`}>
          <motion.div className={styles.cardStack}>
            {firstCardMoved && secondCardMoved && (
              <SelectMatchCard teamA={firstTeam} teamB={secondTeam} />
            )}
          </motion.div>
          {/* Card 2 */}
          <motion.div
            className={styles.cardStack}
            data-tour-id="scrolling-calendar"
            style={{
              boxShadow: secondCardMoved
                ? "none"
                : "0 10px 20px rgba(0, 0, 0, 0.2)",
            }}
            animate={{
              x: secondCardMoved
                ? `${Math.max(Math.max(window.innerWidth * 0.35, 200), 500)}px`
                : "0",
              zIndex: secondCardMoved ? 2 : 1,
              // width: secondCardMoved ? "20%" : "45%", // Adjust width dynamically
              backgroundColor: secondCardMoved ? "transparent" : "white",
            }}
            transition={{
              duration: 0.6,
              ease: "easeInOut",
            }}
          >
            <TeamSearchCard
              setTeam={setSecondTeam}
              moveCard={setSecondCardMoved}
              id="team-2-search-card"
              remove={[firstTeam?.name]}
              allTeams={allTeams}
            />
          </motion.div>

          {/* Card 1 */}
          <motion.div
            className={styles.cardStack}
            data-tour-id="search-team"
            style={{
              boxShadow: firstCardMoved
                ? "none"
                : "0 10px 20px rgba(0, 0, 0, 0.2)",
            }}
            animate={{
              x: firstCardMoved
                ? `-${Math.max(Math.max(window.innerWidth * 0.35, 200), 500)}px`
                : "0", // Move left by 50% of viewport width
              zIndex: firstCardMoved ? 2 : 1,
              // width: firstCardMoved ? "20%" : "45", // Adjust width dynamically
              backgroundColor: firstCardMoved ? "transparent" : "white",
            }}
            transition={{
              duration: 0.6,
              ease: "easeInOut",
            }}
          >
            <TeamSearchCard
              setTeam={setFirstTeam}
              moveCard={setFirstCardMoved}
              id="team-1-search-card"
              remove={""}
              allTeams={allTeams}
            />
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default CardStack;
