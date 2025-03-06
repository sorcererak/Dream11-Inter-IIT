import React, { useState, useEffect } from "react";
import { ScrollMenu } from "react-horizontal-scrolling-menu";
import MatchCard from "./MatchCard.jsx";
import "react-horizontal-scrolling-menu/dist/styles.css";
import cricketicon from "../assets/cricket.png";
import { VisibilityContext } from "react-horizontal-scrolling-menu";
import { FaChevronLeft, FaChevronRight } from "react-icons/fa";
import { BASE_URL } from "../constants.jsx";
import MatchDataFeatch from "../component/helper/fetchMatch.jsx";
const Arrow = ({ text, className }) => <div className={className}>{text}</div>;

const LeftArrow = () => {
  const { isFirstItemVisible, scrollPrev } = React.useContext(
    VisibilityContext
  );

  return (
    <button
      disabled={isFirstItemVisible}
      onClick={() => scrollPrev()}
      className="left-arrow"
      style={{
        background: "none",
        border: "none",
        outline: "none",
        cursor: isFirstItemVisible ? "not-allowed" : "pointer",
        opacity: isFirstItemVisible ? 0.5 : 1,
      }}
    >
      <FaChevronLeft size={24} color={isFirstItemVisible ? "gray" : "black"} />
    </button>
  );
};

const RightArrow = () => {
  const { isLastItemVisible, scrollNext } = React.useContext(VisibilityContext);

  return (
    <button
      disabled={isLastItemVisible}
      onClick={() => scrollNext()}
      className="right-arrow"
      style={{
        background: "none",
        border: "none",
        outline: "none",
        cursor: isLastItemVisible ? "not-allowed" : "pointer",
        opacity: isLastItemVisible ? 0.5 : 1,
      }}
    >
      <FaChevronRight size={24} color={isLastItemVisible ? "gray" : "black"} />
    </button>
  );
};

const List = ({ heading, apiCall, date }) => {
  const [matches, setMatchesData] = useState(null);

  useEffect(() => {
    const formattedDate = date.toISOString().split("T")[0];
    const url = `${BASE_URL}/match/date/${apiCall}?date=${formattedDate}`;
    MatchDataFeatch(url, setMatchesData);
  }, [date]);

  const redirectMatchDetails = (match_id) => {
    console.log(`redirect to match details of match id ${match_id}`);
  };

  return (
    <div className="lists" style={{ width: "70vw" }}>
      <img
        src={cricketicon}
        alt="cricket Icon"
        style={{
          marginLeft: "20px",
          marginBottom: "-50px",
          width: "40px",
          height: "40px",
        }}
      />
      <h2 style={{ marginLeft: "70px" }}>{heading}</h2>
      <br />
      <div className="scrollcontainer">
        <ScrollMenu LeftArrow={LeftArrow} RightArrow={RightArrow}>
          {matches &&
            matches.map((match) => (
              <div key={match.id} className="menu-item">
                <MatchCard
                  match={match}
                  onClick={() => {
                    redirectMatchDetails(match.id);
                  }}
                />
              </div>
            ))}
        </ScrollMenu>
      </div>
    </div>
  );
};

export default List;
