import styles from "../../css/HomePage/FeaturedMatches.module.css";
import MatchCard from "./MatchCard";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import PropTypes from "prop-types";
import cricicon from '../../assets/HomePage/cricket.svg';
import NoMatches from "./NoMatches";

const NextArrow = (props) => {
  const { onClick } = props;
  return <div className={styles.nextArrow} onClick={onClick} />;
};

NextArrow.propTypes = {
  onClick: PropTypes.func.isRequired,
};

const PrevArrow = (props) => {
  const { onClick } = props;
  return <div className={styles.prevArrow} onClick={onClick} />;
};

PrevArrow.propTypes = {
  onClick: PropTypes.func.isRequired,
};

const FeaturedMatches = ({ matches }) => {
  // Reorder matches to prioritize teams
  const prioritizedTeams = [
    "India",
    "England",
    "Australia",
    "Pakistan",
    "South Africa",
    "New Zealand",
    "Sri Lanka",
    "West Indies",
    "Bangladesh",
    "Afghanistan",
    "Zimbabwe",
    "Ireland",
    "Scotland",
    "Netherlands",
    "United Arab Emirates",
    "Oman",
    "Nepal",
    "Hong Kong",
    "United States of America",
    "Canada",
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Lucknow Super Giants",
    "Gujarat Titans",
    "Melbourne Stars",
    "Melbourne Renegades",
    "Sydney Sixers",
    "Sydney Thunder",
    "Perth Scorchers",
    "Brisbane Heat",
    "Adelaide Strikers",
    "Hobart Hurricanes",
    "Karachi Kings",
    "Lahore Qalandars",
    "Islamabad United",
    "Multan Sultans",
    "Peshawar Zalmi",
    "Quetta Gladiators",
    "Guyana Amazon Warriors",
    "Trinbago Knight Riders",
    "Barbados Royals",
    "St Kitts and Nevis Patriots",
    "St Lucia Kings"
  ];

  const sortedMatches = matches.slice().sort((a, b) => {
    const aHasPriorityTeam = prioritizedTeams.some(team =>
      a.teams.includes(team)
    );
    const bHasPriorityTeam = prioritizedTeams.some(team =>
      b.teams.includes(team)
    );
    return bHasPriorityTeam - aHasPriorityTeam; // Place matches with priority teams first
  });

  const settings = {
    dots: sortedMatches.length > 1,
    arrows: sortedMatches.length > 1,
    infinite: sortedMatches.length > 1,
    speed: 600,
    slidesToShow: Math.min(3, sortedMatches.length),
    slidesToScroll: 1,
    autoplay: sortedMatches.length > 1,
    autoplaySpeed: 3000,
    pauseOnHover: true,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: Math.min(2, sortedMatches.length),
          slidesToScroll: 1,
        },
      },
      {
        breakpoint: 600,
        settings: {
          slidesToShow: 1,
          slidesToScroll: 1,
        },
      },
    ],
    nextArrow: <NextArrow />,
    prevArrow: <PrevArrow />,
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.heading}>
        <img src={cricicon} alt="Icon" className={styles.cricicon} />
        Featured Matches
      </h2>
      {sortedMatches.length === 0 ? (
        <NoMatches
          message="There are no featured matches"
          customLink="/custommatch"
        />
      ) : (
        <Slider {...settings}>
          {sortedMatches.map((match, index) => (
            <div key={index} >
              <MatchCard match={match} />
            </div>
          ))}
        </Slider>
      )}
    </div>
  );
};


FeaturedMatches.propTypes = {
  matches: PropTypes.arrayOf(
    PropTypes.shape({
      team1Flag: PropTypes.string.isRequired,
      team2Flag: PropTypes.string.isRequired,
      time: PropTypes.string.isRequired,
      series: PropTypes.string.isRequired,
      date: PropTypes.string.isRequired,
      stadium: PropTypes.string.isRequired,
      isFeatured: PropTypes.bool,
    })
  ).isRequired,
};

export default FeaturedMatches;
