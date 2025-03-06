import PropTypes from "prop-types";
import styles from "../../css/HomePage/NoMatches.module.css";

const NoMatches = ({ message, customLink }) => {
  return (
    <div className={styles.noMatches}>
      <h3>{message}</h3>
      <p>
        <a href={customLink}>Click here to custom match</a>
      </p>
    </div>
  );
};

NoMatches.propTypes = {
  message: PropTypes.string.isRequired,
  customLink: PropTypes.string.isRequired,
};

export default NoMatches;
