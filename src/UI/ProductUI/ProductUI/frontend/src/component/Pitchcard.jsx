import React from "react";
import styles from "../css/PitchCard.module.css"; 

const PitchCard = ({ pitch }) => {
  return (
    <div className={styles.pitchCard}>
      <h3 className={styles.pitchTitle}>Pitch Details</h3>
      <div className={styles.pitchContent}>
     <p><strong>Pitch Type:</strong> {pitch}</p>
       {/* <p><strong>Condition:</strong> {pitch.condition}</p>
        <p><strong>Location:</strong> {pitch.location}</p>
        <p><strong>Weather:</strong> {pitch.weather}</p>
        <p><strong>Additional Info:</strong> {pitch.additionalInfo}</p> */}
      </div>
    </div>
  );
};

export default PitchCard;
