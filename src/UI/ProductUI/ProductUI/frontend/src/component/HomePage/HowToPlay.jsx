import styles from "../../css/HomePage/HowToPlay.module.css";


const HowToPlay = ( { startTour }) => {

  return (
    <div className={styles.container} onClick={startTour} style={{ cursor: "pointer" }}>
      <h2>How To Play</h2>
      <p>Explore our guides to master Dream11 and enhance your experience.</p>
    </div>
  );
};

export default HowToPlay;
