import styles from "../../css/cardStack.module.css";

const ProgressBar = ({ currentStep, steps }) => {
  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <section className={styles.progressContainer}>
      <div className={styles.barmarkerContainer}>
        {steps.map((step, i) => (
          <div
            key={i}
            className={`barmarker ${i <= currentStep ? "active" : ""}`}
          >
            {/* Step Icon */}
            <div className={styles.iconContainer}>{step.icon}</div>
            {/* Step Name */}
            <span>{step.label}</span>
          </div>
        ))}
      </div>
      <progress
        className={styles.progress}
        max="100"
        value={progress}
      ></progress>
    </section>
  );
};

export default ProgressBar;
