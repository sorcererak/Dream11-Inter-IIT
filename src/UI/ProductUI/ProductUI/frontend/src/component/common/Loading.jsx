import React, { useState, useEffect } from "react";
import Lottie from "react-lottie";
import { useSpring, animated } from "react-spring";
import loadingAnimation from "./loadinganimation.json"; 

const Loading = () => {
  const [progress, setProgress] = useState(0);
  const [textIndex, setTextIndex] = useState(0);

  const texts = [
    "Hold tight, something awesome is coming!",
    "Hang on, we're setting things up just for you!",
    "Get ready, we’re almost there!",
    "The magic is happening—almost done!"
  ];
  
  // Lottie animation options
  const lottieOptions = {
    loop: true,
    autoplay: true,
    animationData: loadingAnimation,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice",
    },
  };

  // Linear loader bar animation
  const loaderBarProps = useSpring({
    width: `${progress}%`,
    from: { width: "0%" },
    config: { duration: 1000 },
  });

  // Text animation to change the text every 3 seconds
  useEffect(() => {
    const textInterval = setInterval(() => {
      setTextIndex((prevIndex) => (prevIndex + 1) % texts.length);
    }, 3000);
    return () => clearInterval(textInterval);
  }, []);

  useEffect(() => {
    const progressInterval = setInterval(() => {
      setProgress((prevProgress) => {
        if (prevProgress < 100) {
          return prevProgress + 1;
        } else {
          clearInterval(progressInterval);
          return 100;
        }
      });
    }, 100);
  }, []);

  return (
    <div style={styles.container}>
      <Lottie options={lottieOptions} height={250} width={250} />
      <div style={styles.loaderBarContainer}>
        <animated.div style={{ ...styles.loaderBar, ...loaderBarProps }} />
      </div>
      <animated.div style={styles.text}>{texts[textIndex]}</animated.div>
    </div>
  );
};

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    backgroundColor: "white", 
  },
  loaderBarContainer: {
    width: "60%", 
    height: "5px", 
    backgroundColor: "#e0e0e0",
    marginTop: "20px",
  },
  loaderBar: {
    height: "100%",
    backgroundColor: "#4caf50",
  },
  text: {
    marginTop: "20px",
    fontSize: "18px",
    fontWeight: "bold",
    color: "#333",
  },
};

export default Loading;
