import React, { useState, useEffect } from "react";
import Animate from "react-smooth";
import Button from "@mui/material/Button";
import image1 from "../../assets/HomePage/header4.png";
import image2 from "../../assets/HomePage/header2.png";
import image3 from "../../assets/HomePage/header3.png";
import image4 from "../../assets/HomePage/header1.png";
import { useNavigate } from "react-router-dom";

const Header = () => {
  const navigate = useNavigate();
  const showcaseImages = [image1, image2, image3, image4];
  const timeoutTime = 5000;

  const [currentIndex, setCurrentIndex] = useState(0);

  // Timeout for changing the background image
  useEffect(() => {
    const timeout = setTimeout(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % showcaseImages.length);
    }, timeoutTime);

    return () => clearTimeout(timeout); // Cleanup timeout on component unmount
  }, [currentIndex]);

  const handleDotClick = (index) => {
    setCurrentIndex(index);
  };

  const navigateTo = (path) => {
    navigate(path);
  };

  return (
    <header>
      <Animate to="1" from="0.2" attributeName="opacity">
        <div
          style={{
            background: "rgba(0,0,0,1)",
            height: "6vh",
            position: "relative",
          }}
        ></div>

        <div
          style={{
            backgroundImage: `url(${showcaseImages[currentIndex]})`,
            backgroundColor: "black",
            backgroundSize: "cover",
            backgroundPosition: "center",
            backgroundRepeat: "no-repeat",
            height: "69vh",
            width: "100%",
            position: "relative",
          }}
        >
      
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              zIndex: 3,
              backgroundColor: "rgba(0, 0, 0, 0.8)",
              clipPath: "polygon(0% 0%, 50% 100%, 0% 100%)",
              backdropFilter: "blur(10px)",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
              border: "1px solid rgba(255, 255, 255, 0.2)",
            }}
          ></div>

     
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              backgroundColor: "rgba(0, 0, 0, 0.2)",
              zIndex: 1,
            }}
          />

          {/* Text Content */}
          <div
            style={{
              position: "absolute",
              top: "70%",
              left: "2%",
              transform: "translateY(-50%)",
              textAlign: "left",
              color: "white",
              padding: "20px",
              borderRadius: "12px",
              maxWidth: "400px",
              zIndex: "3",
            }}
          >
            <h1
              style={{
                fontSize: "3rem",
                marginBottom: "10px",
                textShadow: "0px 4px 6px rgba(0, 0, 0, 0.6)",
              }}
            >
              Start Playing
            </h1>
            <p
              style={{
                fontSize: "1.2rem",
                marginBottom: "20px",
                lineHeight: "1.6",
                color: "white",
                textShadow: "0px 2px 4px rgba(0, 0, 0, 0.5)",
              }}
            >
              Build your dream team, challenge your friends, and showcase your
              skills. Are you ready to win big?
            </p>
            <div
              style={{
                display: "flex",
                gap: "15px",
                marginTop: "20px",
              }}
            >
              <Button
                variant="contained"
                data-tour-id="create-match" 
                style={{
                  backgroundColor: "var(--red)",
                  color: "var(--bg)",
                  fontWeight: "bold",
                  padding: "0px 12px 0px 12px",
                  fontSize: "1rem",
                  borderRadius: "8px",
                  transition: "transform 0.2s ease, background-color 0.2s ease",
                  boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.3)",
                }}
                onMouseEnter={(e) => (e.target.style.transform = "scale(1.05)")}
                onMouseLeave={(e) => (e.target.style.transform = "scale(1)")}
                onClick={() => navigateTo("/custommatch")}
              >
                Create Match
              </Button>
              <Button
                variant="outlined"
                data-tour-id="select-match" 
                style={{
                  color: "var(--bg)",
                  fontWeight: "bold",
                  padding: "12px 25px",
                  fontSize: "1rem",
                  borderRadius: "8px",
                  transition: "transform 0.2s ease, background-color 0.2s ease",
                  boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.3)",
                  borderColor: "var(--bg)"
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = "scale(1.05)";
                  e.target.style.borderColor = "var(--red)";  // Border color on hover
                }}
                
                onMouseLeave={(e) => {
                  e.target.style.transform = "scale(1)";
                  e.target.style.borderColor = "var(--bg)";  // Revert border color when not hovered
                }}
                onClick={() => navigateTo("/teamSelect")}
              >
                Select Match
              </Button>
            </div>
          </div>

          {/* Navigation Dots */}
          <div
            style={{
              position: "absolute",
              bottom: "20px",
              left: "50%",
              transform: "translateX(-50%)",
              display: "flex",
              gap: "10px",
            }}
          >
            {showcaseImages.map((_, index) => (
              <div
                key={index}
                style={{
                  width: "15px",
                  height: "15px",
                  borderRadius: "50%",
                  backgroundColor: currentIndex === index ? "white" : "gray",
                  cursor: "pointer",
                }}
                onClick={() => handleDotClick(index)}
              />
            ))}
          </div>
        </div>
      </Animate>
    </header>
  );
};

export default Header;
