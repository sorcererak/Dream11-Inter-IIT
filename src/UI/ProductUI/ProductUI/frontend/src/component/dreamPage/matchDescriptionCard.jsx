/* eslint-disable react/prop-types */
import { useState, useRef, useEffect, CSSProperties } from "react";
import { Tooltip } from "@mui/material";
import {
  BsFillVolumeUpFill,
  BsArrowsExpand,
  BsArrowsCollapse,
} from "react-icons/bs";
import {
  FaUndo,
  FaShareAlt,
  FaSave,
  FaInfoCircle,
  FaHome,
} from "react-icons/fa";
import styles from "../../css/DescriptionCard.module.css";
import { BASE_URL } from "../../constants";
import ClipLoader from "react-spinners/ClipLoader";
import { useNavigate } from "react-router-dom";

export default function DescriptionCard({
  match_id,
  onUndo,
  onMatchDetails,
  onSave,
  onSS,
  info,
  expanded,
  handleExpandToggle,
  reason,
}) {
  const [audioUrl, setAudioUrl] = useState(null);
  const audioRef = useRef(null);
  const [audioLoading, setAudioLoading] = useState(false);

  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.play().catch((error) => {
        console.error("Error playing audio:", error);
      });
    }
  }, [audioUrl]);

  const infoRef = useRef(null);

  const navigate = useNavigate();
  const handleMatchDetailsroute = () => {
    navigate(`/matchdetails/${match_id}`);
  };

  const fetchAudio = async () => {
    setAudioLoading(true); // Show the loader while fetching
    const text = document.getElementById("infoSectionText").textContent;
    console.log("Text to be converted to audio:", text);
    try {
      const body = {
        target_language_code: localStorage.getItem("lang") || "hi-IN",
        message: text,
      };
      console.log("Body:", body);
      const response = await fetch(`${BASE_URL}/ai/audio`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error("Failed to fetch audio");
      }
      // Create a blob URL from the audio data
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      // Play audio after URL is set
      if (audioRef.current) {
        audioRef.current.src = url; // Set the audio source
        await audioRef.current.play(); // Play the audio
      }
    } catch (err) {
      console.log("Error in fetching audio:", err);
    } finally {
      setAudioLoading(false); // Hide the loader
    }
  };

  const handleVoiceClick = async () => {
    console.log("Voice button clicked");
    await fetchAudio(); // Ensure fetchAudio is awaited
  };

  // // Effect to run whenever 'expanded' changes
  // useEffect(() => {
  //   console.log("The expanded state has changed:", expanded);
  //   // Additional logic when expanded changes can go here
  // }, [expanded]); // Dependency array ensures this effect runs on 'expanded' change

  return (
    <div className={`${styles.descriptionCard}`}>
      <audio ref={audioRef} style={{ display: "none" }} />
      <div className={styles.bgBlur}>
        <div className={styles.cardHeader}>
          <p>INFO SECTION</p>
          <div className="flex flex-row justify-end">
            <Tooltip title="Listen" placement="top">
              <button className={styles.iconButton} onClick={handleVoiceClick}>
                {audioLoading ? (
                  <ClipLoader
                    color={"white"}
                    className={styles.loader}
                    loading={audioLoading}
                    aria-label="Loading Spinner"
                    data-testid="loader"
                  />
                ) : (
                  <BsFillVolumeUpFill />
                )}
              </button>
            </Tooltip>
            <Tooltip title={expanded ? "Minimize" : "Expand"} placement="top">
              <button
                className={styles.iconButton}
                onClick={handleExpandToggle}
              >
                {expanded ? <BsArrowsCollapse /> : <BsArrowsExpand />}{" "}
              </button>
            </Tooltip>
          </div>
        </div>

        {/* Info Section */}
        <div
          ref={infoRef}
          className={`${styles.infoSection} ${expanded ? styles.expanded : ""}`}
          id="infoSectionText"
        >
          {match_id && (
            <div>
              <h3>Weather Effect</h3>
              <p>{info}</p>
            </div>
          )}
          <div>
            <h3>Why this team??</h3>
            <p>{reason}</p>
          </div>
        </div>
      </div>
      <div className={styles.buttonRow}>
        <Tooltip title="Undo" placement="top">
          <button className={styles.actionButton} onClick={onUndo}>
            <FaUndo />
          </button>
        </Tooltip>
        <Tooltip title="Share" placement="top">
          <button className={styles.actionButton} onClick={onSS}>
            <FaShareAlt />
          </button>
        </Tooltip>
        {match_id && (
          <Tooltip title="Match Details" placement="top">
            <button
              className={styles.actionButton}
              onClick={handleMatchDetailsroute}
            >
              <FaInfoCircle />
            </button>
          </Tooltip>
        )}
        <Tooltip title="Save" placement="top">
          <button className={styles.actionButton} onClick={onSave}>
            <FaSave />
          </button>
        </Tooltip>
        <Tooltip title="Home" placement="top">
          <button
            className={styles.actionButton}
            onClick={() => navigate("/home", { replace: true })}
          >
            <FaHome />
          </button>
        </Tooltip>
      </div>
    </div>
  );
}
