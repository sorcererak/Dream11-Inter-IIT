import React, { useState } from "react";
import RadarChartComponent from "../charts/radarChart";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

import {
  Tabs,
  TabsHeader,
  TabsBody,
  Tab,
  TabPanel,
} from "@material-tailwind/react";

import styles from "./accordian.module.css";

const TransparentTabs = React.memo(({ inputData, dataType }) => {
  const data = Object.entries(inputData)
    .map(([type, stats]) => {
      if (dataType == "batting") {
        const baseData = {
          "Total Runs": stats.totalRuns,
          "Total Matches": stats.totalMatches,
          "Highest Score": stats.highestScore,
          "Career Strike Rate": stats.careerStrikeRate,
          "Career Avg": stats.careerAvg,
          "Last 10 Matches Avg": stats.last10Avg,
          "Last 10 Matches Strike Rate": stats.last10StrikeRate,
          "50s": stats.fifty,
          "100s": stats.hundred,
          "4s": stats.fours,
          "6s": stats.sixes,
          Stumpings: stats.stumpings,
          Catches: stats.catches,
        };
        const chartData = {
          Adaptability: stats.adaptability,
          Consistency: stats.battingConsistency,
          Form: stats.battingForm,
          FieldingPerformance: stats.fieldingPerformance,
          "Career Strike Rate": stats.careerStrikeRate,
        };
        const filteredBaseData = Object.fromEntries(
          Object.entries(baseData).filter(([_, value]) => value)
        );
        if (Object.keys(filteredBaseData).length > 0) {
          return {
            label: type.toUpperCase(),
            value: type.toLowerCase(),
            ...filteredBaseData,
            chartData,
          };
        }
        return null;
      } else if (dataType == "bowling") {
        const baseData = {
          "Total Matches": stats.totalMatches,
          "Total Wickets": stats.totalWickets,
          "Career Economy Rate": stats.careerEconomyRate,
          "Career Avg": stats.careerAvg,
          "No of 5 Wicket Hauls": stats.fiveWicketHauls,
          "No of 10 Wicket Hauls": stats.tenWicketHauls,
          "Last 10 Matches Avg": stats.last10Avg,
          "Last 10 Matches Eco": stats.last10EconomyRate,
          "No of Maiden Overs": stats.maidens,
        };
        const chartData = {
          Adaptability: stats.adaptability,
          Consistency: stats.bowlingConsistency,
          Form: stats.bowlingForm,
          FieldingPerformance: stats.fieldingPerformance,
          "Career SR": stats.careerSR,
        };
        const filteredBaseData = Object.fromEntries(
          Object.entries(baseData).filter(([_, value]) => value)
        );
        if (Object.keys(filteredBaseData).length > 0) {
          return {
            label: type.toUpperCase(),
            value: type.toLowerCase(),
            ...filteredBaseData,
            chartData,
          };
        }
        return null;
      }
      return null;
    })
    .filter((item) => item !== null); // Filter out null entries

  if (data.length === 0) {
    return <div>No data available</div>; // Handle case when no valid data exists
  }

  return (
    <Tabs value={data[0]?.value}>
      <TabsHeader
        className={`${styles.tabHeader}`}
        indicatorProps={{
          className: "bg-gray-900/10 shadow-none !text-gray-900",
        }}
      >
        {data.map(({ label, value }) => (
          <Tab key={value} value={value}>
            {label}
          </Tab>
        ))}
      </TabsHeader>
      <TabsBody>
        {data.map(({ label, value, chartData, ...otherStats }) => (
          <TabPanel key={value} value={value}>
            <div className="flex flex-row-reverse overflow-auto">
              {/* Stats Columns */}
              <div className="flex flex-wrap w-1/2">
                {/* Iterate over stats and display them in 3 columns */}
                {Object.entries(otherStats)
                  .reduce((result, [statLabel, statValue], index) => {
                    const chunkIndex = Math.floor(index / 3); // Calculate which chunk this entry belongs to
                    if (!result[chunkIndex]) {
                      result[chunkIndex] = []; // Initialize a new chunk
                    }
                    result[chunkIndex].push([statLabel, statValue.toFixed(2)]); // Add the entry to the appropriate chunk
                    return result;
                  }, [])
                  .map((chunk, chunkIndex) => (
                    <div
                      key={chunkIndex}
                      className="flex space-x-4 w-full mb-4"
                    >
                      {chunk.map(([statLabel, statValue]) => (
                        <div key={statLabel} className="w-1/3">
                          <div className="text-lg font-bold text-left pl-7">
                            {statValue}
                          </div>
                          <div className="text-base text-left pl-8 ">
                            {statLabel}
                          </div>
                        </div>
                      ))}
                    </div>
                  ))}
              </div>

              {/* Radar Chart Component */}
              <div className="w-1/2 flex flex-col  items-center">
                <h5>Player Performance</h5>
                <RadarChartComponent fields={chartData} />
              </div>
            </div>
          </TabPanel>
        ))}
      </TabsBody>
    </Tabs>
  );
});

const PlayerStatsAccordion = ({ playerType, data }) => {
  const [expanded, setExpanded] = useState(playerType); // To handle which accordion is expanded

  const handleAccordionChange = (panel) => {
    setExpanded(expanded === panel ? null : panel); // Toggle between the panels
  };

  return (
    <div className={styles.accordionDiv}>
      {/* Batting Accordion */}
      {data.bat && (
        <Accordion
          expanded={expanded === "batting"} // Expand if 'batting' is the current expanded panel
          onChange={() => handleAccordionChange("batting")} // Toggle batting panel
          className={styles.accordionItem}
        >
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            className={styles.accordionSummary}
          >
            <Typography
              variant="h5" // Adjust the variant to suit your design requirements
              sx={{
                fontWeight: "bold",
                textAlign: "left",
                paddingLeft: "16px", // Padding to align the text properly
                color: "var(--red)", // Adjust the color to match your app's theme (e.g., blue)
                textTransform: "uppercase", // Converts the text to uppercase for emphasis
                letterSpacing: "1px", // Adds some letter spacing for better readability
                textShadow: "1px 1px 3px rgba(0, 0, 0, 0.3)", // Creates a slight shadow for depth
              }}
            >
              Batting Stats
            </Typography>

          </AccordionSummary>
          <AccordionDetails>
            <TransparentTabs inputData={data.bat} dataType="batting" />{" "}
            {/* Send batting data */}
          </AccordionDetails>
        </Accordion>
      )}

      {/* Bowling Accordion */}
      {data.bowl && (
        <Accordion
          expanded={expanded === "bowling"} // Expand if 'bowling' is the current expanded panel
          onChange={() => handleAccordionChange("bowling")} // Toggle bowling panel
          className={styles.accordionItem}
        >
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            className={styles.accordionSummary}
          >
            <Typography
              variant="h5" // Adjust the variant to suit your design requirements
              sx={{
                fontWeight: "bold",
                textAlign: "left",
                paddingLeft: "16px", // Padding to align the text properly
                color: "var(--red)", // Adjust the color to match your app's theme (e.g., blue)
                textTransform: "uppercase", // Converts the text to uppercase for emphasis
                letterSpacing: "1px", // Adds some letter spacing for better readability
                textShadow: "1px 1px 3px rgba(0, 0, 0, 0.3)", // Creates a slight shadow for depth
              }}
            >
              Bowling Stats
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <TransparentTabs inputData={data.bowl} dataType="bowling" />{" "}
            {/* Send bowling data */}
          </AccordionDetails>
        </Accordion>
      )}
    </div>
  );
};

export default PlayerStatsAccordion;
