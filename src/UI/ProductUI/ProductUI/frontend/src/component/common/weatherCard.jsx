import React, { useEffect, useState } from "react";
import { MDBCard, MDBCardBody, MDBIcon, MDBTypography } from "mdb-react-ui-kit";
import { BASE_URL } from "../../constants";
import WhichWeather from "../helper/weatherIdentifier";

export default function WeatherCard({ matchId, place, setEffect, showEffect }) {
  const [weatherData, setWeatherData] = useState(null);
  const [effect, setEffectLocal] = useState(null);
  useEffect(() => {
    if (weatherData) {
      setEffect(weatherData.effect);
      setEffectLocal(weatherData.effect);
    }
  }, [weatherData]);
  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const response = await fetch(`${BASE_URL}/match/weather/${matchId}`);
        if (!response.ok) {
          throw new Error(
            `HTTP Error: ${response.status} - ${response.statusText}`
          );
        }
        const rawData = await response.json();
        console.log("weather", rawData);
        // temp and humidity
        setWeatherData({
          ...WhichWeather(rawData.data[0][0], rawData.data[0][1]),
          temp: rawData.data[0][0].toFixed(1),
          humidity: rawData.data[0][1].toFixed(2),
        });
      } catch (error) {
        console.error("Error fetching weather:", error);
      }
    };
    fetchWeather();
  }, [matchId]);

  return (
    weatherData && (
      <MDBCard
        style={{
          color: "white",
          borderRadius: "10px",
          backgroundColor: "transparent",
        }}
      >
        <MDBCardBody style={{ padding: "10px" }}>
          {/* Header Section */}
          <div className="d-flex justify-content-between">
            <MDBTypography tag="h6" className="flex-grow-1">
              {place}
            </MDBTypography>
          </div>

          {/* Temperature Section */}
          <div className="d-flex mt-1">
            <img
              className="pt-2"
              src={weatherData?.url}
              style={{
                height: "50px",
                width: "50px",
                // backgroundColor: "black",
              }}
              alt="weather-icon"
            />

            <MDBTypography
              tag="h6"
              className="display-5 mb-0 font-weight-bold ml-4 "
              style={{ color: "white", fontSize: "2.3rem" }}
            >
              {weatherData.temp}°C
            </MDBTypography>
          </div>
          <div className="flex flex-row w-full justify-between">
            <span className="small" style={{ color: "#868B94" }}>
              {weatherData.name}
            </span>
            <span className="small">
              <MDBIcon fas icon="tint fa-fw" style={{ color: "#868B94" }} />
              <span className="ms-2">{weatherData.humidity}</span>
            </span>
          </div>
          {showEffect && <h6 style={{ marginTop: "2%" }}>{effect}</h6>}
        </MDBCardBody>
      </MDBCard>
    )
  );
}
