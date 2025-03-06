import { Radar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";

// Register chart.js components
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const RadarChartComponent = ({ fields }) => {
  // Ensure fields is an object and filter out null or empty values
  if (!fields || typeof fields !== "object") {
    console.error("Invalid fields data:", fields);
    return <p style={{ color: "red" }}>Invalid data provided for chart</p>;
  }

  const radarFields = Object.entries(fields)
    .filter(([_, value]) => value !== null && value !== "") // Filter out null or empty values
    .map(([key, value]) => ({
      label: key,
      value: !isNaN(parseFloat(value)) ? parseFloat(value) : 0, // Ensure valid numbers
    }));

  // If no valid data exists, show a fallback message
  if (radarFields.length === 0) {
    return <p style={{ textAlign: "center", color: "#fff" }}>No valid data available</p>;
  }

  const chartData = {
    labels: radarFields.map((field) => field.label),
    datasets: [
      {
        label: "Performance", // You can modify this label if necessary
        data: radarFields.map((field) => field.value),
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 2,
      },
    ],
  };

  const options = {
    plugins: {
      legend: { display: false }, // Hide legend if not needed
    },
    scales: {
      r: {
        angleLines: { color: "rgba(0, 0, 0, 0.4)" }, // Axis lines visible
        grid: { display: false },
        ticks: { display: false }, // Hide ticks
        suggestedMin: 0,
        suggestedMax: 10, // Set a reasonable max for the radar chart
      },
    },
  };

  return (
    <div
      style={{
        borderRadius: "8px",
        maxWidth: "600px", // Limit chart size for responsiveness
        margin: "auto",
        padding: "20px",
        backgroundColor: "rgba(255, 255, 255, 0.1)", // Optional: add styling
      }}
    >
      <Radar data={chartData} options={options} />
    </div>
  );
};

export default RadarChartComponent;
